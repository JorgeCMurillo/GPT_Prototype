"""
Memory-mapped row-packed loader for BOS-aligned GPT pretraining.

Input format expectations:
- uint16 `.bin` shards named `{split}_*.bin`
- each row has exactly (seq_len + 1) tokens
- every row starts with BOS

Batching:
- take B contiguous rows
- x = rows[:, :-1]
- y = rows[:, 1:]
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


# -----------------------------
# Helpers

def _natural_sort(paths: List[str]) -> List[str]:
    return sorted(paths)


def _load_meta(data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def _list_shards(data_dir: str, split: str) -> List[str]:
    pat = os.path.join(data_dir, f"{split}_*.bin")
    shards = _natural_sort(glob.glob(pat))
    if not shards:
        raise FileNotFoundError(f"No shards found for split='{split}' with pattern: {pat}")
    return shards


def _memmap_uint16(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def _count_tokens_in_shard(path: str) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte-size not divisible by 2 (uint16): {path}")
    return nbytes // 2


def _split_work(total: int, worker_id: int, num_workers: int) -> Tuple[int, int]:
    base = total // num_workers
    rem = total % num_workers
    start = worker_id * base + min(worker_id, rem)
    end = start + base + (1 if worker_id < rem else 0)
    return start, end


def _get_dist_info() -> Tuple[int, int]:
    rank = os.environ.get("RANK", None)
    world = os.environ.get("WORLD_SIZE", None)

    if rank is None:
        rank = os.environ.get("ACCELERATE_PROCESS_INDEX", None)
    if world is None:
        world = os.environ.get("ACCELERATE_NUM_PROCESSES", None)
    if world is None:
        world = os.environ.get("ACCELERATE_PROCESS_COUNT", None)

    try:
        r = int(rank) if rank is not None else 0
    except ValueError:
        r = 0
    try:
        w = int(world) if world is not None else 1
    except ValueError:
        w = 1

    return max(0, r), max(1, w)


# -----------------------------
# Core dataset

@dataclass(frozen=True)
class BOSRowDatasetConfig:
    data_dir: str
    split: str = "train"
    batch_size: int = 16
    seq_len: int = 1024

    shuffle_blocks: bool = True
    seed: int = 1337
    max_blocks: Optional[int] = None
    shard_by_rank: bool = True
    return_meta: bool = False


BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
]


class MemmapBOSRowDataset(IterableDataset):
    """IterableDataset yielding (x, y) or (x, y, meta) with x,y shape (B, T)."""

    def __init__(self, cfg: BOSRowDatasetConfig):
        super().__init__()
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if cfg.seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.cfg = cfg
        self.meta = _load_meta(cfg.data_dir)
        self.shards = _list_shards(cfg.data_dir, cfg.split)

        self.row_tokens = int(cfg.seq_len + 1)
        meta_seq_len = self.meta.get("seq_len", None)
        meta_row_tokens = self.meta.get("row_tokens", None)
        if meta_seq_len is not None and int(meta_seq_len) != int(cfg.seq_len):
            raise ValueError(
                f"seq_len mismatch: loader seq_len={cfg.seq_len}, dataset meta seq_len={meta_seq_len}. "
                "Rebuild dataset or run with matching seq_len."
            )
        if meta_row_tokens is not None and int(meta_row_tokens) != int(self.row_tokens):
            raise ValueError(
                f"row_tokens mismatch: loader expects {self.row_tokens}, dataset meta has {meta_row_tokens}."
            )

        self.shard_sizes = [_count_tokens_in_shard(p) for p in self.shards]
        for path, n_tokens in zip(self.shards, self.shard_sizes):
            if n_tokens % self.row_tokens != 0:
                raise ValueError(
                    f"Shard token count ({n_tokens}) not divisible by row_tokens ({self.row_tokens}): {path}"
                )

        self.rows_per_shard = [n // self.row_tokens for n in self.shard_sizes]
        self.block_rows = cfg.batch_size
        self.block_tokens = self.block_rows * self.row_tokens
        self.blocks_per_shard = [n_rows // self.block_rows for n_rows in self.rows_per_shard]
        self.total_blocks = sum(self.blocks_per_shard)

        if self.total_blocks == 0:
            raise ValueError(
                f"No valid blocks: need shards with at least {self.block_rows} rows. "
                f"Got rows_per_shard={self.rows_per_shard[:5]}"
            )

    def __iter__(self) -> Iterator[BatchType]:
        cfg = self.cfg
        worker = get_worker_info()

        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        rank, world_size = _get_dist_info()
        shard_by_rank = cfg.shard_by_rank and world_size > 1

        share_shards = False
        if shard_by_rank:
            if len(self.shards) >= world_size:
                rank_shards = [i for i in range(len(self.shards)) if (i % world_size) == rank]
            else:
                rank_shards = list(range(len(self.shards)))
                share_shards = True
        else:
            rank_shards = list(range(len(self.shards)))

        if not rank_shards:
            return iter(())

        ws, we = _split_work(len(rank_shards), worker_id, num_workers)
        shard_indices = rank_shards[ws:we]
        if not shard_indices:
            return iter(())

        rng = np.random.default_rng(cfg.seed + 1000 * rank + worker_id)

        local_shards = [
            (
                i,
                self.shards[i],
                _memmap_uint16(self.shards[i]),
                self.shard_sizes[i],
                self.rows_per_shard[i],
                self.blocks_per_shard[i],
            )
            for i in shard_indices
        ]

        schedule: List[Tuple[int, int]] = []
        for local_pos, (_, _, _, _, _, n_blocks) in enumerate(local_shards):
            if n_blocks <= 0:
                continue
            if share_shards:
                blocks = range(rank, n_blocks, world_size)
            else:
                blocks = range(n_blocks)
            schedule.extend((local_pos, bidx) for bidx in blocks)

        if not schedule:
            return iter(())

        blocks_yielded = 0
        while True:
            if cfg.shuffle_blocks:
                rng.shuffle(schedule)

            for (local_pos, bidx) in schedule:
                global_i, path, mm, _, _, _ = local_shards[local_pos]
                start_row = bidx * self.block_rows
                end_row = start_row + self.block_rows
                start = start_row * self.row_tokens
                end = end_row * self.row_tokens

                chunk = mm[start:end]
                if chunk.size != self.block_tokens:
                    continue

                rows = torch.from_numpy(np.asarray(chunk, dtype=np.int64)).reshape(
                    cfg.batch_size,
                    self.row_tokens,
                )
                x = rows[:, :-1]
                y = rows[:, 1:]

                if cfg.return_meta:
                    meta: Dict[str, Any] = {
                        "split": cfg.split,
                        "batch_size": cfg.batch_size,
                        "seq_len": cfg.seq_len,
                        "row_tokens": self.row_tokens,
                        "block_tokens": self.block_tokens,
                        "shard_idx": int(global_i),
                        "shard_path": path,
                        "block_idx": int(bidx),
                        "start": int(start),
                        "end": int(end),
                        "row_start": int(start_row),
                        "row_end": int(end_row),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "worker_id": int(worker_id),
                        "num_workers": int(num_workers),
                        "share_shards": bool(share_shards),
                    }
                    yield x, y, meta
                else:
                    yield x, y

                blocks_yielded += 1
                if cfg.max_blocks is not None and blocks_yielded >= cfg.max_blocks:
                    return


# -----------------------------
# Public factory

def make_bos_row_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    seq_len: int = 1024,
    shuffle_blocks: bool = True,
    seed: int = 1337,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    max_blocks: Optional[int] = None,
    shard_by_rank: bool = True,
    return_meta: bool = False,
) -> DataLoader:
    cfg = BOSRowDatasetConfig(
        data_dir=data_dir,
        split=split,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle_blocks=shuffle_blocks,
        seed=seed,
        max_blocks=max_blocks,
        shard_by_rank=shard_by_rank,
        return_meta=return_meta,
    )
    ds = MemmapBOSRowDataset(cfg)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )


# Alias for convenience when porting scripts.
make_dataloader = make_bos_row_dataloader


if __name__ == "__main__":
    dl = make_bos_row_dataloader(
        data_dir="/home/jorge/tokenPred/moonshotGPT/fineweb_edu_10B",
        split="train",
        batch_size=4,
        seq_len=1024,
        shuffle_blocks=True,
        seed=123,
        num_workers=0,
        max_blocks=2,
        shard_by_rank=True,
        return_meta=True,
    )
    for batch in dl:
        if len(batch) == 2:
            x, y = batch
            print(x.shape, y.shape)
        else:
            x, y, meta = batch
            print(x.shape, y.shape, meta["shard_idx"], meta["start"], meta["end"])
