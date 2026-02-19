
"""
Memory-mapped token shard loader for GPT-style pretraining.

Designed for raw uint16 `.bin` shards produced by your pretokenizer:
  - Each shard is a 1D stream of token ids (no header), dtype=uint16.
  - Each *document* in the stream is prepended with BOS (= GPT-2 EOT token).

This loader builds batches the "llm.c vibe" way:
  - Take a contiguous chunk of length (B*T + 1) tokens from the stream
  - x = chunk[:-1].reshape(B, T)
  - y = chunk[1: ].reshape(B, T)
  - Then advance by stride = B*T (streaming) or pick the next block start from a shuffled list.

Key features:
  - Memory-mapped shards (no full RAM load)
  - PyTorch IterableDataset (fast, simple)
  - Safe with DataLoader(num_workers>0): each worker opens its own memmaps
  - Optional block-level shuffle (shuffle the order of non-overlapping blocks)
  - Default seq_len=1024, configurable
  - Rank/world_size shard partitioning (torchrun / accelerate)
    - Each rank gets a disjoint subset of shards when possible
    - If shards < world_size, ranks share shards but take disjoint blocks
  - Optional meta yield (for replay / exposure tracking)

Usage:
  from shard_loader import make_dataloader

  # default: yields (x, y)
  train_dl = make_dataloader(...)

  # meta: yields (x, y, meta)
  train_dl = make_dataloader(..., return_meta=True)
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


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
    """
    Return [start, end) range for worker_id among num_workers over total items.
    Even-ish partition. Deterministic.
    """
    base = total // num_workers
    rem = total % num_workers
    start = worker_id * base + min(worker_id, rem)
    end = start + base + (1 if worker_id < rem else 0)
    return start, end


def _get_dist_info() -> Tuple[int, int]:
    """
    Best-effort (rank, world_size) from env vars for torchrun or accelerate.
    Defaults to (0, 1).
    """
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
class ShardDatasetConfig:
    data_dir: str
    split: str = "train"
    batch_size: int = 16
    seq_len: int = 1024

    shuffle_blocks: bool = True
    seed: int = 1337

    stride_tokens: Optional[int] = None
    max_blocks: Optional[int] = None

    allow_cross_shard: bool = False
    shard_by_rank: bool = True

    # NEW: yield meta for replay/debug
    return_meta: bool = False


BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
]


class MemmapTokenShardDataset(IterableDataset):
    """
    IterableDataset yielding (x, y) batches of shape (B, T).
    If cfg.return_meta=True, yields (x, y, meta).
    """

    def __init__(self, cfg: ShardDatasetConfig):
        super().__init__()
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if cfg.seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.cfg = cfg
        self.meta = _load_meta(cfg.data_dir)
        self.shards = _list_shards(cfg.data_dir, cfg.split)

        self.shard_sizes = [_count_tokens_in_shard(p) for p in self.shards]

        self.block_tokens = cfg.batch_size * cfg.seq_len + 1
        self.stride = cfg.stride_tokens if cfg.stride_tokens is not None else cfg.batch_size * cfg.seq_len
        if self.stride <= 0:
            raise ValueError("stride_tokens must be > 0")

        self.blocks_per_shard = [
            max(0, 1 + (n - self.block_tokens) // self.stride) if n >= self.block_tokens else 0
            for n in self.shard_sizes
        ]
        self.total_blocks = sum(self.blocks_per_shard)
        if self.total_blocks == 0:
            raise ValueError(
                f"No valid blocks: need shards with at least {self.block_tokens} tokens. "
                f"Got shard sizes: {self.shard_sizes[:5]}{'...' if len(self.shard_sizes)>5 else ''}"
            )

        if cfg.allow_cross_shard:
            raise NotImplementedError("allow_cross_shard=True not supported in this minimal loader.")

    def __iter__(self) -> Iterator[BatchType]:
        cfg = self.cfg
        worker = get_worker_info()

        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        # Distributed rank partitioning
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

        # Split this rank's shard list across dataloader workers
        ws, we = _split_work(len(rank_shards), worker_id, num_workers)
        shard_indices = rank_shards[ws:we]
        if not shard_indices:
            return iter(())

        # RNG per (rank, worker)
        rng = np.random.default_rng(cfg.seed + 1000 * rank + worker_id)

        # Open memmaps
        local_shards = [
            (i, self.shards[i], _memmap_uint16(self.shards[i]), self.shard_sizes[i], self.blocks_per_shard[i])
            for i in shard_indices
        ]

        # Build schedule: (local_pos, block_idx_within_shard)
        schedule: List[Tuple[int, int]] = []
        for local_pos, (global_i, path, mm, n_tokens, n_blocks) in enumerate(local_shards):
            if n_blocks <= 0:
                continue

            if share_shards:
                blocks = range(rank, n_blocks, world_size)
            else:
                blocks = range(n_blocks)

            schedule.extend([(local_pos, b) for b in blocks])

        if not schedule:
            return iter(())

        blocks_yielded = 0
        while True:
            if cfg.shuffle_blocks:
                rng.shuffle(schedule)

            for (local_pos, bidx) in schedule:
                global_i, path, mm, n_tokens, n_blocks = local_shards[local_pos]
                start = bidx * self.stride
                end = start + self.block_tokens
                if end > n_tokens:
                    continue

                chunk = mm[start:end]  # uint16 view length B*T+1
                chunk_t = torch.from_numpy(np.asarray(chunk, dtype=np.int64))

                x = chunk_t[:-1].reshape(cfg.batch_size, cfg.seq_len)
                y = chunk_t[1:].reshape(cfg.batch_size, cfg.seq_len)

                if cfg.return_meta:
                    meta: Dict[str, Any] = {
                        "split": cfg.split,
                        "batch_size": cfg.batch_size,
                        "seq_len": cfg.seq_len,
                        "stride_tokens": self.stride,
                        "block_tokens": self.block_tokens,
                        "shard_idx": int(global_i),
                        "shard_path": path,
                        "block_idx": int(bidx),
                        "start": int(start),
                        "end": int(end),
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
            # infinite if max_blocks is None


# -----------------------------
# Convenience: DataLoader factory

def make_dataloader(
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
    return_meta: bool = False,   # NEW
) -> DataLoader:
    """
    Create a DataLoader that yields (x, y) or (x, y, meta).

    Notes:
    - This DataLoader returns already-batched tensors (batch_size is part of dataset).
      So DataLoader(batch_size=...) is not used; keep it at default.
    - shard_by_rank partitions shards/blocks across ranks using env vars.
    - return_meta adds a small dict per batch for replay/debug.
    """
    cfg = ShardDatasetConfig(
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
    ds = MemmapTokenShardDataset(cfg)

    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )
    return dl


if __name__ == "__main__":
    dl = make_dataloader(
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
