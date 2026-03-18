"""Finite row manifests and per-row datasets for BOS-packed shards.

This module turns the BOS row-packed binary shards into a deterministic,
indexable view of training rows. It defines the stable global row IDs, the
mapping back to shard-local offsets, and the finite dataset and batching
helpers that the TRAK backend uses when featurizing exposed training rows.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_shards(data_dir: Path, split: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(str(data_dir / f"{split}_*.bin")))
    if not paths:
        raise FileNotFoundError(f"No shards found for split={split!r} under {data_dir}")
    return paths


def _count_tokens(path: Path) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte size is not divisible by 2: {path}")
    return nbytes // 2


@dataclass(frozen=True)
class RowRef:
    global_row_id: int
    shard_idx: int
    shard_path: str
    local_row_idx: int
    token_offset_start: int
    token_offset_end: int


@dataclass(frozen=True)
class PreparedRowBatch:
    batch: tuple[torch.Tensor, torch.Tensor]
    local_inds: np.ndarray
    row_ids: tuple[int, ...]
    row_refs: tuple[RowRef, ...]


@dataclass(frozen=True)
class RowManifest:
    data_dir: Path
    split: str
    seq_len: int
    row_tokens: int
    shard_paths: tuple[Path, ...]
    rows_per_shard: tuple[int, ...]
    shard_row_offsets: tuple[int, ...]
    rows: tuple[RowRef, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def row_ref(self, global_row_id: int) -> RowRef:
        return self.rows[int(global_row_id)]


def build_row_manifest(data_dir: str | Path, split: str = "train") -> RowManifest:
    data_path = Path(data_dir).expanduser().resolve()
    meta = _load_meta(data_path)
    row_tokens = int(meta["row_tokens"])
    seq_len = int(meta["seq_len"])
    shard_paths = tuple(_list_shards(data_path, split))

    rows_per_shard: list[int] = []
    shard_row_offsets: list[int] = []
    row_refs: list[RowRef] = []
    next_row_id = 0

    for shard_idx, shard_path in enumerate(shard_paths):
        shard_row_offsets.append(next_row_id)
        n_tokens = _count_tokens(shard_path)
        if n_tokens % row_tokens != 0:
            raise ValueError(
                f"Shard token count {n_tokens} is not divisible by row_tokens {row_tokens}: {shard_path}"
            )
        n_rows = n_tokens // row_tokens
        rows_per_shard.append(n_rows)
        for local_row_idx in range(n_rows):
            token_offset_start = local_row_idx * row_tokens
            token_offset_end = token_offset_start + row_tokens
            row_refs.append(
                RowRef(
                    global_row_id=next_row_id,
                    shard_idx=shard_idx,
                    shard_path=str(shard_path),
                    local_row_idx=local_row_idx,
                    token_offset_start=token_offset_start,
                    token_offset_end=token_offset_end,
                )
            )
            next_row_id += 1

    return RowManifest(
        data_dir=data_path,
        split=split,
        seq_len=seq_len,
        row_tokens=row_tokens,
        shard_paths=shard_paths,
        rows_per_shard=tuple(rows_per_shard),
        shard_row_offsets=tuple(shard_row_offsets),
        rows=tuple(row_refs),
    )


class FiniteBOSRowDataset(Dataset):
    """Finite per-row dataset aligned to the BOS row-packed shard format."""

    def __init__(self, manifest: RowManifest, row_ids: Sequence[int]) -> None:
        self.manifest = manifest
        self.row_ids = tuple(int(row_id) for row_id in row_ids)
        self._memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.row_ids)

    def _memmap_for_shard(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._memmaps:
            self._memmaps[shard_idx] = np.memmap(
                self.manifest.shard_paths[shard_idx],
                dtype=np.uint16,
                mode="r",
            )
        return self._memmaps[shard_idx]

    def __getitem__(self, index: int) -> dict:
        row_id = self.row_ids[index]
        row_ref = self.manifest.row_ref(row_id)
        mm = self._memmap_for_shard(row_ref.shard_idx)
        chunk = mm[row_ref.token_offset_start : row_ref.token_offset_end]
        row = np.asarray(chunk, dtype=np.int64)
        if row.size != self.manifest.row_tokens:
            raise ValueError(
                f"Expected row with {self.manifest.row_tokens} tokens, got {row.size} "
                f"for row_id={row_id}"
            )
        tensor = torch.from_numpy(row.copy())
        return {
            "input_ids": tensor[:-1],
            "labels": tensor[1:],
            "row_id": row_ref.global_row_id,
            "shard_idx": row_ref.shard_idx,
            "shard_path": row_ref.shard_path,
            "local_row_idx": row_ref.local_row_idx,
        }


def iter_row_batches(dataset: FiniteBOSRowDataset, batch_size: int) -> Iterator[PreparedRowBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    total = len(dataset)
    for start in range(0, total, batch_size):
        samples = [dataset[idx] for idx in range(start, min(start + batch_size, total))]
        input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
        labels = torch.stack([sample["labels"] for sample in samples], dim=0)
        row_ids = tuple(int(sample["row_id"]) for sample in samples)
        row_refs = tuple(dataset.manifest.row_ref(row_id) for row_id in row_ids)
        yield PreparedRowBatch(
            batch=(input_ids, labels),
            local_inds=np.arange(start, start + len(samples), dtype=np.int64),
            row_ids=row_ids,
            row_refs=row_refs,
        )


__all__ = [
    "FiniteBOSRowDataset",
    "PreparedRowBatch",
    "RowManifest",
    "RowRef",
    "build_row_manifest",
    "iter_row_batches",
]
