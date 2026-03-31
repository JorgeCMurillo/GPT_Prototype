"""Generic training-example manifests for attribution backends.

This module is the shared bridge between the model's *actual* training-example
surface and the attribution backends. Historically the attribution code only
handled BOS-packed rows, so many helper names elsewhere still say "row". The
classes here intentionally generalize that idea:

- BOS-packed runs use fixed-width row examples.
- Stream-trained runs use exact contiguous stream windows of length
  ``seq_len + 1`` tokens, matching the next-token-shifted training example.

Keeping both modes behind one manifest/dataset surface lets the TRAK and
TrackStar backends stay focused on gradient scoring instead of on how training
examples are reconstructed from disk.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from research.bos_aligned_proto.pipeline.bos_packed_index import (
        PACKED_INDEX_FORMAT,
        PackedIndexView,
    )
except ImportError:
    from ....pipeline.bos_packed_index import PACKED_INDEX_FORMAT, PackedIndexView


CandidateKind = Literal["bos_packed_row", "stream_window"]


def _load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_shards(data_dir: Path, split: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(str(data_dir / f"{split}_*.bin")))
    if not paths:
        raise FileNotFoundError(f"No shards found for split={split!r} under {data_dir}")
    return paths


def _list_virtual_shards(data_dir: Path, split: str) -> list[dict]:
    manifest_path = data_dir / f"{split}.virtual_shards.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Virtual shard manifest not found for split={split!r} under {data_dir}")
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if not rows:
        raise FileNotFoundError(f"Virtual shard manifest is empty for split={split!r} under {data_dir}")
    return rows


def _count_tokens(path: Path) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte size is not divisible by 2: {path}")
    return nbytes // 2


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def _stream_examples_per_shard(n_tokens: int, seq_len: int) -> int:
    """Return the number of exact stream windows in one raw shard.

    The stream loader trains on windows of length ``seq_len + 1`` tokens and
    advances by ``seq_len`` tokens between consecutive per-sequence examples.
    That makes the example starts ``0, seq_len, 2 * seq_len, ...`` up to the
    largest start whose full ``seq_len + 1`` token span fits in the shard.
    """

    example_tokens = int(seq_len) + 1
    if n_tokens < example_tokens:
        return 0
    return 1 + (n_tokens - example_tokens) // int(seq_len)


@dataclass(frozen=True)
class ExampleRef:
    """Stable identity and on-disk location for one candidate training example."""

    global_example_id: int
    candidate_kind: CandidateKind
    shard_idx: int
    shard_path: str
    local_example_idx: int
    token_offset_start: int
    token_offset_end: int

    @property
    def global_row_id(self) -> int:
        """Backward-compatible alias used by older BOS-row-only code paths."""

        return int(self.global_example_id)

    @property
    def local_row_idx(self) -> int:
        """Backward-compatible alias used by older BOS-row-only code paths."""

        return int(self.local_example_idx)


@dataclass(frozen=True)
class PreparedExampleBatch:
    batch: tuple[torch.Tensor, torch.Tensor]
    local_inds: np.ndarray
    example_ids: tuple[int, ...]
    example_refs: tuple[ExampleRef, ...]

    @property
    def row_ids(self) -> tuple[int, ...]:
        return self.example_ids

    @property
    def row_refs(self) -> tuple[ExampleRef, ...]:
        return self.example_refs


@dataclass(frozen=True)
class ExampleManifest:
    """Finite, deterministic view of the training examples attribution can score."""

    data_dir: Path
    split: str
    format: str
    candidate_kind: CandidateKind
    seq_len: int
    example_tokens: int
    token_stride: int
    shard_paths: tuple[Path, ...]
    examples_per_shard: tuple[int, ...]
    shard_example_offsets: tuple[int, ...]
    examples: tuple[ExampleRef, ...]

    def __len__(self) -> int:
        return len(self.examples)

    def example_ref(self, global_example_id: int) -> ExampleRef:
        return self.examples[int(global_example_id)]

    @property
    def row_tokens(self) -> int:
        return int(self.example_tokens)

    @property
    def rows_per_shard(self) -> tuple[int, ...]:
        return self.examples_per_shard

    @property
    def shard_row_offsets(self) -> tuple[int, ...]:
        return self.shard_example_offsets

    @property
    def rows(self) -> tuple[ExampleRef, ...]:
        return self.examples

    def row_ref(self, global_row_id: int) -> ExampleRef:
        return self.example_ref(global_row_id)


def infer_candidate_kind(
    data_dir: str | Path,
    *,
    preferred_kind: str = "auto",
) -> CandidateKind:
    """Infer whether attribution candidates are BOS-packed rows or stream windows."""

    resolved = str(preferred_kind).strip().lower()
    if resolved in {"bos", "bos_packed", "bos_packed_row", "bos_packed_rows"}:
        return "bos_packed_row"
    if resolved in {"stream", "stream_window", "stream_windows"}:
        return "stream_window"
    if resolved != "auto":
        raise ValueError(
            f"Unsupported candidate kind override {preferred_kind!r}; expected 'auto', 'bos_packed', or 'stream'."
        )

    meta = _load_meta(Path(data_dir).expanduser().resolve())
    data_format = str(meta.get("format", ""))
    if data_format == PACKED_INDEX_FORMAT or "row_tokens" in meta:
        return "bos_packed_row"
    return "stream_window"


def build_example_manifest(
    data_dir: str | Path,
    split: str = "train",
    *,
    candidate_kind: str = "auto",
    seq_len: int | None = None,
    show_progress: bool = False,
) -> ExampleManifest:
    """Build a manifest for BOS-packed rows or exact stream windows.

    ``seq_len`` is only required for stream-mode data, where raw shard metadata
    records document/tokenization facts but not the model context length used at
    training time.
    """

    data_path = Path(data_dir).expanduser().resolve()
    meta = _load_meta(data_path)
    resolved_kind = infer_candidate_kind(data_path, preferred_kind=candidate_kind)

    if resolved_kind == "bos_packed_row":
        data_format = str(meta.get("format", "bos_row_packed_bestfit"))
        resolved_seq_len = int(meta["seq_len"])
        example_tokens = int(meta["row_tokens"])
        token_stride = int(example_tokens)
        if data_format == PACKED_INDEX_FORMAT:
            virtual_shards = _list_virtual_shards(data_path, split)
            shard_paths = tuple(Path(str(row["shard_path"])) for row in virtual_shards)
            shard_example_counts = [int(row["row_count"]) for row in virtual_shards]
        else:
            shard_paths = tuple(_list_shards(data_path, split))
            shard_example_counts = []
    else:
        data_format = str(meta.get("format", "token_stream"))
        if seq_len is None or int(seq_len) <= 0:
            raise ValueError(
                "Stream-window attribution requires seq_len so exact training windows can be reconstructed."
            )
        resolved_seq_len = int(seq_len)
        example_tokens = int(resolved_seq_len) + 1
        token_stride = int(resolved_seq_len)
        shard_paths = tuple(_list_shards(data_path, split))
        shard_example_counts = []

    examples_per_shard: list[int] = []
    shard_example_offsets: list[int] = []
    example_refs: list[ExampleRef] = []
    next_example_id = 0

    progress = _build_tqdm(
        enabled=show_progress and len(shard_paths) > 1,
        total=len(shard_paths),
        desc=f"Building {split} example manifest",
        unit="shard",
    )
    try:
        for shard_idx, shard_path in enumerate(shard_paths):
            if progress is not None:
                progress.set_postfix_str(shard_path.name)
            shard_example_offsets.append(next_example_id)
            if resolved_kind == "bos_packed_row" and data_format == PACKED_INDEX_FORMAT:
                n_examples = int(shard_example_counts[shard_idx])
            else:
                n_tokens = _count_tokens(shard_path)
                if resolved_kind == "bos_packed_row":
                    if n_tokens % example_tokens != 0:
                        raise ValueError(
                            f"Shard token count {n_tokens} is not divisible by row_tokens {example_tokens}: {shard_path}"
                        )
                    n_examples = n_tokens // example_tokens
                else:
                    n_examples = _stream_examples_per_shard(n_tokens, resolved_seq_len)

            examples_per_shard.append(n_examples)
            for local_example_idx in range(n_examples):
                if resolved_kind == "bos_packed_row":
                    token_offset_start = local_example_idx * example_tokens
                else:
                    token_offset_start = local_example_idx * token_stride
                token_offset_end = token_offset_start + example_tokens
                example_refs.append(
                    ExampleRef(
                        global_example_id=next_example_id,
                        candidate_kind=resolved_kind,
                        shard_idx=shard_idx,
                        shard_path=str(shard_path),
                        local_example_idx=local_example_idx,
                        token_offset_start=token_offset_start,
                        token_offset_end=token_offset_end,
                    )
                )
                next_example_id += 1
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return ExampleManifest(
        data_dir=data_path,
        split=split,
        format=data_format,
        candidate_kind=resolved_kind,
        seq_len=resolved_seq_len,
        example_tokens=example_tokens,
        token_stride=token_stride,
        shard_paths=shard_paths,
        examples_per_shard=tuple(examples_per_shard),
        shard_example_offsets=tuple(shard_example_offsets),
        examples=tuple(example_refs),
    )


class FiniteTrainingExampleDataset(Dataset):
    """Finite per-example dataset for both BOS-packed rows and stream windows."""

    def __init__(self, manifest: ExampleManifest, example_ids: Sequence[int]) -> None:
        self.manifest = manifest
        self.example_ids = tuple(int(example_id) for example_id in example_ids)
        self._memmaps: dict[int, np.memmap] = {}
        self._packed_index_view: PackedIndexView | None = None

    def __len__(self) -> int:
        return len(self.example_ids)

    def _memmap_for_shard(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._memmaps:
            self._memmaps[shard_idx] = np.memmap(
                self.manifest.shard_paths[shard_idx],
                dtype=np.uint16,
                mode="r",
            )
        return self._memmaps[shard_idx]

    def _packed_index(self) -> PackedIndexView:
        if self._packed_index_view is None:
            self._packed_index_view = PackedIndexView(self.manifest.data_dir)
        return self._packed_index_view

    def __getitem__(self, index: int) -> dict:
        example_id = self.example_ids[index]
        example_ref = self.manifest.example_ref(example_id)
        if self.manifest.format == PACKED_INDEX_FORMAT:
            tokens = self._packed_index().reconstruct_row(self.manifest.split, example_id).astype(np.int64, copy=False)
        else:
            mm = self._memmap_for_shard(example_ref.shard_idx)
            chunk = mm[example_ref.token_offset_start : example_ref.token_offset_end]
            tokens = np.asarray(chunk, dtype=np.int64)

        if tokens.size != self.manifest.example_tokens:
            raise ValueError(
                f"Expected example with {self.manifest.example_tokens} tokens, got {tokens.size} "
                f"for example_id={example_id}"
            )

        tensor = torch.from_numpy(tokens.copy())
        sample = {
            "input_ids": tensor[:-1],
            "labels": tensor[1:],
            "candidate_id": example_ref.global_example_id,
            "candidate_kind": example_ref.candidate_kind,
            "shard_idx": example_ref.shard_idx,
            "shard_path": example_ref.shard_path,
            "local_example_idx": example_ref.local_example_idx,
            "token_offset_start": example_ref.token_offset_start,
            "token_offset_end": example_ref.token_offset_end,
            # Backward-compatible aliases used by older export/notebook code.
            "row_id": example_ref.global_example_id,
            "local_row_idx": example_ref.local_example_idx,
        }
        return sample


def iter_example_batches(dataset: FiniteTrainingExampleDataset, batch_size: int) -> Iterator[PreparedExampleBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    total = len(dataset)
    for start in range(0, total, batch_size):
        samples = [dataset[idx] for idx in range(start, min(start + batch_size, total))]
        input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
        labels = torch.stack([sample["labels"] for sample in samples], dim=0)
        example_ids = tuple(int(sample["candidate_id"]) for sample in samples)
        example_refs = tuple(dataset.manifest.example_ref(example_id) for example_id in example_ids)
        yield PreparedExampleBatch(
            batch=(input_ids, labels),
            local_inds=np.arange(start, start + len(samples), dtype=np.int64),
            example_ids=example_ids,
            example_refs=example_refs,
        )


__all__ = [
    "CandidateKind",
    "ExampleManifest",
    "ExampleRef",
    "FiniteTrainingExampleDataset",
    "PreparedExampleBatch",
    "PACKED_INDEX_FORMAT",
    "PackedIndexView",
    "build_example_manifest",
    "infer_candidate_kind",
    "iter_example_batches",
]
