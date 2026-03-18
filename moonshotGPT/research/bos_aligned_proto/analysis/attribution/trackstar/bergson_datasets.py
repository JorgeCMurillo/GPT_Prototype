"""Dataset adapters for the Bergson BOS attribution backend.

This module keeps the BOS row manifest and candidate ordering logic unchanged
while exposing a dataset schema that is convenient for Bergson-style gradient
collection and for backend-local caching.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.checkpoints import CheckpointRef
from ..common.row_dataset import FiniteBOSRowDataset, RowManifest


@dataclass(frozen=True)
class CandidateIndexMetadata:
    """Small cache contract for one checkpoint-local Bergson candidate index.

    Bergson's gradient index is only reusable when the checkpoint, candidate
    ordering, and index-affecting settings match exactly. We store the ordered
    row ids rather than just a count so a stale cache cannot silently survive a
    different candidate subset with the same cardinality.
    """

    backend: str
    checkpoint_step: int
    checkpoint_path: str
    fingerprint: str
    candidate_count: int
    candidate_row_ids: tuple[int, ...]
    projection_dim: int
    use_fast_jl: bool
    adam_second_moment_correction: bool

    def to_json(self) -> dict:
        """Return a plain JSON-serializable representation for cache metadata."""

        return {
            "backend": self.backend,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": self.checkpoint_path,
            "fingerprint": self.fingerprint,
            "candidate_count": self.candidate_count,
            "candidate_row_ids": list(self.candidate_row_ids),
            "projection_dim": self.projection_dim,
            "use_fast_jl": self.use_fast_jl,
            "adam_second_moment_correction": self.adam_second_moment_correction,
        }


class _BergsonDatasetView:
    """Minimal Hugging Face Dataset-like view for Bergson teardown.

    Bergson's collector reads examples from our custom dataset during gradient
    collection, but once collection finishes it mutates `self.data` using a
    Hugging Face Dataset-style API:

    - `remove_columns(...)`
    - `add_column(...)`
    - `save_to_disk(...)`

    The BOS attribution code does not otherwise need a Hugging Face dataset, so
    this lightweight view lets us satisfy Bergson's teardown contract without
    replacing the existing manifest-backed row dataset.
    """

    def __init__(
        self,
        *,
        base_rows: list[dict[str, Any]],
        dropped_columns: frozenset[str] = frozenset(),
        added_columns: dict[str, list[Any]] | None = None,
    ) -> None:
        self._base_rows = base_rows
        self._dropped_columns = dropped_columns
        self._added_columns = {} if added_columns is None else dict(added_columns)

    def __len__(self) -> int:
        return len(self._base_rows)

    @property
    def column_names(self) -> list[str]:
        names = list(self._base_rows[0].keys()) if self._base_rows else []
        visible = [name for name in names if name not in self._dropped_columns]
        for name in self._added_columns:
            if name not in visible:
                visible.append(name)
        return visible

    def remove_columns(self, columns: Sequence[str]) -> "_BergsonDatasetView":
        return _BergsonDatasetView(
            base_rows=self._base_rows,
            dropped_columns=self._dropped_columns | frozenset(str(name) for name in columns),
            added_columns=self._added_columns,
        )

    def add_column(
        self,
        name: str,
        column,
        *,
        feature=None,
        new_fingerprint=None,
    ) -> "_BergsonDatasetView":
        del feature, new_fingerprint
        normalized = _normalize_python_column(column)
        if len(normalized) != len(self._base_rows):
            raise ValueError(
                f"Added column {name!r} has length {len(normalized)} but dataset has {len(self._base_rows)} rows"
            )
        updated = dict(self._added_columns)
        updated[str(name)] = normalized
        return _BergsonDatasetView(
            base_rows=self._base_rows,
            dropped_columns=self._dropped_columns,
            added_columns=updated,
        )

    def save_to_disk(self, path: str) -> None:
        from datasets import Dataset as HFDataset

        HFDataset.from_list(self._materialize_rows()).save_to_disk(str(path))

    def _materialize_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(self._base_rows):
            materialized = {
                key: value
                for key, value in row.items()
                if key not in self._dropped_columns
            }
            for key, values in self._added_columns.items():
                materialized[key] = values[row_index]
            rows.append(materialized)
        return rows


def _normalize_python_column(column) -> list[Any]:
    """Convert a tensor/array/sequence column payload into plain Python values."""

    if isinstance(column, torch.Tensor):
        return column.detach().cpu().tolist()
    if isinstance(column, np.ndarray):
        return column.tolist()
    if isinstance(column, SequenceABC) and not isinstance(column, (str, bytes)):
        return list(column)
    raise TypeError(f"Unsupported column payload type: {type(column)!r}")


class BergsonCandidateDataset(Dataset):
    """Checkpoint-local BOS row dataset with the indexing semantics Bergson expects.

    The BOS attribution pipeline already has a stable row manifest and a stable
    candidate selection step. This adapter deliberately reuses that logic and
    only changes the surface area presented to Bergson.

    The important Bergson-specific detail is that its collector does *not*
    always fetch one example at a time. During gradient collection it calls
    `dataset[indices]` where `indices` is often a list of integers for a whole
    batch. Our first pass assumed standard integer-only indexing, which caused
    the `tuple indices must be integers or slices, not list` failure when the
    list propagated into `FiniteBOSRowDataset`.

    To match Bergson's collector, batched indexing here returns Python
    `list[list[int]]` payloads for `input_ids` and `labels`, not tensors. The
    collector then pads and tensors those sequences on its own.

    A second subtle requirement appears after collection completes: Bergson's
    teardown path expects `self.data` to behave like a Hugging Face dataset so
    it can drop columns, append losses, and save the candidate metadata to
    disk. The small adapter methods below provide that bridge.
    """

    _BASE_COLUMN_NAMES = (
        "input_ids",
        "labels",
        "attention_mask",
        "row_id",
        "shard_idx",
        "shard_path",
        "local_row_idx",
        "candidate_idx",
    )

    def __init__(self, manifest: RowManifest, row_ids: Sequence[int]) -> None:
        self._base = FiniteBOSRowDataset(manifest, row_ids)
        self.row_ids = tuple(int(row_id) for row_id in row_ids)
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self._base)

    def _single_item(self, index: int) -> dict:
        """Return one candidate row in the richer schema TrackStar needs.

        Bergson itself mainly needs token ids and labels for gradient
        collection, but we also preserve row-manifest metadata so downstream
        exports can still map scores back to BOS rows, shards, and local row
        indices without redesigning the existing attribution pipeline.
        """

        sample = self._base[index]
        input_ids = sample["input_ids"]
        return {
            "input_ids": input_ids,
            "labels": sample["labels"],
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            "row_id": int(sample["row_id"]),
            "shard_idx": int(sample["shard_idx"]),
            "shard_path": str(sample["shard_path"]),
            "local_row_idx": int(sample["local_row_idx"]),
            "candidate_idx": int(index),
        }

    @staticmethod
    def _pythonify_scalar_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Convert one scalar sample into plain Python container types."""

        return {
            "input_ids": sample["input_ids"].tolist(),
            "labels": sample["labels"].tolist(),
            "attention_mask": sample["attention_mask"].tolist(),
            "row_id": int(sample["row_id"]),
            "shard_idx": int(sample["shard_idx"]),
            "shard_path": str(sample["shard_path"]),
            "local_row_idx": int(sample["local_row_idx"]),
            "candidate_idx": int(sample["candidate_idx"]),
        }

    def _dataset_view(self) -> _BergsonDatasetView:
        """Materialize a Dataset-like view for Bergson's teardown/save path."""

        rows = [self._pythonify_scalar_sample(self._single_item(index)) for index in range(len(self))]
        return _BergsonDatasetView(base_rows=rows)

    @staticmethod
    def _batched_indices(index) -> list[int] | None:
        """Normalize Bergson-style batched indexing into a list of integers.

        Bergson may hand us slices, Python lists, tuples, numpy arrays, or
        torch tensors. Returning `None` means "treat this as scalar indexing";
        otherwise the caller should build and return a whole batch.
        """

        if isinstance(index, slice):
            start = 0 if index.start is None else int(index.start)
            stop = int(index.stop)
            step = 1 if index.step is None else int(index.step)
            return list(range(start, stop, step))
        if isinstance(index, torch.Tensor):
            if index.ndim == 0:
                return None
            return [int(value) for value in index.detach().cpu().tolist()]
        if isinstance(index, np.ndarray):
            if index.ndim == 0:
                return None
            return [int(value) for value in index.tolist()]
        if isinstance(index, SequenceABC) and not isinstance(index, (str, bytes)):
            return [int(value) for value in index]
        return None

    def __getitem__(self, index) -> dict:
        """Support both scalar and collector-driven batched access.

        Scalar access keeps the dataset pleasant to inspect and easy to reuse in
        tests. Batched access is the critical Bergson compatibility path:
        `collector.run_with_collector_hooks(...)` calls `dataset[indices]` and
        expects a dict whose `input_ids`/`labels` values are lists of token
        sequences. Returning tensors here would force an extra, conflicting
        collation step inside Bergson.
        """

        batched_indices = self._batched_indices(index)
        if batched_indices is None:
            return self._single_item(int(index))

        samples = [self._single_item(int(sample_index)) for sample_index in batched_indices]
        return {
            "input_ids": [sample["input_ids"].tolist() for sample in samples],
            "labels": [sample["labels"].tolist() for sample in samples],
            "attention_mask": [sample["attention_mask"].tolist() for sample in samples],
            "row_id": [int(sample["row_id"]) for sample in samples],
            "shard_idx": [int(sample["shard_idx"]) for sample in samples],
            "shard_path": [str(sample["shard_path"]) for sample in samples],
            "local_row_idx": [int(sample["local_row_idx"]) for sample in samples],
            "candidate_idx": [int(sample["candidate_idx"]) for sample in samples],
        }

    @property
    def column_names(self) -> list[str]:
        """Expose a Hugging Face Dataset-like column list when Bergson asks."""

        return list(self._BASE_COLUMN_NAMES)

    def remove_columns(self, columns: Sequence[str]) -> _BergsonDatasetView:
        """Return a Dataset-like view with the requested columns removed.

        Bergson calls this during teardown on rank 0 before it appends the
        per-document loss column and saves the resulting dataset to disk.
        """

        return self._dataset_view().remove_columns(columns)

    def add_column(
        self,
        name: str,
        column,
        *,
        feature=None,
        new_fingerprint=None,
    ) -> _BergsonDatasetView:
        """Return a Dataset-like view with an added column.

        This supports the Bergson teardown path when `drop_columns=False`, in
        which case it appends the per-document loss directly to the original
        candidate dataset view before saving it.
        """

        return self._dataset_view().add_column(
            name,
            column,
            feature=feature,
            new_fingerprint=new_fingerprint,
        )

    def save_to_disk(self, path: str) -> None:
        """Persist the candidate dataset in Hugging Face dataset format."""

        self._dataset_view().save_to_disk(path)


def build_candidate_index_fingerprint(
    *,
    checkpoint: CheckpointRef,
    candidate_row_ids: Sequence[int],
    projection_dim: int,
    use_fast_jl: bool,
    adam_second_moment_correction: bool,
) -> str:
    """Build a deterministic short hash for one candidate-index configuration.

    The hash intentionally includes the ordered candidate row ids. Reordering
    candidates changes the meaning of every score column, so cache reuse is
    only valid when the order is identical, not merely when the same set of row
    ids appears.
    """

    payload = {
        "checkpoint_step": int(checkpoint.step),
        "checkpoint_path": str(checkpoint.path),
        "candidate_row_ids": [int(row_id) for row_id in candidate_row_ids],
        "projection_dim": int(projection_dim),
        "use_fast_jl": bool(use_fast_jl),
        "adam_second_moment_correction": bool(adam_second_moment_correction),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def build_candidate_index_metadata(
    *,
    checkpoint: CheckpointRef,
    candidate_row_ids: Sequence[int],
    projection_dim: int,
    use_fast_jl: bool,
    adam_second_moment_correction: bool,
) -> CandidateIndexMetadata:
    """Package the full cache identity for a checkpoint-local Bergson index."""

    ordered_row_ids = tuple(int(row_id) for row_id in candidate_row_ids)
    fingerprint = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_row_ids=ordered_row_ids,
        projection_dim=projection_dim,
        use_fast_jl=use_fast_jl,
        adam_second_moment_correction=adam_second_moment_correction,
    )
    return CandidateIndexMetadata(
        backend="trackstar",
        checkpoint_step=int(checkpoint.step),
        checkpoint_path=str(checkpoint.path),
        fingerprint=fingerprint,
        candidate_count=len(ordered_row_ids),
        candidate_row_ids=ordered_row_ids,
        projection_dim=int(projection_dim),
        use_fast_jl=bool(use_fast_jl),
        adam_second_moment_correction=bool(adam_second_moment_correction),
    )


__all__ = [
    "BergsonCandidateDataset",
    "CandidateIndexMetadata",
    "build_candidate_index_fingerprint",
    "build_candidate_index_metadata",
]
