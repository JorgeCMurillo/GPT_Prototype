"""Dataset adapters for the Bergson attribution backend.

This module keeps the shared candidate-example manifest and ordering logic
unchanged while exposing a dataset schema that is convenient for Bergson-style
gradient collection and for backend-local caching.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.checkpoints import CheckpointRef
from ..common.training_examples import ExampleManifest, FiniteTrainingExampleDataset


@dataclass(frozen=True)
class CandidateIndexMetadata:
    """Small cache contract for one checkpoint-local Bergson candidate index.

    Bergson's gradient index is only reusable when the checkpoint, candidate
    ordering, and index-affecting settings match exactly. We store the ordered
    candidate ids rather than just a count so a stale cache cannot silently
    survive a different candidate subset with the same cardinality.
    """

    backend: str
    checkpoint_step: int
    checkpoint_path: str
    fingerprint: str
    candidate_count: int
    candidate_ids: tuple[int, ...]
    projection_dim: int
    use_fast_jl: bool
    adam_second_moment_correction: bool
    projection_layout: str = "module"
    paper_block_features: int = 0
    paper_block_side: int = 0

    def to_json(self) -> dict:
        """Return a plain JSON-serializable representation for cache metadata."""

        return {
            "backend": self.backend,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": self.checkpoint_path,
            "fingerprint": self.fingerprint,
            "candidate_count": self.candidate_count,
            "candidate_ids": list(self.candidate_ids),
            "projection_dim": self.projection_dim,
            "use_fast_jl": self.use_fast_jl,
            "adam_second_moment_correction": self.adam_second_moment_correction,
            "projection_layout": self.projection_layout,
            "paper_block_features": self.paper_block_features,
            "paper_block_side": self.paper_block_side,
        }


class _BergsonDatasetView:
    """Minimal Hugging Face Dataset-like view for Bergson teardown.

    Bergson's collector reads examples from our custom dataset during gradient
    collection, but once collection finishes it mutates `self.data` using a
    Hugging Face Dataset-style API:

    - `remove_columns(...)`
    - `add_column(...)`
    - `save_to_disk(...)`

    The attribution code does not otherwise need a Hugging Face dataset, so
    this lightweight view lets us satisfy Bergson's teardown contract without
    replacing the existing manifest-backed candidate dataset.
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
    """Checkpoint-local training-example dataset with Bergson-compatible indexing.

    The attribution pipeline already has a stable manifest and a stable
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
        "candidate_id",
        "candidate_kind",
        "shard_idx",
        "shard_path",
        "local_example_idx",
        "token_offset_start",
        "token_offset_end",
        "candidate_idx",
        # Legacy aliases retained so older notebooks keep working.
        "row_id",
        "local_row_idx",
    )

    def __init__(self, manifest: ExampleManifest, candidate_ids: Sequence[int]) -> None:
        self._base = FiniteTrainingExampleDataset(manifest, candidate_ids)
        self.candidate_ids = tuple(int(candidate_id) for candidate_id in candidate_ids)
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self._base)

    def _single_item(self, index: int) -> dict:
        """Return one candidate example in the richer schema TrackStar needs.

        Bergson itself mainly needs token ids and labels for gradient
        collection, but we also preserve manifest metadata so downstream
        exports can still map scores back to candidate examples, shards, and
        local indices without redesigning the existing attribution pipeline.

        Important implementation detail: `FiniteTrainingExampleDataset`
        materializes the repo's standard shifted `(input_ids, labels)` training
        pair. Bergson's causal-LM CE path applies its own shift internally, so
        handing it those already-shifted labels would double-shift the target
        sequence. To match standard causal-LM next-token scoring, this adapter
        reconstructs the full unshifted token chunk and passes that chunk as
        both `input_ids` and `labels`.
        """

        sample = self._base[index]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        full_tokens = torch.cat([input_ids, labels[-1:].clone()], dim=0)
        return {
            "input_ids": full_tokens,
            "labels": full_tokens.clone(),
            "attention_mask": torch.ones_like(full_tokens, dtype=torch.long),
            "candidate_id": int(sample["candidate_id"]),
            "candidate_kind": str(sample["candidate_kind"]),
            "shard_idx": int(sample["shard_idx"]),
            "shard_path": str(sample["shard_path"]),
            "local_example_idx": int(sample["local_example_idx"]),
            "token_offset_start": int(sample["token_offset_start"]),
            "token_offset_end": int(sample["token_offset_end"]),
            "candidate_idx": int(index),
            "row_id": int(sample["row_id"]),
            "local_row_idx": int(sample["local_row_idx"]),
        }

    @staticmethod
    def _pythonify_scalar_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Convert one scalar sample into plain Python container types."""

        return {
            "input_ids": sample["input_ids"].tolist(),
            "labels": sample["labels"].tolist(),
            "attention_mask": sample["attention_mask"].tolist(),
            "candidate_id": int(sample["candidate_id"]),
            "candidate_kind": str(sample["candidate_kind"]),
            "shard_idx": int(sample["shard_idx"]),
            "shard_path": str(sample["shard_path"]),
            "local_example_idx": int(sample["local_example_idx"]),
            "token_offset_start": int(sample["token_offset_start"]),
            "token_offset_end": int(sample["token_offset_end"]),
            "candidate_idx": int(sample["candidate_idx"]),
            "row_id": int(sample["row_id"]),
            "local_row_idx": int(sample["local_row_idx"]),
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
            "candidate_id": [int(sample["candidate_id"]) for sample in samples],
            "candidate_kind": [str(sample["candidate_kind"]) for sample in samples],
            "shard_idx": [int(sample["shard_idx"]) for sample in samples],
            "shard_path": [str(sample["shard_path"]) for sample in samples],
            "local_example_idx": [int(sample["local_example_idx"]) for sample in samples],
            "token_offset_start": [int(sample["token_offset_start"]) for sample in samples],
            "token_offset_end": [int(sample["token_offset_end"]) for sample in samples],
            "candidate_idx": [int(sample["candidate_idx"]) for sample in samples],
            "row_id": [int(sample["row_id"]) for sample in samples],
            "local_row_idx": [int(sample["local_row_idx"]) for sample in samples],
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
    candidate_ids: Sequence[int],
    projection_dim: int,
    use_fast_jl: bool,
    adam_second_moment_correction: bool,
    projection_layout: str = "module",
    paper_block_features: int = 0,
    paper_block_side: int = 0,
) -> str:
    """Build a deterministic short hash for one candidate-index configuration.

    The hash intentionally includes the ordered candidate ids. Reordering
    candidates changes the meaning of every score column, so cache reuse is
    only valid when the order is identical, not merely when the same set of
    ids appears.
    """

    payload = {
        "checkpoint_step": int(checkpoint.step),
        "checkpoint_path": str(checkpoint.path),
        "candidate_ids": [int(candidate_id) for candidate_id in candidate_ids],
        "projection_dim": int(projection_dim),
        "use_fast_jl": bool(use_fast_jl),
        "adam_second_moment_correction": bool(adam_second_moment_correction),
        "projection_layout": str(projection_layout),
        "paper_block_features": int(paper_block_features),
        "paper_block_side": int(paper_block_side),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def build_candidate_index_metadata(
    *,
    checkpoint: CheckpointRef,
    candidate_ids: Sequence[int],
    projection_dim: int,
    use_fast_jl: bool,
    adam_second_moment_correction: bool,
    projection_layout: str = "module",
    paper_block_features: int = 0,
    paper_block_side: int = 0,
) -> CandidateIndexMetadata:
    """Package the full cache identity for a checkpoint-local Bergson index."""

    ordered_candidate_ids = tuple(int(candidate_id) for candidate_id in candidate_ids)
    fingerprint = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_ids=ordered_candidate_ids,
        projection_dim=projection_dim,
        use_fast_jl=use_fast_jl,
        adam_second_moment_correction=adam_second_moment_correction,
        projection_layout=projection_layout,
        paper_block_features=paper_block_features,
        paper_block_side=paper_block_side,
    )
    return CandidateIndexMetadata(
        backend="trackstar",
        checkpoint_step=int(checkpoint.step),
        checkpoint_path=str(checkpoint.path),
        fingerprint=fingerprint,
        candidate_count=len(ordered_candidate_ids),
        candidate_ids=ordered_candidate_ids,
        projection_dim=int(projection_dim),
        use_fast_jl=bool(use_fast_jl),
        adam_second_moment_correction=bool(adam_second_moment_correction),
        projection_layout=str(projection_layout),
        paper_block_features=int(paper_block_features),
        paper_block_side=int(paper_block_side),
    )


def load_flat_gradient_index(index_dir: str | Path) -> dict[str, np.ndarray]:
    """Load a flat gradient memmap written by Bergson's Builder.

    The TrackStar integration always writes unstructured `[num_grads, total_dim]`
    indices plus `info.json`/`grad_sizes`. Loading locally avoids assuming the
    installed Bergson version understands every repo-local pooled-block layout.
    """

    root = Path(index_dir)
    info = json.loads((root / "info.json").read_text(encoding="utf-8"))
    num_grads = int(info["num_grads"])
    grad_sizes = info.get("grad_sizes")
    if not isinstance(grad_sizes, dict) or not grad_sizes:
        raise ValueError(f"Gradient index at {root} is missing non-empty grad_sizes metadata")
    base_dtype = np.dtype(str(info.get("base_dtype", "float32")))
    total_dim = int(sum(int(size) for size in grad_sizes.values()))
    mmap = np.memmap(
        root / "gradients.bin",
        dtype=base_dtype,
        mode="r",
        shape=(num_grads, total_dim),
    )
    array_view = mmap.view(np.ndarray)
    grads: dict[str, np.ndarray] = {}
    start = 0
    for name, size in grad_sizes.items():
        width = int(size)
        grads[str(name)] = array_view[:, start : start + width]
        start += width
    return grads


__all__ = [
    "BergsonCandidateDataset",
    "CandidateIndexMetadata",
    "build_candidate_index_fingerprint",
    "build_candidate_index_metadata",
    "load_flat_gradient_index",
]
