"""Backward-compatible wrappers around generic training-example manifests.

Historically the attribution package only handled BOS-packed rows, so a number
of modules still import `row_dataset`. The generic implementation now lives in
`training_examples.py`; this file intentionally re-exports the old names so the
rest of the package can migrate incrementally without breaking imports.
"""

from __future__ import annotations

from .training_examples import (
    ExampleManifest as RowManifest,
    ExampleRef as RowRef,
    FiniteTrainingExampleDataset,
    PreparedExampleBatch as PreparedRowBatch,
    build_example_manifest,
    iter_example_batches,
)


class FiniteBOSRowDataset(FiniteTrainingExampleDataset):
    """Backward-compatible row-centric constructor for older callers/tests."""

    def __init__(self, manifest: RowManifest, row_ids=None, example_ids=None) -> None:
        if row_ids is not None and example_ids is not None:
            raise TypeError("Pass either row_ids or example_ids, not both")
        super().__init__(manifest, example_ids if example_ids is not None else row_ids)


def build_row_manifest(*args, **kwargs):
    return build_example_manifest(*args, **kwargs)


def iter_row_batches(*args, **kwargs):
    return iter_example_batches(*args, **kwargs)


__all__ = [
    "FiniteBOSRowDataset",
    "PreparedRowBatch",
    "RowManifest",
    "RowRef",
    "build_row_manifest",
    "iter_row_batches",
]
