"""Grouped train/validation/test splits for EWoK probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import EWOKProbePair


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(value <= 0.0 for value in ratios):
            raise ValueError(f"Split ratios must be positive, got {ratios!r}")
        total = sum(ratios)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def _split_group_counts(n_groups: int, config: SplitConfig) -> tuple[int, int, int]:
    if n_groups < 3:
        raise ValueError("Need at least 3 EWoK rows for grouped train/val/test splits.")

    n_train = int(np.floor(n_groups * config.train_ratio))
    n_val = int(np.floor(n_groups * config.val_ratio))
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n_groups - n_train - n_val

    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Could not allocate non-empty train/val/test splits.")
        n_test = n_groups - n_train - n_val
    return n_train, n_val, n_test


def assign_grouped_splits(
    pairs: Sequence[EWOKProbePair],
    config: SplitConfig,
) -> dict[int, str]:
    """Assign split names per EWoK row while keeping each four-way item intact."""

    config.validate()
    row_indices = np.array(sorted({int(pair.row_index) for pair in pairs}), dtype=np.int64)
    n_train, n_val, _ = _split_group_counts(len(row_indices), config)

    rng = np.random.default_rng(int(config.seed))
    shuffled = row_indices.copy()
    rng.shuffle(shuffled)

    train_rows = set(int(x) for x in shuffled[:n_train])
    val_rows = set(int(x) for x in shuffled[n_train : n_train + n_val])
    test_rows = set(int(x) for x in shuffled[n_train + n_val :])

    assignments: dict[int, str] = {}
    for row_index in row_indices:
        idx = int(row_index)
        if idx in train_rows:
            assignments[idx] = "train"
        elif idx in val_rows:
            assignments[idx] = "val"
        elif idx in test_rows:
            assignments[idx] = "test"
        else:
            raise RuntimeError(f"Missing split assignment for row_index={idx}")
    return assignments


def pair_split_labels(
    pairs: Sequence[EWOKProbePair],
    row_splits: Mapping[int, str],
) -> np.ndarray:
    return np.asarray([str(row_splits[int(pair.row_index)]) for pair in pairs], dtype=object)


def split_assignment_rows(
    pairs: Sequence[EWOKProbePair],
    row_splits: Mapping[int, str],
) -> list[dict]:
    return [
        {
            "pair_id": pair.pair_id,
            "row_index": int(pair.row_index),
            "domain": pair.domain,
            "role": pair.role,
            "label": int(pair.label),
            "split": str(row_splits[int(pair.row_index)]),
        }
        for pair in pairs
    ]


__all__ = [
    "SplitConfig",
    "assign_grouped_splits",
    "pair_split_labels",
    "split_assignment_rows",
]
