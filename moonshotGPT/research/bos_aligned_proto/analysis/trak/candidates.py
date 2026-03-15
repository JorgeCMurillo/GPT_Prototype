"""Candidate row selection for checkpoint-local TRAK runs.

This module translates row-level exposure history into the actual training-row
pool that a checkpoint will featurize. It owns the supported selection
strategies, the metadata that records what was chosen, and the deterministic
subsampling step used when the raw exposed set is too large to score directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .exposures import ExposureIndex


@dataclass(frozen=True)
class CandidateSelection:
    strategy: str
    checkpoint_step: int
    previous_step: int | None
    source_count: int
    selected_count: int
    row_ids: tuple[int, ...]


def _deterministic_subsample(
    row_ids: tuple[int, ...],
    *,
    max_candidate_rows: int,
    seed: int,
    checkpoint_step: int,
) -> tuple[int, ...]:
    if len(row_ids) <= max_candidate_rows:
        return row_ids
    rng = random.Random(seed + checkpoint_step)
    sampled = rng.sample(list(row_ids), max_candidate_rows)
    return tuple(sorted(int(row_id) for row_id in sampled))


def _rows_for_strategy(
    exposure_index: ExposureIndex,
    *,
    strategy: str,
    checkpoint_step: int,
    previous_step: int | None,
    recent_window_steps: int,
) -> tuple[int, ...]:
    if strategy == "between_checkpoints":
        return exposure_index.rows_exposed_between_steps(previous_step, checkpoint_step)
    if strategy == "up_to_step":
        return exposure_index.rows_exposed_up_to_step(checkpoint_step)
    if strategy == "recent_window":
        lower = max(0, int(checkpoint_step) - int(recent_window_steps))
        return exposure_index.rows_exposed_between_steps(lower, checkpoint_step)
    if strategy == "new_since_prev":
        return exposure_index.rows_first_seen_between_steps(previous_step, checkpoint_step)
    raise ValueError(f"Unknown candidate strategy: {strategy}")


def select_candidate_rows(
    exposure_index: ExposureIndex,
    *,
    strategy: str,
    checkpoint_step: int,
    previous_step: int | None,
    max_candidate_rows: int,
    seed: int,
    recent_window_steps: int,
) -> CandidateSelection:
    source_row_ids = _rows_for_strategy(
        exposure_index,
        strategy=strategy,
        checkpoint_step=checkpoint_step,
        previous_step=previous_step,
        recent_window_steps=recent_window_steps,
    )
    selected_row_ids = _deterministic_subsample(
        source_row_ids,
        max_candidate_rows=max_candidate_rows,
        seed=seed,
        checkpoint_step=checkpoint_step,
    )
    return CandidateSelection(
        strategy=strategy,
        checkpoint_step=int(checkpoint_step),
        previous_step=None if previous_step is None else int(previous_step),
        source_count=len(source_row_ids),
        selected_count=len(selected_row_ids),
        row_ids=selected_row_ids,
    )


__all__ = ["CandidateSelection", "select_candidate_rows"]
