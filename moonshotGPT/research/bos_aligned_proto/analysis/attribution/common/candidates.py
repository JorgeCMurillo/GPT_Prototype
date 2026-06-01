"""Candidate training-example selection for checkpoint-local attribution runs.

This module translates exposure history into the actual training-example
pool that a checkpoint will featurize. It owns the supported selection
strategies, the metadata that records what was chosen, and the deterministic
subsampling step used when the raw exposed set is too large to score directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .exposures import ExposureIndex
from .training_examples import ExampleManifest


RAW_WINDOW_CANDIDATE_STRATEGIES = (
    "raw_window_range",
    "raw_window_random",
)


@dataclass(frozen=True)
class CandidateSelection:
    strategy: str
    checkpoint_step: int
    previous_step: int | None
    candidate_to_step: int
    source_count: int
    selected_count: int
    candidate_ids: tuple[int, ...]

    @property
    def row_ids(self) -> tuple[int, ...]:
        """Backward-compatible alias for older row-centric call sites."""

        return self.candidate_ids


def _deterministic_subsample(
    candidate_ids: tuple[int, ...],
    *,
    max_candidate_rows: int,
    seed: int,
    sample_key: int | str,
) -> tuple[int, ...]:
    if len(candidate_ids) <= max_candidate_rows:
        return candidate_ids
    rng_seed = int(seed) + int(sample_key) if isinstance(sample_key, int) else f"{seed}:{sample_key}"
    rng = random.Random(rng_seed)
    sampled = rng.sample(list(candidate_ids), max_candidate_rows)
    return tuple(sorted(int(candidate_id) for candidate_id in sampled))


def _rows_for_strategy(
    exposure_index: ExposureIndex,
    *,
    strategy: str,
    candidate_to_step: int,
    previous_step: int | None,
    recent_window_steps: int,
) -> tuple[int, ...]:
    if strategy == "between_checkpoints":
        return exposure_index.ids_exposed_between_steps(previous_step, candidate_to_step)
    if strategy == "up_to_step":
        return exposure_index.ids_exposed_up_to_step(candidate_to_step)
    if strategy == "recent_window":
        lower = max(0, int(candidate_to_step) - int(recent_window_steps))
        return exposure_index.ids_exposed_between_steps(lower, candidate_to_step)
    if strategy == "new_since_prev":
        return exposure_index.ids_first_seen_between_steps(previous_step, candidate_to_step)
    raise ValueError(f"Unknown candidate strategy: {strategy}")


def uses_raw_window_candidates(strategy: str) -> bool:
    return str(strategy) in RAW_WINDOW_CANDIDATE_STRATEGIES


def select_candidate_rows(
    exposure_index: ExposureIndex,
    *,
    strategy: str,
    checkpoint_step: int,
    previous_step: int | None,
    candidate_to_step: int | None = None,
    max_candidate_rows: int,
    seed: int,
    recent_window_steps: int,
) -> CandidateSelection:
    effective_candidate_to_step = int(checkpoint_step if candidate_to_step is None else candidate_to_step)
    source_candidate_ids = _rows_for_strategy(
        exposure_index,
        strategy=strategy,
        candidate_to_step=effective_candidate_to_step,
        previous_step=previous_step,
        recent_window_steps=recent_window_steps,
    )
    sample_key: int | str = int(checkpoint_step)
    if effective_candidate_to_step != int(checkpoint_step):
        lower_text = "none" if previous_step is None else str(int(previous_step))
        sample_key = f"{lower_text}:{effective_candidate_to_step}"
    selected_candidate_ids = _deterministic_subsample(
        source_candidate_ids,
        max_candidate_rows=max_candidate_rows,
        seed=seed,
        sample_key=sample_key,
    )
    return CandidateSelection(
        strategy=strategy,
        checkpoint_step=int(checkpoint_step),
        previous_step=None if previous_step is None else int(previous_step),
        candidate_to_step=effective_candidate_to_step,
        source_count=len(source_candidate_ids),
        selected_count=len(selected_candidate_ids),
        candidate_ids=selected_candidate_ids,
    )


def select_manifest_candidate_rows(
    manifest: ExampleManifest,
    *,
    strategy: str,
    checkpoint_step: int,
    max_candidate_rows: int,
    seed: int,
    raw_window_start_id: int = 0,
    raw_window_end_id: int | None = None,
) -> CandidateSelection:
    if strategy not in RAW_WINDOW_CANDIDATE_STRATEGIES:
        raise ValueError(f"Unknown raw-window candidate strategy: {strategy}")
    if manifest.candidate_kind != "stream_window":
        raise ValueError(
            "Raw-window candidate strategies require candidate_kind='stream_window'. "
            f"Got candidate_kind={manifest.candidate_kind!r}."
        )

    start_id = int(raw_window_start_id)
    requested_end_id = len(manifest.examples) if raw_window_end_id is None else int(raw_window_end_id)
    end_id = min(requested_end_id, len(manifest.examples))
    if start_id < 0:
        raise ValueError("raw_window_start_id must be >= 0")
    if requested_end_id <= start_id:
        raise ValueError("raw_window_end_id must be greater than raw_window_start_id")
    if start_id >= len(manifest.examples):
        raise ValueError(
            f"raw_window_start_id={start_id} is beyond the manifest length "
            f"{len(manifest.examples)}"
        )
    if end_id <= start_id:
        raise ValueError(
            f"Raw-window interval [{start_id}, {requested_end_id}) does not include "
            "any manifest examples"
        )

    source_count = end_id - start_id
    if strategy == "raw_window_range":
        selected_count = min(source_count, int(max_candidate_rows))
        selected_candidate_ids = tuple(range(start_id, start_id + selected_count))
    elif source_count <= int(max_candidate_rows):
        selected_candidate_ids = tuple(range(start_id, end_id))
    else:
        rng = random.Random(f"{seed}:{strategy}:{start_id}:{end_id}")
        sampled = rng.sample(range(start_id, end_id), int(max_candidate_rows))
        selected_candidate_ids = tuple(
            sorted(int(candidate_id) for candidate_id in sampled)
        )

    return CandidateSelection(
        strategy=strategy,
        checkpoint_step=int(checkpoint_step),
        previous_step=None,
        candidate_to_step=int(checkpoint_step),
        source_count=source_count,
        selected_count=len(selected_candidate_ids),
        candidate_ids=selected_candidate_ids,
    )


__all__ = [
    "CandidateSelection",
    "RAW_WINDOW_CANDIDATE_STRATEGIES",
    "select_candidate_rows",
    "select_manifest_candidate_rows",
    "uses_raw_window_candidates",
]
