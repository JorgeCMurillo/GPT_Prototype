"""Exposure-log parsing and reusable row-level exposure indexes.

This module reads the per-rank exposure JSONL files emitted during BOS-row
training and converts token offsets back into global row identifiers using a
row manifest. Candidate selection code depends on these indexes to answer which
rows were seen before, within, or between checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .row_dataset import RowManifest


@dataclass(frozen=True)
class ExposureIndex:
    run_dir: Path
    step_to_row_ids: dict[int, tuple[int, ...]]
    first_seen_step_by_row_id: dict[int, int]

    def rows_exposed_up_to_step(self, step: int) -> tuple[int, ...]:
        row_ids: set[int] = set()
        for exposure_step, ids in self.step_to_row_ids.items():
            if exposure_step <= int(step):
                row_ids.update(ids)
        return tuple(sorted(row_ids))

    def rows_exposed_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        lo_value = -1 if lo is None else int(lo)
        row_ids: set[int] = set()
        for exposure_step, ids in self.step_to_row_ids.items():
            if lo_value < exposure_step <= int(hi):
                row_ids.update(ids)
        return tuple(sorted(row_ids))

    def rows_first_seen_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        lo_value = -1 if lo is None else int(lo)
        row_ids = [
            row_id
            for row_id, first_step in self.first_seen_step_by_row_id.items()
            if lo_value < first_step <= int(hi)
        ]
        return tuple(sorted(row_ids))


def _iter_exposure_files(run_dir: Path) -> Iterable[Path]:
    exposure_dir = run_dir / "exposures"
    if not exposure_dir.is_dir():
        raise FileNotFoundError(f"Exposure directory not found: {exposure_dir}")
    yield from sorted(exposure_dir.glob("exposures_rank*.jsonl"))


def _row_ids_from_micro_batch(micro_batch: dict, manifest: RowManifest) -> range:
    shard_idx = int(micro_batch["shard_idx"])
    start = int(micro_batch["start"])
    end = int(micro_batch["end"])
    if start % manifest.row_tokens != 0 or end % manifest.row_tokens != 0:
        raise ValueError(
            f"Exposure token offsets must align to row_tokens={manifest.row_tokens}: "
            f"start={start}, end={end}"
        )
    row_start = start // manifest.row_tokens
    row_end = end // manifest.row_tokens
    shard_offset = manifest.shard_row_offsets[shard_idx]
    return range(shard_offset + row_start, shard_offset + row_end)


def build_exposure_index(run_dir: str | Path, manifest: RowManifest) -> ExposureIndex:
    run_path = Path(run_dir).expanduser().resolve()
    step_to_row_ids: dict[int, set[int]] = {}
    first_seen_step_by_row_id: dict[int, int] = {}

    for exposure_file in _iter_exposure_files(run_path):
        with exposure_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                step = int(payload["step"])
                step_rows = step_to_row_ids.setdefault(step, set())
                for micro_batch in payload.get("micro_batches", []):
                    for row_id in _row_ids_from_micro_batch(micro_batch, manifest):
                        step_rows.add(int(row_id))
                        if row_id not in first_seen_step_by_row_id or step < first_seen_step_by_row_id[row_id]:
                            first_seen_step_by_row_id[row_id] = step

    return ExposureIndex(
        run_dir=run_path,
        step_to_row_ids={step: tuple(sorted(ids)) for step, ids in sorted(step_to_row_ids.items())},
        first_seen_step_by_row_id=first_seen_step_by_row_id,
    )


__all__ = ["ExposureIndex", "build_exposure_index"]
