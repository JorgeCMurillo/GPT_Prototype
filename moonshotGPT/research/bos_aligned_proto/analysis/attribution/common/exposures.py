"""Exposure-log parsing and reusable candidate-example exposure indexes.

The exposure files emitted during training record the shard-local token spans
seen by each micro-batch. This module maps those spans back to the attribution
candidate unit:

- BOS-packed runs map one logged span to a contiguous range of row examples.
- Stream-trained runs expand one logged batch block into the exact per-sequence
  windows that produced the SGD updates.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .training_examples import ExampleManifest


@dataclass(frozen=True)
class ExposureIndex:
    run_dir: Path
    step_to_example_ids: dict[int, tuple[int, ...]]
    first_seen_step_by_example_id: dict[int, int]

    @property
    def step_to_row_ids(self) -> dict[int, tuple[int, ...]]:
        """Backward-compatible alias for older row-centric logging code."""

        return self.step_to_example_ids

    def ids_exposed_up_to_step(self, step: int) -> tuple[int, ...]:
        example_ids: set[int] = set()
        for exposure_step, ids in self.step_to_example_ids.items():
            if exposure_step <= int(step):
                example_ids.update(ids)
        return tuple(sorted(example_ids))

    def ids_exposed_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        lo_value = -1 if lo is None else int(lo)
        example_ids: set[int] = set()
        for exposure_step, ids in self.step_to_example_ids.items():
            if lo_value < exposure_step <= int(hi):
                example_ids.update(ids)
        return tuple(sorted(example_ids))

    def ids_first_seen_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        lo_value = -1 if lo is None else int(lo)
        example_ids = [
            example_id
            for example_id, first_step in self.first_seen_step_by_example_id.items()
            if lo_value < first_step <= int(hi)
        ]
        return tuple(sorted(example_ids))

    # Backward-compatible aliases used by older row-centric code.
    def rows_exposed_up_to_step(self, step: int) -> tuple[int, ...]:
        return self.ids_exposed_up_to_step(step)

    def rows_exposed_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        return self.ids_exposed_between_steps(lo, hi)

    def rows_first_seen_between_steps(self, lo: int | None, hi: int) -> tuple[int, ...]:
        return self.ids_first_seen_between_steps(lo, hi)

    @property
    def first_seen_step_by_row_id(self) -> dict[int, int]:
        return self.first_seen_step_by_example_id


@dataclass(frozen=True)
class _DocumentAlignedLookup:
    starts_by_shard: dict[int, tuple[int, ...]]
    ends_by_shard: dict[int, tuple[int, ...]]
    first_id_by_shard: dict[int, int]

    @classmethod
    def from_manifest(cls, manifest: ExampleManifest) -> "_DocumentAlignedLookup":
        starts_by_shard: dict[int, list[int]] = {}
        ends_by_shard: dict[int, list[int]] = {}
        first_id_by_shard: dict[int, int] = {}
        for example in manifest.examples:
            shard_idx = int(example.shard_idx)
            starts_by_shard.setdefault(shard_idx, [])
            ends_by_shard.setdefault(shard_idx, [])
            first_id_by_shard.setdefault(shard_idx, int(example.global_example_id))
            starts_by_shard[shard_idx].append(int(example.token_offset_start))
            ends_by_shard[shard_idx].append(
                int(
                    example.document_token_offset_end
                    if example.document_token_offset_end is not None
                    else example.token_offset_end
                )
            )
        return cls(
            starts_by_shard={key: tuple(value) for key, value in starts_by_shard.items()},
            ends_by_shard={key: tuple(value) for key, value in ends_by_shard.items()},
            first_id_by_shard=first_id_by_shard,
        )

    def ids_overlapping_span(self, shard_idx: int, start: int, end: int) -> range:
        starts = self.starts_by_shard.get(int(shard_idx), ())
        ends = self.ends_by_shard.get(int(shard_idx), ())
        if not starts:
            return range(0, 0)
        left = bisect_right(ends, int(start))
        right = bisect_left(starts, int(end))
        if right <= left:
            return range(0, 0)
        first_id = self.first_id_by_shard[int(shard_idx)]
        return range(first_id + left, first_id + right)


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def _iter_exposure_files(run_dir: Path) -> Iterable[Path]:
    exposure_dir = run_dir / "exposures"
    if not exposure_dir.is_dir():
        raise FileNotFoundError(f"Exposure directory not found: {exposure_dir}")
    yield from sorted(exposure_dir.glob("exposures_rank*.jsonl"))


def _bos_packed_ids_from_micro_batch(micro_batch: dict, manifest: ExampleManifest) -> range:
    shard_idx = int(micro_batch["shard_idx"])
    start = int(micro_batch["start"])
    end = int(micro_batch["end"])
    if start % manifest.example_tokens != 0 or end % manifest.example_tokens != 0:
        raise ValueError(
            f"Exposure token offsets must align to example_tokens={manifest.example_tokens}: "
            f"start={start}, end={end}"
        )
    example_start = start // manifest.example_tokens
    example_end = end // manifest.example_tokens
    shard_offset = manifest.shard_example_offsets[shard_idx]
    return range(shard_offset + example_start, shard_offset + example_end)


def _stream_window_ids_from_micro_batch(micro_batch: dict, manifest: ExampleManifest) -> range:
    """Expand one logged stream batch block into the exact per-sequence windows."""

    shard_idx = int(micro_batch["shard_idx"])
    start = int(micro_batch["start"])
    end = int(micro_batch["end"])
    span = int(end - start)
    if span < manifest.example_tokens:
        raise ValueError(
            f"Stream exposure span {span} is shorter than one training example of {manifest.example_tokens} tokens."
        )
    if start % manifest.token_stride != 0:
        raise ValueError(
            f"Stream exposure start={start} does not align to token_stride={manifest.token_stride}."
        )
    if (span - manifest.example_tokens) % manifest.token_stride != 0:
        raise ValueError(
            "Stream exposure span does not match the expected batch-block pattern "
            f"for seq_len={manifest.seq_len}: start={start}, end={end}, span={span}."
        )

    local_example_start = start // manifest.token_stride
    num_examples = 1 + (span - manifest.example_tokens) // manifest.token_stride
    shard_offset = manifest.shard_example_offsets[shard_idx]
    return range(shard_offset + local_example_start, shard_offset + local_example_start + num_examples)


def _document_aligned_ids_from_micro_batch(
    micro_batch: dict,
    lookup: _DocumentAlignedLookup,
) -> range:
    shard_idx = int(micro_batch["shard_idx"])
    start = int(micro_batch["start"])
    end = int(micro_batch["end"])
    return lookup.ids_overlapping_span(shard_idx, start, end)


def _example_ids_from_micro_batch(
    micro_batch: dict,
    manifest: ExampleManifest,
    document_lookup: _DocumentAlignedLookup | None,
) -> range:
    if manifest.candidate_kind == "bos_packed_row":
        return _bos_packed_ids_from_micro_batch(micro_batch, manifest)
    if manifest.candidate_kind == "document_aligned_row":
        if document_lookup is None:
            raise ValueError("document_lookup is required for document_aligned_row exposure indexing")
        return _document_aligned_ids_from_micro_batch(micro_batch, document_lookup)
    return _stream_window_ids_from_micro_batch(micro_batch, manifest)


def build_exposure_index(
    run_dir: str | Path,
    manifest: ExampleManifest,
    *,
    show_progress: bool = False,
) -> ExposureIndex:
    run_path = Path(run_dir).expanduser().resolve()
    step_to_example_ids: dict[int, set[int]] = {}
    first_seen_step_by_example_id: dict[int, int] = {}
    document_lookup = (
        _DocumentAlignedLookup.from_manifest(manifest)
        if manifest.candidate_kind == "document_aligned_row"
        else None
    )

    exposure_files = tuple(_iter_exposure_files(run_path))
    progress = _build_tqdm(
        enabled=show_progress and len(exposure_files) > 1,
        total=len(exposure_files),
        desc="Indexing exposure logs",
        unit="file",
    )
    try:
        for exposure_file in exposure_files:
            if progress is not None:
                progress.set_postfix_str(exposure_file.name)
            with exposure_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    step = int(payload["step"])
                    step_examples = step_to_example_ids.setdefault(step, set())
                    for micro_batch in payload.get("micro_batches", []):
                        for example_id in _example_ids_from_micro_batch(
                            micro_batch,
                            manifest,
                            document_lookup,
                        ):
                            step_examples.add(int(example_id))
                            if (
                                example_id not in first_seen_step_by_example_id
                                or step < first_seen_step_by_example_id[example_id]
                            ):
                                first_seen_step_by_example_id[int(example_id)] = step
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return ExposureIndex(
        run_dir=run_path,
        step_to_example_ids={step: tuple(sorted(ids)) for step, ids in sorted(step_to_example_ids.items())},
        first_seen_step_by_example_id=first_seen_step_by_example_id,
    )


__all__ = [
    "ExposureIndex",
    "build_exposure_index",
]
