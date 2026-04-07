#!/usr/bin/env python3
"""Sample exact training examples directly from a sharded dataset.

This is the dataset-native companion to ``mine_candidate_pools.py``.
It writes the same candidate schema that the miner already accepts, but
without requiring a pre-existing attribution ``row_summary`` export.

Sampling is exact and reproducible:

- every training example in the chosen split is assigned a global id;
- a fixed RNG seed samples global ids uniformly without replacement;
- sampled ids are mapped back to shard/local offsets using shard counts.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

import pandas as pd

from ..attribution.common.training_examples import PACKED_INDEX_FORMAT, infer_candidate_kind


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_shards(data_dir: Path, split: str) -> tuple[Path, ...]:
    paths = sorted(data_dir.glob(f"{split}_*.bin"))
    if not paths:
        raise FileNotFoundError(f"No shards found for split={split!r} under {data_dir}")
    return tuple(paths)


def _list_virtual_shards(data_dir: Path, split: str) -> tuple[dict[str, Any], ...]:
    manifest_path = data_dir / f"{split}.virtual_shards.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Virtual shard manifest not found for split={split!r} under {data_dir}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if not rows:
        raise FileNotFoundError(f"Virtual shard manifest is empty for split={split!r} under {data_dir}")
    return tuple(rows)


def _count_tokens(path: Path) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte size is not divisible by 2: {path}")
    return nbytes // 2


def _stream_examples_per_shard(n_tokens: int, seq_len: int) -> int:
    example_tokens = int(seq_len) + 1
    if n_tokens < example_tokens:
        return 0
    return 1 + (n_tokens - example_tokens) // int(seq_len)


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _resolve_candidate_kind_text(raw: str | None) -> str:
    if raw is None:
        return "auto"
    value = str(raw).strip().lower()
    if value in {"auto", "bos_packed", "bos", "bos_packed_row", "stream", "stream_window"}:
        return value
    raise ValueError(f"Unsupported candidate_kind={raw!r}; expected auto, bos_packed, or stream_window")


def _sample_global_ids(*, total_examples: int, num_samples: int, seed: int) -> tuple[list[int], dict[int, int]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if total_examples <= 0:
        raise ValueError("Dataset split exposes zero training examples")
    if num_samples > total_examples:
        raise ValueError(f"Requested {num_samples} samples, but only {total_examples} examples exist")

    rng = random.Random(int(seed))
    sampled_in_rng_order = [int(value) for value in rng.sample(range(int(total_examples)), int(num_samples))]
    sample_order = {int(global_id): int(order) for order, global_id in enumerate(sampled_in_rng_order)}
    sampled_sorted = sorted(sampled_in_rng_order)
    return sampled_sorted, sample_order


def _build_sample_frame(
    *,
    data_dir: str | Path,
    split: str,
    candidate_kind: str,
    seq_len: int | None,
    num_samples: int,
    seed: int,
    show_progress: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path = Path(data_dir).expanduser().resolve()
    meta = _load_json(data_path / "meta.json")
    resolved_kind = infer_candidate_kind(data_path, preferred_kind=_resolve_candidate_kind_text(candidate_kind))
    data_format = str(meta.get("format", ""))

    rows: list[dict[str, Any]] = []
    shard_paths: tuple[Path, ...]
    example_counts: list[int] = []

    if resolved_kind == "stream_window":
        if seq_len is None or int(seq_len) <= 0:
            raise ValueError("Stream-window sampling requires --seq_len")
        resolved_seq_len = int(seq_len)
        example_tokens = int(resolved_seq_len) + 1
        token_stride = int(resolved_seq_len)
        shard_paths = _list_shards(data_path, split)
        progress = _build_tqdm(enabled=show_progress, total=len(shard_paths), desc="Counting shard windows", unit="shard")
        try:
            for shard_path in shard_paths:
                example_counts.append(_stream_examples_per_shard(_count_tokens(shard_path), resolved_seq_len))
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    else:
        resolved_seq_len = int(meta["seq_len"])
        example_tokens = int(meta["row_tokens"])
        token_stride = int(example_tokens)
        if data_format == PACKED_INDEX_FORMAT:
            virtual_shards = _list_virtual_shards(data_path, split)
            shard_paths = tuple(Path(str(row["shard_path"])) for row in virtual_shards)
            example_counts = [int(row["row_count"]) for row in virtual_shards]
            running = 0
            for shard_row, count in zip(virtual_shards, example_counts):
                expected_start = int(shard_row.get("row_start", running))
                if expected_start != running:
                    raise ValueError(
                        "Packed-index virtual shards are not contiguous as expected: "
                        f"row_start={expected_start} vs running_offset={running}"
                    )
                running += int(count)
        else:
            shard_paths = _list_shards(data_path, split)
            progress = _build_tqdm(enabled=show_progress, total=len(shard_paths), desc="Counting row shards", unit="shard")
            try:
                for shard_path in shard_paths:
                    n_tokens = _count_tokens(shard_path)
                    if n_tokens % int(example_tokens) != 0:
                        raise ValueError(
                            f"Shard token count {n_tokens} is not divisible by row_tokens {example_tokens}: {shard_path}"
                        )
                    example_counts.append(n_tokens // int(example_tokens))
                    if progress is not None:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

    shard_offsets = [0]
    for count in example_counts:
        shard_offsets.append(int(shard_offsets[-1]) + int(count))
    total_examples = int(shard_offsets[-1])

    sampled_global_ids, sample_order = _sample_global_ids(
        total_examples=total_examples,
        num_samples=int(num_samples),
        seed=int(seed),
    )

    progress = _build_tqdm(enabled=show_progress, total=len(sampled_global_ids), desc="Mapping sampled examples", unit="sample")
    try:
        for global_id in sampled_global_ids:
            shard_idx = bisect.bisect_right(shard_offsets, int(global_id)) - 1
            local_example_idx = int(global_id) - int(shard_offsets[shard_idx])
            token_offset_start = int(local_example_idx) * int(token_stride)
            token_offset_end = int(token_offset_start) + int(example_tokens)
            rows.append(
                {
                    "candidate_id": int(global_id),
                    "candidate_kind": str(resolved_kind),
                    "shard_idx": int(shard_idx),
                    "shard_path": str(shard_paths[shard_idx]),
                    "local_example_idx": int(local_example_idx),
                    "token_offset_start": int(token_offset_start),
                    "token_offset_end": int(token_offset_end),
                    "token_count": int(example_tokens),
                    "sample_seed": int(seed),
                    "sample_order": int(sample_order[int(global_id)]),
                }
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    frame = pd.DataFrame.from_records(rows).sort_values(
        ["shard_idx", "local_example_idx", "candidate_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary = {
        "data_dir": str(data_path),
        "split": str(split),
        "candidate_kind": str(resolved_kind),
        "seq_len": int(resolved_seq_len),
        "example_tokens": int(example_tokens),
        "token_stride": int(token_stride),
        "data_format": data_format,
        "num_shards": int(len(shard_paths)),
        "total_examples": int(total_examples),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "shards_touched": int(frame["shard_idx"].nunique()) if not frame.empty else 0,
    }
    return frame, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample exact training examples directly from a sharded dataset with "
            "a fixed seed and export the candidate CSV schema used by "
            "mine_candidate_pools.py."
        )
    )
    parser.add_argument("--data_dir", required=True, help="Dataset root, for example fineweb_edu_10B or a BOS-packed view")
    parser.add_argument("--output_csv", required=True, help="Where to write the sampled candidate CSV")
    parser.add_argument("--output_summary", default=None, help="Optional summary JSON path; defaults next to output_csv")
    parser.add_argument("--split", default="train", help="Dataset split, usually train")
    parser.add_argument(
        "--candidate_kind",
        default="auto",
        help="auto, stream_window, or bos_packed_row",
    )
    parser.add_argument("--seq_len", type=int, default=None, help="Required for stream-window sampling")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of exact training examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Fixed RNG seed for exact reproducibility")
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary = (
        Path(args.output_summary).expanduser().resolve()
        if args.output_summary is not None
        else output_csv.with_suffix(".summary.json")
    )

    frame, summary = _build_sample_frame(
        data_dir=args.data_dir,
        split=str(args.split),
        candidate_kind=str(args.candidate_kind),
        seq_len=args.seq_len,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        show_progress=bool(args.show_progress),
    )
    frame.to_csv(output_csv, index=False)
    summary["artifacts"] = {
        "candidate_csv": str(output_csv),
        "summary_json": str(output_summary),
    }
    _write_json(output_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
