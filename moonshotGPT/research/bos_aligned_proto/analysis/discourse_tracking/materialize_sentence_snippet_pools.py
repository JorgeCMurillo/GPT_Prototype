#!/usr/bin/env python3
"""Materialize selector-scored sentence snippets into CPT ablation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..attribution.common.checkpoints import load_tokenizer_from_checkpoint
from ..attribution.trackstar.cpt_ablation import load_checkpoint_model_config
from .selector_recipes import SELECTOR_NAMES, selector_sort_columns


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must expose bos_token_id, eos_token_id, or pad_token_id")


def _tokenize_text(tokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return [int(value) for value in tokens]


def _unique_in_order(values: Sequence[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(str(value))
        result.append(str(value))
    return result


def _to_uint16_array(values: Sequence[int]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.uint16)
    max_value = max(int(value) for value in values)
    min_value = min(int(value) for value in values)
    if min_value < 0 or max_value > np.iinfo(np.uint16).max:
        raise ValueError(f"Token ids must fit uint16, got min={min_value}, max={max_value}")
    return np.asarray(values, dtype=np.uint16)


def pack_snippets_to_rows(
    snippets: pd.DataFrame,
    *,
    text_lookup: dict[str, str],
    tokenizer,
    seq_len: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Pack BOS-separated snippets into fixed `seq_len + 1` token rows."""
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be > 0")
    row_tokens = int(seq_len) + 1
    bos_token_id = _resolve_bos_token_id(tokenizer)
    token_stream: list[int] = []
    source_stream: list[str | None] = []
    used_snippet_ids: list[str] = []
    skipped_overlong: list[str] = []
    tokenized_snippet_count = 0

    for row in snippets.itertuples(index=False):
        window_id = str(row.window_id)
        text = text_lookup.get(window_id)
        if text is None:
            raise KeyError(f"Missing snippet text for window_id={window_id!r}")
        snippet_tokens = _tokenize_text(tokenizer, text)
        if len(snippet_tokens) > int(seq_len):
            skipped_overlong.append(window_id)
            continue
        token_stream.append(int(bos_token_id))
        source_stream.append(None)
        token_stream.extend(snippet_tokens)
        source_stream.extend([window_id] * len(snippet_tokens))
        used_snippet_ids.append(window_id)
        tokenized_snippet_count += 1

    # Each emitted row starts with BOS for bos_row compatibility. The remaining
    # seq_len positions are filled from the BOS-separated snippet token stream.
    full_row_count = len(token_stream) // int(seq_len)
    used_token_count = full_row_count * int(seq_len)
    dropped_tail_tokens = len(token_stream) - used_token_count
    token_stream = token_stream[:used_token_count]
    source_stream = source_stream[:used_token_count]
    if full_row_count <= 0:
        return (
            np.empty((0, row_tokens), dtype=np.uint16),
            [],
            {
                "tokenized_snippet_count": int(tokenized_snippet_count),
                "used_snippet_count": 0,
                "skipped_overlong_count": int(len(skipped_overlong)),
                "dropped_tail_tokens": int(dropped_tail_tokens),
                "packed_row_count": 0,
            },
        )

    packed_rows: list[list[int]] = []
    row_records: list[dict[str, Any]] = []
    used_in_rows: set[str] = set()
    for row_idx in range(full_row_count):
        start = row_idx * int(seq_len)
        end = start + int(seq_len)
        packed_rows.append([int(bos_token_id), *token_stream[start:end]])
        snippet_ids = _unique_in_order(source_stream[start:end])
        used_in_rows.update(snippet_ids)
        row_records.append(
            {
                "pool_row_index": int(row_idx),
                "packed_snippet_ids": snippet_ids,
                "packed_snippet_count": int(len(snippet_ids)),
                "token_stream_start": int(start),
                "token_stream_end": int(end),
            }
        )
    rows = _to_uint16_array([token for row in packed_rows for token in row]).reshape(full_row_count, row_tokens)

    stats = {
        "tokenized_snippet_count": int(tokenized_snippet_count),
        "used_snippet_count": int(len(used_in_rows)),
        "skipped_overlong_count": int(len(skipped_overlong)),
        "skipped_overlong_examples": skipped_overlong[:20],
        "dropped_tail_tokens": int(dropped_tail_tokens),
        "packed_row_count": int(full_row_count),
        "row_tokens": int(row_tokens),
    }
    return rows, row_records, stats


def _write_row_shards(
    *,
    rows: np.ndarray,
    row_records: list[dict[str, Any]],
    output_dir: Path,
    rows_per_shard: int,
    pool_role: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    rows_per_shard = max(1, int(rows_per_shard))
    for shard_idx, start in enumerate(range(0, len(rows), rows_per_shard)):
        end = min(len(rows), start + rows_per_shard)
        path = output_dir / f"train_{shard_idx:06d}.bin"
        rows[start:end].reshape(-1).astype(np.uint16, copy=False).tofile(path)
        shard_paths.append(str(path))

    records = []
    for record in row_records:
        records.append({**record, "pool_role": pool_role})
    _write_jsonl(output_dir / "rows.jsonl", records)
    _write_json(output_dir / "meta.json", meta)
    return {
        "data_dir": str(output_dir),
        "meta": str(output_dir / "meta.json"),
        "rows": str(output_dir / "rows.jsonl"),
        "train_shards": shard_paths,
    }


def _load_text_lookup(snippet_text_jsonl: str | Path) -> dict[str, str]:
    rows = _load_jsonl(snippet_text_jsonl)
    lookup: dict[str, str] = {}
    for row in rows:
        lookup[str(row["window_id"])] = str(row["text"])
    return lookup


def _select_pools(
    frame: pd.DataFrame,
    *,
    selector: str,
    num_treated_snippets: int | None,
    num_control_snippets: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated_col = f"is_treated_{selector}"
    if treated_col not in frame.columns:
        raise ValueError(f"Snippet feature CSV is missing {treated_col!r}")
    selector_control_col = f"is_control_{selector}"
    if selector_control_col in frame.columns:
        control_col = selector_control_col
    elif "is_random_control_pool" in frame.columns:
        control_col = "is_random_control_pool"
    else:
        raise ValueError(
            "Snippet feature CSV is missing selector-specific control column "
            f"{selector_control_col!r} and fallback 'is_random_control_pool'"
        )

    sort_columns, ascending = selector_sort_columns(selector)
    treated = frame.loc[frame[treated_col].astype(bool)].sort_values(sort_columns, ascending=ascending).copy()
    control = frame.loc[frame[control_col].astype(bool)].copy()
    if num_treated_snippets is not None and int(num_treated_snippets) > 0:
        treated = treated.head(int(num_treated_snippets)).copy()
    if num_control_snippets is not None and int(num_control_snippets) > 0:
        control = control.sample(
            n=min(int(num_control_snippets), len(control)),
            replace=False,
            random_state=int(seed),
        ).copy()
    if treated.empty:
        raise ValueError(f"Selector {selector!r} has no treated snippets")
    if control.empty:
        raise ValueError("Random control pool is empty")
    return treated.reset_index(drop=True), control.reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack mined sentence snippets into treated/control bos_row-compatible "
            "datasets for the existing TrackStar CPT ablation runner."
        )
    )
    parser.add_argument("--snippet_features_csv", required=True)
    parser.add_argument("--snippet_text_jsonl", required=True)
    parser.add_argument("--selector", required=True, choices=SELECTOR_NAMES)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--rows_per_shard", type=int, default=50_000)
    parser.add_argument("--num_treated_snippets", type=int, default=None)
    parser.add_argument("--num_control_snippets", type=int, default=None)
    parser.add_argument("--selection_seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    model_config = load_checkpoint_model_config(checkpoint_dir)
    seq_len = int(args.seq_len or model_config.seq_len)
    if seq_len <= 0:
        raise ValueError("--seq_len must be > 0")
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)
    text_lookup = _load_text_lookup(args.snippet_text_jsonl)
    frame = pd.read_csv(args.snippet_features_csv)

    treated, control = _select_pools(
        frame,
        selector=str(args.selector),
        num_treated_snippets=args.num_treated_snippets,
        num_control_snippets=args.num_control_snippets,
        seed=int(args.selection_seed),
    )

    treated_rows, treated_row_records, treated_stats = pack_snippets_to_rows(
        treated,
        text_lookup=text_lookup,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
    control_rows, control_row_records, control_stats = pack_snippets_to_rows(
        control,
        text_lookup=text_lookup,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    equal_rows = min(len(treated_rows), len(control_rows))
    if equal_rows <= 0:
        raise ValueError(
            "Could not pack at least one full row for both arms. "
            f"treated_rows={len(treated_rows)}, control_rows={len(control_rows)}"
        )
    treated_rows = treated_rows[:equal_rows]
    control_rows = control_rows[:equal_rows]
    treated_row_records = treated_row_records[:equal_rows]
    control_row_records = control_row_records[:equal_rows]

    base_meta = {
        "format": "sentence_snippet_row_packed",
        "seq_len": int(seq_len),
        "row_tokens": int(seq_len) + 1,
        "num_rows": int(equal_rows),
        "rows_per_shard": int(args.rows_per_shard),
        "row_semantics": "packed_sentence_snippets",
        "packing_strategy": "bos_separated_token_stream_drop_tail",
        "tokenizer": str(checkpoint_dir),
        "use_fast": True,
        "source_snippet_features_csv": str(Path(args.snippet_features_csv).expanduser().resolve()),
        "source_snippet_text_jsonl": str(Path(args.snippet_text_jsonl).expanduser().resolve()),
        "selector": str(args.selector),
    }
    treated_artifacts = _write_row_shards(
        rows=treated_rows,
        row_records=treated_row_records,
        output_dir=output_dir / "treated_dataset",
        rows_per_shard=int(args.rows_per_shard),
        pool_role="treated",
        meta={**base_meta, "pool_role": "treated", "packing_stats": treated_stats},
    )
    control_artifacts = _write_row_shards(
        rows=control_rows,
        row_records=control_row_records,
        output_dir=output_dir / "control_dataset",
        rows_per_shard=int(args.rows_per_shard),
        pool_role="control",
        meta={**base_meta, "pool_role": "control", "packing_stats": control_stats},
    )

    treated.to_csv(output_dir / "treated_snippets.csv", index=False)
    control.to_csv(output_dir / "control_snippets.csv", index=False)

    summary = {
        "selector": str(args.selector),
        "checkpoint_dir": str(checkpoint_dir),
        "seq_len": int(seq_len),
        "row_tokens": int(seq_len) + 1,
        "num_rows": int(equal_rows),
        "treated_input_snippets": int(len(treated)),
        "control_input_snippets": int(len(control)),
        "treated_packing_stats": treated_stats,
        "control_packing_stats": control_stats,
        "truncated_to_equal_rows": {
            "treated_rows_before_truncation": int(treated_stats["packed_row_count"]),
            "control_rows_before_truncation": int(control_stats["packed_row_count"]),
            "equal_rows": int(equal_rows),
        },
        "artifacts": {
            "treated_dataset": treated_artifacts["data_dir"],
            "control_dataset": control_artifacts["data_dir"],
            "treated_snippets": str(output_dir / "treated_snippets.csv"),
            "control_snippets": str(output_dir / "control_snippets.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
