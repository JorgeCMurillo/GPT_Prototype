#!/usr/bin/env python3
"""Materialize mined discourse-tracking pools into exact CPT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..attribution.build_matched_cpt_pools import match_control_candidates
from ..attribution.common.training_examples import FiniteTrainingExampleDataset, build_example_manifest


POOL_TO_COLUMN = {
    "positive": "is_positive_pool",
    "random_control": "is_random_control_pool",
    "negative_low_binding": "is_negative_low_binding_pool",
    "negative_repetition": "is_negative_repetition_pool",
    "selected_cluster": "is_selected_cluster_pool",
}


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


def _full_tokens_from_sample(sample: dict[str, Any]) -> np.ndarray:
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Expected tensor-valued input_ids and labels from FiniteTrainingExampleDataset")
    full_tokens = torch.cat([input_ids, labels[-1:].clone()], dim=0)
    return np.asarray(full_tokens.detach().cpu(), dtype=np.uint16)


def _infer_seq_len(frame: pd.DataFrame, seq_len: int | None) -> int:
    if seq_len is not None:
        return int(seq_len)
    token_counts = sorted(int(value) for value in frame["token_count"].dropna().unique().tolist())
    if len(token_counts) != 1:
        raise ValueError(
            "Could not infer one seq_len from token_count values. "
            f"Found {token_counts!r}; pass --seq_len explicitly."
        )
    token_count = int(token_counts[0])
    if token_count <= 1:
        raise ValueError(f"token_count must be > 1, got {token_count}")
    return token_count - 1


def _write_row_packed_shards(
    *,
    output_dir: Path,
    manifest,
    candidate_frame: pd.DataFrame,
    rows_per_shard: int,
    pool_role: str,
    extra_meta: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = tuple(int(value) for value in candidate_frame["candidate_id"].tolist())
    dataset = FiniteTrainingExampleDataset(manifest, candidate_ids)
    shard_paths: list[Path] = []
    row_records: list[dict[str, Any]] = []
    buffer: list[np.ndarray] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        path = output_dir / f"train_{shard_idx:06d}.bin"
        np.concatenate(buffer, axis=0).astype(np.uint16, copy=False).tofile(path)
        shard_paths.append(path)
        shard_idx += 1
        buffer = []

    for row_order, row in enumerate(candidate_frame.itertuples(index=False)):
        sample = dataset[row_order]
        full_tokens = _full_tokens_from_sample(sample)
        buffer.append(full_tokens)
        row_records.append(
            {
                "pool_role": pool_role,
                "pool_row_index": int(row_order),
                "candidate_id": int(row.candidate_id),
                "selection_score": float(row.selection_score),
                "priority_score": float(row.priority_score),
                "candidate_kind": str(row.candidate_kind),
                "shard_path": str(row.shard_path),
                "local_example_idx": int(row.local_example_idx),
                "token_offset_start": int(row.token_offset_start),
                "token_offset_end": int(row.token_offset_end),
                "token_count": int(row.token_count),
                "pool_label": str(row.pool_label),
                "text_preview": str(getattr(row, "text_preview", "")),
            }
        )
        if len(buffer) >= int(rows_per_shard):
            flush()

    flush()
    meta = {
        "format": "exact_window_row_packed",
        "seq_len": int(manifest.seq_len),
        "row_tokens": int(manifest.example_tokens),
        "candidate_kind": str(manifest.candidate_kind),
        "num_rows": int(len(candidate_frame)),
        "rows_per_shard": int(rows_per_shard),
        "source_data_dir": str(manifest.data_dir),
        "source_format": str(manifest.format),
        "row_semantics": "exact_training_example",
        "pool_role": pool_role,
        **extra_meta,
    }
    meta_path = output_dir / "meta.json"
    rows_path = output_dir / "rows.jsonl"
    _write_json(meta_path, meta)
    _write_jsonl(rows_path, row_records)
    return {
        "data_dir": output_dir,
        "meta": meta_path,
        "rows": rows_path,
    }


def _build_balance_report(*, treated: pd.DataFrame, control: pd.DataFrame, pairings: pd.DataFrame) -> dict[str, Any]:
    score_gap = pairings["score_gap"].astype(float) if "score_gap" in pairings.columns and not pairings.empty else pd.Series(dtype=float)
    position_distance = (
        pairings["position_distance"].astype(float)
        if "position_distance" in pairings.columns and not pairings.empty
        else pd.Series(dtype=float)
    )
    return {
        "treated_total_tokens": int(treated["token_count"].sum()),
        "control_total_tokens": int(control["token_count"].sum()),
        "token_count_balance_exact": bool(
            treated["token_count"].sort_values().reset_index(drop=True).equals(
                control["token_count"].sort_values().reset_index(drop=True)
            )
        ),
        "same_shard_pair_fraction": (
            float((pairings["treated_shard_path"] == pairings["control_shard_path"]).mean()) if not pairings.empty else 0.0
        ),
        "mean_position_distance": float(position_distance.mean()) if not position_distance.empty else 0.0,
        "max_position_distance": float(position_distance.max()) if not position_distance.empty else 0.0,
        "mean_score_gap": float(score_gap.mean()) if not score_gap.empty else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn mined discourse-tracking pools into exact treated/control datasets "
            "for short continued-pretraining interventions."
        )
    )
    parser.add_argument("--features_csv", required=True, help="candidate_features.csv from mine_candidate_pools.py")
    parser.add_argument("--data_dir", required=True, help="Training data directory used by the candidate frame")
    parser.add_argument("--output_dir", required=True, help="Where to write treated/control datasets and matching reports")
    parser.add_argument("--treated_pool", choices=tuple(POOL_TO_COLUMN), default="positive")
    parser.add_argument("--control_pool", choices=tuple(POOL_TO_COLUMN), default="random_control")
    parser.add_argument("--num_treated", type=int, default=1024)
    parser.add_argument("--rows_per_shard", type=int, default=50_000)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument(
        "--max_control_score",
        type=float,
        default=None,
        help="Optional upper bound on control priority score. Leave unset for random or negative controls.",
    )
    parser.add_argument("--allow_relaxed_shard_match", action="store_true", default=True)
    parser.add_argument("--strict_shard_match", action="store_false", dest="allow_relaxed_shard_match")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.features_csv)
    missing_columns = [column for column in ("candidate_id", "candidate_kind", "token_count", "priority_score") if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Feature CSV is missing required columns: {missing_columns}")
    if frame["candidate_kind"].nunique() != 1:
        raise ValueError(
            "This materializer expects one candidate_kind per feature CSV. "
            f"Found {sorted(str(value) for value in frame['candidate_kind'].unique().tolist())!r}"
        )

    treated_col = POOL_TO_COLUMN[str(args.treated_pool)]
    control_col = POOL_TO_COLUMN[str(args.control_pool)]
    for column in (treated_col, control_col):
        if column not in frame.columns:
            raise ValueError(f"Feature CSV is missing pool column {column!r}")

    treated = frame.loc[frame[treated_col].astype(bool)].copy()
    control_candidates = frame.loc[frame[control_col].astype(bool)].copy()
    if treated.empty:
        raise ValueError(f"Treated pool {args.treated_pool!r} is empty")
    if control_candidates.empty:
        raise ValueError(f"Control pool {args.control_pool!r} is empty")

    treated["selection_score"] = treated["priority_score"].astype(float)
    control_candidates["selection_score"] = control_candidates["priority_score"].astype(float)
    treated = treated.sort_values(["selection_score", "candidate_id"], ascending=[False, True]).head(int(args.num_treated)).reset_index(drop=True)
    pairings, control = match_control_candidates(
        treated=treated,
        candidates=control_candidates,
        max_control_score=args.max_control_score,
        allow_relaxed_shard_match=bool(args.allow_relaxed_shard_match),
    )

    seq_len = _infer_seq_len(frame, args.seq_len)
    manifest = build_example_manifest(
        args.data_dir,
        split="train",
        candidate_kind=str(frame["candidate_kind"].iloc[0]),
        seq_len=seq_len,
    )

    treated_artifacts = _write_row_packed_shards(
        output_dir=output_dir / "treated_dataset",
        manifest=manifest,
        candidate_frame=treated,
        rows_per_shard=int(args.rows_per_shard),
        pool_role="treated",
        extra_meta={
            "selection": {
                "treated_pool": str(args.treated_pool),
                "num_treated": int(len(treated)),
            }
        },
    )
    control_artifacts = _write_row_packed_shards(
        output_dir=output_dir / "control_dataset",
        manifest=manifest,
        candidate_frame=control,
        rows_per_shard=int(args.rows_per_shard),
        pool_role="control",
        extra_meta={
            "selection": {
                "control_pool": str(args.control_pool),
                "matched_from": str(args.features_csv),
                "max_control_score": None if args.max_control_score is None else float(args.max_control_score),
            }
        },
    )

    treated.to_csv(output_dir / "treated_candidates.csv", index=False)
    control.to_csv(output_dir / "control_candidates.csv", index=False)
    pairings.to_csv(output_dir / "pairings.csv", index=False)

    summary = {
        "features_csv": str(Path(args.features_csv).expanduser().resolve()),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "treated_pool": str(args.treated_pool),
        "control_pool": str(args.control_pool),
        "num_treated": int(len(treated)),
        "num_control": int(len(control)),
        "candidate_kind": str(manifest.candidate_kind),
        "seq_len": int(manifest.seq_len),
        "allow_relaxed_shard_match": bool(args.allow_relaxed_shard_match),
        "max_control_score": None if args.max_control_score is None else float(args.max_control_score),
        "balance_report": _build_balance_report(treated=treated, control=control, pairings=pairings),
        "artifacts": {
            "treated_dataset": str(treated_artifacts["data_dir"]),
            "control_dataset": str(control_artifacts["data_dir"]),
            "treated_candidates": str(output_dir / "treated_candidates.csv"),
            "control_candidates": str(output_dir / "control_candidates.csv"),
            "pairings": str(output_dir / "pairings.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
