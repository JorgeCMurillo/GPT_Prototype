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


def _load_source_tokenizer_meta(source_data_dir: Path) -> dict[str, Any]:
    meta_path = source_data_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    tokenizer = str(meta.get("tokenizer") or "").strip()
    if not tokenizer:
        return {}
    payload: dict[str, Any] = {"tokenizer": tokenizer}
    if "use_fast" in meta:
        payload["use_fast"] = bool(meta["use_fast"])
    return payload


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
                "cluster_id": (
                    None
                    if pd.isna(getattr(row, "cluster_id", None))
                    else int(getattr(row, "cluster_id"))
                ),
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
        **_load_source_tokenizer_meta(Path(manifest.data_dir).expanduser().resolve()),
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


def _parse_cluster_ids(raw: str | None) -> list[int]:
    if raw is None:
        return []
    values: list[int] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _rank_treated_frame(
    candidate_frame: pd.DataFrame,
    *,
    selection_score: str,
    selection_seed: int,
) -> pd.DataFrame:
    ranked = candidate_frame.copy()
    if str(selection_score) == "random":
        rng = np.random.default_rng(int(selection_seed))
        ranked["selection_score"] = rng.random(len(ranked))
    else:
        ranked["selection_score"] = ranked["priority_score"].astype(float)
    return ranked.sort_values(["selection_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)


def _select_treated_subset(
    candidate_frame: pd.DataFrame,
    *,
    num_treated: int | None,
    selection_score: str,
    selection_seed: int,
) -> pd.DataFrame:
    ranked = _rank_treated_frame(
        candidate_frame,
        selection_score=selection_score,
        selection_seed=selection_seed,
    )
    if num_treated is None:
        return ranked.reset_index(drop=True)
    return ranked.head(min(int(num_treated), len(ranked))).reset_index(drop=True)


def _load_cluster_assignments(cluster_assignments_csv: str) -> pd.DataFrame:
    cluster_frame = pd.read_csv(cluster_assignments_csv)
    missing = [column for column in ("candidate_id", "cluster_id") if column not in cluster_frame.columns]
    if missing:
        raise ValueError(f"Cluster assignments CSV is missing required columns: {missing}")
    if cluster_frame["candidate_id"].duplicated().any():
        raise ValueError("Cluster assignments CSV has duplicate candidate_id rows")
    keep_columns = [column for column in ("candidate_id", "cluster_id", "is_selected_cluster") if column in cluster_frame.columns]
    return cluster_frame[keep_columns].copy()


def _attach_cluster_assignments(frame: pd.DataFrame, cluster_assignments_csv: str) -> pd.DataFrame:
    cluster_frame = _load_cluster_assignments(cluster_assignments_csv)
    merged = frame.merge(cluster_frame, on="candidate_id", how="left")
    if "cluster_id" in merged.columns:
        merged["cluster_id"] = merged["cluster_id"].astype("Int64")
    return merged


def _build_balanced_cluster_mix(
    cluster_frames: dict[int, pd.DataFrame],
    *,
    target_total: int,
    selection_score: str,
    selection_seed: int,
) -> tuple[pd.DataFrame, dict[int, int]]:
    if not cluster_frames:
        raise ValueError("cluster_frames must not be empty")
    ranked_by_cluster: dict[int, pd.DataFrame] = {}
    for offset, cluster_id in enumerate(sorted(cluster_frames)):
        ranked_by_cluster[int(cluster_id)] = _rank_treated_frame(
            cluster_frames[int(cluster_id)],
            selection_score=selection_score,
            selection_seed=int(selection_seed) + 1009 * (offset + 1),
        )

    total_available = int(sum(len(frame) for frame in ranked_by_cluster.values()))
    if total_available <= 0:
        raise ValueError("No treated examples available across requested clusters")
    remaining = min(int(target_total), total_available)
    allocation = {int(cluster_id): 0 for cluster_id in ranked_by_cluster}
    active = [int(cluster_id) for cluster_id in sorted(ranked_by_cluster)]

    while remaining > 0 and active:
        quota = max(1, remaining // len(active))
        progress = False
        next_active: list[int] = []
        for cluster_id in active:
            available = len(ranked_by_cluster[cluster_id]) - allocation[cluster_id]
            if available <= 0:
                continue
            take = min(quota, available, remaining)
            if take > 0:
                allocation[cluster_id] += take
                remaining -= take
                progress = True
            if allocation[cluster_id] < len(ranked_by_cluster[cluster_id]):
                next_active.append(cluster_id)
            if remaining <= 0:
                break
        if not progress:
            break
        active = next_active

    selected_parts: list[pd.DataFrame] = []
    for cluster_id in sorted(ranked_by_cluster):
        take = int(allocation[cluster_id])
        if take <= 0:
            continue
        part = ranked_by_cluster[cluster_id].head(take).copy()
        part["cluster_mix_source_id"] = int(cluster_id)
        part["cluster_mix_rank"] = np.arange(len(part), dtype=int)
        selected_parts.append(part)

    if not selected_parts:
        raise ValueError("Balanced cluster mix selected zero examples")

    mixed = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(["cluster_mix_rank", "cluster_id", "selection_score"], ascending=[True, True, False])
        .drop(columns=["cluster_mix_rank"], errors="ignore")
        .reset_index(drop=True)
    )
    return mixed, allocation


def _materialize_condition(
    *,
    output_dir: Path,
    args,
    frame: pd.DataFrame,
    treated: pd.DataFrame,
    control_candidates: pd.DataFrame,
    extra_summary: dict[str, Any] | None = None,
    treated_meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control_ranked = control_candidates.copy()
    control_ranked["selection_score"] = control_ranked["priority_score"].astype(float)
    pairings, control = match_control_candidates(
        treated=treated,
        candidates=control_ranked,
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

    treated_selection_meta = {
        "treated_pool": str(args.treated_pool),
        "num_treated": int(len(treated)),
        "selection_score": str(args.selection_score),
        "selection_seed": int(args.selection_seed),
    }
    if treated_meta_extra:
        treated_selection_meta.update(treated_meta_extra)

    treated_artifacts = _write_row_packed_shards(
        output_dir=output_dir / "treated_dataset",
        manifest=manifest,
        candidate_frame=treated,
        rows_per_shard=int(args.rows_per_shard),
        pool_role="treated",
        extra_meta={"selection": treated_selection_meta},
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
        "selection_score": str(args.selection_score),
        "selection_seed": int(args.selection_seed),
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
    if extra_summary:
        summary.update(extra_summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


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
    parser.add_argument(
        "--cluster_assignments_csv",
        type=str,
        default=None,
        help="Optional cluster_assignments.csv from mine_candidate_pools.py. Required for --cluster_ids batch materialization.",
    )
    parser.add_argument(
        "--cluster_ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated cluster IDs to materialize individually. "
            "When set, the script writes one matched-pool subdirectory per cluster plus a balanced mixed-cluster pool."
        ),
    )
    parser.add_argument(
        "--per_cluster_num_treated",
        type=int,
        default=None,
        help="Optional cap for each individual cluster condition. Defaults to using all available rows from that cluster.",
    )
    parser.add_argument(
        "--cluster_mix_num_treated",
        type=int,
        default=None,
        help="Target treated size for the balanced mixed-cluster condition. Defaults to --num_treated.",
    )
    parser.add_argument(
        "--skip_cluster_mix",
        action="store_true",
        default=False,
        help="Only materialize individual clusters, not the balanced mixed-cluster condition.",
    )
    parser.add_argument(
        "--selection_score",
        choices=("priority_score", "random"),
        default="priority_score",
        help=(
            "How to choose treated examples when num_treated is smaller than the treated pool. "
            "'priority_score' keeps the highest-priority rows; 'random' draws a reproducible random subset."
        ),
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Random seed used when --selection_score random.",
    )
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

    cluster_ids = _parse_cluster_ids(args.cluster_ids)
    if cluster_ids:
        if args.cluster_assignments_csv is None:
            raise ValueError("--cluster_assignments_csv is required when --cluster_ids is set")
        frame = _attach_cluster_assignments(frame, args.cluster_assignments_csv)
        treated = frame.loc[frame[treated_col].astype(bool)].copy()
        control_candidates = frame.loc[frame[control_col].astype(bool)].copy()
        if "cluster_id" not in treated.columns:
            raise ValueError("Cluster-aware materialization requires cluster_id assignments")

        batch_summary: dict[str, Any] = {
            "mode": "cluster_batch_materialization",
            "features_csv": str(Path(args.features_csv).expanduser().resolve()),
            "cluster_assignments_csv": str(Path(args.cluster_assignments_csv).expanduser().resolve()),
            "treated_pool": str(args.treated_pool),
            "control_pool": str(args.control_pool),
            "selection_score": str(args.selection_score),
            "selection_seed": int(args.selection_seed),
            "requested_cluster_ids": [int(value) for value in cluster_ids],
            "conditions": [],
        }

        cluster_frames: dict[int, pd.DataFrame] = {}
        for cluster_id in cluster_ids:
            cluster_frame = treated.loc[treated["cluster_id"] == int(cluster_id)].copy()
            if cluster_frame.empty:
                raise ValueError(
                    f"Cluster {cluster_id} has no rows in treated pool {args.treated_pool!r}. "
                    "Make sure the cluster assignments CSV matches the mined run used for features_csv."
                )
            cluster_frames[int(cluster_id)] = cluster_frame
            cluster_output_dir = output_dir / f"cluster_{int(cluster_id)}"
            cluster_treated = _select_treated_subset(
                cluster_frame,
                num_treated=args.per_cluster_num_treated,
                selection_score=str(args.selection_score),
                selection_seed=int(args.selection_seed) + 97 * int(cluster_id),
            )
            cluster_summary = _materialize_condition(
                output_dir=cluster_output_dir,
                args=args,
                frame=frame,
                treated=cluster_treated,
                control_candidates=control_candidates,
                extra_summary={
                    "condition_kind": "single_cluster",
                    "cluster_id": int(cluster_id),
                    "cluster_available": int(len(cluster_frame)),
                },
                treated_meta_extra={
                    "cluster_id": int(cluster_id),
                    "cluster_available": int(len(cluster_frame)),
                },
            )
            batch_summary["conditions"].append(
                {
                    "condition_kind": "single_cluster",
                    "cluster_id": int(cluster_id),
                    "available": int(len(cluster_frame)),
                    "num_treated": int(cluster_summary["num_treated"]),
                    "output_dir": str(cluster_output_dir),
                }
            )

        if not bool(args.skip_cluster_mix):
            mix_target = int(args.cluster_mix_num_treated or args.num_treated)
            mixed_treated, allocation = _build_balanced_cluster_mix(
                cluster_frames,
                target_total=mix_target,
                selection_score=str(args.selection_score),
                selection_seed=int(args.selection_seed),
            )
            mix_output_dir = output_dir / "cluster_mix"
            mix_summary = _materialize_condition(
                output_dir=mix_output_dir,
                args=args,
                frame=frame,
                treated=mixed_treated,
                control_candidates=control_candidates,
                extra_summary={
                    "condition_kind": "cluster_mix",
                    "cluster_ids": [int(value) for value in sorted(cluster_frames)],
                    "cluster_mix_target": int(mix_target),
                    "cluster_mix_allocation": {str(k): int(v) for k, v in allocation.items()},
                },
                treated_meta_extra={
                    "cluster_ids": [int(value) for value in sorted(cluster_frames)],
                    "cluster_mix_target": int(mix_target),
                    "cluster_mix_allocation": {str(k): int(v) for k, v in allocation.items()},
                },
            )
            batch_summary["conditions"].append(
                {
                    "condition_kind": "cluster_mix",
                    "cluster_ids": [int(value) for value in sorted(cluster_frames)],
                    "num_treated": int(mix_summary["num_treated"]),
                    "cluster_mix_target": int(mix_target),
                    "cluster_mix_allocation": {str(k): int(v) for k, v in allocation.items()},
                    "output_dir": str(mix_output_dir),
                }
            )

        _write_json(output_dir / "summary.json", batch_summary)
        print(json.dumps(batch_summary, indent=2))
        return 0

    treated = _select_treated_subset(
        treated,
        num_treated=int(args.num_treated),
        selection_score=str(args.selection_score),
        selection_seed=int(args.selection_seed),
    )
    summary = _materialize_condition(
        output_dir=output_dir,
        args=args,
        frame=frame,
        treated=treated,
        control_candidates=control_candidates,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
