"""Build matched continued-pretraining pools from attribution outputs.

This utility is meant for small follow-up continued-pretraining experiments
where we want the treated and control pools to differ mainly in attribution
score, not in obvious confounders such as token budget or shard/file mixture.

The current implementation supports both main candidate surfaces used by the
`research/bos_aligned_proto` workflow:

- `bos_packed_row`
- `stream_window`

Regardless of the source candidate kind, the builder materializes the selected
examples into fixed-width row shards for continued pretraining. That keeps the
treated/control examples exact during ablation, rather than feeding them back
through a raw contiguous stream loader that could change the training windows.

It supports two main selection modes:

- `positive_pooled`: rank candidates by `positive_score_sum` from
  `row_summary_stepXXXXXXXX.csv`
- `per_query`: rank candidates by one target-specific row of the dense score
  matrix, selected by `target_id`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .common.export import write_json, write_jsonl
from .common.training_examples import (
    CandidateKind,
    FiniteTrainingExampleDataset,
    build_example_manifest,
)


SUPPORTED_SCORE_MODES = (
    "positive_pooled",
    "mean_score",
    "mean_abs_score",
    "per_query",
)

SCORE_MODE_TO_ROW_SUMMARY_COLUMN = {
    "positive_pooled": "positive_score_sum",
    "mean_score": "mean_score",
    "mean_abs_score": "mean_abs_score",
}


@dataclass(frozen=True)
class StepArtifactPaths:
    row_summary_path: Path
    dense_scores_path: Path | None
    target_items_path: Path | None


def _step_tag(step: int) -> str:
    return f"step{int(step):08d}"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_step_artifacts(attribution_dir: str | Path, step: int) -> StepArtifactPaths:
    root = Path(attribution_dir).expanduser().resolve()
    step_tag = _step_tag(step)
    row_summary_path = root / f"row_summary_{step_tag}.csv"
    if not row_summary_path.exists():
        raise FileNotFoundError(f"Missing row summary file: {row_summary_path}")
    dense_scores_path = root / f"dense_scores_{step_tag}.npy"
    target_items_path = root / "target_items.jsonl"
    return StepArtifactPaths(
        row_summary_path=row_summary_path,
        dense_scores_path=dense_scores_path if dense_scores_path.exists() else None,
        target_items_path=target_items_path if target_items_path.exists() else None,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _score_mode_source_column(score_mode: str) -> str:
    try:
        return SCORE_MODE_TO_ROW_SUMMARY_COLUMN[str(score_mode)]
    except KeyError as exc:
        raise ValueError(
            f"score_mode={score_mode!r} does not map to a row-summary column; "
            f"expected one of {tuple(SCORE_MODE_TO_ROW_SUMMARY_COLUMN)}"
        ) from exc


def _resolve_source_candidate_kind(frame: pd.DataFrame) -> CandidateKind:
    if "candidate_kind" not in frame.columns:
        raise ValueError("Row summary is missing candidate_kind, which is required for CPT pool materialization")
    unique_kinds = sorted(str(value) for value in frame["candidate_kind"].dropna().unique().tolist())
    if len(unique_kinds) != 1:
        raise ValueError(
            "Matched CPT pool building expects one candidate_kind in the row summary, "
            f"but found {unique_kinds!r}"
        )
    candidate_kind = unique_kinds[0]
    if candidate_kind not in {"bos_packed_row", "stream_window"}:
        raise ValueError(
            f"Unsupported candidate_kind={candidate_kind!r} in row summary; expected 'bos_packed_row' or 'stream_window'"
        )
    return candidate_kind  # type: ignore[return-value]


def _infer_seq_len_from_scored_frame(frame: pd.DataFrame) -> int:
    if "token_count" not in frame.columns:
        raise ValueError("Scored candidate frame is missing token_count")
    token_counts = sorted(int(value) for value in frame["token_count"].dropna().unique().tolist())
    if len(token_counts) != 1:
        raise ValueError(
            "Expected one token_count across attribution candidates when materializing CPT pools, "
            f"but found {token_counts!r}"
        )
    token_count = int(token_counts[0])
    if token_count <= 1:
        raise ValueError(f"token_count must be > 1, got {token_count}")
    return token_count - 1


def _build_balance_report(*, treated: pd.DataFrame, control: pd.DataFrame, pairings: pd.DataFrame) -> dict[str, Any]:
    score_gap = (
        pairings["score_gap"].astype(float)
        if "score_gap" in pairings.columns and not pairings.empty
        else pd.Series(dtype=float)
    )
    position_distance = (
        pairings["position_distance"].astype(float)
        if "position_distance" in pairings.columns and not pairings.empty
        else pd.Series(dtype=float)
    )
    return {
        "matching_principles": [
            "treated and control rows are matched on candidate_kind",
            "treated and control rows are matched on exact token_count",
            "controls are chosen from the same shard_path when possible",
            "when multiple controls satisfy the stratum, the closest local_example_idx is preferred",
            "controls are preferentially drawn from rows whose selection score is at or below max_control_score",
        ],
        "treated_total_tokens": int(treated["token_count"].sum()),
        "control_total_tokens": int(control["token_count"].sum()),
        "treated_token_count_values": sorted(int(value) for value in treated["token_count"].unique().tolist()),
        "control_token_count_values": sorted(int(value) for value in control["token_count"].unique().tolist()),
        "token_count_balance_exact": bool(
            treated["token_count"].sort_values().reset_index(drop=True).equals(
                control["token_count"].sort_values().reset_index(drop=True)
            )
        ),
        "treated_shard_counts": {
            str(key): int(value) for key, value in treated["shard_path"].value_counts().sort_index().items()
        },
        "control_shard_counts": {
            str(key): int(value) for key, value in control["shard_path"].value_counts().sort_index().items()
        },
        "same_shard_pair_fraction": (
            float((pairings["treated_shard_path"] == pairings["control_shard_path"]).mean()) if not pairings.empty else 0.0
        ),
        "mean_position_distance": float(position_distance.mean()) if not position_distance.empty else 0.0,
        "max_position_distance": float(position_distance.max()) if not position_distance.empty else 0.0,
        "mean_score_gap": float(score_gap.mean()) if not score_gap.empty else 0.0,
        "min_score_gap": float(score_gap.min()) if not score_gap.empty else 0.0,
        "max_score_gap": float(score_gap.max()) if not score_gap.empty else 0.0,
    }


def load_candidate_score_frame(
    *,
    attribution_dir: str | Path,
    step: int,
    score_mode: str,
    target_id: str | None = None,
) -> pd.DataFrame:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected one of {SUPPORTED_SCORE_MODES!r}")

    artifacts = resolve_step_artifacts(attribution_dir, step)
    frame = pd.read_csv(artifacts.row_summary_path)
    if "group" in frame.columns:
        frame = frame.loc[frame["group"] == "overall"].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Row summary {artifacts.row_summary_path} has no overall candidate rows")

    _require_columns(
        frame,
        (
            "candidate_id",
            "candidate_kind",
            "shard_path",
            "local_example_idx",
            "token_offset_start",
            "token_offset_end",
        ),
        label="row_summary",
    )
    frame["token_count"] = frame["token_offset_end"].astype(int) - frame["token_offset_start"].astype(int)
    frame["selection_score"] = np.nan
    frame["selection_target_id"] = None

    if score_mode == "per_query":
        if not target_id:
            raise ValueError("score_mode='per_query' requires --target_id")
        if artifacts.dense_scores_path is None:
            raise FileNotFoundError(
                "Per-query selection requires dense scores. Re-run attribution with --write_dense_scores."
            )
        if artifacts.target_items_path is None:
            raise FileNotFoundError(
                f"Per-query selection requires target_items.jsonl under {Path(attribution_dir).expanduser().resolve()}"
            )
        dense_scores = np.load(artifacts.dense_scores_path)
        if dense_scores.ndim != 2:
            raise ValueError(
                f"Expected dense score matrix to be 2D, got shape {dense_scores.shape} from {artifacts.dense_scores_path}"
            )
        target_items = _load_jsonl(artifacts.target_items_path)
        target_ids = [str(row["target_id"]) for row in target_items]
        try:
            target_index = target_ids.index(str(target_id))
        except ValueError as exc:
            raise ValueError(f"target_id={target_id!r} was not found in {artifacts.target_items_path}") from exc
        if dense_scores.shape[1] != len(frame):
            raise ValueError(
                "Dense-score candidate dimension does not match row summary ordering: "
                f"{dense_scores.shape[1]} vs {len(frame)}"
            )
        frame["selection_score"] = np.asarray(dense_scores[target_index], dtype=np.float64)
        frame["selection_target_id"] = str(target_id)
        return frame

    source_column = _score_mode_source_column(score_mode)
    if source_column not in frame.columns:
        raise ValueError(
            f"Row summary does not contain {source_column!r}. Available columns: {sorted(frame.columns.tolist())}"
        )
    frame["selection_score"] = frame[source_column].astype(float)
    return frame


def select_treated_candidates(
    frame: pd.DataFrame,
    *,
    num_treated: int,
    min_score: float | None = None,
) -> pd.DataFrame:
    if num_treated <= 0:
        raise ValueError("num_treated must be > 0")
    working = frame.copy()
    if min_score is not None:
        working = working.loc[working["selection_score"] >= float(min_score)].copy()
    working = working.dropna(subset=["selection_score"])
    working = working.sort_values(["selection_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    if len(working) < int(num_treated):
        raise ValueError(
            f"Requested {num_treated} treated candidates, but only {len(working)} candidates satisfy the selection criteria"
        )
    return working.head(int(num_treated)).reset_index(drop=True)


def _candidate_match_levels(
    *,
    treated_row: pd.Series,
    available: pd.DataFrame,
    max_control_score: float | None,
    allow_relaxed_shard_match: bool,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    base_mask = (
        (available["candidate_kind"] == treated_row["candidate_kind"])
        & (available["token_count"] == treated_row["token_count"])
    )
    same_shard_mask = base_mask & (available["shard_path"] == treated_row["shard_path"])

    below_threshold_mask = pd.Series(True, index=available.index)
    if max_control_score is not None:
        below_threshold_mask = available["selection_score"] <= float(max_control_score)

    levels: list[tuple[str, pd.DataFrame]] = [
        ("exact_shard_below_threshold", available.loc[same_shard_mask & below_threshold_mask].copy()),
        ("exact_shard", available.loc[same_shard_mask].copy()),
    ]
    if allow_relaxed_shard_match:
        levels.extend(
            [
                ("matched_kind_tokens_below_threshold", available.loc[base_mask & below_threshold_mask].copy()),
                ("matched_kind_tokens", available.loc[base_mask].copy()),
            ]
        )
    return tuple(levels)


def match_control_candidates(
    *,
    treated: pd.DataFrame,
    candidates: pd.DataFrame,
    max_control_score: float | None = 0.0,
    allow_relaxed_shard_match: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = candidates.loc[~candidates["candidate_id"].isin(treated["candidate_id"])].copy()
    used_candidate_ids: set[int] = set()
    pairing_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for pair_idx, treated_row in enumerate(treated.itertuples(index=False), start=1):
        treated_series = pd.Series(treated_row._asdict())
        pool = available.loc[~available["candidate_id"].isin(used_candidate_ids)].copy()
        chosen_row: pd.Series | None = None
        chosen_level: str | None = None

        for level_name, level_frame in _candidate_match_levels(
            treated_row=treated_series,
            available=pool,
            max_control_score=max_control_score,
            allow_relaxed_shard_match=allow_relaxed_shard_match,
        ):
            if level_frame.empty:
                continue
            level_frame["position_distance"] = (
                level_frame["local_example_idx"].astype(int) - int(treated_series["local_example_idx"])
            ).abs()
            level_frame = level_frame.sort_values(
                ["position_distance", "selection_score", "candidate_id"],
                ascending=[True, True, True],
            )
            chosen_row = level_frame.iloc[0]
            chosen_level = level_name
            break

        if chosen_row is None or chosen_level is None:
            raise ValueError(
                "Could not find a matched control candidate for treated candidate "
                f"{int(treated_series['candidate_id'])}. Try relaxing shard matching or control-score filtering."
            )

        chosen_candidate_id = int(chosen_row["candidate_id"])
        used_candidate_ids.add(chosen_candidate_id)
        control_rows.append(dict(chosen_row))
        pairing_rows.append(
            {
                "pair_index": int(pair_idx),
                "match_level": chosen_level,
                "treated_candidate_id": int(treated_series["candidate_id"]),
                "control_candidate_id": chosen_candidate_id,
                "treated_selection_score": float(treated_series["selection_score"]),
                "control_selection_score": float(chosen_row["selection_score"]),
                "score_gap": float(treated_series["selection_score"] - chosen_row["selection_score"]),
                "token_count": int(treated_series["token_count"]),
                "treated_shard_path": str(treated_series["shard_path"]),
                "control_shard_path": str(chosen_row["shard_path"]),
                "treated_local_example_idx": int(treated_series["local_example_idx"]),
                "control_local_example_idx": int(chosen_row["local_example_idx"]),
                "position_distance": abs(
                    int(treated_series["local_example_idx"]) - int(chosen_row["local_example_idx"])
                ),
            }
        )

    return (
        pd.DataFrame.from_records(pairing_rows),
        pd.DataFrame.from_records(control_rows).reset_index(drop=True),
    )


def _full_tokens_from_sample(sample: dict[str, Any]) -> np.ndarray:
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Expected tensor-valued input_ids and labels from FiniteTrainingExampleDataset")
    full_tokens = torch.cat([input_ids, labels[-1:].clone()], dim=0)
    return np.asarray(full_tokens.detach().cpu(), dtype=np.uint16)


def _write_row_packed_shards(
    *,
    output_dir: Path,
    manifest,
    candidate_ids: Sequence[int],
    candidate_frame: pd.DataFrame,
    rows_per_shard: int,
    pool_role: str,
    extra_meta: dict[str, Any],
) -> dict[str, Path]:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = FiniteTrainingExampleDataset(manifest, candidate_ids)
    shard_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    buffer: list[np.ndarray] = []
    shard_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        shard_path = output_dir / f"train_{shard_index:06d}.bin"
        np.concatenate(buffer, axis=0).astype(np.uint16, copy=False).tofile(shard_path)
        shard_paths.append(shard_path)
        shard_index += 1
        buffer = []

    for row_order, row in enumerate(candidate_frame.itertuples(index=False)):
        sample = dataset[row_order]
        full_tokens = _full_tokens_from_sample(sample)
        if full_tokens.size != int(manifest.example_tokens):
            raise ValueError(
                f"Expected {manifest.example_tokens} tokens, got {full_tokens.size} for candidate_id={row.candidate_id}"
            )
        buffer.append(full_tokens)
        manifest_rows.append(
            {
                "pool_role": pool_role,
                "pool_row_index": int(row_order),
                "candidate_id": int(row.candidate_id),
                "selection_score": float(row.selection_score),
                "candidate_kind": str(row.candidate_kind),
                "shard_path": str(row.shard_path),
                "local_example_idx": int(row.local_example_idx),
                "token_offset_start": int(row.token_offset_start),
                "token_offset_end": int(row.token_offset_end),
                "token_count": int(row.token_count),
            }
        )
        if len(buffer) >= int(rows_per_shard):
            flush_buffer()

    flush_buffer()
    meta = {
        "format": "exact_window_row_packed",
        "seq_len": int(manifest.seq_len),
        "row_tokens": int(manifest.example_tokens),
        "candidate_kind": str(manifest.candidate_kind),
        "num_rows": int(len(candidate_ids)),
        "rows_per_shard": int(rows_per_shard),
        "source_data_dir": str(manifest.data_dir),
        "source_format": str(manifest.format),
        "row_semantics": "exact_training_example",
        "pool_role": pool_role,
        **extra_meta,
    }
    meta_path = output_dir / "meta.json"
    rows_path = output_dir / "rows.jsonl"
    write_json(meta_path, meta)
    write_jsonl(rows_path, manifest_rows)
    return {
        "data_dir": output_dir,
        "meta": meta_path,
        "rows": rows_path,
    }


def build_matched_cpt_pools(
    *,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    output_dir: str | Path,
    score_mode: str,
    num_treated: int,
    target_id: str | None = None,
    min_treated_score: float | None = None,
    max_control_score: float | None = 0.0,
    allow_relaxed_shard_match: bool = True,
    rows_per_shard: int = 50_000,
    seq_len: int | None = None,
) -> dict[str, Path]:
    scored = load_candidate_score_frame(
        attribution_dir=attribution_dir,
        step=step,
        score_mode=score_mode,
        target_id=target_id,
    )
    source_candidate_kind = _resolve_source_candidate_kind(scored)
    resolved_seq_len = int(seq_len) if seq_len is not None else _infer_seq_len_from_scored_frame(scored)
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=source_candidate_kind,
        seq_len=resolved_seq_len,
    )
    if str(manifest.candidate_kind) != str(source_candidate_kind):
        raise ValueError(
            "Materialized manifest candidate kind does not match row summary candidate kind: "
            f"{manifest.candidate_kind!r} vs {source_candidate_kind!r}"
        )

    treated = select_treated_candidates(
        scored,
        num_treated=num_treated,
        min_score=min_treated_score,
    )
    pairings, control = match_control_candidates(
        treated=treated,
        candidates=scored,
        max_control_score=max_control_score,
        allow_relaxed_shard_match=allow_relaxed_shard_match,
    )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    treated_dir = output_root / "treated_dataset"
    control_dir = output_root / "control_dataset"

    treated_artifacts = _write_row_packed_shards(
        output_dir=treated_dir,
        manifest=manifest,
        candidate_ids=tuple(int(value) for value in treated["candidate_id"].tolist()),
        candidate_frame=treated,
        rows_per_shard=rows_per_shard,
        pool_role="treated",
        extra_meta={
            "selection": {
                "score_mode": score_mode,
                "target_id": target_id,
                "step": int(step),
                "min_treated_score": None if min_treated_score is None else float(min_treated_score),
            },
            "source_candidate_kind": str(source_candidate_kind),
        },
    )
    control_artifacts = _write_row_packed_shards(
        output_dir=control_dir,
        manifest=manifest,
        candidate_ids=tuple(int(value) for value in control["candidate_id"].tolist()),
        candidate_frame=control,
        rows_per_shard=rows_per_shard,
        pool_role="control",
        extra_meta={
            "selection": {
                "score_mode": score_mode,
                "target_id": target_id,
                "step": int(step),
                "max_control_score": None if max_control_score is None else float(max_control_score),
            },
            "source_candidate_kind": str(source_candidate_kind),
        },
    )

    treated_csv = output_root / "treated_candidates.csv"
    control_csv = output_root / "control_candidates.csv"
    pairings_csv = output_root / "pairings.csv"
    treated.to_csv(treated_csv, index=False)
    control.to_csv(control_csv, index=False)
    pairings.to_csv(pairings_csv, index=False)

    summary_path = output_root / "summary.json"
    balance_report = _build_balance_report(treated=treated, control=control, pairings=pairings)
    write_json(
        summary_path,
        {
            "score_mode": score_mode,
            "target_id": target_id,
            "step": int(step),
            "num_treated": int(len(treated)),
            "num_control": int(len(control)),
            "treated_total_tokens": int(treated["token_count"].sum()),
            "control_total_tokens": int(control["token_count"].sum()),
            "treated_score_mean": float(treated["selection_score"].mean()),
            "control_score_mean": float(control["selection_score"].mean()),
            "treated_score_min": float(treated["selection_score"].min()),
            "treated_score_max": float(treated["selection_score"].max()),
            "control_score_min": float(control["selection_score"].min()),
            "control_score_max": float(control["selection_score"].max()),
            "exact_shard_matches": int((pairings["match_level"] == "exact_shard_below_threshold").sum())
            + int((pairings["match_level"] == "exact_shard").sum()),
            "relaxed_matches": int((pairings["match_level"] == "matched_kind_tokens_below_threshold").sum())
            + int((pairings["match_level"] == "matched_kind_tokens").sum()),
            "selection_source": (
                {"kind": "dense_scores", "target_id": target_id}
                if score_mode == "per_query"
                else {"kind": "row_summary", "column": _score_mode_source_column(score_mode)}
            ),
            "source_candidate_kind": str(source_candidate_kind),
            "materialized_format": "exact_window_row_packed",
            "balance_report": balance_report,
        },
    )

    return {
        "root": output_root,
        "treated_candidates": treated_csv,
        "control_candidates": control_csv,
        "pairings": pairings_csv,
        "summary": summary_path,
        "treated_data_dir": treated_artifacts["data_dir"],
        "control_data_dir": control_artifacts["data_dir"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched treated/control continued-pretraining pools from TrackStar or TRAK attribution outputs."
        )
    )
    parser.add_argument("attribution_dir", help="Attribution output directory containing row_summary_*.csv")
    parser.add_argument("--data_dir", required=True, help="Training-data directory that matches the attribution run")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to use")
    parser.add_argument("--output_dir", required=True, help="Directory where matched pools will be written")
    parser.add_argument(
        "--score_mode",
        choices=SUPPORTED_SCORE_MODES,
        default="positive_pooled",
        help="How to rank candidates before matching",
    )
    parser.add_argument(
        "--target_id",
        type=str,
        default=None,
        help="Required when score_mode=per_query; selects one EWoK target row from target_items.jsonl",
    )
    parser.add_argument("--num_treated", type=int, required=True, help="Number of treated candidates to select")
    parser.add_argument(
        "--min_treated_score",
        type=float,
        default=None,
        help="Optional minimum selection score for treated candidates",
    )
    parser.add_argument(
        "--max_control_score",
        type=float,
        default=0.0,
        help=(
            "Preferred upper bound for control selection score. When no candidate satisfies the bound "
            "inside the preferred match stratum, the matcher falls back gracefully."
        ),
    )
    parser.add_argument(
        "--no_relaxed_shard_match",
        action="store_true",
        help="Disable fallback from exact-shard matching to global same-kind/same-token matching",
    )
    parser.add_argument(
        "--rows_per_shard",
        type=int,
        default=50_000,
        help="Maximum number of rows per output train_*.bin shard",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help=(
            "Optional seq_len override used when reconstructing source candidates. "
            "Normally this is inferred from the row-summary token_count column."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    artifacts = build_matched_cpt_pools(
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=args.step,
        output_dir=args.output_dir,
        score_mode=args.score_mode,
        num_treated=args.num_treated,
        target_id=args.target_id,
        min_treated_score=args.min_treated_score,
        max_control_score=args.max_control_score,
        allow_relaxed_shard_match=not args.no_relaxed_shard_match,
        rows_per_shard=args.rows_per_shard,
        seq_len=args.seq_len,
    )
    print(f"wrote matched continued-pretraining pools under {artifacts['root']}")
    print(f"treated dataset: {artifacts['treated_data_dir']}")
    print(f"control dataset: {artifacts['control_data_dir']}")
    print(f"pairings: {artifacts['pairings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
