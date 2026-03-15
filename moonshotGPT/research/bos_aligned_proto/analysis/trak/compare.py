"""Comparison helpers for exported TRAK checkpoint summaries.

This module works on already-exported row summary tables rather than on raw
TRAK internals. Its job is to quantify how attribution changes across adjacent
checkpoints using overlap, correlation, sign-flip, and newly influential-row
metrics that are easy to inspect downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    value = left_rank.corr(right_rank, method="pearson")
    return None if pd.isna(value) else float(value)


def compare_adjacent_row_summaries(
    row_summaries: dict[int, pd.DataFrame],
    *,
    topk: int,
) -> pd.DataFrame:
    steps = sorted(row_summaries)
    records: list[dict] = []
    for prev_step, curr_step in zip(steps, steps[1:]):
        prev_df = row_summaries[prev_step].set_index("row_id").sort_index()
        curr_df = row_summaries[curr_step].set_index("row_id").sort_index()
        shared_ids = prev_df.index.intersection(curr_df.index)
        if shared_ids.empty:
            records.append(
                {
                    "previous_step": prev_step,
                    "current_step": curr_step,
                    "shared_row_count": 0,
                    "spearman_mean_score": None,
                    "topk_abs_overlap": 0,
                    "sign_flip_count": 0,
                    "new_topk_abs_row_ids": json.dumps([]),
                }
            )
            continue

        prev_shared = prev_df.loc[shared_ids]
        curr_shared = curr_df.loc[shared_ids]
        prev_top = set(prev_df.nlargest(topk, "mean_abs_score").index.astype(int))
        curr_top = set(curr_df.nlargest(topk, "mean_abs_score").index.astype(int))
        sign_flip_count = int(
            (
                (prev_shared["mean_score"] > 0) & (curr_shared["mean_score"] < 0)
            ).sum()
            + ((prev_shared["mean_score"] < 0) & (curr_shared["mean_score"] > 0)).sum()
        )
        records.append(
            {
                "previous_step": prev_step,
                "current_step": curr_step,
                "shared_row_count": int(shared_ids.size),
                "spearman_mean_score": _spearman_correlation(
                    prev_shared["mean_score"],
                    curr_shared["mean_score"],
                ),
                "topk_abs_overlap": int(len(prev_top & curr_top)),
                "sign_flip_count": sign_flip_count,
                "new_topk_abs_row_ids": json.dumps(sorted(curr_top - prev_top)),
            }
        )
    return pd.DataFrame.from_records(records)


def write_checkpoint_compare_csv(path: str | Path, row_summaries: dict[int, pd.DataFrame], *, topk: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = compare_adjacent_row_summaries(row_summaries, topk=topk)
    df.to_csv(out_path, index=False)
    return out_path


__all__ = ["compare_adjacent_row_summaries", "write_checkpoint_compare_csv"]
