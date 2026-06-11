"""EWoK context-sensitivity metrics and probe-vs-LM case buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np

from .data import EWOKProbePair


CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
ROLE_ORDER = ("c1t1", "c1t2", "c2t2", "c2t1")
CASE_BUCKETS = (
    "probe_correct_lm_correct",
    "probe_correct_lm_wrong",
    "probe_wrong_lm_correct",
    "probe_wrong_lm_wrong",
)


def _rows_from_scores(
    pairs: Sequence[EWOKProbePair],
    scores: np.ndarray,
    *,
    split_labels: Sequence[str] | None = None,
    split: str | None = None,
) -> list[dict]:
    score_by_pair = {pair.pair_id: float(scores[idx]) for idx, pair in enumerate(pairs)}
    split_by_pair = {}
    if split_labels is not None:
        split_by_pair = {pair.pair_id: str(split_labels[idx]) for idx, pair in enumerate(pairs)}

    pairs_by_row: dict[int, dict[str, EWOKProbePair]] = defaultdict(dict)
    for pair in pairs:
        if split is not None and split_by_pair.get(pair.pair_id) != split:
            continue
        pairs_by_row[int(pair.row_index)][pair.role] = pair

    rows: list[dict] = []
    for row_index in sorted(pairs_by_row):
        role_map = pairs_by_row[row_index]
        missing = [role for role in ROLE_ORDER if role not in role_map]
        if missing:
            raise ValueError(f"row_index={row_index} is missing EWoK pair roles: {missing!r}")
        c1t1 = role_map["c1t1"]
        c1t2 = role_map["c1t2"]
        c2t2 = role_map["c2t2"]
        c2t1 = role_map["c2t1"]
        k1 = score_by_pair[c1t1.pair_id] - score_by_pair[c2t1.pair_id]
        k2 = score_by_pair[c2t2.pair_id] - score_by_pair[c1t2.pair_id]
        rows.append(
            {
                "row_index": int(row_index),
                "domain": c1t1.domain,
                "probe_context1": c1t1.context,
                "probe_context2": c2t2.context,
                "target1": c1t1.target,
                "target2": c2t2.target,
                "k1": float(k1),
                "k2": float(k2),
                "combined_margin": float(0.5 * (k1 + k2)),
                "min_margin": float(min(k1, k2)),
                "k1_correct": bool(k1 > 0.0),
                "k2_correct": bool(k2 > 0.0),
                "row_correct": bool(k1 > 0.0 and k2 > 0.0),
            }
        )
    return rows


def compute_context_sensitivity_metrics(
    pairs: Sequence[EWOKProbePair],
    scores: np.ndarray,
    *,
    split_labels: Sequence[str] | None = None,
    split: str | None = None,
) -> dict:
    rows = _rows_from_scores(pairs, np.asarray(scores, dtype=np.float64), split_labels=split_labels, split=split)
    if not rows:
        return {
            "n_rows": 0,
            "k1_accuracy": float("nan"),
            "k2_accuracy": float("nan"),
            "row_strict_accuracy": float("nan"),
            "mean_combined_margin": float("nan"),
            "mean_min_margin": float("nan"),
        }
    return {
        "n_rows": int(len(rows)),
        "k1_accuracy": float(np.mean([row["k1_correct"] for row in rows])),
        "k2_accuracy": float(np.mean([row["k2_correct"] for row in rows])),
        "row_strict_accuracy": float(np.mean([row["row_correct"] for row in rows])),
        "mean_combined_margin": float(np.mean([row["combined_margin"] for row in rows])),
        "mean_min_margin": float(np.mean([row["min_margin"] for row in rows])),
    }


def compute_context_sensitivity_rows(
    pairs: Sequence[EWOKProbePair],
    scores: np.ndarray,
    *,
    split_labels: Sequence[str] | None = None,
    split: str | None = None,
) -> list[dict]:
    """Return row-level EWoK context-sensitivity margins and correctness flags."""

    return _rows_from_scores(
        pairs,
        np.asarray(scores, dtype=np.float64),
        split_labels=split_labels,
        split=split,
    )


def pair_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
) -> float:
    active = np.asarray(mask, dtype=bool)
    if active.sum() == 0:
        return float("nan")
    predicted = (np.asarray(scores)[active] > 0.0).astype(np.int64)
    truth = np.asarray(labels, dtype=np.int64)[active]
    return float(np.mean(predicted == truth))


def _bucket_name(probe_correct: bool, lm_correct: bool) -> str:
    if probe_correct and lm_correct:
        return "probe_correct_lm_correct"
    if probe_correct and not lm_correct:
        return "probe_correct_lm_wrong"
    if not probe_correct and lm_correct:
        return "probe_wrong_lm_correct"
    return "probe_wrong_lm_wrong"


def compute_probe_lm_cases(
    pairs: Sequence[EWOKProbePair],
    *,
    probe_scores: np.ndarray,
    lm_scores: np.ndarray,
    split_labels: Sequence[str],
    split: str = "test",
) -> tuple[list[dict], dict[str, int], list[dict]]:
    probe_rows = {
        row["row_index"]: row
        for row in _rows_from_scores(pairs, probe_scores, split_labels=split_labels, split=split)
    }
    lm_rows = {
        row["row_index"]: row
        for row in _rows_from_scores(pairs, lm_scores, split_labels=split_labels, split=split)
    }

    case_rows: list[dict] = []
    for row_index in sorted(probe_rows):
        probe = probe_rows[row_index]
        lm = lm_rows[row_index]
        bucket = _bucket_name(bool(probe["row_correct"]), bool(lm["row_correct"]))
        rank_score = float(probe["min_margin"] - lm["min_margin"])
        combined_gap = float(probe["combined_margin"] - lm["combined_margin"])
        case_rows.append(
            {
                "row_index": int(row_index),
                "split": split,
                "domain": probe["domain"],
                "bucket": bucket,
                "probe_correct": bool(probe["row_correct"]),
                "lm_correct": bool(lm["row_correct"]),
                "probe_k1": float(probe["k1"]),
                "probe_k2": float(probe["k2"]),
                "probe_combined_margin": float(probe["combined_margin"]),
                "probe_min_margin": float(probe["min_margin"]),
                "lm_k1": float(lm["k1"]),
                "lm_k2": float(lm["k2"]),
                "lm_combined_margin": float(lm["combined_margin"]),
                "lm_min_margin": float(lm["min_margin"]),
                "rank_score": rank_score,
                "combined_margin_gap": combined_gap,
                "context1": probe["probe_context1"],
                "context2": probe["probe_context2"],
                "target1": probe["target1"],
                "target2": probe["target2"],
            }
        )

    bucket_counts = {bucket: 0 for bucket in CASE_BUCKETS}
    bucket_counts.update(Counter(row["bucket"] for row in case_rows))

    domains = sorted({str(row["domain"]) for row in case_rows})
    domain_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in case_rows:
        domain_counts[(str(row["domain"]), str(row["bucket"]))] += 1
    domain_rows = [
        {"domain": domain, "bucket": bucket, "count": int(domain_counts[(domain, bucket)])}
        for domain in domains
        for bucket in CASE_BUCKETS
    ]
    return case_rows, bucket_counts, domain_rows


def rows_by_bucket(case_rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    grouped = {bucket: [] for bucket in CASE_BUCKETS}
    for row in case_rows:
        grouped[str(row["bucket"])].append(dict(row))
    for bucket, rows in grouped.items():
        rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
    return grouped


__all__ = [
    "CASE_BUCKETS",
    "CONTEXT_SENSITIVITY",
    "compute_context_sensitivity_metrics",
    "compute_context_sensitivity_rows",
    "compute_probe_lm_cases",
    "pair_accuracy",
    "rows_by_bucket",
]
