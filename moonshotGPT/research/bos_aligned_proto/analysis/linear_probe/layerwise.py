"""Layer-wise domain diagnostics for EWoK linear probes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .activations import ActivationCache
from .data import EWOKProbePair
from .evaluation import compute_context_sensitivity_rows, pair_accuracy
from .probes import _decision_scores, _fit_estimator, _selection_key, _table_row


def load_probe_pairs_jsonl(path: str | Path) -> tuple[EWOKProbePair, ...]:
    """Load probe-pair records written by `run_ewok_linear_probe`."""

    allowed = {field.name for field in fields(EWOKProbePair)}
    pairs: list[EWOKProbePair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            pairs.append(EWOKProbePair(**{key: payload[key] for key in allowed}))
    return tuple(pairs)


def load_split_labels_csv(
    path: str | Path,
    pairs: Sequence[EWOKProbePair],
) -> np.ndarray:
    """Load split labels in the same order as `pairs`."""

    split_by_pair_id: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split_by_pair_id[str(row["pair_id"])] = str(row["split"])

    missing = [pair.pair_id for pair in pairs if pair.pair_id not in split_by_pair_id]
    if missing:
        raise ValueError(f"Split assignments are missing {len(missing)} pair_id(s), first={missing[0]!r}")
    return np.asarray([split_by_pair_id[pair.pair_id] for pair in pairs], dtype=object)


def _metric_from_rows(rows: Sequence[Mapping]) -> dict:
    if not rows:
        return {
            "n_rows": 0,
            "k1_accuracy": float("nan"),
            "k2_accuracy": float("nan"),
            "directional_avg": float("nan"),
            "row_strict_accuracy": float("nan"),
            "mean_combined_margin": float("nan"),
            "mean_min_margin": float("nan"),
        }
    k1 = float(np.mean([bool(row["k1_correct"]) for row in rows]))
    k2 = float(np.mean([bool(row["k2_correct"]) for row in rows]))
    strict = float(np.mean([bool(row["row_correct"]) for row in rows]))
    return {
        "n_rows": int(len(rows)),
        "k1_accuracy": k1,
        "k2_accuracy": k2,
        "directional_avg": 0.5 * (k1 + k2),
        "row_strict_accuracy": strict,
        "mean_combined_margin": float(np.mean([float(row["combined_margin"]) for row in rows])),
        "mean_min_margin": float(np.mean([float(row["min_margin"]) for row in rows])),
    }


def _macro_average(rows: Sequence[Mapping]) -> dict:
    values = [row for row in rows if str(row["domain"]) != "average"]
    if not values:
        return {
            "n_rows": 0,
            "k1_accuracy": float("nan"),
            "k2_accuracy": float("nan"),
            "directional_avg": float("nan"),
            "row_strict_accuracy": float("nan"),
            "mean_combined_margin": float("nan"),
            "mean_min_margin": float("nan"),
        }
    metric_keys = [
        "k1_accuracy",
        "k2_accuracy",
        "directional_avg",
        "row_strict_accuracy",
        "mean_combined_margin",
        "mean_min_margin",
    ]
    out = {"n_rows": int(sum(int(row["n_rows"]) for row in values))}
    for key in metric_keys:
        clean = [float(row[key]) for row in values if not np.isnan(float(row[key]))]
        out[key] = float(np.mean(clean)) if clean else float("nan")
    return out


def compute_layer_domain_curves(
    *,
    cache: ActivationCache,
    pairs: Sequence[EWOKProbePair],
    split_labels: np.ndarray,
    c_grid: Sequence[float],
    seed: int,
    score_split: str = "test",
    layer_c_values: Mapping[int, float] | None = None,
) -> tuple[dict, ...]:
    """Train per-layer probes and return domain metrics for each layer.

    For every layer, the best C is selected on the overall validation split.
    Domain curves are then computed on `score_split`.
    """

    labels = cache.labels.astype(np.int64, copy=False)
    train_mask = np.asarray(split_labels == "train", dtype=bool)
    score_mask = np.asarray(split_labels == score_split, dtype=bool)
    if train_mask.sum() == 0 or score_mask.sum() == 0:
        raise ValueError(f"Need non-empty train and {score_split!r} splits for layer-domain curves.")

    out_rows: list[dict] = []
    for layer_position, layer_index in enumerate(cache.layer_indices):
        X = cache.features[:, layer_position, :].astype(np.float32, copy=False)
        best_row = None
        best_scores = None
        best_C = None
        active_c_grid = (
            (float(layer_c_values[int(layer_index)]),)
            if layer_c_values is not None and int(layer_index) in layer_c_values
            else tuple(c_grid)
        )
        for C in active_c_grid:
            estimator = _fit_estimator(
                X[train_mask],
                labels[train_mask],
                C=float(C),
                seed=int(seed),
            )
            scores = _decision_scores(estimator, X)
            row = _table_row(
                layer_index=int(layer_index),
                layer_position=int(layer_position),
                C=float(C),
                pairs=pairs,
                labels=labels,
                scores=scores,
                split_labels=split_labels,
            )
            if best_row is None or _selection_key(row) > _selection_key(best_row):
                best_row = row
                best_scores = scores
                best_C = float(C)

        assert best_scores is not None and best_C is not None
        cs_rows = compute_context_sensitivity_rows(
            pairs,
            best_scores,
            split_labels=split_labels,
            split=score_split,
        )
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for row in cs_rows:
            by_domain[str(row["domain"])].append(row)

        layer_domain_rows = []
        for domain in sorted(by_domain):
            metrics = _metric_from_rows(by_domain[domain])
            domain_pair_mask = np.asarray(
                [(split == score_split and pair.domain == domain) for pair, split in zip(pairs, split_labels)],
                dtype=bool,
            )
            layer_domain_rows.append(
                {
                    "layer_index": int(layer_index),
                    "layer_position": int(layer_position),
                    "selected_C": float(best_C),
                    "score_split": score_split,
                    "domain": domain,
                    "pair_accuracy": pair_accuracy(labels, best_scores, domain_pair_mask),
                    **metrics,
                }
            )

        avg = _macro_average(layer_domain_rows)
        avg_pair_mask = np.asarray(split_labels == score_split, dtype=bool)
        layer_domain_rows.append(
            {
                "layer_index": int(layer_index),
                "layer_position": int(layer_position),
                "selected_C": float(best_C),
                "score_split": score_split,
                "domain": "average",
                "pair_accuracy": pair_accuracy(labels, best_scores, avg_pair_mask),
                **avg,
            }
        )
        out_rows.extend(layer_domain_rows)
    return tuple(out_rows)


__all__ = [
    "compute_layer_domain_curves",
    "load_probe_pairs_jsonl",
    "load_split_labels_csv",
]
