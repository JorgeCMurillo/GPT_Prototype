"""Linear probe fitting, validation selection, and shuffled-label controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .activations import ActivationCache
from .data import EWOKProbePair
from .evaluation import compute_context_sensitivity_metrics, pair_accuracy


DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class SelectedProbe:
    estimator: Pipeline
    layer_index: int
    layer_position: int
    C: float
    validation_metrics: dict
    test_metrics: dict
    validation_table: tuple[dict, ...]
    probe_scores: np.ndarray


def parse_c_grid(value: str | None) -> tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return DEFAULT_C_GRID
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _fit_estimator(X: np.ndarray, y: np.ndarray, *, C: float, seed: int) -> Pipeline:
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(C),
                    penalty="l2",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    estimator.fit(X.astype(np.float32, copy=False), y.astype(np.int64, copy=False))
    return estimator


def _decision_scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    scores = estimator.decision_function(X.astype(np.float32, copy=False))
    return np.asarray(scores, dtype=np.float64)


def _split_mask(split_labels: np.ndarray, split: str) -> np.ndarray:
    return np.asarray(split_labels == split, dtype=bool)


def _table_row(
    *,
    layer_index: int,
    layer_position: int,
    C: float,
    pairs: Sequence[EWOKProbePair],
    labels: np.ndarray,
    scores: np.ndarray,
    split_labels: np.ndarray,
) -> dict:
    train_mask = _split_mask(split_labels, "train")
    val_mask = _split_mask(split_labels, "val")
    train_metrics = compute_context_sensitivity_metrics(
        pairs,
        scores,
        split_labels=split_labels,
        split="train",
    )
    val_metrics = compute_context_sensitivity_metrics(
        pairs,
        scores,
        split_labels=split_labels,
        split="val",
    )
    return {
        "layer_index": int(layer_index),
        "layer_position": int(layer_position),
        "C": float(C),
        "train_pair_accuracy": pair_accuracy(labels, scores, train_mask),
        "train_row_strict_accuracy": train_metrics["row_strict_accuracy"],
        "train_k1_accuracy": train_metrics["k1_accuracy"],
        "train_k2_accuracy": train_metrics["k2_accuracy"],
        "val_pair_accuracy": pair_accuracy(labels, scores, val_mask),
        "val_row_strict_accuracy": val_metrics["row_strict_accuracy"],
        "val_k1_accuracy": val_metrics["k1_accuracy"],
        "val_k2_accuracy": val_metrics["k2_accuracy"],
        "val_mean_min_margin": val_metrics["mean_min_margin"],
    }


def _selection_key(row: dict) -> tuple[float, float, float, float, int]:
    return (
        float(row["val_row_strict_accuracy"]),
        float(row["val_pair_accuracy"]),
        float(row["val_mean_min_margin"]),
        -float(row["C"]),
        -int(row["layer_index"]),
    )


def fit_validation_selected_probe(
    *,
    cache: ActivationCache,
    pairs: Sequence[EWOKProbePair],
    split_labels: np.ndarray,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = 42,
    train_labels_override: np.ndarray | None = None,
) -> SelectedProbe:
    labels = cache.labels.astype(np.int64, copy=False)
    train_labels = labels if train_labels_override is None else np.asarray(train_labels_override, dtype=np.int64)
    train_mask = _split_mask(split_labels, "train")
    test_mask = _split_mask(split_labels, "test")
    if train_mask.sum() == 0 or _split_mask(split_labels, "val").sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Train, validation, and test splits must all contain at least one pair.")

    best_row: dict | None = None
    best_estimator: Pipeline | None = None
    best_scores: np.ndarray | None = None
    table_rows: list[dict] = []

    for layer_position, layer_index in enumerate(cache.layer_indices):
        X = cache.features[:, layer_position, :].astype(np.float32, copy=False)
        for C in c_grid:
            estimator = _fit_estimator(
                X[train_mask],
                train_labels[train_mask],
                C=float(C),
                seed=seed,
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
            table_rows.append(row)
            if best_row is None or _selection_key(row) > _selection_key(best_row):
                best_row = row
                best_estimator = estimator
                best_scores = scores

    assert best_row is not None and best_estimator is not None and best_scores is not None
    test_cs = compute_context_sensitivity_metrics(
        pairs,
        best_scores,
        split_labels=split_labels,
        split="test",
    )
    test_metrics = {
        **test_cs,
        "pair_accuracy": pair_accuracy(labels, best_scores, test_mask),
    }
    validation_metrics = {
        key: best_row[key]
        for key in (
            "val_pair_accuracy",
            "val_row_strict_accuracy",
            "val_k1_accuracy",
            "val_k2_accuracy",
            "val_mean_min_margin",
        )
    }
    return SelectedProbe(
        estimator=best_estimator,
        layer_index=int(best_row["layer_index"]),
        layer_position=int(best_row["layer_position"]),
        C=float(best_row["C"]),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_table=tuple(table_rows),
        probe_scores=best_scores,
    )


def run_shuffle_controls(
    *,
    cache: ActivationCache,
    pairs: Sequence[EWOKProbePair],
    split_labels: np.ndarray,
    c_grid: Sequence[float],
    repeats: int,
    seed: int,
) -> tuple[dict, ...]:
    if repeats <= 0:
        return ()
    labels = cache.labels.astype(np.int64, copy=False)
    train_mask = _split_mask(split_labels, "train")
    controls: list[dict] = []
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + repeat + 1)
        shuffled = labels.copy()
        shuffled_train = shuffled[train_mask].copy()
        rng.shuffle(shuffled_train)
        shuffled[train_mask] = shuffled_train

        selected = fit_validation_selected_probe(
            cache=cache,
            pairs=pairs,
            split_labels=split_labels,
            c_grid=c_grid,
            seed=int(seed) + repeat + 1,
            train_labels_override=shuffled,
        )
        controls.append(
            {
                "repeat": int(repeat),
                "selected_layer_index": int(selected.layer_index),
                "selected_layer_position": int(selected.layer_position),
                "selected_C": float(selected.C),
                "val_row_strict_accuracy": float(selected.validation_metrics["val_row_strict_accuracy"]),
                "val_pair_accuracy": float(selected.validation_metrics["val_pair_accuracy"]),
                "test_row_strict_accuracy": float(selected.test_metrics["row_strict_accuracy"]),
                "test_pair_accuracy": float(selected.test_metrics["pair_accuracy"]),
                "test_k1_accuracy": float(selected.test_metrics["k1_accuracy"]),
                "test_k2_accuracy": float(selected.test_metrics["k2_accuracy"]),
            }
        )
    return tuple(controls)


__all__ = [
    "DEFAULT_C_GRID",
    "SelectedProbe",
    "fit_validation_selected_probe",
    "parse_c_grid",
    "run_shuffle_controls",
]
