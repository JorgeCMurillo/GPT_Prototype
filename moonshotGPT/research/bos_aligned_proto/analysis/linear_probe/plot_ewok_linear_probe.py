#!/usr/bin/env python3
"""Plot artifacts from an EWoK linear-probe run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from .activations import load_activation_cache
from .artifacts import atomic_write_json
from .artifacts import atomic_write_csv
from .evaluation import CASE_BUCKETS
from .layerwise import compute_layer_domain_curves, load_probe_pairs_jsonl, load_split_labels_csv


BUCKET_LABELS = {
    "probe_correct_lm_correct": "probe ok\nLM ok",
    "probe_correct_lm_wrong": "probe ok\nLM wrong",
    "probe_wrong_lm_correct": "probe wrong\nLM ok",
    "probe_wrong_lm_wrong": "probe wrong\nLM wrong",
}
BUCKET_COLORS = {
    "probe_correct_lm_correct": "#2a9d8f",
    "probe_correct_lm_wrong": "#e9c46a",
    "probe_wrong_lm_correct": "#457b9d",
    "probe_wrong_lm_wrong": "#d16666",
}
PROBE_COLOR = "#2a9d8f"
LM_COLOR = "#457b9d"
SHUFFLE_COLOR = "#9a8c98"
CHANCE_COLOR = "#6c757d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Linear-probe output directory containing test_metrics.json and CSV artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Plot directory. Defaults to <run-dir>/plots.",
    )
    parser.add_argument("--label", default=None, help="Optional label for plot titles.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--formats",
        default="png,svg",
        help="Comma-separated output formats, e.g. 'png' or 'png,svg'.",
    )
    parser.add_argument(
        "--layer-score-split",
        choices=("val", "test"),
        default="test",
        help="Split to use for the domain-by-layer 4x3 score plot.",
    )
    parser.add_argument(
        "--layer-score-metric",
        choices=("directional_avg", "row_strict_accuracy", "pair_accuracy"),
        default="directional_avg",
        help="Metric to draw in the domain-by-layer 4x3 score plot.",
    )
    parser.add_argument(
        "--skip-layer-domain",
        action="store_true",
        help="Skip recomputing per-layer per-domain curves from the activation cache.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for refitting layer-domain probes.")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "plot"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def _formats(value: str) -> tuple[str, ...]:
    out = tuple(_safe_name(part.lower()) for part in str(value).split(",") if part.strip())
    return out or ("png",)


def _best_c_by_layer(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        layer = _int(row, "layer_index")
        current = best.get(layer)
        key = (
            _float(row, "val_row_strict_accuracy"),
            _float(row, "val_pair_accuracy"),
            _float(row, "val_mean_min_margin"),
            -_float(row, "C"),
        )
        current_key = (
            _float(current, "val_row_strict_accuracy"),
            _float(current, "val_pair_accuracy"),
            _float(current, "val_mean_min_margin"),
            -_float(current, "C"),
        ) if current is not None else None
        if current is None or key > current_key:
            best[layer] = dict(row)
    return {layer: _float(row, "C") for layer, row in best.items()}


def _save(fig, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _title(label: str | None, suffix: str) -> str:
    if label:
        return f"{label}: {suffix}"
    return suffix


def plot_layer_validation_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_layer: int | None,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    if not rows:
        return []

    by_layer: dict[int, dict[str, Any]] = {}
    for row in rows:
        layer = _int(row, "layer_index")
        current = by_layer.get(layer)
        if current is None:
            by_layer[layer] = dict(row)
            continue
        key = (
            _float(row, "val_row_strict_accuracy"),
            _float(row, "val_pair_accuracy"),
            _float(row, "val_mean_min_margin"),
        )
        current_key = (
            _float(current, "val_row_strict_accuracy"),
            _float(current, "val_pair_accuracy"),
            _float(current, "val_mean_min_margin"),
        )
        if key > current_key:
            by_layer[layer] = dict(row)

    layers = sorted(by_layer)
    strict = [_float(by_layer[layer], "val_row_strict_accuracy") for layer in layers]
    pair = [_float(by_layer[layer], "val_pair_accuracy") for layer in layers]
    k1 = [_float(by_layer[layer], "val_k1_accuracy") for layer in layers]
    k2 = [_float(by_layer[layer], "val_k2_accuracy") for layer in layers]

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.plot(layers, pair, marker="o", linewidth=1.8, color=PROBE_COLOR, label="pair accuracy")
    ax.plot(layers, strict, marker="s", linewidth=1.8, color="#e76f51", label="strict row accuracy")
    ax.plot(layers, k1, marker=".", linewidth=1.2, color="#7b2cbf", alpha=0.8, label="k1 accuracy")
    ax.plot(layers, k2, marker=".", linewidth=1.2, color="#f77f00", alpha=0.8, label="k2 accuracy")
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.65)
    if selected_layer is not None:
        ax.axvline(int(selected_layer), color="#222222", linestyle=(0, (5, 3)), linewidth=1.0, label="selected layer")
    ax.set_title(_title(label, "Validation-Selected Layer Curve"))
    ax.set_xlabel("Hidden-state layer index")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(layers)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, "layer_validation_curve", formats, dpi)


def plot_test_summary(
    test_metrics: Mapping[str, Any],
    shuffle_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    probe = test_metrics.get("probe", {})
    lm = test_metrics.get("lm", {})
    if not isinstance(probe, Mapping) or not isinstance(lm, Mapping):
        return []

    probe_directional = 0.5 * (_float(probe, "k1_accuracy") + _float(probe, "k2_accuracy"))
    lm_directional = 0.5 * (_float(lm, "k1_accuracy") + _float(lm, "k2_accuracy"))
    probe_strict = _float(probe, "row_strict_accuracy")
    lm_strict = _float(lm, "row_strict_accuracy")

    shuffle_directional = None
    shuffle_strict = None
    if shuffle_rows:
        shuffle_directional_vals = [
            0.5 * (_float(row, "test_k1_accuracy") + _float(row, "test_k2_accuracy"))
            for row in shuffle_rows
        ]
        shuffle_strict_vals = [_float(row, "test_row_strict_accuracy") for row in shuffle_rows]
        shuffle_directional = float(np_nanmean(shuffle_directional_vals))
        shuffle_strict = float(np_nanmean(shuffle_strict_vals))

    labels = ["directional avg", "strict row"]
    x = [0, 1]
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.bar([value - width for value in x], [probe_directional, probe_strict], width=width, color=PROBE_COLOR, label="probe")
    ax.bar(x, [lm_directional, lm_strict], width=width, color=LM_COLOR, label="LM score")
    if shuffle_directional is not None and shuffle_strict is not None:
        ax.bar(
            [value + width for value in x],
            [shuffle_directional, shuffle_strict],
            width=width,
            color=SHUFFLE_COLOR,
            label="shuffle mean",
        )
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out test accuracy")
    ax.set_title(_title(label, "Probe vs LM Test Summary"))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    return _save(fig, output_dir, "probe_lm_test_summary", formats, dpi)


def plot_bucket_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    counts = {str(row.get("bucket")): _int(row, "count") for row in rows}
    values = [counts.get(bucket, 0) for bucket in CASE_BUCKETS]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(
        [BUCKET_LABELS[bucket] for bucket in CASE_BUCKETS],
        values,
        color=[BUCKET_COLORS[bucket] for bucket in CASE_BUCKETS],
        width=0.65,
    )
    ax.set_title(_title(label, "Probe-vs-LM Case Buckets"))
    ax.set_ylabel("Test rows")
    ax.grid(True, axis="y", alpha=0.25)
    ax.bar_label(bars, fontsize=9, padding=3)
    fig.tight_layout()
    return _save(fig, output_dir, "probe_vs_lm_bucket_counts", formats, dpi)


def plot_domain_bucket_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    domains = sorted({str(row.get("domain")) for row in rows if row.get("domain")})
    if not domains:
        return []
    counts = {(str(row.get("domain")), str(row.get("bucket"))): _int(row, "count") for row in rows}
    matrix = []
    annotations = []
    for domain in domains:
        domain_counts = [counts.get((domain, bucket), 0) for bucket in CASE_BUCKETS]
        total = sum(domain_counts) or 1
        matrix.append([count / total for count in domain_counts])
        annotations.append(domain_counts)
    data = np.asarray(matrix, dtype=np.float64)

    fig_height = max(4.8, 0.42 * len(domains) + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    image = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.01, float(data.max())))
    ax.set_title(_title(label, "Domain Case-Bucket Rates"))
    ax.set_xticks(range(len(CASE_BUCKETS)), [BUCKET_LABELS[bucket] for bucket in CASE_BUCKETS], fontsize=8)
    ax.set_yticks(range(len(domains)), domains, fontsize=9)
    ax.set_xlabel("Case bucket")
    ax.set_ylabel("EWoK domain")
    for row_idx, domain_counts in enumerate(annotations):
        total = sum(domain_counts) or 1
        for col_idx, count in enumerate(domain_counts):
            rate = count / total
            text_color = "white" if rate > 0.45 * float(data.max()) else "#1f2933"
            ax.text(col_idx, row_idx, f"{count}\n{rate:.0%}", ha="center", va="center", fontsize=7, color=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Within-domain rate")
    fig.tight_layout()
    return _save(fig, output_dir, "domain_bucket_heatmap", formats, dpi)


def plot_probe_correct_lm_wrong_domains(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    domains = sorted({str(row.get("domain")) for row in rows if row.get("domain")})
    if not domains:
        return []
    by_domain: dict[str, dict[str, int]] = {domain: {bucket: 0 for bucket in CASE_BUCKETS} for domain in domains}
    for row in rows:
        domain = str(row.get("domain"))
        bucket = str(row.get("bucket"))
        by_domain.setdefault(domain, {b: 0 for b in CASE_BUCKETS})[bucket] = _int(row, "count")

    frame = []
    for domain in domains:
        counts = by_domain[domain]
        total = sum(counts.values()) or 1
        target = counts.get("probe_correct_lm_wrong", 0)
        frame.append((domain, target, target / total, total))
    frame.sort(key=lambda item: (item[2], item[1]), reverse=True)

    fig_height = max(4.8, 0.36 * len(frame) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    y = list(range(len(frame)))
    rates = [item[2] for item in frame]
    bars = ax.barh(y, rates, color="#e9c46a", edgecolor="#8a6d1f", linewidth=0.7)
    ax.set_yticks(y, [item[0] for item in frame], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(0.01, min(1.0, max(rates) * 1.2)))
    ax.set_xlabel("probe-correct / LM-wrong rate")
    ax.set_title(_title(label, "Where the Probe Succeeds and LM Scoring Fails"))
    ax.grid(True, axis="x", alpha=0.25)
    for bar, (_, count, rate, total) in zip(bars, frame):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{total} ({rate:.0%})",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    return _save(fig, output_dir, "probe_correct_lm_wrong_by_domain", formats, dpi)


def plot_layer_domain_4x3(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    selected_layer: int | None,
    output_dir: Path,
    label: str | None,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    if not rows:
        return []

    domains = sorted({str(row.get("domain")) for row in rows if row.get("domain") and str(row.get("domain")) != "average"})
    panel_domains = domains + ["average"]
    if len(panel_domains) > 12:
        panel_domains = domains[:11] + ["average"]

    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in panel_domains}
    for row in rows:
        domain = str(row.get("domain"))
        if domain in by_domain:
            by_domain[domain].append(dict(row))

    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.0), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = axes.flatten()
    for axis, domain in zip(flat_axes, panel_domains):
        domain_rows = sorted(by_domain.get(domain, []), key=lambda row: _int(row, "layer_index"))
        layers = [_int(row, "layer_index") for row in domain_rows]
        values = [_float(row, metric) for row in domain_rows]
        if domain == "average":
            axis.plot(layers, values, marker="o", linewidth=2.2, color="#264653")
            axis.set_title("average", fontsize=10, fontweight="bold")
        else:
            axis.plot(layers, values, marker="o", linewidth=1.8, color=PROBE_COLOR)
            axis.set_title(domain, fontsize=10)
        axis.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=0.8, alpha=0.65)
        if selected_layer is not None:
            axis.axvline(int(selected_layer), color="#222222", linestyle=(0, (4, 3)), linewidth=0.8, alpha=0.75)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, axis="y", alpha=0.22)
        if layers:
            axis.set_xticks(layers)
            axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=8)

    for axis in flat_axes[len(panel_domains) :]:
        axis.axis("off")

    metric_label = {
        "directional_avg": "Directional Avg",
        "row_strict_accuracy": "Strict Row Accuracy",
        "pair_accuracy": "Pair Accuracy",
    }.get(metric, metric)
    fig.suptitle(_title(label, f"EWoK Domain Scores by Layer ({metric_label})"), fontsize=14)
    fig.supxlabel("Hidden-state layer index")
    fig.supylabel("Held-out score")
    return _save(fig, output_dir, f"layer_domain_{_safe_name(metric)}_4x3", formats, dpi)


def np_nanmean(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def plot_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    label: str | None = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 150,
    layer_score_split: str = "test",
    layer_score_metric: str = "directional_avg",
    skip_layer_domain: bool = False,
    seed: int = 42,
) -> list[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this Python environment.")

    run_path = Path(run_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _read_json(run_path / "selected_probe_summary.json")
    test_metrics = _read_json(run_path / "test_metrics.json")
    layer_rows = _read_csv(run_path / "layer_validation_table.csv")
    bucket_rows = _read_csv(run_path / "probe_vs_lm_bucket_counts.csv")
    domain_rows = _read_csv(run_path / "domain_bucket_counts.csv")
    shuffle_path = run_path / "shuffle_controls.csv"
    shuffle_rows = _read_csv(shuffle_path) if shuffle_path.exists() else []

    title_label = label or Path(run_path).name
    created: list[Path] = []
    created.extend(
        plot_layer_validation_curve(
            layer_rows,
            selected_layer=selected.get("selected_layer_index"),
            output_dir=out_dir,
            label=title_label,
            formats=formats,
            dpi=dpi,
        )
    )
    created.extend(
        plot_test_summary(
            test_metrics,
            shuffle_rows,
            output_dir=out_dir,
            label=title_label,
            formats=formats,
            dpi=dpi,
        )
    )
    created.extend(
        plot_bucket_counts(
            bucket_rows,
            output_dir=out_dir,
            label=title_label,
            formats=formats,
            dpi=dpi,
        )
    )
    created.extend(
        plot_domain_bucket_heatmap(
            domain_rows,
            output_dir=out_dir,
            label=title_label,
            formats=formats,
            dpi=dpi,
        )
    )
    created.extend(
        plot_probe_correct_lm_wrong_domains(
            domain_rows,
            output_dir=out_dir,
            label=title_label,
            formats=formats,
            dpi=dpi,
        )
    )
    if not skip_layer_domain:
        cache_path = run_path / "activation_cache.fp16.npz"
        items_path = run_path / "items.jsonl"
        splits_path = run_path / "split_assignments.csv"
        if cache_path.exists() and items_path.exists() and splits_path.exists():
            pairs = load_probe_pairs_jsonl(items_path)
            cache = load_activation_cache(cache_path, pairs)
            split_labels = load_split_labels_csv(splits_path, pairs)
            c_grid = selected.get("c_grid") or [row.get("C") for row in layer_rows if row.get("C")]
            c_grid = tuple(float(value) for value in c_grid)
            layer_domain_rows = compute_layer_domain_curves(
                cache=cache,
                pairs=pairs,
                split_labels=split_labels,
                c_grid=c_grid,
                seed=int(seed),
                score_split=layer_score_split,
                layer_c_values=_best_c_by_layer(layer_rows),
            )
            atomic_write_csv(out_dir / "layer_domain_scores.csv", list(layer_domain_rows))
            created.extend(
                plot_layer_domain_4x3(
                    layer_domain_rows,
                    metric=layer_score_metric,
                    selected_layer=selected.get("selected_layer_index"),
                    output_dir=out_dir,
                    label=title_label,
                    formats=formats,
                    dpi=dpi,
                )
            )

    manifest = {
        "run_dir": str(run_path),
        "output_dir": str(out_dir),
        "label": title_label,
        "formats": list(formats),
        "selected_layer_index": selected.get("selected_layer_index"),
        "selected_C": selected.get("selected_C"),
        "layer_score_split": layer_score_split,
        "layer_score_metric": layer_score_metric,
        "seed": int(seed),
        "plots": [str(path) for path in created],
    }
    atomic_write_json(out_dir / "plot_manifest.json", manifest)
    return created


def main() -> None:
    args = parse_args()
    created = plot_run(
        args.run_dir,
        output_dir=args.output_dir,
        label=args.label,
        formats=_formats(args.formats),
        dpi=int(args.dpi),
        layer_score_split=args.layer_score_split,
        layer_score_metric=args.layer_score_metric,
        skip_layer_domain=bool(args.skip_layer_domain),
        seed=int(args.seed),
    )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
