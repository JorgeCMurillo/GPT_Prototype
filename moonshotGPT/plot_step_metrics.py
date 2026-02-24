#!/usr/bin/env python3
"""Generate plots from a run's step_metrics.json.

This script focuses on EWOK-centric records in step_metrics.json and:
1) Removes duplicate EWOK entries for the same optimizer step
   (prefers non-final records, which avoids the duplicated last step).
2) Plots training scalars carried in EWOK records (train_loss, lr, norms, tokens).
3) Plots EWOK official scores across steps (all domains + one plot per domain).
4) Plots EWOK full scores across steps (one plot per domain, both components).
5) Plots EWOK category subplots (TargetDiff, ContextDiff, ContextType) if present.
6) Plots EWOK full average across domains (sum vs mean) if present.
7) Optionally plots EWOK full-mean average comparison against another run.
8) Optionally plots HellaSwag if present in step_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

EWOK_WORD2VEC_MEAN_BASELINES: Dict[str, float] = {
    "social-interactions": 0.69,
    "social-properties": 0.73,
    "material-dynamics": 0.62,
    "social-relations": 0.51,
    "quantitative-properties": 0.54,
    "physical-dynamics": 0.62,
    "agent-properties": 0.51,
    "physical-interactions": 0.54,
    "material-properties": 0.52,
    "physical-relations": 0.51,
    "spatial-relations": 0.42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EWOK/train metrics from step_metrics.json")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to step_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write PNG plots (default: <metrics_dir>/plots_from_step_metrics)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="PNG DPI",
    )
    parser.add_argument(
        "--overlay-word2vec-ewok-mean",
        action="store_true",
        help="Overlay Word2Vec baselines on ewok_full_mean_domains_4x3 plot and save a separate PNG",
    )
    parser.add_argument(
        "--compare-metrics",
        type=str,
        default=None,
        help="Optional second step_metrics.json to compare against the primary run",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default="primary",
        help="Legend label for --metrics in comparison plots",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="compare",
        help="Legend label for --compare-metrics in comparison plots",
    )
    return parser.parse_args()


def _iso_or_min(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_records(metrics_path: Path) -> List[Dict]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{metrics_path} must contain a JSON list")
    return [x for x in data if isinstance(x, dict)]


def is_ewok_record(record: Dict) -> bool:
    return (
        "eval_official" in record
        or "eval_official_sum" in record
        or "eval_official_mean" in record
    )


def dedupe_ewok_by_step(records: Iterable[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
    """Group EWOK records by step and choose one preferred record per step.

    Preference rule:
    - prefer non-final records if available
    - otherwise use the latest timestamp in the group
    """
    by_step: Dict[int, List[Dict]] = defaultdict(list)
    for r in records:
        step = r.get("step")
        if isinstance(step, int):
            by_step[step].append(r)

    deduped: List[Dict] = []
    duplicates: Dict[int, int] = {}
    for step, group in by_step.items():
        if len(group) > 1:
            duplicates[step] = len(group)
        preferred = [r for r in group if not bool(r.get("final", False))]
        candidates = preferred if preferred else group
        chosen = sorted(candidates, key=lambda x: _iso_or_min(x.get("timestamp", "")))[-1]
        deduped.append(chosen)

    deduped.sort(key=lambda r: int(r["step"]))
    return deduped, duplicates


def get_ewok_payload(record: Dict, reduction: str) -> Tuple[Dict | None, Dict | None]:
    if reduction == "sum":
        return (
            record.get("eval_official_sum", record.get("eval_official")),
            record.get("eval_full_sum", record.get("eval_full")),
        )
    if reduction == "mean":
        return (
            record.get("eval_official_mean"),
            record.get("eval_full_mean"),
        )
    raise ValueError(f"unsupported reduction: {reduction}")


def detect_reductions(ewok_records: Iterable[Dict]) -> List[str]:
    seen_sum = False
    seen_mean = False
    for r in ewok_records:
        if "eval_official" in r or "eval_official_sum" in r:
            seen_sum = True
        if "eval_official_mean" in r:
            seen_mean = True
    out = []
    if seen_sum:
        out.append("sum")
    if seen_mean:
        out.append("mean")
    return out


def plot_training_scalars(ewok_records: List[Dict], output_dir: Path, dpi: int) -> List[Path]:
    fields = [
        ("train_loss_last", "Train Loss (Last)", "loss"),
        ("lr", "Learning Rate", "lr"),
        ("grad_norm_l2", "Grad Norm L2", "norm"),
        ("param_norm_l2", "Param Norm L2", "norm"),
        ("tokens_seen_global_approx", "Tokens Seen (Global Approx)", "tokens"),
    ]
    created: List[Path] = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.flatten()
    used_axes = 0
    for field, title, ylabel in fields:
        xs = []
        ys = []
        for r in ewok_records:
            step = r.get("step")
            val = r.get(field)
            if isinstance(step, int) and _is_number(val):
                xs.append(step)
                ys.append(float(val))
        if not xs:
            continue
        ax = axes[used_axes]
        used_axes += 1
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    for idx in range(used_axes, len(axes)):
        axes[idx].axis("off")

    if used_axes > 0:
        out = output_dir / "training_scalars_from_step_metrics.png"
        fig.savefig(out, dpi=dpi)
        created.append(out)
    plt.close(fig)
    return created


def plot_ewok_official(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
) -> List[Path]:
    by_domain: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in ewok_records:
        step = r.get("step")
        off, _ = get_ewok_payload(r, reduction)
        if not isinstance(step, int) or not isinstance(off, dict):
            continue
        for domain, value in off.items():
            if _is_number(value):
                by_domain[str(domain)].append((step, float(value)))

    created: List[Path] = []
    if not by_domain:
        return created

    # All domains on one figure
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1)
    for domain in sorted(by_domain):
        pts = sorted(by_domain[domain], key=lambda t: t[0])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=domain)
    ax.set_title(f"EWOK Official by Domain ({reduction})")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.2, label="random chance = 50%")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    out_all = output_dir / f"ewok_official_{reduction}_all_domains.png"
    fig.tight_layout()
    fig.savefig(out_all, dpi=dpi)
    plt.close(fig)
    created.append(out_all)

    return created


def _extract_pair(value) -> Tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1]):
        return float(value[0]), float(value[1])
    return None


def plot_ewok_full(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
    word2vec_baselines: Dict[str, float] | None = None,
) -> List[Path]:
    # For each domain in eval_full, compute average of both components:
    # avg_component = 0.5 * (component_1 + component_2)
    # Then render all 11 domains in a single 4x3 grid.
    by_domain: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for r in ewok_records:
        step = r.get("step")
        _, full = get_ewok_payload(r, reduction)
        if not isinstance(step, int) or not isinstance(full, dict):
            continue
        for domain, value in full.items():
            if str(domain) == "average":
                continue
            pair = _extract_pair(value)
            if pair is None:
                continue
            by_domain[str(domain)].append((step, pair[0], pair[1]))

    created: List[Path] = []
    if not by_domain:
        return created

    domains = sorted(by_domain)
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14), constrained_layout=True)
    flat_axes = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = flat_axes[idx]
        pts = sorted(by_domain[domain], key=lambda t: t[0])
        xs = [x for x, _, _ in pts]
        y_avg = [0.5 * (a + b) for _, a, b in pts]
        ax.plot(xs, y_avg, marker="o", linewidth=1.6, markersize=3.5, color="#2a6f97", label="avg(full)")
        ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
        w2v = word2vec_baselines.get(domain) if word2vec_baselines else None
        if _is_number(w2v):
            ax.axhline(
                float(w2v),
                color="#1b9e77",
                linestyle=(0, (3, 2)),
                linewidth=1.2,
                label=f"word2vec {float(w2v):.2f}",
            )
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Acc", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(f"EWOK Full (avg of both components) by Domain ({reduction})", fontsize=14)
    suffix = "_word2vec" if word2vec_baselines else ""
    out = output_dir / f"ewok_full_{reduction}_domains_4x3{suffix}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    created.append(out)
    return created


def _pair_to_scalar(value) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1]):
        return 0.5 * (float(value[0]) + float(value[1]))
    if _is_number(value):
        return float(value)
    return None


def _extract_full_average_scalar(full_payload: Dict) -> float | None:
    if not isinstance(full_payload, dict):
        return None

    avg = _pair_to_scalar(full_payload.get("average"))
    if avg is not None:
        return avg

    # Fallback for payloads that omit explicit "average": compute across domains.
    vals: List[float] = []
    for domain, value in full_payload.items():
        if str(domain) == "average":
            continue
        y = _pair_to_scalar(value)
        if y is not None:
            vals.append(y)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _full_average_series(
    ewok_records: List[Dict],
    reduction: str,
) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in ewok_records:
        step = r.get("step")
        _, full = get_ewok_payload(r, reduction)
        if not isinstance(step, int) or not isinstance(full, dict):
            continue
        y = _extract_full_average_scalar(full)
        if y is None:
            continue
        out.append((step, y))
    return sorted(out, key=lambda t: t[0])


def plot_ewok_full_average_all_domains(
    ewok_records: List[Dict],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    series: Dict[str, List[Tuple[int, float]]] = {
        "sum": _full_average_series(ewok_records, "sum"),
        "mean": _full_average_series(ewok_records, "mean"),
    }

    if not series["sum"] and not series["mean"]:
        return []

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(1, 1, 1)
    style = {
        "sum": {"color": "#1f77b4", "marker": "o"},
        "mean": {"color": "#2ca02c", "marker": "s"},
    }
    for reduction in ("sum", "mean"):
        pts = sorted(series[reduction], key=lambda t: t[0])
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            markersize=3.5,
            marker=style[reduction]["marker"],
            color=style[reduction]["color"],
            label=f"full_{reduction}_average",
        )

    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
    ax.set_title("EWOK Full Average Across Domains")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = output_dir / "ewok_full_average_all_domains_sum_vs_mean.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def plot_ewok_full_mean_average_compare(
    primary_ewok_records: List[Dict],
    compare_ewok_records: List[Dict],
    output_dir: Path,
    dpi: int,
    primary_label: str,
    compare_label: str,
) -> List[Path]:
    primary = _full_average_series(primary_ewok_records, "mean")
    compare = _full_average_series(compare_ewok_records, "mean")
    if not primary and not compare:
        return []

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    if primary:
        xs = [x for x, _ in primary]
        ys = [y for _, y in primary]
        ax.plot(xs, ys, linewidth=1.9, marker="o", markersize=3.5, color="#1f77b4", label=primary_label)

    if compare:
        xs = [x for x, _ in compare]
        ys = [y for _, y in compare]
        ax.plot(xs, ys, linewidth=1.9, marker="s", markersize=3.5, color="#ff7f0e", label=compare_label)

    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
    ax.set_title("EWOK Full Mean Average Across Domains: Run Comparison")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = output_dir / "ewok_full_mean_average_compare_runs.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def plot_ewok_category_subplots(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
) -> List[Path]:
    metric_key = f"eval_by_category_full_{reduction}"
    category_records = [
        r
        for r in ewok_records
        if isinstance(r.get("step"), int) and isinstance(r.get(metric_key), dict)
    ]
    if not category_records:
        return []

    last_by_col = category_records[-1].get(metric_key, {})
    if not isinstance(last_by_col, dict) or not last_by_col:
        return []

    created: List[Path] = []
    for column in sorted(last_by_col.keys(), key=lambda x: str(x)):
        col_last = last_by_col.get(column, {})
        if not isinstance(col_last, dict):
            continue

        categories = sorted(
            (k for k in col_last.keys() if str(k) != "average"),
            key=lambda x: str(x),
        )
        if not categories:
            continue

        avg_epochs: List[int] = []
        avg_vals: List[float] = []
        category_series: Dict[str, Tuple[List[int], List[float]]] = {}

        for rec in category_records:
            by_col = rec.get(metric_key, {})
            if not isinstance(by_col, dict):
                continue
            col_map = by_col.get(column, {})
            if not isinstance(col_map, dict):
                continue
            y_avg = _pair_to_scalar(col_map.get("average"))
            if y_avg is None:
                continue
            avg_epochs.append(rec["step"])
            avg_vals.append(y_avg)

        for category in categories:
            xs: List[int] = []
            ys: List[float] = []
            for rec in category_records:
                by_col = rec.get(metric_key, {})
                if not isinstance(by_col, dict):
                    continue
                col_map = by_col.get(column, {})
                if not isinstance(col_map, dict):
                    continue
                y = _pair_to_scalar(col_map.get(category))
                if y is None:
                    continue
                xs.append(rec["step"])
                ys.append(y)
            if xs:
                category_series[str(category)] = (xs, ys)

        if not category_series:
            continue

        ordered_categories = sorted(category_series.keys())
        ncols = min(3, max(1, len(ordered_categories)))
        nrows = int(math.ceil(len(ordered_categories) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, max(4, nrows * 3.8)),
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        for idx, category in enumerate(ordered_categories):
            ax = axes_flat[idx]
            xs, ys = category_series[category]
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color="#2a6f97", label=category)

            if avg_epochs and avg_vals:
                ax.plot(
                    avg_epochs,
                    avg_vals,
                    linewidth=1.0,
                    linestyle="--",
                    color="#808080",
                    alpha=0.28,
                    label="column_average",
                )

            ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
            ax.set_title(category, fontsize=10)
            ax.set_xlabel("Optimizer Step", fontsize=9)
            ax.set_ylabel("Acc", fontsize=9)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)

        for idx in range(len(ordered_categories), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"EWOK Category Accuracy by {column} ({reduction})", fontsize=14)
        slug = _safe_name(str(column).lower())
        out = output_dir / f"ewok_category_{slug}_{reduction}_subplots.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        created.append(out)

    return created


def plot_hellaswag(records: List[Dict], output_dir: Path, dpi: int) -> List[Path]:
    pts = []
    for r in records:
        hs = r.get("hellaswag")
        step = r.get("step")
        if isinstance(step, int) and isinstance(hs, dict):
            a = hs.get("accuracy")
            an = hs.get("accuracy_norm")
            if _is_number(a) and _is_number(an):
                pts.append((step, float(a), float(an)))
    pts.sort(key=lambda t: t[0])
    if not pts:
        return []

    xs = [x for x, _, _ in pts]
    y_acc = [a for _, a, _ in pts]
    y_norm = [b for _, _, b in pts]
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, y_acc, marker="o", linewidth=1.6, markersize=4, label="accuracy")
    ax.plot(xs, y_norm, marker="s", linewidth=1.4, markersize=3.5, label="accuracy_norm")
    ax.set_title("HellaSwag Across Steps")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    out = output_dir / "hellaswag_scores.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.exists():
        raise SystemExit(f"--metrics not found: {metrics_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (metrics_path.parent / "plots_from_step_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(metrics_path)
    ewok_all = [r for r in records if is_ewok_record(r)]
    ewok_records, dup_info = dedupe_ewok_by_step(ewok_all)
    reductions = detect_reductions(ewok_records)

    compare_ewok_records: List[Dict] = []
    compare_dup_info: Dict[int, int] = {}
    if args.compare_metrics:
        compare_path = Path(args.compare_metrics).expanduser().resolve()
        if not compare_path.exists():
            raise SystemExit(f"--compare-metrics not found: {compare_path}")
        compare_records = load_records(compare_path)
        compare_ewok_all = [r for r in compare_records if is_ewok_record(r)]
        compare_ewok_records, compare_dup_info = dedupe_ewok_by_step(compare_ewok_all)

    created: List[Path] = []
    created.extend(plot_training_scalars(ewok_records, output_dir, args.dpi))
    for reduction in reductions:
        created.extend(plot_ewok_official(ewok_records, output_dir, reduction, args.dpi))
        word2vec = EWOK_WORD2VEC_MEAN_BASELINES if (args.overlay_word2vec_ewok_mean and reduction == "mean") else None
        created.extend(plot_ewok_full(ewok_records, output_dir, reduction, args.dpi, word2vec_baselines=word2vec))
        created.extend(plot_ewok_category_subplots(ewok_records, output_dir, reduction, args.dpi))
    created.extend(plot_ewok_full_average_all_domains(ewok_records, output_dir, args.dpi))
    if compare_ewok_records:
        created.extend(
            plot_ewok_full_mean_average_compare(
                ewok_records,
                compare_ewok_records,
                output_dir,
                args.dpi,
                args.primary_label,
                args.compare_label,
            )
        )
    created.extend(plot_hellaswag(records, output_dir, args.dpi))

    print(f"Loaded records: total={len(records)}, ewok={len(ewok_all)}, ewok_deduped={len(ewok_records)}")
    if dup_info:
        dup_str = ", ".join(f"{k}x{v}" for k, v in sorted(dup_info.items()))
        print(f"Dropped duplicate EWOK steps (kept preferred record): {dup_str}")
    else:
        print("No duplicate EWOK steps found.")
    if args.compare_metrics:
        print(f"Loaded comparison EWOK records: {len(compare_ewok_records)}")
        if compare_dup_info:
            dup_str = ", ".join(f"{k}x{v}" for k, v in sorted(compare_dup_info.items()))
            print(f"Dropped duplicate comparison EWOK steps: {dup_str}")
        else:
            print("No duplicate comparison EWOK steps found.")
    print(f"Detected EWOK reductions: {', '.join(reductions) if reductions else 'none'}")
    print(f"Wrote {len(created)} plot(s) to: {output_dir}")
    for p in created:
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
