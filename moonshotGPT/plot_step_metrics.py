#!/usr/bin/env python3
"""Generate plots from a run's step_metrics.json.

This script focuses on EWOK-centric records in step_metrics.json and:
1) Removes duplicate EWOK entries for the same optimizer step
   (prefers non-final records, which avoids the duplicated last step).
2) Plots training scalars carried in EWOK records (train_loss, lr, norms, tokens).
3) Plots EWOK official scores across steps (all domains + one plot per domain).
4) Plots EWOK full scores across steps (one plot per domain, both components).
5) Optionally plots HellaSwag if present in step_metrics.json.
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


def _fit_line(xs: List[int], ys: List[float]) -> Tuple[float, float] | None:
    """Return (slope, intercept) for y = slope*x + intercept."""
    if len(xs) < 2 or len(ys) < 2:
        return None
    n = float(len(xs))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 0:
        return None
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _format_slope(value: float) -> str:
    # Keep roughly 4-5 decimals while preserving readability for small slopes.
    if abs(value) >= 1e-4:
        return f"{value:.5f}"
    return f"{value:.5e}"


def _fit_label(slope: float, intercept: float) -> str:
    return f"fit: y={_format_slope(slope)}x + {intercept:.5f}"


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
    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.2, label="baseline 0.50")
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
        fit = _fit_line(xs, y_avg)
        if fit is not None:
            slope, intercept = fit
            y_fit = [slope * x + intercept for x in xs]
            ax.plot(xs, y_fit, linewidth=1.5, color="#d95f02", label=_fit_label(slope, intercept))
        ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="baseline 0.50")
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Acc", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(f"EWOK Full (avg of both components) by Domain ({reduction})", fontsize=14)
    out = output_dir / f"ewok_full_{reduction}_domains_4x3.png"
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

    created: List[Path] = []
    created.extend(plot_training_scalars(ewok_records, output_dir, args.dpi))
    for reduction in reductions:
        created.extend(plot_ewok_official(ewok_records, output_dir, reduction, args.dpi))
        created.extend(plot_ewok_full(ewok_records, output_dir, reduction, args.dpi))
    created.extend(plot_hellaswag(records, output_dir, args.dpi))

    print(f"Loaded records: total={len(records)}, ewok={len(ewok_all)}, ewok_deduped={len(ewok_records)}")
    if dup_info:
        dup_str = ", ".join(f"{k}x{v}" for k, v in sorted(dup_info.items()))
        print(f"Dropped duplicate EWOK steps (kept preferred record): {dup_str}")
    else:
        print("No duplicate EWOK steps found.")
    print(f"Detected EWOK reductions: {', '.join(reductions) if reductions else 'none'}")
    print(f"Wrote {len(created)} plot(s) to: {output_dir}")
    for p in created:
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
