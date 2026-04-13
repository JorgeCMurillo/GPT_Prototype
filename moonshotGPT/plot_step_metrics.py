#!/usr/bin/env python3
"""Generate plots from a run's step_metrics.json.

This script focuses on EWOK-centric records in step_metrics.json and:
1) Removes duplicate EWOK entries for the same optimizer step
   (prefers non-final records, which avoids the duplicated last step).
2) Plots training scalars carried in EWOK records (train_loss, lr, norms, tokens).
3) Plots EWOK full/category metrics for the enabled reductions.
   By default this means mean-reduction plots only; sum plots are opt-in.
4) Plots EWOK mean margins (signed and absolute) when present for the enabled reductions.
5) Plots EWOK full average across domains (sum vs mean) if sum plots are enabled.
6) Optionally plots EWOK full-mean average comparison against another run.
7) Optionally plots HellaSwag if present in step_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_WORD2VEC_GLOB = _PROJECT_DIR / "runs" / "research" / "w2v_lexical_probe"
WORD2VEC_BASELINE_LABEL = "FineWeb-Edu Word2Vec"


def _default_word2vec_interval_metrics() -> Path:
    env_path = os.environ.get("MOONSHOT_WORD2VEC_INTERVAL_METRICS")
    if env_path:
        return Path(env_path).expanduser()

    candidates = sorted(_DEFAULT_WORD2VEC_GLOB.glob("*/ewok_interval_metrics.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    return _DEFAULT_WORD2VEC_GLOB / "<run_name>" / "ewok_interval_metrics.jsonl"


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
        "--word2vec-interval-metrics",
        type=str,
        default=str(_default_word2vec_interval_metrics()),
        help="Path to Word2Vec ewok_interval_metrics.jsonl used for the EWOK mean baseline overlay.",
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
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="If set, only keep EWOK records with step <= this value before plotting",
    )
    parser.add_argument(
        "--compare-full-mean-domains-4x3",
        action="store_true",
        help="When --compare-metrics is set, plot both runs in a per-domain EWOK full-mean grid",
    )
    parser.add_argument(
        "--compare-category-columns",
        type=str,
        default="",
        help="Comma-separated EWOK category columns for mean comparison (e.g. ContextDiff,ContextType)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window for plotted lines (1 disables smoothing)",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable point markers on line plots",
    )
    parser.add_argument(
        "--include-ewok-sum-plots",
        action="store_true",
        help="Include EWOK sum-reduction plots; default behavior is mean-only",
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


def _parse_csv_list(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    out: List[str] = []
    for part in value.split(","):
        item = part.strip()
        if item:
            out.append(item)
    return out


def _load_word2vec_mean_baselines(interval_metrics_path: Path) -> Dict[str, float]:
    rows = []
    with interval_metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get("domain_scores_full"), dict):
                rows.append(payload)
    if not rows:
        raise ValueError(f"No domain_scores_full payloads found in {interval_metrics_path}")

    final_payload = rows[-1]["domain_scores_full"]
    out: Dict[str, float] = {}
    for domain, value in final_payload.items():
        if str(domain) == "average":
            continue
        scalar = _pair_to_scalar(value)
        if scalar is not None:
            out[str(domain)] = float(scalar)
    if not out:
        raise ValueError(f"Could not extract Word2Vec baselines from {interval_metrics_path}")
    return out


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _format_compact_count(value, _pos=None) -> str:
    if not _is_number(value):
        return ""
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"{sign}{value / scale:.3g}{suffix}"
    return f"{sign}{value:.3g}"


def _ewok_use_token_axis(records: Iterable[Dict]) -> bool:
    xs = [
        record.get("tokens_seen_global_approx")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("step"), int)
    ]
    return bool(xs) and all(_is_number(x) for x in xs)


def _ewok_x_value(record: Dict, use_tokens: bool) -> float | None:
    if use_tokens:
        value = record.get("tokens_seen_global_approx")
        if _is_number(value):
            return float(value)
    step = record.get("step")
    if isinstance(step, int):
        return float(step)
    return None


def _apply_ewok_x_axis(ax, use_tokens: bool, fontsize: int | None = None) -> None:
    if use_tokens:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_compact_count))
        label = "Tokens Observed"
    else:
        label = "Optimizer Step"
    if fontsize is None:
        ax.set_xlabel(label)
    else:
        ax.set_xlabel(label, fontsize=fontsize)


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    w = max(1, int(window))
    out: List[float] = []
    running = 0.0
    q: List[float] = []
    for y in values:
        q.append(float(y))
        running += float(y)
        if len(q) > w:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def _smoothed_xy(points: List[Tuple[int, float]], smooth_window: int) -> Tuple[List[int], List[float]]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return xs, _moving_average(ys, smooth_window)


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
        or "eval_babylm_completion_choice_official" in record
        or "eval_babylm_completion_choice_official_sum" in record
        or "eval_babylm_completion_choice_official_mean" in record
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


def filter_records_by_max_step(ewok_records: List[Dict], max_step: int | None) -> List[Dict]:
    if not isinstance(max_step, int):
        return list(ewok_records)
    return [r for r in ewok_records if isinstance(r.get("step"), int) and int(r["step"]) <= max_step]


def get_ewok_payload(record: Dict, reduction: str) -> Tuple[Dict | None, Dict | None]:
    if reduction == "sum":
        return (
            record.get(
                "eval_babylm_completion_choice_official_sum",
                record.get("eval_official_sum", record.get("eval_official")),
            ),
            record.get(
                "eval_babylm_completion_choice_full_sum",
                record.get("eval_full_sum", record.get("eval_full")),
            ),
        )
    if reduction == "mean":
        return (
            record.get("eval_babylm_completion_choice_official_mean", record.get("eval_official_mean")),
            record.get("eval_babylm_completion_choice_full_mean", record.get("eval_full_mean")),
        )
    raise ValueError(f"unsupported reduction: {reduction}")


def get_margin_payload(record: Dict, reduction: str) -> Dict | None:
    if reduction == "sum":
        return record.get(
            "eval_babylm_completion_choice_margin_stats_sum",
            record.get("eval_margin_stats_sum", record.get("eval_margin_stats")),
        )
    if reduction == "mean":
        return record.get(
            "eval_babylm_completion_choice_margin_stats_mean",
            record.get("eval_margin_stats_mean"),
        )
    raise ValueError(f"unsupported reduction: {reduction}")


def get_category_payload(record: Dict, reduction: str) -> Dict | None:
    if reduction == "sum":
        return record.get(
            "eval_babylm_completion_choice_by_category_full_sum",
            record.get("eval_by_category_full_sum"),
        )
    if reduction == "mean":
        return record.get(
            "eval_babylm_completion_choice_by_category_full_mean",
            record.get("eval_by_category_full_mean"),
        )
    raise ValueError(f"unsupported reduction: {reduction}")


def detect_reductions(ewok_records: Iterable[Dict]) -> List[str]:
    seen_sum = False
    seen_mean = False
    for r in ewok_records:
        if (
            "eval_official" in r
            or "eval_official_sum" in r
            or "eval_babylm_completion_choice_official" in r
            or "eval_babylm_completion_choice_official_sum" in r
        ):
            seen_sum = True
        if "eval_official_mean" in r or "eval_babylm_completion_choice_official_mean" in r:
            seen_mean = True
    out = []
    if seen_sum:
        out.append("sum")
    if seen_mean:
        out.append("mean")
    return out


def filter_enabled_reductions(reductions: Iterable[str], include_sum: bool) -> List[str]:
    enabled: List[str] = []
    for reduction in reductions:
        if reduction == "sum" and not include_sum:
            continue
        enabled.append(reduction)
    return enabled


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
    use_tokens = _ewok_use_token_axis(ewok_records)
    by_domain: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for r in ewok_records:
        x_value = _ewok_x_value(r, use_tokens)
        _, full = get_ewok_payload(r, reduction)
        if x_value is None or not isinstance(full, dict):
            continue
        for domain, value in full.items():
            if str(domain) == "average":
                continue
            pair = _extract_pair(value)
            if pair is None:
                continue
            by_domain[str(domain)].append((x_value, pair[0], pair[1]))

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
                label=WORD2VEC_BASELINE_LABEL,
            )
        ax.set_title(domain, fontsize=10)
        _apply_ewok_x_axis(ax, use_tokens, fontsize=9)
        ax.set_ylabel("Acc", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    title = f"EWOK Full (avg of both components) by Domain ({reduction})"
    if word2vec_baselines:
        title += " with Word2Vec baseline"
    fig.suptitle(title, fontsize=14)
    out = output_dir / f"ewok_full_{reduction}_domains_4x3.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    created.append(out)
    return created


def plot_ewok_margin_domains(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
) -> List[Path]:
    # Render all domains in a 4x3 grid with both signed and absolute mean margin.
    use_tokens = _ewok_use_token_axis(ewok_records)
    by_domain: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for r in ewok_records:
        x_value = _ewok_x_value(r, use_tokens)
        margin_payload = get_margin_payload(r, reduction)
        if x_value is None or not isinstance(margin_payload, dict):
            continue
        for domain, stats in margin_payload.items():
            if str(domain) == "average" or not isinstance(stats, dict):
                continue
            y_signed = stats.get("mean_signed_m")
            y_abs = stats.get("mean_abs_m")
            if _is_number(y_signed) and _is_number(y_abs):
                by_domain[str(domain)].append((x_value, float(y_signed), float(y_abs)))

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
        ys_signed = [a for _, a, _ in pts]
        ys_abs = [b for _, _, b in pts]
        ax.plot(xs, ys_signed, marker="o", linewidth=1.6, markersize=3.5, color="#1f77b4", label="mean signed margin")
        ax.plot(
            xs,
            ys_abs,
            marker="s",
            linewidth=1.4,
            markersize=3.2,
            linestyle=(0, (4, 2)),
            color="#ff7f0e",
            label="mean abs margin",
        )
        ax.axhline(0.0, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="zero margin")
        ax.set_title(domain, fontsize=10)
        _apply_ewok_x_axis(ax, use_tokens, fontsize=9)
        ax.set_ylabel("Margin", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(f"EWOK Mean Margins by Domain ({reduction})", fontsize=14)
    out = output_dir / f"ewok_margin_{reduction}_domains_4x3.png"
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
    use_tokens: bool = False,
) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in ewok_records:
        x_value = _ewok_x_value(r, use_tokens)
        _, full = get_ewok_payload(r, reduction)
        if x_value is None or not isinstance(full, dict):
            continue
        y = _extract_full_average_scalar(full)
        if y is None:
            continue
        out.append((x_value, y))
    return sorted(out, key=lambda t: t[0])


def _extract_margin_average_scalar(margin_payload: Dict, metric_key: str) -> float | None:
    if not isinstance(margin_payload, dict):
        return None

    avg = margin_payload.get("average")
    if isinstance(avg, dict) and _is_number(avg.get(metric_key)):
        return float(avg[metric_key])

    vals: List[float] = []
    for domain, stats in margin_payload.items():
        if str(domain) == "average" or not isinstance(stats, dict):
            continue
        if _is_number(stats.get(metric_key)):
            vals.append(float(stats[metric_key]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _margin_average_series(
    ewok_records: List[Dict],
    reduction: str,
    metric_key: str,
    use_tokens: bool = False,
) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in ewok_records:
        x_value = _ewok_x_value(r, use_tokens)
        margin_payload = get_margin_payload(r, reduction)
        if x_value is None or not isinstance(margin_payload, dict):
            continue
        y = _extract_margin_average_scalar(margin_payload, metric_key)
        if y is None:
            continue
        out.append((x_value, y))
    return sorted(out, key=lambda t: t[0])


def plot_ewok_margin_average_all_domains(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
) -> List[Path]:
    use_tokens = _ewok_use_token_axis(ewok_records)
    signed = _margin_average_series(ewok_records, reduction, "mean_signed_m", use_tokens=use_tokens)
    abs_margin = _margin_average_series(ewok_records, reduction, "mean_abs_m", use_tokens=use_tokens)
    if not signed and not abs_margin:
        return []

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    if signed:
        xs = [x for x, _ in signed]
        ys = [y for _, y in signed]
        ax.plot(xs, ys, linewidth=1.9, marker="o", markersize=3.5, color="#1f77b4", label="mean signed margin")

    if abs_margin:
        xs = [x for x, _ in abs_margin]
        ys = [y for _, y in abs_margin]
        ax.plot(
            xs,
            ys,
            linewidth=1.6,
            marker="s",
            markersize=3.4,
            linestyle=(0, (4, 2)),
            color="#ff7f0e",
            alpha=0.45,
            label="mean abs margin (reference)",
        )

    ax.axhline(0.0, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="zero margin")
    ax.set_title(f"EWOK Mean Margins Across Domains ({reduction})")
    _apply_ewok_x_axis(ax, use_tokens)
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Add a compact math + intuition note directly on the chart.
    reduction_label = (
        r"$s(C,T)=\sum_t \log P_{\theta}(t\mid C)$"
        if reduction == "sum"
        else r"$s(C,T)=\frac{1}{|T|}\sum_t \log P_{\theta}(t\mid C)$"
    )
    expl = (
        r"$m_1=s(C_1,T_1)-s(C_1,T_2),\ m_2=s(C_2,T_2)-s(C_2,T_1),\ m=\frac{1}{2}(m_1+m_2)$"
        "\n"
        r"$\mu_d=\mathbb{E}_i[m_i],\ \mathrm{plotted}=\frac{1}{D}\sum_d \mu_d$"
        "\n"
        + reduction_label
        + "; "
        + r"$\mathrm{abs\ ref}=\frac{1}{D}\sum_d \mathbb{E}_i[|m_i|]$"
        + "\n"
        + "Intuition: signed > 0 favors the correct direction; near 0 with high abs can indicate strong but inconsistent or biased discrimination."
    )
    ax.text(
        0.015,
        0.015,
        expl,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.75},
    )

    out = output_dir / f"ewok_margin_{reduction}_average_all_domains.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def plot_ewok_full_average_all_domains(
    ewok_records: List[Dict],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    use_tokens = _ewok_use_token_axis(ewok_records)
    series: Dict[str, List[Tuple[int, float]]] = {
        "sum": _full_average_series(ewok_records, "sum", use_tokens=use_tokens),
        "mean": _full_average_series(ewok_records, "mean", use_tokens=use_tokens),
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
    _apply_ewok_x_axis(ax, use_tokens)
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
    smooth_window: int = 1,
    show_markers: bool = True,
) -> List[Path]:
    use_tokens = _ewok_use_token_axis(primary_ewok_records) and _ewok_use_token_axis(compare_ewok_records)
    primary = _full_average_series(primary_ewok_records, "mean", use_tokens=use_tokens)
    compare = _full_average_series(compare_ewok_records, "mean", use_tokens=use_tokens)
    if not primary and not compare:
        return []

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    if primary:
        xs, ys = _smoothed_xy(primary, smooth_window)
        if show_markers:
            ax.plot(xs, ys, linewidth=1.9, marker="o", markersize=3.5, color="#1f77b4", label=primary_label)
        else:
            ax.plot(xs, ys, linewidth=1.9, color="#1f77b4", label=primary_label)

    if compare:
        xs, ys = _smoothed_xy(compare, smooth_window)
        if show_markers:
            ax.plot(xs, ys, linewidth=1.9, marker="s", markersize=3.5, color="#ff7f0e", label=compare_label)
        else:
            ax.plot(xs, ys, linewidth=1.9, color="#ff7f0e", label=compare_label)

    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
    ax.set_title("EWOK Full Mean Average Across Domains: Run Comparison")
    _apply_ewok_x_axis(ax, use_tokens)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = output_dir / "ewok_full_mean_average_compare_runs.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def _full_mean_domain_series(
    ewok_records: List[Dict],
    use_tokens: bool = False,
) -> Dict[str, List[Tuple[int, float]]]:
    by_domain: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in ewok_records:
        x_value = _ewok_x_value(r, use_tokens)
        full = r.get("eval_babylm_completion_choice_full_mean", r.get("eval_full_mean"))
        if x_value is None or not isinstance(full, dict):
            continue
        for domain, value in full.items():
            if str(domain) == "average":
                continue
            y = _pair_to_scalar(value)
            if y is None:
                continue
            by_domain[str(domain)].append((x_value, y))
    for domain in list(by_domain.keys()):
        by_domain[domain] = sorted(by_domain[domain], key=lambda t: t[0])
    return by_domain


def plot_ewok_full_mean_domains_compare(
    primary_ewok_records: List[Dict],
    compare_ewok_records: List[Dict],
    output_dir: Path,
    dpi: int,
    primary_label: str,
    compare_label: str,
    word2vec_baselines: Dict[str, float] | None = None,
    max_step: int | None = None,
    smooth_window: int = 1,
    show_markers: bool = True,
) -> List[Path]:
    use_tokens = _ewok_use_token_axis(primary_ewok_records) and _ewok_use_token_axis(compare_ewok_records)
    primary_by_domain = _full_mean_domain_series(primary_ewok_records, use_tokens=use_tokens)
    compare_by_domain = _full_mean_domain_series(compare_ewok_records, use_tokens=use_tokens)
    domains = set(primary_by_domain.keys()) | set(compare_by_domain.keys())
    if word2vec_baselines:
        domains |= set(word2vec_baselines.keys())
    domains = sorted(domains)
    if not domains:
        return []

    ncols = 3
    nrows = 4 if len(domains) <= 12 else int(math.ceil(len(domains) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(14, nrows * 3.5)), constrained_layout=True)
    flat_axes = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = flat_axes[idx]
        pts_primary = primary_by_domain.get(domain, [])
        pts_compare = compare_by_domain.get(domain, [])

        if pts_primary:
            xs, ys = _smoothed_xy(pts_primary, smooth_window)
            if show_markers:
                ax.plot(xs, ys, linewidth=1.8, marker="o", markersize=3.5, color="#1f77b4", label=primary_label)
            else:
                ax.plot(xs, ys, linewidth=1.8, color="#1f77b4", label=primary_label)
        if pts_compare:
            xs, ys = _smoothed_xy(pts_compare, smooth_window)
            if show_markers:
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.8,
                    marker="s",
                    markersize=3.3,
                    linestyle=(0, (4, 2)),
                    color="#ff7f0e",
                    label=compare_label,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.8,
                    linestyle=(0, (4, 2)),
                    color="#ff7f0e",
                    label=compare_label,
                )

        w2v = word2vec_baselines.get(domain) if word2vec_baselines else None
        if _is_number(w2v):
            ax.axhline(
                float(w2v),
                color="#1b9e77",
                linestyle=(0, (3, 2)),
                linewidth=1.3,
                label=WORD2VEC_BASELINE_LABEL,
            )
        ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="random chance = 50%")

        ax.set_title(domain, fontsize=10)
        _apply_ewok_x_axis(ax, use_tokens, fontsize=9)
        ax.set_ylabel("Acc", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        if isinstance(max_step, int) and not use_tokens:
            ax.set_xlim(0, max_step)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    step_note = f" (<= step {max_step})" if isinstance(max_step, int) else ""
    fig.suptitle(f"EWOK Full Mean by Domain: {primary_label} vs {compare_label}{step_note}", fontsize=14)
    suffix_w2v = "_word2vec" if word2vec_baselines else ""
    suffix_step = f"_to_step{max_step}" if isinstance(max_step, int) else ""
    out = output_dir / f"ewok_full_mean_domains_4x3_compare_runs{suffix_w2v}{suffix_step}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def _category_column_series(
    ewok_records: List[Dict],
    reduction: str,
    column: str,
    use_tokens: bool = False,
) -> Dict[str, List[Tuple[int, float]]]:
    by_category: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for rec in ewok_records:
        x_value = _ewok_x_value(rec, use_tokens)
        by_col = get_category_payload(rec, reduction)
        if x_value is None or not isinstance(by_col, dict):
            continue
        col_map = by_col.get(column)
        if not isinstance(col_map, dict):
            continue
        for category, value in col_map.items():
            if str(category) == "average":
                continue
            y = _pair_to_scalar(value)
            if y is None:
                continue
            by_category[str(category)].append((x_value, y))
    for category in list(by_category.keys()):
        by_category[category] = sorted(by_category[category], key=lambda t: t[0])
    return by_category


def _available_category_columns(ewok_records: List[Dict], reduction: str = "mean") -> List[str]:
    columns: set[str] = set()
    for rec in ewok_records:
        payload = get_category_payload(rec, reduction)
        if isinstance(payload, dict):
            columns.update(str(k) for k in payload.keys())
    return sorted(columns)


def _resolve_requested_columns(
    requested_columns: List[str],
    primary_ewok_records: List[Dict],
    compare_ewok_records: List[Dict],
    reduction: str = "mean",
) -> Tuple[List[str], List[str]]:
    available = set(_available_category_columns(primary_ewok_records, reduction)) | set(
        _available_category_columns(compare_ewok_records, reduction)
    )
    lower_to_canonical = {c.lower(): c for c in available}
    resolved: List[str] = []
    missing: List[str] = []
    for item in requested_columns:
        key = item.strip()
        if not key:
            continue
        canonical = lower_to_canonical.get(key.lower())
        if canonical:
            if canonical not in resolved:
                resolved.append(canonical)
        else:
            missing.append(key)
    return resolved, missing


def plot_ewok_category_column_mean_compare(
    primary_ewok_records: List[Dict],
    compare_ewok_records: List[Dict],
    column: str,
    output_dir: Path,
    dpi: int,
    primary_label: str,
    compare_label: str,
    max_step: int | None = None,
    smooth_window: int = 1,
    show_markers: bool = True,
) -> List[Path]:
    use_tokens = _ewok_use_token_axis(primary_ewok_records) and _ewok_use_token_axis(compare_ewok_records)
    primary_by_category = _category_column_series(primary_ewok_records, "mean", column, use_tokens=use_tokens)
    compare_by_category = _category_column_series(compare_ewok_records, "mean", column, use_tokens=use_tokens)
    categories = sorted(set(primary_by_category.keys()) | set(compare_by_category.keys()))
    if not categories:
        return []

    ncols = min(3, max(1, len(categories)))
    nrows = int(math.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, max(4, nrows * 3.8)),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for idx, category in enumerate(categories):
        ax = axes_flat[idx]
        primary_pts = primary_by_category.get(category, [])
        compare_pts = compare_by_category.get(category, [])

        if primary_pts:
            xs, ys = _smoothed_xy(primary_pts, smooth_window)
            if show_markers:
                ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color="#1f77b4", label=primary_label)
            else:
                ax.plot(xs, ys, linewidth=1.8, color="#1f77b4", label=primary_label)

        if compare_pts:
            xs, ys = _smoothed_xy(compare_pts, smooth_window)
            if show_markers:
                ax.plot(
                    xs,
                    ys,
                    marker="s",
                    linewidth=1.8,
                    markersize=3.3,
                    linestyle=(0, (4, 2)),
                    color="#ff7f0e",
                    label=compare_label,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.8,
                    linestyle=(0, (4, 2)),
                    color="#ff7f0e",
                    label=compare_label,
                )

        ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
        ax.set_title(category, fontsize=10)
        _apply_ewok_x_axis(ax, use_tokens, fontsize=9)
        ax.set_ylabel("Acc", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        if isinstance(max_step, int) and not use_tokens:
            ax.set_xlim(0, max_step)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(categories), len(axes_flat)):
        axes_flat[idx].axis("off")

    step_note = f" (<= step {max_step})" if isinstance(max_step, int) else ""
    fig.suptitle(f"EWOK Mean by {column}: {primary_label} vs {compare_label}{step_note}", fontsize=14)
    step_suffix = f"_to_step{max_step}" if isinstance(max_step, int) else ""
    out = output_dir / f"ewok_category_{_safe_name(column.lower())}_mean_compare_runs{step_suffix}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return [out]


def plot_ewok_category_subplots(
    ewok_records: List[Dict],
    output_dir: Path,
    reduction: str,
    dpi: int,
) -> List[Path]:
    category_records = [
        r
        for r in ewok_records
        if isinstance(r.get("step"), int) and isinstance(get_category_payload(r, reduction), dict)
    ]
    if not category_records:
        return []
    use_tokens = _ewok_use_token_axis(category_records)

    last_by_col = get_category_payload(category_records[-1], reduction) or {}
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
            by_col = get_category_payload(rec, reduction) or {}
            if not isinstance(by_col, dict):
                continue
            col_map = by_col.get(column, {})
            if not isinstance(col_map, dict):
                continue
            y_avg = _pair_to_scalar(col_map.get("average"))
            if y_avg is None:
                continue
            x_value = _ewok_x_value(rec, use_tokens)
            if x_value is None:
                continue
            avg_epochs.append(x_value)
            avg_vals.append(y_avg)

        for category in categories:
            xs: List[int] = []
            ys: List[float] = []
            for rec in category_records:
                by_col = get_category_payload(rec, reduction) or {}
                if not isinstance(by_col, dict):
                    continue
                col_map = by_col.get(column, {})
                if not isinstance(col_map, dict):
                    continue
                y = _pair_to_scalar(col_map.get(category))
                if y is None:
                    continue
                x_value = _ewok_x_value(rec, use_tokens)
                if x_value is None:
                    continue
                xs.append(x_value)
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
            _apply_ewok_x_axis(ax, use_tokens, fontsize=9)
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
    ewok_records = filter_records_by_max_step(ewok_records, args.max_step)
    available_reductions = detect_reductions(ewok_records)
    reductions = filter_enabled_reductions(available_reductions, args.include_ewok_sum_plots)

    compare_ewok_records: List[Dict] = []
    compare_dup_info: Dict[int, int] = {}
    if args.compare_metrics:
        compare_path = Path(args.compare_metrics).expanduser().resolve()
        if not compare_path.exists():
            raise SystemExit(f"--compare-metrics not found: {compare_path}")
        compare_records = load_records(compare_path)
        compare_ewok_all = [r for r in compare_records if is_ewok_record(r)]
        compare_ewok_records, compare_dup_info = dedupe_ewok_by_step(compare_ewok_all)
        compare_ewok_records = filter_records_by_max_step(compare_ewok_records, args.max_step)

    word2vec_baselines = None
    if args.overlay_word2vec_ewok_mean:
        word2vec_baselines = _load_word2vec_mean_baselines(Path(args.word2vec_interval_metrics).expanduser().resolve())

    created: List[Path] = []
    created.extend(plot_training_scalars(ewok_records, output_dir, args.dpi))
    for reduction in reductions:
        word2vec = word2vec_baselines if (word2vec_baselines is not None and reduction == "mean") else None
        created.extend(plot_ewok_full(ewok_records, output_dir, reduction, args.dpi, word2vec_baselines=word2vec))
        created.extend(plot_ewok_margin_domains(ewok_records, output_dir, reduction, args.dpi))
        created.extend(plot_ewok_margin_average_all_domains(ewok_records, output_dir, reduction, args.dpi))
        created.extend(plot_ewok_category_subplots(ewok_records, output_dir, reduction, args.dpi))
    if args.include_ewok_sum_plots:
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
                smooth_window=args.smooth_window,
                show_markers=not args.no_markers,
            )
        )
        if args.compare_full_mean_domains_4x3:
            compare_word2vec = word2vec_baselines if args.overlay_word2vec_ewok_mean else None
            created.extend(
                plot_ewok_full_mean_domains_compare(
                    ewok_records,
                    compare_ewok_records,
                    output_dir,
                    args.dpi,
                    args.primary_label,
                    args.compare_label,
                    word2vec_baselines=compare_word2vec,
                    max_step=args.max_step,
                    smooth_window=args.smooth_window,
                    show_markers=not args.no_markers,
                )
            )

        requested_columns = _parse_csv_list(args.compare_category_columns)
        if requested_columns:
            resolved_columns, missing_columns = _resolve_requested_columns(
                requested_columns,
                ewok_records,
                compare_ewok_records,
                reduction="mean",
            )
            for column in resolved_columns:
                created.extend(
                    plot_ewok_category_column_mean_compare(
                        ewok_records,
                        compare_ewok_records,
                        column,
                        output_dir,
                        args.dpi,
                        args.primary_label,
                        args.compare_label,
                        max_step=args.max_step,
                        smooth_window=args.smooth_window,
                        show_markers=not args.no_markers,
                    )
                )
            if missing_columns:
                print(
                    "Requested comparison category columns not found: "
                    + ", ".join(missing_columns)
                )
    created.extend(plot_hellaswag(records, output_dir, args.dpi))

    print(f"Loaded records: total={len(records)}, ewok={len(ewok_all)}, ewok_deduped={len(ewok_records)}")
    if dup_info:
        dup_str = ", ".join(f"{k}x{v}" for k, v in sorted(dup_info.items()))
        print(f"Dropped duplicate EWOK steps (kept preferred record): {dup_str}")
    else:
        print("No duplicate EWOK steps found.")
    if isinstance(args.max_step, int):
        print(f"Applied EWOK step filter: <= {args.max_step}")
    if args.smooth_window > 1:
        print(f"Applied moving-average smoothing window: {args.smooth_window}")
    if args.no_markers:
        print("Disabled line markers for comparison plots.")
    if args.compare_metrics:
        print(f"Loaded comparison EWOK records: {len(compare_ewok_records)}")
        if compare_dup_info:
            dup_str = ", ".join(f"{k}x{v}" for k, v in sorted(compare_dup_info.items()))
            print(f"Dropped duplicate comparison EWOK steps: {dup_str}")
        else:
            print("No duplicate comparison EWOK steps found.")
    print(
        f"Detected EWOK reductions in metrics: "
        f"{', '.join(available_reductions) if available_reductions else 'none'}"
    )
    if not args.include_ewok_sum_plots and "sum" in available_reductions:
        print("Skipping EWOK sum plots by default. Use --include-ewok-sum-plots to enable them.")
    print(f"Plotted EWOK reductions: {', '.join(reductions) if reductions else 'none'}")
    print(f"Wrote {len(created)} plot(s) to: {output_dir}")
    for p in created:
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
