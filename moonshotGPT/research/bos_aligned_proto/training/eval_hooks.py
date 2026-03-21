"""Evaluation-time helpers and plotting hooks for the research trainer."""

from __future__ import annotations

import gc
import math
import os

import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from research.bos_aligned_proto.evaluation.ewok import (
        BABYLM_COMPLETION_CHOICE,
        EWOK_CONTEXT_SENSITIVITY,
        evaluate,
    )
    from research.bos_aligned_proto.evaluation.ewok_category import (
        plot_ewok_category_subplots,
    )
except ImportError:
    try:
        from ..evaluation.ewok import (
            BABYLM_COMPLETION_CHOICE,
            EWOK_CONTEXT_SENSITIVITY,
            evaluate,
        )
        from ..evaluation.ewok_category import plot_ewok_category_subplots
    except ImportError:
        from evaluation.ewok import (
            BABYLM_COMPLETION_CHOICE,
            EWOK_CONTEXT_SENSITIVITY,
            evaluate,
        )
        from evaluation.ewok_category import plot_ewok_category_subplots


def release_eval_memory() -> None:
    """Best-effort cleanup before heavy eval runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unpack_ewok_per_item(result):
    """
    Backward/forward compatible unpack for ewok_eval.evaluate(return_per_item=True).
    Older API returns 3 values, newer API returns 4 (with margin stats).
    """
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected EWoK return type: {type(result)}")
    if len(result) == 3:
        eval_off, eval_full, per_item = result
        return eval_off, eval_full, per_item, None
    if len(result) == 4:
        eval_off, eval_full, per_item, margin_stats = result
        return eval_off, eval_full, per_item, margin_stats
    raise ValueError(f"Unexpected EWoK return tuple length: {len(result)}")


def evaluate_ewok_all_methods(model, tokenizer, *, batch_size: int, score_reduction: str):
    try:
        result = evaluate(
            model,
            tokenizer,
            batch_size=batch_size,
            return_per_item=True,
            score_reduction=score_reduction,
            return_all_methods=True,
        )
    except TypeError:
        eval_off, eval_full, per_item, margin_stats = unpack_ewok_per_item(
            evaluate(
                model,
                tokenizer,
                batch_size=batch_size,
                return_per_item=True,
                score_reduction=score_reduction,
            )
        )
        return {
            BABYLM_COMPLETION_CHOICE: {
                "domain_scores_official": eval_off,
                "domain_scores_full": eval_full,
                "domain_margin_stats": margin_stats,
            },
            EWOK_CONTEXT_SENSITIVITY: None,
        }, per_item

    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError(
            "Expected evaluate(return_all_methods=True, return_per_item=True) "
            f"to return (metrics_by_method, per_item), got: {type(result)}"
        )

    metrics_by_method, per_item = result
    if not isinstance(metrics_by_method, dict):
        raise TypeError(f"Unexpected metrics_by_method type: {type(metrics_by_method)}")
    if not isinstance(per_item, list):
        raise TypeError(f"Unexpected per_item type: {type(per_item)}")
    if BABYLM_COMPLETION_CHOICE not in metrics_by_method:
        raise KeyError(f"Missing {BABYLM_COMPLETION_CHOICE} in EWoK metrics.")
    return metrics_by_method, per_item


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pair_to_scalar(value):
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and is_finite_number(value[0])
        and is_finite_number(value[1])
    ):
        return 0.5 * (float(value[0]) + float(value[1]))
    if is_finite_number(value):
        return float(value)
    return None


def extract_full_average_scalar(full_payload):
    if not isinstance(full_payload, dict):
        return None

    avg = pair_to_scalar(full_payload.get("average"))
    if avg is not None:
        return avg

    vals = []
    for domain, value in full_payload.items():
        if str(domain) == "average":
            continue
        scalar = pair_to_scalar(value)
        if scalar is not None:
            vals.append(scalar)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def plot_ewok_full_mean_average(step_metrics, out_dir: str) -> None:
    """Plot EWOK full-mean average across optimizer steps from in-memory step_metrics."""
    if plt is None or not step_metrics:
        return

    by_step = {}
    for record in step_metrics:
        step = record.get("step")
        full_mean = record.get("eval_full_mean")
        if not isinstance(step, int) or not isinstance(full_mean, dict):
            continue
        scalar = extract_full_average_scalar(full_mean)
        if scalar is None:
            continue
        by_step[step] = float(scalar)

    if not by_step:
        return

    points = sorted(by_step.items(), key=lambda item: item[0])
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color="#2a6f97", label="full_mean_average")
    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
    ax.set_title("EWOK Full Mean Average Across Steps")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_path = os.path.join(out_dir, "ewok_full_mean_average_by_step.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def extract_margin_average_scalar(margin_payload, metric_key: str):
    if not isinstance(margin_payload, dict):
        return None

    avg = margin_payload.get("average")
    if isinstance(avg, dict) and is_finite_number(avg.get(metric_key)):
        return float(avg[metric_key])

    vals = []
    for domain, stats in margin_payload.items():
        if str(domain) == "average" or not isinstance(stats, dict):
            continue
        if is_finite_number(stats.get(metric_key)):
            vals.append(float(stats[metric_key]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def plot_ewok_margin_domains(step_metrics, out_dir: str, metric_key: str = "eval_margin_stats_mean") -> None:
    if plt is None or not step_metrics:
        return

    ewok_records = [
        record for record in step_metrics
        if isinstance(record, dict) and isinstance(record.get("step"), int) and isinstance(record.get(metric_key), dict)
    ]
    if not ewok_records:
        return

    reduction_suffix = metric_key.replace("eval_margin_stats_", "").strip("_") or "unknown"
    by_domain = {}
    for record in ewok_records:
        margin_payload = record.get(metric_key, {})
        for domain, stats in margin_payload.items():
            if str(domain) == "average" or not isinstance(stats, dict):
                continue
            y_signed = stats.get("mean_signed_m")
            y_abs = stats.get("mean_abs_m")
            if is_finite_number(y_signed) and is_finite_number(y_abs):
                by_domain.setdefault(str(domain), []).append(
                    (record["step"], float(y_signed), float(y_abs))
                )

    if not by_domain:
        return

    domains = sorted(by_domain)
    ncols = 3
    nrows = max(1, int(math.ceil(len(domains) / ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, max(14, nrows * 3.5)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        points = sorted(by_domain[domain], key=lambda item: item[0])
        xs = [x for x, _, _ in points]
        ys_signed = [signed for _, signed, _ in points]
        ys_abs = [abs_value for _, _, abs_value in points]
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
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Margin", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"EWOK Mean Margins by Domain ({reduction_suffix})", fontsize=14)
    out_path = os.path.join(out_dir, f"ewok_margin_{reduction_suffix}_domains_4x3.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def plot_ewok_margin_average_all_domains(
    step_metrics,
    out_dir: str,
    metric_key: str = "eval_margin_stats_mean",
) -> None:
    if plt is None or not step_metrics:
        return

    reduction_suffix = metric_key.replace("eval_margin_stats_", "").strip("_") or "unknown"
    signed = []
    abs_margin = []
    for record in step_metrics:
        step = record.get("step")
        margin_payload = record.get(metric_key)
        if not isinstance(step, int) or not isinstance(margin_payload, dict):
            continue
        y_signed = extract_margin_average_scalar(margin_payload, "mean_signed_m")
        y_abs = extract_margin_average_scalar(margin_payload, "mean_abs_m")
        if y_signed is not None:
            signed.append((step, float(y_signed)))
        if y_abs is not None:
            abs_margin.append((step, float(y_abs)))

    if not signed and not abs_margin:
        return

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
    ax.set_title(f"EWOK Mean Margins Across Domains ({reduction_suffix})")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.25)
    ax.legend()

    reduction_label = (
        r"$s(C,T)=\sum_t \log P_{\theta}(t\mid C)$"
        if reduction_suffix == "sum"
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

    out_path = os.path.join(out_dir, f"ewok_margin_{reduction_suffix}_average_all_domains.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def refresh_ewok_analysis_plots(step_metrics, out_dir: str, include_sum_plots: bool = False) -> None:
    plot_ewok_full_mean_average(step_metrics, out_dir)
    plot_ewok_category_subplots(step_metrics, out_dir, metric_key="eval_by_category_full_mean")
    plot_ewok_margin_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean")
    plot_ewok_margin_average_all_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean")
    if include_sum_plots:
        plot_ewok_margin_domains(step_metrics, out_dir, metric_key="eval_margin_stats_sum")
        plot_ewok_margin_average_all_domains(step_metrics, out_dir, metric_key="eval_margin_stats_sum")
