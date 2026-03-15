"""EWoK category aggregation and plotting helpers for the BOS prototype."""

import math
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _canonicalize_category_value(value) -> str:
    if value is None:
        return "<NA>"
    if isinstance(value, str):
        return value.strip().replace("_", " ")
    try:
        if np.isnan(value):
            return "<NA>"
    except Exception:
        pass
    return str(value)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "unknown"


def _pair_to_scalar(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return 0.5 * (float(value[0]) + float(value[1]))
        except Exception:
            return None
    if isinstance(value, (float, int)):
        return float(value)
    return None


def build_ewok_row_category_lookup(
    ewok_df,
    category_columns=("TargetDiff", "ContextDiff", "ContextType"),
):
    lookup = {}
    if ewok_df is None:
        return lookup
    for row_idx, row in ewok_df.iterrows():
        row_key = int(row_idx)
        lookup[row_key] = {
            col: _canonicalize_category_value(row.get(col))
            for col in category_columns
        }
    return lookup


def aggregate_eval_full_by_category(
    per_item_records,
    row_category_lookup,
    category_columns=("TargetDiff", "ContextDiff", "ContextType"),
):
    counters = {col: {} for col in category_columns}

    for rec in per_item_records:
        row_idx = rec.get("row_index")
        if not isinstance(row_idx, int):
            continue
        row_meta = row_category_lookup.get(row_idx)
        if not isinstance(row_meta, dict):
            continue

        off_ok = 1 if bool(rec.get("correct_official", False)) else 0
        sym_ok = 1 if bool(rec.get("correct_symmetric", False)) else 0

        for col in category_columns:
            cat = row_meta.get(col, "<NA>")
            bucket = counters[col].setdefault(cat, {"off_ok": 0, "sym_ok": 0, "n": 0})
            bucket["off_ok"] += off_ok
            bucket["sym_ok"] += sym_ok
            bucket["n"] += 1

    out = {}
    for col in category_columns:
        col_map = {}
        acc1_vals = []
        acc2_vals = []
        for cat in sorted(counters[col].keys()):
            b = counters[col][cat]
            n = int(b["n"])
            if n <= 0:
                continue
            acc1 = float(b["off_ok"] / n)
            acc2 = float(b["sym_ok"] / n)
            col_map[str(cat)] = (acc1, acc2)
            acc1_vals.append(acc1)
            acc2_vals.append(acc2)
        if acc1_vals:
            col_map["average"] = (float(np.mean(acc1_vals)), float(np.mean(acc2_vals)))
        out[col] = col_map

    return out


def plot_ewok_category_subplots(step_metrics, out_dir, metric_key="eval_by_category_full_mean"):
    """
    One PNG per metadata column, one subplot per category.
    Scalar per point is avg_eval2_acc = 0.5 * (acc1 + acc2).
    """
    if plt is None:
        return
    if not step_metrics:
        return

    ewok_records = [
        r for r in step_metrics
        if isinstance(r, dict) and isinstance(r.get("step"), int) and isinstance(r.get(metric_key), dict)
    ]
    if not ewok_records:
        return

    last_by_col = ewok_records[-1].get(metric_key, {})
    if not isinstance(last_by_col, dict) or not last_by_col:
        return

    reduction_suffix = metric_key.replace("eval_by_category_full_", "").strip("_") or "unknown"

    for column in sorted(last_by_col.keys()):
        col_last = last_by_col.get(column, {})
        if not isinstance(col_last, dict):
            continue

        categories = sorted(k for k in col_last.keys() if str(k) != "average")
        if not categories:
            continue

        avg_epochs = []
        avg_vals = []
        category_series = {}
        y_top = 0.7

        for rec in ewok_records:
            by_col = rec.get(metric_key, {})
            if not isinstance(by_col, dict):
                continue
            c = by_col.get(column, {})
            if not isinstance(c, dict):
                continue

            y_avg = _pair_to_scalar(c.get("average"))
            if y_avg is not None:
                avg_epochs.append(rec["step"])
                avg_vals.append(y_avg)
                y_top = max(y_top, y_avg)

        for category in categories:
            xs, ys = [], []
            for rec in ewok_records:
                by_col = rec.get(metric_key, {})
                if not isinstance(by_col, dict):
                    continue
                c = by_col.get(column, {})
                if not isinstance(c, dict):
                    continue
                y = _pair_to_scalar(c.get(category))
                if y is None:
                    continue
                xs.append(rec["step"])
                ys.append(y)
            if xs:
                category_series[category] = (xs, ys)
                y_top = max(y_top, max(ys))

        if not category_series:
            continue

        categories_sorted = sorted(category_series.keys())
        cols = min(3, max(1, len(categories_sorted)))
        rows = int(math.ceil(len(categories_sorted) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, max(4, rows * 3.8)), squeeze=False)
        axes_flat = axes.flatten()

        for idx, category in enumerate(categories_sorted):
            ax = axes_flat[idx]
            xs, ys = category_series[category]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2a6f97", label=str(category))

            if avg_epochs and avg_vals:
                ax.plot(
                    avg_epochs,
                    avg_vals,
                    marker=None,
                    linewidth=1.0,
                    linestyle="--",
                    color="#808080",
                    alpha=0.28,
                    label="column_average",
                )

            ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
            ax.set_title(str(category), fontsize=10)
            ax.set_xlabel("Optimizer Step", fontsize=9)
            ax.set_ylabel("avg_eval2_acc", fontsize=9)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)

        for idx in range(len(categories_sorted), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"EWOK Category Accuracy by {column} ({reduction_suffix})", fontsize=14)
        slug = _safe_name(str(column).lower())
        out_path = os.path.join(out_dir, f"ewok_category_{slug}_{reduction_suffix}_subplots.png")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path)
        plt.close(fig)


__all__ = [
    "aggregate_eval_full_by_category",
    "build_ewok_row_category_lookup",
    "plot_ewok_category_subplots",
]
