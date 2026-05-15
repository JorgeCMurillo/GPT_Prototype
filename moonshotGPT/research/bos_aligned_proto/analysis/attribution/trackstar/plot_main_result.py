#!/usr/bin/env python3
"""Plot checkpoint-wise treated/control final eval2_acc for CPT ablations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from .cpt_ablation import EWOK_DF
except ImportError:
    from research.bos_aligned_proto.analysis.attribution.trackstar.cpt_ablation import EWOK_DF


ARM_COLORS = {
    "control": "#457b9d",
    "treated": "#d62828",
}
BASELINE_COLOR = "#6c757d"
PAIR_COLOR = "#adb5bd"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot small-multiple checkpoint summaries for final CPT eval2_acc."
    )
    parser.add_argument(
        "--ablation_root",
        required=True,
        help="Root directory containing one subdirectory per checkpoint with ablation outputs.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="PNG path to write. Defaults to <ablation_root>/main_result_<group>.png",
    )
    parser.add_argument(
        "--checkpoint_order",
        default="ckpt_periodic_step0008000,ckpt_periodic_step0012000,ckpt_periodic_step0016000",
        help="Comma-separated checkpoint subdirectory order.",
    )
    parser.add_argument(
        "--group_by",
        choices=("overall", "Domain", "TargetDiff", "ContextDiff", "ContextType"),
        default="overall",
        help="Which EWoK grouping to aggregate.",
    )
    parser.add_argument(
        "--group_name",
        default="overall",
        help="Group label inside --group_by. Use 'overall' when group_by=overall.",
    )
    parser.add_argument(
        "--error_mode",
        choices=("none", "sd", "se", "ci95"),
        default="ci95",
        help="How to draw vertical whiskers across seeds.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _normalize_group_name(group_by: str, group_name: str) -> str:
    if group_by != "ContextDiff":
        return str(group_name)
    normalized = str(group_name).replace("_", " ").strip().lower()
    return " ".join(normalized.split())


def _item_eval2_acc(item: dict[str, Any]) -> float | None:
    value = item.get("eval2_acc")
    if isinstance(value, (int, float)):
        return float(value)
    official = item.get("babylm_completion_choice_correct_official")
    symmetric = item.get("babylm_completion_choice_correct_symmetric")
    if isinstance(official, (int, float)) and isinstance(symmetric, (int, float)):
        return 0.5 * (float(official) + float(symmetric))
    return None


def _item_matches_group(item: dict[str, Any], *, group_by: str, group_name: str) -> bool:
    if group_by == "overall":
        return True
    row_index = item.get("row_index")
    if not isinstance(row_index, int) or row_index < 0 or row_index >= len(EWOK_DF):
        return False
    meta = EWOK_DF.iloc[int(row_index)]
    raw_value = meta[group_by]
    if group_by == "ContextDiff":
        return _normalize_group_name(group_by, raw_value) == _normalize_group_name(group_by, group_name)
    return str(raw_value) == str(group_name)


def _final_items(path: Path) -> list[dict[str, Any]]:
    rows = [
        item
        for item in _load_jsonl(path)
        if str(item.get("type", "")).startswith("ewok_item")
        or str(item.get("type", "")).startswith("baseline_ewok_item")
    ]
    if not rows:
        return []
    if any("step" in item for item in rows):
        max_step = max(int(item.get("step", -1)) for item in rows)
        rows = [item for item in rows if int(item.get("step", -1)) == max_step]
    return rows


def _seed_group_mean(items: Iterable[dict[str, Any]], *, group_by: str, group_name: str) -> float:
    values = [
        value
        for item in items
        if _item_matches_group(item, group_by=group_by, group_name=group_name)
        for value in [_item_eval2_acc(item)]
        if value is not None
    ]
    if not values:
        raise ValueError(f"No eval2_acc items found for group {group_by}:{group_name}")
    return float(mean(values))


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df <= 0:
        return 0.0
    return table[df] if df in table else 1.96


def _error_size(values: list[float], *, mode: str) -> float:
    if mode == "none" or len(values) <= 1:
        return 0.0
    sd = float(stdev(values))
    if mode == "sd":
        return sd
    se = sd / math.sqrt(len(values))
    if mode == "se":
        return se
    return _t_critical_95(len(values) - 1) * se


def _checkpoint_label(name: str) -> str:
    if "step" in name:
        raw = name.split("step", 1)[1].lstrip("0") or "0"
        if raw.endswith("000"):
            return f"{int(raw) // 1000}k"
        return raw
    return name


def _lr_label(value: float) -> str:
    return f"{float(value):.0e}"


def load_plot_frame(root: Path, *, checkpoint_order: list[str], group_by: str, group_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for checkpoint_name in checkpoint_order:
        checkpoint_dir = root / checkpoint_name
        runs_path = checkpoint_dir / "ablation_runs.json"
        baseline_path = checkpoint_dir / "baseline" / "baseline_ewok_items.jsonl"
        if not runs_path.exists():
            continue
        runs = _load_json(runs_path)
        for run in runs:
            items = _final_items(Path(run["ewok_items_path"]))
            score = _seed_group_mean(items, group_by=group_by, group_name=group_name)
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "lr": float(run["learning_rate"]),
                    "arm": str(run["arm"]),
                    "seed": int(run["seed"]),
                    "value": score,
                }
            )
        if baseline_path.exists():
            baseline_items = _final_items(baseline_path)
            baseline_score = _seed_group_mean(baseline_items, group_by=group_by, group_name=group_name)
            baselines.append(
                {
                    "checkpoint": checkpoint_name,
                    "value": baseline_score,
                }
            )
    return rows, baselines


def _format_group_title(group_by: str, group_name: str) -> str:
    if group_by == "overall":
        return "overall"
    return f"{group_by}: {group_name}"


def plot_main_result(
    *,
    rows: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    checkpoint_order: list[str],
    group_by: str,
    group_name: str,
    error_mode: str,
    output_path: Path,
    dpi: int,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to render plots")
    if not rows:
        raise ValueError("No ablation rows were loaded")

    baseline_map = {row["checkpoint"]: float(row["value"]) for row in baselines}
    frame = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lr_values: set[float] = set()
    for row in rows:
        checkpoint = str(row["checkpoint"])
        lr = float(row["lr"])
        arm = str(row["arm"])
        frame[checkpoint][lr][arm].append(float(row["value"]))
        lr_values.add(lr)
    ordered_lrs = sorted(lr_values)

    fig, axes = plt.subplots(1, len(checkpoint_order), figsize=(5.1 * len(checkpoint_order), 4.8), sharey=True)
    if len(checkpoint_order) == 1:
        axes = [axes]

    x_positions = list(range(len(ordered_lrs)))
    pair_offset = 0.13

    for axis, checkpoint_name in zip(axes, checkpoint_order):
        axis.set_title(_checkpoint_label(checkpoint_name))
        baseline_value = baseline_map.get(checkpoint_name)
        if baseline_value is not None:
            axis.axhline(
                baseline_value,
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.4,
                zorder=1,
            )

        checkpoint_frame = frame.get(checkpoint_name, {})
        for idx, lr in enumerate(ordered_lrs):
            x_center = x_positions[idx]
            control_values = checkpoint_frame.get(lr, {}).get("control", [])
            treated_values = checkpoint_frame.get(lr, {}).get("treated", [])
            if not control_values or not treated_values:
                continue

            control_mean = float(mean(control_values))
            treated_mean = float(mean(treated_values))
            control_err = _error_size(control_values, mode=error_mode)
            treated_err = _error_size(treated_values, mode=error_mode)

            x_control = x_center - pair_offset
            x_treated = x_center + pair_offset
            axis.plot(
                [x_control, x_treated],
                [control_mean, treated_mean],
                color=PAIR_COLOR,
                linewidth=1.2,
                zorder=2,
            )
            axis.errorbar(
                [x_control],
                [control_mean],
                yerr=[control_err],
                fmt="o",
                markersize=6.4,
                color=ARM_COLORS["control"],
                ecolor=ARM_COLORS["control"],
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            axis.errorbar(
                [x_treated],
                [treated_mean],
                yerr=[treated_err],
                fmt="o",
                markersize=6.4,
                color=ARM_COLORS["treated"],
                ecolor=ARM_COLORS["treated"],
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels([_lr_label(value) for value in ordered_lrs])
        axis.set_xlabel("Learning rate")
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("eval2_acc")

    handles = [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=1.4, label="base checkpoint"),
        plt.Line2D([0], [0], color=ARM_COLORS["control"], marker="o", linestyle="None", markersize=6.4, label="control"),
        plt.Line2D([0], [0], color=ARM_COLORS["treated"], marker="o", linestyle="None", markersize=6.4, label="treated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    title = f"Main Result Plot: {_format_group_title(group_by, group_name)}"
    fig.suptitle(title, y=1.08, fontsize=13)
    if error_mode != "none":
        note = {
            "sd": "whiskers = +/-1 SD across seeds",
            "se": "whiskers = +/-1 SE across seeds",
            "ci95": "whiskers = 95% CI across seeds",
        }[error_mode]
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color="#495057")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(args.ablation_root).expanduser().resolve()
    checkpoint_order = [part.strip() for part in str(args.checkpoint_order).split(",") if part.strip()]
    group_by = str(args.group_by)
    group_name = str(args.group_name)
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else root / f"main_result_{group_by}_{group_name.replace(' ', '_')}.png"
    )

    rows, baselines = load_plot_frame(
        root,
        checkpoint_order=checkpoint_order,
        group_by=group_by,
        group_name=group_name,
    )
    created = plot_main_result(
        rows=rows,
        baselines=baselines,
        checkpoint_order=checkpoint_order,
        group_by=group_by,
        group_name=group_name,
        error_mode=str(args.error_mode),
        output_path=output_path,
        dpi=int(args.dpi),
    )
    print(f"wrote plot: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
