#!/usr/bin/env python3
"""Plot selector-wise checkpoint CPT ablation summaries.

This is a selector-aware sibling of ``plot_main_result.py``.  It expects an
ablation root laid out as:

    <root>/<checkpoint>/<selector>/ablation_runs.json
    <root>/<checkpoint>/<selector>/baseline/baseline_ewok_items.jsonl

For a fixed learning rate, it draws one panel per checkpoint with selectors on
the x-axis, paired control/treated points, and a dashed base-checkpoint line.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from .cpt_ablation import EWOK_DF
except ImportError:
    from research.bos_aligned_proto.analysis.attribution.trackstar.cpt_ablation import EWOK_DF


ARM_COLORS = {
    "control": "#e76f51",
    "treated": "#457b9d",
}
BASELINE_COLOR = "#6c757d"
PAIR_COLOR = "#adb5bd"
DEFAULT_CHECKPOINT_ORDER = (
    "ckpt_periodic_step0008000",
    "ckpt_periodic_step0012000",
    "ckpt_periodic_step0016000",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot selector-wise checkpoint summaries for final corrected EWoK accuracy."
    )
    parser.add_argument(
        "--ablation_root",
        required=True,
        help="Root containing checkpoint/selector ablation directories.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write plots. Defaults to <ablation_root>/selector_checkpoint_plots.",
    )
    parser.add_argument(
        "--checkpoint_order",
        default=",".join(DEFAULT_CHECKPOINT_ORDER),
        help="Comma-separated checkpoint directory order.",
    )
    parser.add_argument(
        "--selectors",
        default=None,
        help="Comma-separated selectors. Defaults to all selector subdirectories found under the first checkpoint.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate to plot.",
    )
    parser.add_argument(
        "--group_by",
        choices=("overall", "Domain", "TargetDiff", "ContextDiff", "ContextType"),
        default="overall",
    )
    parser.add_argument(
        "--group_name",
        default="overall",
        help="Group label to plot. Ignored when --all_groups is used.",
    )
    parser.add_argument(
        "--all_groups",
        action="store_true",
        help="Write one plot for every group in --group_by.",
    )
    parser.add_argument(
        "--error_mode",
        choices=("none", "sd", "se", "ci95"),
        default="ci95",
        help="How to draw vertical whiskers across seeds.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned.strip("_") or "unknown"


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _normalize_group_name(group_by: str, group_name: str) -> str:
    if group_by != "ContextDiff":
        return str(group_name)
    normalized = str(group_name).replace("_", " ").strip().lower()
    return " ".join(normalized.split())


def _item_value(item: dict[str, Any]) -> float | None:
    """Return corrected combined completion-choice accuracy for an EWoK item."""
    value = item.get("eval2_acc")
    if isinstance(value, (int, float)):
        return float(value)
    official = item.get("babylm_completion_choice_correct_official")
    symmetric = item.get("babylm_completion_choice_correct_symmetric")
    if isinstance(official, (int, float, bool)) and isinstance(symmetric, (int, float, bool)):
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


@lru_cache(maxsize=None)
def _final_items(path: Path) -> tuple[dict[str, Any], ...]:
    rows = [
        item
        for item in _load_jsonl(path)
        if str(item.get("type", "")).startswith("ewok_item")
        or str(item.get("type", "")).startswith("baseline_ewok_item")
    ]
    if not rows:
        return tuple()
    if any("step" in item for item in rows):
        max_step = max(int(item.get("step", -1)) for item in rows)
        rows = [item for item in rows if int(item.get("step", -1)) == max_step]
    return tuple(rows)


@lru_cache(maxsize=None)
def _final_step_metrics(path: Path) -> dict[str, Any]:
    records = _load_json(path)
    if isinstance(records, dict):
        return records
    if not isinstance(records, list) or not records:
        return {}
    candidates = [record for record in records if isinstance(record, dict) and isinstance(record.get("step"), int)]
    if not candidates:
        return {}
    return max(candidates, key=lambda record: int(record["step"]))


def _mean_pair(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return 0.5 * (float(value[0]) + float(value[1]))
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _lookup_normalized(mapping: dict[str, Any], *, group_by: str, group_name: str) -> Any:
    if group_name in mapping:
        return mapping[group_name]
    if group_by == "ContextDiff":
        wanted = _normalize_group_name(group_by, group_name)
        for key, value in mapping.items():
            if _normalize_group_name(group_by, str(key)) == wanted:
                return value
    return None


def _step_metric_score(path: Path, *, group_by: str, group_name: str) -> float | None:
    record = _final_step_metrics(path)
    if not record:
        return None
    if group_by == "overall":
        full = record.get("eval_babylm_completion_choice_full_mean") or record.get("eval_full_mean")
        if isinstance(full, dict):
            return _mean_pair(full.get("average"))
        return None
    if group_by == "Domain":
        full = record.get("eval_babylm_completion_choice_full_mean") or record.get("eval_full_mean")
        if isinstance(full, dict):
            return _mean_pair(full.get(group_name))
        return None
    by_category = record.get("eval_babylm_completion_choice_by_category_full_mean") or record.get("eval_by_category_full_mean")
    if not isinstance(by_category, dict):
        return None
    group_values = by_category.get(group_by)
    if not isinstance(group_values, dict):
        return None
    return _mean_pair(_lookup_normalized(group_values, group_by=group_by, group_name=group_name))


def _group_mean(items: Iterable[dict[str, Any]], *, group_by: str, group_name: str) -> float:
    values = [
        value
        for item in items
        if _item_matches_group(item, group_by=group_by, group_name=group_name)
        for value in [_item_value(item)]
        if value is not None
    ]
    if not values:
        raise ValueError(f"No corrected accuracy items found for {group_by}:{group_name}")
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


def _lr_dir_name(learning_rate: float) -> str:
    text = f"{float(learning_rate):.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _discover_selectors(root: Path, checkpoint_order: Sequence[str]) -> list[str]:
    for checkpoint in checkpoint_order:
        checkpoint_dir = root / checkpoint
        if checkpoint_dir.is_dir():
            selectors = [
                child.name
                for child in checkpoint_dir.iterdir()
                if child.is_dir() and (child / "ablation_runs.json").exists()
            ]
            if selectors:
                return sorted(selectors)
    return []


def _available_group_names(group_by: str) -> list[str]:
    if group_by == "overall":
        return ["overall"]
    values = [str(value) for value in EWOK_DF[group_by].dropna().tolist()]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = _normalize_group_name(group_by, value) if group_by == "ContextDiff" else value
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def load_plot_frame(
    root: Path,
    *,
    checkpoint_order: Sequence[str],
    selectors: Sequence[str],
    learning_rate: float,
    group_by: str,
    group_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    lr_target = float(learning_rate)
    for checkpoint_name in checkpoint_order:
        checkpoint_baselines: list[float] = []
        for selector in selectors:
            selector_dir = root / checkpoint_name / selector
            runs_path = selector_dir / "ablation_runs.json"
            baseline_path = selector_dir / "baseline" / "baseline_ewok_items.jsonl"
            if not runs_path.exists():
                continue
            runs = _load_json(runs_path)
            for run in runs:
                run_lr = float(run["learning_rate"])
                if not math.isclose(run_lr, lr_target, rel_tol=1e-9, abs_tol=1e-12):
                    continue
                score = None
                step_metrics_path = run.get("step_metrics_path")
                if step_metrics_path:
                    score = _step_metric_score(
                        Path(step_metrics_path),
                        group_by=group_by,
                        group_name=group_name,
                    )
                if score is None:
                    items = _final_items(Path(run["ewok_items_path"]))
                    score = _group_mean(items, group_by=group_by, group_name=group_name)
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "selector": selector,
                        "arm": str(run["arm"]),
                        "seed": int(run["seed"]),
                        "value": score,
                    }
                )
            if baseline_path.exists():
                baseline_items = _final_items(baseline_path)
                baseline_score = _group_mean(baseline_items, group_by=group_by, group_name=group_name)
                checkpoint_baselines.append(baseline_score)
        if checkpoint_baselines:
            baselines.append(
                {
                    "checkpoint": checkpoint_name,
                    "value": float(mean(checkpoint_baselines)),
                }
            )
    return rows, baselines


def _format_group_title(group_by: str, group_name: str) -> str:
    if group_by == "overall":
        return "overall"
    return f"{group_by}: {group_name}"


def plot_selector_checkpoint_result(
    *,
    rows: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    checkpoint_order: Sequence[str],
    selectors: Sequence[str],
    learning_rate: float,
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
    frame: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        frame[str(row["checkpoint"])][str(row["selector"])][str(row["arm"])].append(float(row["value"]))

    fig_width = max(5.2 * len(checkpoint_order), 12.0)
    fig, axes = plt.subplots(1, len(checkpoint_order), figsize=(fig_width, 4.9), sharey=True)
    if len(checkpoint_order) == 1:
        axes = [axes]

    x_positions = list(range(len(selectors)))
    pair_offset = 0.14

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
        for idx, selector in enumerate(selectors):
            control_values = checkpoint_frame.get(selector, {}).get("control", [])
            treated_values = checkpoint_frame.get(selector, {}).get("treated", [])
            if not control_values or not treated_values:
                continue

            control_mean = float(mean(control_values))
            treated_mean = float(mean(treated_values))
            control_err = _error_size(control_values, mode=error_mode)
            treated_err = _error_size(treated_values, mode=error_mode)

            x_control = x_positions[idx] - pair_offset
            x_treated = x_positions[idx] + pair_offset
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
                markersize=6.0,
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
                markersize=6.0,
                color=ARM_COLORS["treated"],
                ecolor=ARM_COLORS["treated"],
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels([selector.replace("_", "\n") for selector in selectors], fontsize=8)
        axis.set_xlabel("Selector")
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("corrected EWoK acc")

    handles = [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=1.4, label="base checkpoint"),
        plt.Line2D([0], [0], color=ARM_COLORS["control"], marker="o", linestyle="None", markersize=6.4, label="control"),
        plt.Line2D([0], [0], color=ARM_COLORS["treated"], marker="o", linestyle="None", markersize=6.4, label="treated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    title = f"Selector Result Plot: {_format_group_title(group_by, group_name)}"
    fig.suptitle(f"{title} | lr={learning_rate:.0e}", y=1.08, fontsize=13)
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
    checkpoint_order = _split_csv(args.checkpoint_order)
    selectors = _split_csv(args.selectors) or _discover_selectors(root, checkpoint_order)
    if not selectors:
        raise ValueError("No selectors found. Pass --selectors or check --ablation_root.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "selector_checkpoint_plots"
    )

    group_names = _available_group_names(str(args.group_by)) if args.all_groups else [str(args.group_name)]
    created: list[Path] = []
    for group_name in group_names:
        rows, baselines = load_plot_frame(
            root,
            checkpoint_order=checkpoint_order,
            selectors=selectors,
            learning_rate=float(args.learning_rate),
            group_by=str(args.group_by),
            group_name=group_name,
        )
        output_path = output_dir / (
            f"selector_checkpoint_{args.group_by}_{_safe_name(group_name)}_"
            f"lr{_lr_dir_name(float(args.learning_rate))}_corrected_acc.png"
        )
        created.append(
            plot_selector_checkpoint_result(
                rows=rows,
                baselines=baselines,
                checkpoint_order=checkpoint_order,
                selectors=selectors,
                learning_rate=float(args.learning_rate),
                group_by=str(args.group_by),
                group_name=group_name,
                error_mode=str(args.error_mode),
                output_path=output_path,
                dpi=int(args.dpi),
            )
        )

    for path in created:
        print(f"wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
