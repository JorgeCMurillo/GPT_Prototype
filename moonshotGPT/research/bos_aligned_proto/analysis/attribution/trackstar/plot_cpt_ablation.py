#!/usr/bin/env python3
"""Plot paired treated/control TrackStar continued-pretraining ablations."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from ..common.export import write_json
from .cpt_ablation import (
    DEFAULT_GROUP_BY,
    DEFAULT_METRIC_NAME,
    DEFAULT_REDUCTION,
    SUPPORTED_GROUP_BYS,
    SUPPORTED_REDUCTIONS,
    _group_label_order,
)


ARM_COLORS = {
    "treated": "#1d3557",
    "control": "#e76f51",
}
EFFECT_COLOR = "#2a9d8f"
BASELINE_COLOR = "#6c757d"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot treated/control TrackStar CPT ablation curves from ablation_curves.jsonl"
    )
    parser.add_argument("--ablation_dir", required=True, help="Ablation output root containing ablation_curves.jsonl")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write plots. Defaults to <ablation_dir>/plots/<group_by>_<reduction>",
    )
    parser.add_argument("--group_by", choices=SUPPORTED_GROUP_BYS, default=DEFAULT_GROUP_BY)
    parser.add_argument("--reduction", choices=SUPPORTED_REDUCTIONS, default=DEFAULT_REDUCTION)
    parser.add_argument("--metric_name", type=str, default=DEFAULT_METRIC_NAME)
    parser.add_argument("--x_axis", choices=("epoch", "step"), default="epoch")
    parser.add_argument("--dpi", type=int, default=140)
    return parser


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_curves_frame(ablation_dir: str | Path) -> pd.DataFrame:
    path = Path(ablation_dir).expanduser().resolve() / "ablation_curves.jsonl"
    rows = _load_jsonl(path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def _resolve_selection_metadata(ablation_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
    manifest_path = ablation_dir / "ablation_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return None, None

    matched_pool_dir = manifest.get("matched_pool_dir")
    if not matched_pool_dir:
        return None, None
    summary_path = Path(str(matched_pool_dir)).expanduser().resolve() / "summary.json"
    if not summary_path.exists():
        return None, None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None, None

    score_mode = summary.get("score_mode")
    target_id = summary.get("target_id")
    if score_mode:
        label = f"selection={score_mode}"
        if target_id:
            label = f"{label} target={target_id}"
        return label, summary

    selection_source = summary.get("selection_source")
    if isinstance(selection_source, dict) and selection_source:
        source_kind = selection_source.get("kind", "unknown")
        label = f"selection_source={source_kind}"
        if selection_source.get("target_id"):
            label = f"{label} target={selection_source['target_id']}"
        return label, summary
    return None, summary


def _extract_command_arg(command: Sequence[Any], flag: str) -> str | None:
    parts = [str(part) for part in command]
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _resolve_training_annotation(ablation_dir: Path) -> str | None:
    manifest_path = ablation_dir / "ablation_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return None

    defaults = manifest.get("defaults", {}) if isinstance(manifest, dict) else {}
    planned_runs = manifest.get("planned_runs", []) if isinstance(manifest, dict) else []
    first_run = planned_runs[0] if planned_runs else {}
    budget = first_run.get("budget", {}) if isinstance(first_run, dict) else {}
    command = first_run.get("command", []) if isinstance(first_run, dict) else []

    num_rows = budget.get("num_rows")
    micro_batch_size = defaults.get("micro_batch_size", budget.get("micro_batch_size"))
    effective_batch = budget.get("effective_global_batch_seqs")
    steps_per_epoch = budget.get("steps_per_epoch")
    ewok_every = first_run.get("ewok_every")
    if ewok_every is None:
        raw_ewok_every = _extract_command_arg(command, "--ewok_every")
        if raw_ewok_every is not None:
            try:
                ewok_every = int(raw_ewok_every)
            except Exception:
                ewok_every = None

    lines: list[str] = []
    if num_rows is not None:
        lines.append(f"samples/epoch: {int(num_rows):,}")
    if micro_batch_size is not None:
        lines.append(f"micro batch: {int(micro_batch_size)}")
    if effective_batch is not None:
        lines.append(f"effective batch: {int(effective_batch)} seqs")
    if steps_per_epoch is not None:
        lines.append(f"steps/epoch: {int(steps_per_epoch)}")
    if ewok_every is not None:
        ewok_line = f"EWoK every: {int(ewok_every)} steps"
        if steps_per_epoch:
            ewok_line = f"{ewok_line} ({float(ewok_every) / float(steps_per_epoch):.2f} ep)"
        lines.append(ewok_line)
    return "\n".join(lines) if lines else None


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned.strip("_") or "unknown"


def _curve_figure_size(num_panels: int) -> tuple[float, float]:
    cols = min(2, max(1, int(num_panels)))
    rows = int(math.ceil(float(num_panels) / float(cols)))
    return (7.2 * cols, 4.6 * rows)


def _plot_seed_lines(ax, frame: pd.DataFrame, *, x_axis: str, y_column: str, color: str) -> None:
    for _, seed_frame in frame.groupby("seed", as_index=False):
        ordered = seed_frame.sort_values(x_axis)
        ax.plot(
            ordered[x_axis],
            ordered[y_column],
            color=color,
            linewidth=1.0,
            alpha=0.18,
        )


def _plot_mean_and_band(ax, frame: pd.DataFrame, *, x_axis: str, y_column: str, color: str, label: str) -> None:
    grouped = frame.groupby(x_axis)[y_column].agg(y_mean="mean", y_std="std").reset_index()
    ax.plot(grouped[x_axis], grouped["y_mean"], color=color, linewidth=2.3, label=label)
    if len(frame["seed"].drop_duplicates()) > 1:
        y_std = grouped["y_std"].fillna(0.0)
        ax.fill_between(
            grouped[x_axis],
            grouped["y_mean"] - y_std,
            grouped["y_mean"] + y_std,
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )


def _plot_arm_panel(
    ax,
    frame: pd.DataFrame,
    *,
    x_axis: str,
    title: str,
    baseline_value: float | None,
) -> None:
    for arm in ("treated", "control"):
        arm_frame = frame.loc[frame["arm"] == arm].copy()
        if arm_frame.empty:
            continue
        color = ARM_COLORS[arm]
        _plot_seed_lines(ax, arm_frame, x_axis=x_axis, y_column="value", color=color)
        _plot_mean_and_band(ax, arm_frame, x_axis=x_axis, y_column="value", color=color, label=arm)

    if baseline_value is not None:
        ax.axhline(
            float(baseline_value),
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.4,
            label="baseline",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch" if x_axis == "epoch" else "Optimizer Step")
    ax.set_ylabel("Average EWoK Margin")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0.0)
    ax.legend()


def _plot_effect_panel(ax, frame: pd.DataFrame, *, x_axis: str, title: str) -> None:
    pivot = (
        frame.pivot_table(index=["seed", x_axis], columns="arm", values="value", aggfunc="first")
        .reset_index()
    )
    if "treated" not in pivot.columns or "control" not in pivot.columns:
        pivot = pd.DataFrame()
    else:
        pivot = pivot.dropna(subset=["treated", "control"], how="any")
    if pivot.empty:
        ax.set_title(title)
        ax.set_xlabel("Epoch" if x_axis == "epoch" else "Optimizer Step")
        ax.set_ylabel("treated - control")
        ax.grid(True, alpha=0.25)
        return

    pivot["treated_minus_control"] = pivot["treated"] - pivot["control"]
    _plot_seed_lines(ax, pivot, x_axis=x_axis, y_column="treated_minus_control", color=EFFECT_COLOR)
    _plot_mean_and_band(
        ax,
        pivot,
        x_axis=x_axis,
        y_column="treated_minus_control",
        color=EFFECT_COLOR,
        label="treated - control",
    )
    ax.axhline(0.0, color=BASELINE_COLOR, linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Epoch" if x_axis == "epoch" else "Optimizer Step")
    ax.set_ylabel("treated - control")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0.0)
    ax.legend()


def _baseline_value(frame: pd.DataFrame) -> float | None:
    values = [value for value in frame["baseline_value"].dropna().tolist()]
    if not values:
        return None
    return float(values[0])


def _add_explicit_baseline_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an explicit epoch-0 / step-0 point when curves only store a baseline line.

    Existing ablation outputs already carry `baseline_value` on every curve row, but
    older runs did not emit a true pre-training datapoint. Adding it at plot time
    makes the trajectory visibly start before the first epoch finishes, while still
    working with previously generated ablation directories.
    """

    if frame.empty or "baseline_value" not in frame.columns:
        return frame

    key_cols = [
        column
        for column in ("lr", "seed", "arm", "metric_name", "reduction", "group_by", "group_name")
        if column in frame.columns
    ]
    if not key_cols:
        return frame

    candidate_cols = [column for column in (*key_cols, "run_dir", "baseline_value") if column in frame.columns]
    candidates = (
        frame.loc[frame["baseline_value"].notna(), candidate_cols]
        .drop_duplicates(subset=key_cols, keep="first")
        .to_dict("records")
    )
    if not candidates:
        return frame

    existing_zero_keys = {
        tuple(record[column] for column in key_cols)
        for record in frame.loc[frame["step"].eq(0), key_cols].drop_duplicates().to_dict("records")
    }

    baseline_rows: list[dict[str, Any]] = []
    for record in candidates:
        key = tuple(record[column] for column in key_cols)
        if key in existing_zero_keys:
            continue
        baseline_value = record.get("baseline_value")
        if baseline_value is None or pd.isna(baseline_value):
            continue

        row = {column: pd.NA for column in frame.columns}
        for column in key_cols:
            row[column] = record[column]
        if "run_dir" in frame.columns:
            row["run_dir"] = record.get("run_dir")
        row["step"] = 0
        row["epoch"] = 0.0
        row["final"] = False
        row["value"] = float(baseline_value)
        row["baseline_value"] = float(baseline_value)
        if "delta_from_baseline" in frame.columns:
            row["delta_from_baseline"] = 0.0
        baseline_rows.append(row)

    if not baseline_rows:
        return frame

    augmented = pd.concat([frame, pd.DataFrame.from_records(baseline_rows)], ignore_index=True)
    sort_cols = [column for column in ("lr", "seed", "arm", "group_name", "step", "epoch") if column in augmented.columns]
    if sort_cols:
        augmented = augmented.sort_values(sort_cols).reset_index(drop=True)
    return augmented


def _selection_filename_tag(selection_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(selection_metadata, dict):
        return None
    score_mode = selection_metadata.get("score_mode")
    if score_mode:
        tag = f"selection_{score_mode}"
        target_id = selection_metadata.get("target_id")
        if target_id:
            tag = f"{tag}_{target_id}"
        return _safe_name(str(tag))
    selection_source = selection_metadata.get("selection_source")
    if isinstance(selection_source, dict):
        source_kind = selection_source.get("kind")
        if source_kind:
            return _safe_name(f"selection_{source_kind}")
    return None


def _annotate_figure(fig, note: str | None) -> None:
    if not note:
        return
    fig.text(
        0.995,
        0.012,
        str(note),
        ha="right",
        va="bottom",
        fontsize=8.5,
        family="monospace",
        color="#343a40",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#ced4da",
            "alpha": 0.9,
        },
    )


def _average_plot_paths(
    output_dir: Path,
    group_by: str,
    reduction: str,
    *,
    selection_tag: str | None = None,
) -> tuple[Path, Path]:
    base = f"{_safe_name(group_by)}_{_safe_name(reduction)}"
    if selection_tag:
        base = f"{base}_{selection_tag}"
    return (
        output_dir / f"arms_{base}.png",
        output_dir / f"effect_{base}.png",
    )


def _group_plot_paths(
    output_dir: Path,
    group_by: str,
    reduction: str,
    lr: float,
    *,
    selection_tag: str | None = None,
) -> tuple[Path, Path]:
    lr_tag = _safe_name(f"{float(lr):.0e}")
    base = f"{_safe_name(group_by)}_{_safe_name(reduction)}"
    if selection_tag:
        base = f"{base}_{selection_tag}"
    return (
        output_dir / f"arms_{base}_lr_{lr_tag}.png",
        output_dir / f"effect_{base}_lr_{lr_tag}.png",
    )


def generate_ablation_plots(
    *,
    ablation_dir: str | Path,
    output_dir: str | Path | None = None,
    group_by: str = DEFAULT_GROUP_BY,
    reduction: str = DEFAULT_REDUCTION,
    metric_name: str = DEFAULT_METRIC_NAME,
    x_axis: str = "epoch",
    dpi: int = 140,
) -> dict[str, Path]:
    if group_by not in SUPPORTED_GROUP_BYS:
        raise ValueError(f"Unsupported group_by={group_by!r}; expected one of {SUPPORTED_GROUP_BYS}")
    if reduction not in SUPPORTED_REDUCTIONS:
        raise ValueError(f"Unsupported reduction={reduction!r}; expected one of {SUPPORTED_REDUCTIONS}")
    if x_axis not in {"epoch", "step"}:
        raise ValueError("x_axis must be 'epoch' or 'step'")

    root = Path(ablation_dir).expanduser().resolve()
    plot_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "plots" / f"{group_by}_{reduction}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plot_dir / "plot_manifest.json"
    selection_label, selection_metadata = _resolve_selection_metadata(root)
    selection_filename_tag = _selection_filename_tag(selection_metadata)
    training_annotation = _resolve_training_annotation(root)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "ablation_dir": str(root),
        "output_dir": str(plot_dir),
        "group_by": str(group_by),
        "reduction": str(reduction),
        "metric_name": str(metric_name),
        "x_axis": str(x_axis),
        "selection_label": selection_label,
        "selection_metadata": selection_metadata,
        "selection_filename_tag": selection_filename_tag,
        "training_annotation": training_annotation,
        "matplotlib_available": bool(plt is not None),
        "plots": [],
    }
    if plt is None:
        write_json(manifest_path, manifest)
        return {"manifest_path": manifest_path}

    frame = _load_curves_frame(root)
    if frame.empty:
        write_json(manifest_path, manifest)
        return {"manifest_path": manifest_path}

    frame = frame.loc[
        (frame["metric_name"] == metric_name)
        & (frame["group_by"] == group_by)
        & (frame["reduction"] == reduction)
    ].copy()
    if frame.empty:
        write_json(manifest_path, manifest)
        return {"manifest_path": manifest_path}

    frame["lr"] = frame["lr"].astype(float)
    frame["seed"] = frame["seed"].astype(int)
    frame["step"] = frame["step"].astype(int)
    frame["epoch"] = frame["epoch"].astype(float)
    frame["value"] = frame["value"].astype(float)
    frame["group_name"] = frame["group_name"].astype(str)
    frame = _add_explicit_baseline_points(frame)
    lr_values = sorted(frame["lr"].drop_duplicates().tolist())

    if group_by == "average":
        fig_arms, axes_arms = plt.subplots(
            max(1, len(lr_values)),
            1,
            figsize=(8.4, 4.2 * max(1, len(lr_values))),
            squeeze=False,
        )
        fig_effect, axes_effect = plt.subplots(
            max(1, len(lr_values)),
            1,
            figsize=(8.4, 4.2 * max(1, len(lr_values))),
            squeeze=False,
        )
        for row_index, lr in enumerate(lr_values):
            lr_frame = frame.loc[(frame["lr"] == lr) & (frame["group_name"] == "average")].copy()
            _plot_arm_panel(
                axes_arms[row_index, 0],
                lr_frame,
                x_axis=x_axis,
                title=f"lr={float(lr):.0e}",
                baseline_value=_baseline_value(lr_frame),
            )
            _plot_effect_panel(
                axes_effect[row_index, 0],
                lr_frame,
                x_axis=x_axis,
                title=f"lr={float(lr):.0e}",
            )
        arms_title = "average arm curves"
        effect_title = "average treated - control"
        if selection_label:
            arms_title = f"{arms_title}\n{selection_label}"
            effect_title = f"{effect_title}\n{selection_label}"
        fig_arms.suptitle(arms_title, fontsize=14)
        fig_effect.suptitle(effect_title, fontsize=14)
        _annotate_figure(fig_arms, training_annotation)
        _annotate_figure(fig_effect, training_annotation)
        fig_arms.tight_layout(rect=[0, 0.04, 1, 0.95])
        fig_effect.tight_layout(rect=[0, 0.04, 1, 0.95])
        arms_path, effect_path = _average_plot_paths(
            plot_dir,
            group_by,
            reduction,
            selection_tag=selection_filename_tag,
        )
        fig_arms.savefig(arms_path, dpi=dpi)
        fig_effect.savefig(effect_path, dpi=dpi)
        plt.close(fig_arms)
        plt.close(fig_effect)
        manifest["plots"].append({"kind": "arms", "group_by": group_by, "reduction": reduction, "path": str(arms_path)})
        manifest["plots"].append(
            {"kind": "effect", "group_by": group_by, "reduction": reduction, "path": str(effect_path)}
        )
    else:
        group_names = _group_label_order(group_by, frame["group_name"].drop_duplicates().tolist())
        for lr in lr_values:
            lr_frame = frame.loc[frame["lr"] == lr].copy()
            ordered_groups = [name for name in group_names if name in set(lr_frame["group_name"].tolist())]
            if not ordered_groups:
                continue
            cols = min(3, max(1, len(ordered_groups)))
            rows = int(math.ceil(len(ordered_groups) / cols))
            fig_arms, axes_arms = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.4 * rows), squeeze=False)
            fig_effect, axes_effect = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.4 * rows), squeeze=False)
            axes_arms_flat = list(axes_arms.flatten())
            axes_effect_flat = list(axes_effect.flatten())
            for index, group_name in enumerate(ordered_groups):
                group_frame = lr_frame.loc[lr_frame["group_name"] == group_name].copy()
                _plot_arm_panel(
                    axes_arms_flat[index],
                    group_frame,
                    x_axis=x_axis,
                    title=str(group_name),
                    baseline_value=_baseline_value(group_frame),
                )
                _plot_effect_panel(
                    axes_effect_flat[index],
                    group_frame,
                    x_axis=x_axis,
                    title=str(group_name),
                )
            for axis in axes_arms_flat[len(ordered_groups):]:
                axis.axis("off")
            for axis in axes_effect_flat[len(ordered_groups):]:
                axis.axis("off")
            arms_title = f"{group_by} arm curves @ lr={float(lr):.0e}"
            effect_title = f"{group_by} treated - control @ lr={float(lr):.0e}"
            if selection_label:
                arms_title = f"{arms_title}\n{selection_label}"
                effect_title = f"{effect_title}\n{selection_label}"
            fig_arms.suptitle(arms_title, fontsize=14)
            fig_effect.suptitle(effect_title, fontsize=14)
            _annotate_figure(fig_arms, training_annotation)
            _annotate_figure(fig_effect, training_annotation)
            fig_arms.tight_layout(rect=[0, 0.04, 1, 0.94])
            fig_effect.tight_layout(rect=[0, 0.04, 1, 0.94])
            arms_path, effect_path = _group_plot_paths(
                plot_dir,
                group_by,
                reduction,
                float(lr),
                selection_tag=selection_filename_tag,
            )
            fig_arms.savefig(arms_path, dpi=dpi)
            fig_effect.savefig(effect_path, dpi=dpi)
            plt.close(fig_arms)
            plt.close(fig_effect)
            manifest["plots"].append(
                {
                    "kind": "arms",
                    "group_by": group_by,
                    "reduction": reduction,
                    "lr": float(lr),
                    "path": str(arms_path),
                }
            )
            manifest["plots"].append(
                {
                    "kind": "effect",
                    "group_by": group_by,
                    "reduction": reduction,
                    "lr": float(lr),
                    "path": str(effect_path),
                }
            )

    write_json(manifest_path, manifest)
    return {"manifest_path": manifest_path, "output_dir": plot_dir}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = generate_ablation_plots(
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
        group_by=args.group_by,
        reduction=args.reduction,
        metric_name=args.metric_name,
        x_axis=args.x_axis,
        dpi=int(args.dpi),
    )
    print(f"plot manifest: {outputs['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
