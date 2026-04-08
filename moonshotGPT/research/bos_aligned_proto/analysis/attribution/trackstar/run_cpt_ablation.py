#!/usr/bin/env python3
"""Run paired TrackStar continued-pretraining ablations from matched pools."""

from __future__ import annotations

import argparse
import shlex
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..common.export import write_json
from .cpt_ablation import (
    DEFAULT_EWOK_BATCH_SIZE,
    DEFAULT_GROUP_BY,
    DEFAULT_LRS,
    DEFAULT_METRIC_NAME,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_PROCESSES,
    DEFAULT_REDUCTION,
    DEFAULT_SEEDS,
    DEFAULT_TOTAL_BATCH_TOKENS,
    SUPPORTED_GROUP_BYS,
    SUPPORTED_REDUCTIONS,
    build_ablation_run_specs,
    evaluate_checkpoint_baseline,
    launch_training_run,
    parse_float_list,
    parse_int_list,
    run_ablation_aggregation,
    spec_to_record,
)
from .plot_cpt_ablation import generate_ablation_plots


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str, leave: bool = True):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired TrackStar continued-pretraining ablation from one checkpoint and one matched-pool root."
        )
    )
    parser.add_argument("--base_ckpt", required=True, help="Checkpoint directory used to initialize both ablation arms")
    parser.add_argument(
        "--matched_pool_dir",
        required=True,
        help="Matched-pool root containing treated_dataset/ and control_dataset/",
    )
    parser.add_argument("--output_dir", required=True, help="Top-level ablation output directory")
    parser.add_argument(
        "--learning_rates",
        default=",".join(f"{value:.12g}" for value in DEFAULT_LRS),
        help="Comma-separated LR sweep, for example 1e-5,2e-5,4e-5,8e-5",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
        help="Comma-separated seed list",
    )
    parser.add_argument("--micro_batch_size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--total_batch_tokens", type=int, default=DEFAULT_TOTAL_BATCH_TOKENS)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--num_processes", type=int, default=DEFAULT_NUM_PROCESSES)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ewok_batch_size", type=int, default=DEFAULT_EWOK_BATCH_SIZE)
    parser.add_argument(
        "--save_final_checkpoint",
        action="store_true",
        help="Persist a final child-run checkpoint/model directory. Disabled by default to save disk space.",
    )
    parser.add_argument(
        "--plot_group_by",
        choices=SUPPORTED_GROUP_BYS,
        default=DEFAULT_GROUP_BY,
        help="Primary grouping view to auto-plot after aggregation",
    )
    parser.add_argument(
        "--plot_reduction",
        choices=SUPPORTED_REDUCTIONS,
        default=DEFAULT_REDUCTION,
        help="EWoK score reduction to auto-plot after aggregation",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default=DEFAULT_METRIC_NAME,
        help="Per-item EWoK margin field to aggregate and plot",
    )
    parser.add_argument(
        "--plot_x_axis",
        choices=("epoch", "step"),
        default="epoch",
        help="Horizontal axis for auto-generated plots",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=None,
        help="Override ablation warmup. By default the runner uses 5% of ablation steps.",
    )
    parser.add_argument(
        "--allow_unequal_budgets",
        action="store_false",
        dest="enforce_equal_budgets",
        help="Allow treated/control arms to proceed even if their row or step budgets differ. Strict equal budgets are enforced by default.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_false",
        dest="show_progress",
        help="Disable the outer ablation tqdm progress bar.",
    )
    parser.add_argument(
        "--baseline_device",
        type=str,
        default=None,
        help="Optional device override for step-0 baseline EWoK evaluation",
    )
    parser.add_argument("--dpi", type=int, default=140, help="PNG DPI for auto-generated plots")
    parser.add_argument("--dry_run", action="store_true", help="Plan the ablation runs without launching them")
    return parser


def _render_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _auto_plot_group_bys(primary_group_by: str) -> tuple[str, ...]:
    ordered: list[str] = []
    for group_by in ("average", "domain", "ContextDiff", "TargetDiff", "ContextType", str(primary_group_by)):
        if group_by not in ordered:
            ordered.append(group_by)
    return tuple(ordered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    learning_rates = parse_float_list(args.learning_rates, default=DEFAULT_LRS)
    seeds = parse_int_list(args.seeds, default=DEFAULT_SEEDS)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_manifest.json"

    training_views, specs = build_ablation_run_specs(
        base_ckpt=args.base_ckpt,
        matched_pool_dir=args.matched_pool_dir,
        output_dir=output_dir,
        learning_rates=learning_rates,
        seeds=seeds,
        micro_batch_size=int(args.micro_batch_size),
        total_batch_tokens=int(args.total_batch_tokens),
        num_epochs=int(args.num_epochs),
        num_processes=int(args.num_processes),
        ewok_batch_size=int(args.ewok_batch_size),
        num_workers=int(args.num_workers),
        warmup_iters=args.warmup_iters,
        enforce_equal_budgets=bool(args.enforce_equal_budgets),
        save_final_checkpoint=bool(args.save_final_checkpoint),
    )

    planned_runs = [spec_to_record(spec) for spec in specs]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "base_ckpt": str(Path(args.base_ckpt).expanduser().resolve()),
        "matched_pool_dir": str(Path(args.matched_pool_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "metric_name": str(args.metric_name),
        "learning_rates": [float(value) for value in learning_rates],
        "seeds": [int(value) for value in seeds],
        "defaults": {
            "micro_batch_size": int(args.micro_batch_size),
            "total_batch_tokens": int(args.total_batch_tokens),
            "num_epochs": int(args.num_epochs),
            "num_processes": int(args.num_processes),
            "ewok_batch_size": int(args.ewok_batch_size),
            "save_final_checkpoint": bool(args.save_final_checkpoint),
            "plot_group_by": str(args.plot_group_by),
            "plot_reduction": str(args.plot_reduction),
            "plot_x_axis": str(args.plot_x_axis),
            "warmup_iters": (None if args.warmup_iters is None else int(args.warmup_iters)),
            "enforce_equal_budgets": bool(args.enforce_equal_budgets),
        },
        "training_views": {arm: str(path) for arm, path in training_views.items()},
        "planned_runs": planned_runs,
        "baseline": None,
        "run_records": [],
        "aggregation_outputs": None,
        "plot_outputs": None,
    }
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(f"planned {len(specs)} ablation run(s)")
        for spec in specs:
            print(f"[{spec.arm}] lr={spec.learning_rate:.12g} seed={spec.seed}: {_render_command(spec.command)}")
        print(f"manifest: {manifest_path}")
        return 0

    progress = _build_tqdm(
        enabled=bool(args.show_progress),
        total=len(specs) + 3,
        desc="TrackStar CPT ablation",
        unit="stage",
        leave=True,
    )
    try:
        if progress is not None:
            progress.set_postfix_str("baseline")
        print("running baseline EWoK evaluation for the base checkpoint")
        baseline_artifacts = evaluate_checkpoint_baseline(
            checkpoint_dir=args.base_ckpt,
            output_dir=output_dir / "baseline",
            ewok_batch_size=int(args.ewok_batch_size),
            metric_name=str(args.metric_name),
            device=args.baseline_device,
        )
        manifest["baseline"] = {key: str(value) for key, value in baseline_artifacts.items()}
        write_json(manifest_path, manifest)
        if progress is not None:
            progress.update(1)

        run_records = []
        for index, spec in enumerate(specs, start=1):
            if progress is not None:
                progress.set_postfix_str(
                    f"run {index}/{len(specs)} {spec.arm} lr={spec.learning_rate:.12g} seed={spec.seed}"
                )
            print(
                f"[{index}/{len(specs)}] arm={spec.arm} lr={spec.learning_rate:.12g} "
                f"seed={spec.seed} max_steps={spec.budget.max_train_steps}"
            )
            run_record = launch_training_run(spec, dry_run=False)
            run_records.append(run_record)
            manifest["run_records"] = run_records
            write_json(manifest_path, manifest)
            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.set_postfix_str("aggregation")
        aggregation_outputs = run_ablation_aggregation(
            output_dir=output_dir,
            run_records=run_records,
            baseline_summary_path=baseline_artifacts["summary_path"],
            metric_name=str(args.metric_name),
        )
        manifest["aggregation_outputs"] = {key: str(value) for key, value in aggregation_outputs.items()}
        write_json(manifest_path, manifest)
        if progress is not None:
            progress.update(1)

        if progress is not None:
            progress.set_postfix_str("plotting")
        plot_outputs: dict[str, dict[str, str]] = {}
        for group_by in _auto_plot_group_bys(str(args.plot_group_by)):
            group_output_dir = (
                output_dir / "plots"
                if group_by == str(args.plot_group_by)
                else output_dir / "plots" / f"{group_by}_{str(args.plot_reduction)}"
            )
            outputs = generate_ablation_plots(
                ablation_dir=output_dir,
                output_dir=group_output_dir,
                group_by=group_by,
                reduction=str(args.plot_reduction),
                metric_name=str(args.metric_name),
                x_axis=str(args.plot_x_axis),
                dpi=int(args.dpi),
            )
            plot_outputs[group_by] = {key: str(value) for key, value in outputs.items()}
        manifest["plot_outputs"] = plot_outputs
        write_json(manifest_path, manifest)
        if progress is not None:
            progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    print(f"ablation manifest: {manifest_path}")
    print(f"aggregation summary: {aggregation_outputs['summary_path']}")
    for group_by, outputs in plot_outputs.items():
        if "manifest_path" in outputs:
            print(f"{group_by} plot manifest: {outputs['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
