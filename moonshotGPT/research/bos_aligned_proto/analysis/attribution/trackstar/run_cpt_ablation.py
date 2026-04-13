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
    DEFAULT_EWOK_FRAC_PER_EPOCH,
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
    parse_positive_fraction,
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


def _is_matched_pool_condition_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "treated_dataset").is_dir()
        and (path / "control_dataset").is_dir()
    )


def _discover_matched_pool_conditions(root: Path) -> list[tuple[str, Path]]:
    if _is_matched_pool_condition_dir(root):
        return [(root.name or "matched_pool", root)]
    discovered: list[tuple[str, Path]] = []
    for child in sorted((entry for entry in root.iterdir() if entry.is_dir()), key=lambda entry: entry.name):
        if _is_matched_pool_condition_dir(child):
            discovered.append((child.name, child))
    return discovered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired TrackStar continued-pretraining ablation from one or more checkpoints and one matched-pool root."
        )
    )
    parser.add_argument(
        "--base_ckpt",
        required=False,
        help="Checkpoint directory used to initialize both ablation arms (use --base_ckpts for multiple).",
    )
    parser.add_argument(
        "--base_ckpts",
        type=str,
        default=None,
        help=(
            "Comma-separated list of checkpoint directories to sweep in a single invocation. "
            "If provided, each checkpoint gets its own subdirectory under --output_dir."
        ),
    )
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
        "--ewok_frac_per_epoch",
        type=str,
        default="1/2",
        help=(
            "How often to run EWoK within each epoch, expressed as a positive fraction of an epoch. "
            "Examples: 1/2, 0.5, 1.0. Defaults to half an epoch."
        ),
    )
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


def _run_single_ablation(
    *,
    args: argparse.Namespace,
    matched_pool_dir: Path,
    output_dir: Path,
    base_ckpt: Path,
    condition_name: str | None = None,
) -> dict[str, object]:
    learning_rates = parse_float_list(args.learning_rates, default=DEFAULT_LRS)
    seeds = parse_int_list(args.seeds, default=DEFAULT_SEEDS)
    ewok_frac_per_epoch = parse_positive_fraction(
        args.ewok_frac_per_epoch,
        default=DEFAULT_EWOK_FRAC_PER_EPOCH,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_manifest.json"

    training_views, specs = build_ablation_run_specs(
        base_ckpt=str(base_ckpt),
        matched_pool_dir=str(matched_pool_dir),
        output_dir=output_dir,
        learning_rates=learning_rates,
        seeds=seeds,
        micro_batch_size=int(args.micro_batch_size),
        total_batch_tokens=int(args.total_batch_tokens),
        num_epochs=int(args.num_epochs),
        num_processes=int(args.num_processes),
        ewok_batch_size=int(args.ewok_batch_size),
        ewok_frac_per_epoch=float(ewok_frac_per_epoch),
        num_workers=int(args.num_workers),
        warmup_iters=args.warmup_iters,
        enforce_equal_budgets=bool(args.enforce_equal_budgets),
        save_final_checkpoint=bool(args.save_final_checkpoint),
    )

    planned_runs = [spec_to_record(spec) for spec in specs]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "condition_name": condition_name,
        "base_ckpt": str(base_ckpt),
        "matched_pool_dir": str(matched_pool_dir),
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
            "ewok_frac_per_epoch": float(ewok_frac_per_epoch),
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
        label = condition_name or matched_pool_dir.name or "matched_pool"
        print(f"planned {len(specs)} ablation run(s) for condition {label}")
        for spec in specs:
            print(f"[{spec.arm}] lr={spec.learning_rate:.12g} seed={spec.seed}: {_render_command(spec.command)}")
        print(f"manifest: {manifest_path}")
        return {
            "condition_name": label,
            "matched_pool_dir": str(matched_pool_dir),
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "summary_path": None,
            "plot_manifest_paths": {},
            "dry_run": True,
        }

    progress = _build_tqdm(
        enabled=bool(args.show_progress),
        total=len(specs) + 3,
        desc=("TrackStar CPT ablation" if not condition_name else f"TrackStar CPT {condition_name}"),
        unit="stage",
        leave=True,
    )
    aggregation_outputs: dict[str, Path] | None = None
    plot_outputs: dict[str, dict[str, str]] = {}
    try:
        if progress is not None:
            progress.set_postfix_str("baseline")
        print("running baseline EWoK evaluation for the base checkpoint")
        baseline_artifacts = evaluate_checkpoint_baseline(
            checkpoint_dir=str(base_ckpt),
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

    if aggregation_outputs is None:
        raise RuntimeError("Ablation aggregation did not complete")

    print(f"ablation manifest: {manifest_path}")
    print(f"aggregation summary: {aggregation_outputs['summary_path']}")
    for group_by, outputs in plot_outputs.items():
        if "manifest_path" in outputs:
            print(f"{group_by} plot manifest: {outputs['manifest_path']}")
    return {
        "condition_name": condition_name or matched_pool_dir.name or "matched_pool",
        "matched_pool_dir": str(matched_pool_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(aggregation_outputs["summary_path"]),
        "plot_manifest_paths": {
            group_by: outputs["manifest_path"]
            for group_by, outputs in plot_outputs.items()
            if "manifest_path" in outputs
        },
        "dry_run": False,
    }


def _parse_base_ckpts(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    ckpts: list[str] = []
    if args.base_ckpts:
        ckpts.extend([entry.strip() for entry in args.base_ckpts.split(",") if entry.strip()])
    if args.base_ckpt:
        ckpts.append(str(args.base_ckpt))
    if not ckpts:
        parser.error("Provide --base_ckpt or --base_ckpts (comma-separated).")
    return [Path(entry).expanduser().resolve() for entry in ckpts]


def _label_ckpts(ckpts: list[Path]) -> dict[Path, str]:
    labels: dict[Path, str] = {}
    seen: dict[str, int] = {}
    for ckpt in ckpts:
        base = ckpt.name or "checkpoint"
        count = seen.get(base, 0)
        label = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        labels[ckpt] = label
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base_ckpts = _parse_base_ckpts(args, parser)
    ckpt_labels = _label_ckpts(base_ckpts)
    matched_pool_dir = Path(args.matched_pool_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = _discover_matched_pool_conditions(matched_pool_dir)
    if not conditions:
        parser.error(
            f"--matched_pool_dir must either contain treated_dataset/control_dataset or child condition "
            f"directories that do. Got: {matched_pool_dir}"
        )

    single_condition = len(conditions) == 1 and conditions[0][1] == matched_pool_dir

    def run_conditions_for_ckpt(base_ckpt: Path, ckpt_output_dir: Path) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        if single_condition:
            result = _run_single_ablation(
                args=args,
                matched_pool_dir=matched_pool_dir,
                output_dir=ckpt_output_dir,
                base_ckpt=base_ckpt,
                condition_name=None,
            )
            results.append(result)
            return results

        batch_manifest_path = ckpt_output_dir / "ablation_batch_manifest.json"
        batch_manifest = {
            "created_at": datetime.now().isoformat(),
            "base_ckpt": str(base_ckpt),
            "matched_pool_root": str(matched_pool_dir),
            "output_dir": str(ckpt_output_dir),
            "dry_run": bool(args.dry_run),
            "conditions": [
                {
                    "condition_name": condition_name,
                    "matched_pool_dir": str(condition_dir),
                    "output_dir": str(ckpt_output_dir / condition_name),
                }
                for condition_name, condition_dir in conditions
            ],
            "runs": [],
        }
        write_json(batch_manifest_path, batch_manifest)
        for condition_name, condition_dir in conditions:
            print(f"running batch condition {condition_name} from {condition_dir}")
            result = _run_single_ablation(
                args=args,
                matched_pool_dir=condition_dir,
                output_dir=ckpt_output_dir / condition_name,
                base_ckpt=base_ckpt,
                condition_name=condition_name,
            )
            batch_manifest["runs"].append(result)
            write_json(batch_manifest_path, batch_manifest)
            results.append(result)

        print(f"batch manifest: {batch_manifest_path}")
        return results

    if len(base_ckpts) == 1:
        run_conditions_for_ckpt(base_ckpts[0], output_dir)
        return 0

    multi_manifest_path = output_dir / "ablation_multi_ckpt_manifest.json"
    multi_manifest = {
        "created_at": datetime.now().isoformat(),
        "matched_pool_root": str(matched_pool_dir),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "checkpoints": [
            {
                "label": ckpt_labels[ckpt],
                "base_ckpt": str(ckpt),
                "output_dir": str(output_dir / ckpt_labels[ckpt]),
            }
            for ckpt in base_ckpts
        ],
        "runs": [],
    }
    write_json(multi_manifest_path, multi_manifest)

    for ckpt in base_ckpts:
        label = ckpt_labels[ckpt]
        ckpt_output_dir = output_dir / label
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"running checkpoint {label} from {ckpt}")
        results = run_conditions_for_ckpt(ckpt, ckpt_output_dir)
        multi_manifest["runs"].extend(results)
        write_json(multi_manifest_path, multi_manifest)

    print(f"multi-ckpt manifest: {multi_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
