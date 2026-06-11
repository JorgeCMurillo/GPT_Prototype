#!/usr/bin/env python3
"""Run validation-selected linear probes on EWoK context sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
_PROTO_ROOT = _ANALYSIS_ROOT.parent
_RESEARCH_ROOT = _PROTO_ROOT.parent
_REPO_ROOT = _RESEARCH_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from .activations import (
    extract_activation_cache,
    load_activation_cache,
    parse_layer_arg,
    save_activation_cache,
)
from .artifacts import atomic_write_csv, atomic_write_json, atomic_write_jsonl
from .data import build_ewok_probe_pairs, pair_records_to_json
from .evaluation import (
    CASE_BUCKETS,
    CONTEXT_SENSITIVITY,
    compute_context_sensitivity_metrics,
    compute_probe_lm_cases,
    rows_by_bucket,
)
from .model_loading import load_model_and_tokenizer
from .plot_ewok_linear_probe import plot_run
from .probes import fit_validation_selected_probe, parse_c_grid, run_shuffle_controls
from .splits import SplitConfig, assign_grouped_splits, pair_split_labels, split_assignment_rows


def _parse_plot_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower().lstrip(".") for part in str(value).split(",") if part.strip())
    if not formats:
        raise argparse.ArgumentTypeError("At least one plot format is required.")
    invalid = [fmt for fmt in formats if any(not char.isalnum() for char in fmt)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid plot format(s): {', '.join(invalid)}")
    return formats


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(output_dir: Path, stage: str, **details: Any) -> None:
    atomic_write_json(
        output_dir / "run_status.json",
        {
            "updated_utc": _now_utc(),
            "stage": stage,
            **details,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for probe artifacts.")
    parser.add_argument("--ewok-variant", choices=("fast", "full"), default="fast")
    parser.add_argument(
        "--score-view",
        choices=(CONTEXT_SENSITIVITY,),
        default=CONTEXT_SENSITIVITY,
        help="V1 supports EWoK context sensitivity only.",
    )
    parser.add_argument("--filter-spec", default=None, help="Optional EWoK filter spec JSON.")
    parser.add_argument("--max-targets", type=int, default=0, help="Limit selected EWoK rows for smoke runs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--layers", default=None, help="Comma-separated hidden-state indices, or 'all'.")
    parser.add_argument("--cache-dtype", choices=("float16", "float32", "fp16", "fp32"), default="float16")
    parser.add_argument("--lm-score-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--C-grid", default=None, help="Comma-separated logistic-regression C values.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Report activation extraction progress every N batches. Set 0 to keep extraction quiet.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute activation cache even if it exists.")
    parser.add_argument("--no-plots", action="store_true", help="Skip automatic plot generation after fitting.")
    parser.add_argument("--plot-output-dir", default=None, help="Plot directory. Defaults to <output-dir>/plots.")
    parser.add_argument("--plot-label", default=None, help="Optional label for plot titles.")
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument(
        "--plot-formats",
        type=_parse_plot_formats,
        default=("png", "svg"),
        help="Comma-separated output formats for automatic plots, e.g. 'png' or 'png,svg'.",
    )
    parser.add_argument(
        "--plot-layer-score-split",
        choices=("val", "test"),
        default="test",
        help="Split to use for the domain-by-layer 4x3 score plot.",
    )
    parser.add_argument(
        "--plot-layer-score-metric",
        choices=("directional_avg", "row_strict_accuracy", "pair_accuracy"),
        default="directional_avg",
        help="Metric to draw in the domain-by-layer 4x3 score plot.",
    )
    parser.add_argument(
        "--skip-layer-domain-plots",
        action="store_true",
        help="Generate the lightweight plots but skip the per-layer per-domain 4x3 plot.",
    )
    return parser.parse_args()


def _lm_scores_from_cache(cache, reduction: str) -> np.ndarray:
    if reduction == "mean":
        return cache.lm_score_mean
    if reduction == "sum":
        return cache.lm_score_sum
    raise ValueError(f"Unsupported LM score reduction: {reduction!r}")


def _write_case_tables(output_dir: Path, case_rows: list[dict]) -> None:
    case_dir = output_dir / "case_examples"
    grouped = rows_by_bucket(case_rows)
    fieldnames = [
        "row_index",
        "split",
        "domain",
        "bucket",
        "probe_correct",
        "lm_correct",
        "probe_k1",
        "probe_k2",
        "probe_combined_margin",
        "probe_min_margin",
        "lm_k1",
        "lm_k2",
        "lm_combined_margin",
        "lm_min_margin",
        "rank_score",
        "combined_margin_gap",
        "context1",
        "context2",
        "target1",
        "target2",
    ]
    for bucket, rows in grouped.items():
        atomic_write_csv(case_dir / f"{bucket}.csv", rows, fieldnames=fieldnames)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "activation_cache.fp16.npz"
    _write_status(output_dir, "started", model=args.model, output_path=str(output_dir))

    pairs = build_ewok_probe_pairs(
        variant=args.ewok_variant,
        filter_spec_path=args.filter_spec,
        max_targets=int(args.max_targets),
    )
    atomic_write_jsonl(output_dir / "items.jsonl", pair_records_to_json(pairs))

    split_config = SplitConfig(
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    row_splits = assign_grouped_splits(pairs, split_config)
    split_labels = pair_split_labels(pairs, row_splits)
    atomic_write_csv(output_dir / "split_assignments.csv", split_assignment_rows(pairs, row_splits))
    _write_status(
        output_dir,
        "prepared_data",
        n_pairs=len(pairs),
        n_rows=len(row_splits),
        split_counts={
            "train_pairs": int(np.sum(split_labels == "train")),
            "val_pairs": int(np.sum(split_labels == "val")),
            "test_pairs": int(np.sum(split_labels == "test")),
        },
    )

    if cache_path.exists() and not args.force:
        print(f"[linear_probe] loading activation cache: {cache_path}", flush=True)
        _write_status(output_dir, "loading_activation_cache", cache_path=str(cache_path))
        cache = load_activation_cache(cache_path, pairs)
        _write_status(
            output_dir,
            "loaded_activation_cache",
            cache_path=str(cache_path),
            layer_indices=list(cache.layer_indices),
            n_pairs=cache.n_pairs,
            hidden_dim=cache.hidden_dim,
        )
    else:
        print(f"[linear_probe] extracting activation cache for model: {args.model}", flush=True)
        _write_status(
            output_dir,
            "extracting_activation_cache",
            model=args.model,
            requested_layers=parse_layer_arg(args.layers),
            batch_size=int(args.batch_size),
        )
        loaded = load_model_and_tokenizer(
            args.model,
            device_arg=args.device,
            dtype_arg=args.dtype,
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        cache = extract_activation_cache(
            loaded.model,
            loaded.tokenizer,
            pairs,
            batch_size=int(args.batch_size),
            layers=parse_layer_arg(args.layers),
            cache_dtype=args.cache_dtype,
            progress_every=int(args.progress_every),
            progress_callback=lambda event: (
                print(
                    "[linear_probe] activation "
                    f"{event['completed_batches']}/{event['total_batches']} batches "
                    f"({event['processed_pairs']}/{event['total_pairs']} pairs)",
                    flush=True,
                ),
                _write_status(output_dir, "extracting_activation_cache", **event),
            ),
        )
        save_activation_cache(cache_path, cache)
        _write_status(
            output_dir,
            "wrote_activation_cache",
            cache_path=str(cache_path),
            layer_indices=list(cache.layer_indices),
            n_pairs=cache.n_pairs,
            hidden_dim=cache.hidden_dim,
        )

    c_grid = parse_c_grid(args.C_grid)
    print(
        "[linear_probe] fitting probes "
        f"for {len(cache.layer_indices)} layer(s) x {len(c_grid)} C value(s)",
        flush=True,
    )
    _write_status(
        output_dir,
        "fitting_probes",
        layer_indices=list(cache.layer_indices),
        c_grid=list(c_grid),
        total_fits=int(len(cache.layer_indices) * len(c_grid)),
    )
    selected = fit_validation_selected_probe(
        cache=cache,
        pairs=pairs,
        split_labels=split_labels,
        c_grid=c_grid,
        seed=int(args.seed),
        progress_callback=lambda event: (
            print(
                "[linear_probe] fit "
                f"{event['completed_fits']}/{event['total_fits']} "
                f"layer={event['layer_index']} C={event['C']} "
                f"train_row={event['train_row_strict_accuracy']:.3f} "
                f"val_row={event['val_row_strict_accuracy']:.3f}",
                flush=True,
            ),
            _write_status(output_dir, "fitting_probes", **event),
        ),
    )
    atomic_write_csv(output_dir / "layer_validation_table.csv", list(selected.validation_table))
    _write_status(
        output_dir,
        "selected_probe",
        selected_layer_index=int(selected.layer_index),
        selected_C=float(selected.C),
        train_metrics=selected.train_metrics,
        validation_metrics=selected.validation_metrics,
        test_metrics=selected.test_metrics,
    )

    lm_scores = _lm_scores_from_cache(cache, args.lm_score_reduction)
    case_rows, bucket_counts, domain_bucket_rows = compute_probe_lm_cases(
        pairs,
        probe_scores=selected.probe_scores,
        lm_scores=lm_scores,
        split_labels=split_labels,
        split="test",
    )
    atomic_write_csv(
        output_dir / "probe_vs_lm_bucket_counts.csv",
        [{"bucket": bucket, "count": int(bucket_counts.get(bucket, 0))} for bucket in CASE_BUCKETS],
    )
    atomic_write_csv(output_dir / "domain_bucket_counts.csv", domain_bucket_rows)
    _write_case_tables(output_dir, case_rows)

    test_mask = split_labels == "test"
    lm_test_metrics = compute_context_sensitivity_metrics(
        pairs,
        lm_scores,
        split_labels=split_labels,
        split="test",
    )
    selected_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "score_view": args.score_view,
        "representation": "post_target_last",
        "selected_layer_index": int(selected.layer_index),
        "selected_layer_position": int(selected.layer_position),
        "selected_C": float(selected.C),
        "train_metrics": selected.train_metrics,
        "validation_metrics": selected.validation_metrics,
        "test_metrics": selected.test_metrics,
        "split_counts": {
            "train_pairs": int(np.sum(split_labels == "train")),
            "val_pairs": int(np.sum(split_labels == "val")),
            "test_pairs": int(np.sum(test_mask)),
        },
        "layer_indices": list(cache.layer_indices),
        "c_grid": list(c_grid),
    }
    atomic_write_json(output_dir / "selected_probe_summary.json", selected_summary)

    test_metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "score_view": args.score_view,
        "lm_score_reduction": args.lm_score_reduction,
        "representation": "post_target_last",
        "selected_layer_index": int(selected.layer_index),
        "selected_layer_position": int(selected.layer_position),
        "selected_C": float(selected.C),
        "probe": selected.test_metrics,
        "lm": lm_test_metrics,
        "bucket_counts": bucket_counts,
        "n_test_rows": int(len(case_rows)),
    }
    atomic_write_json(output_dir / "test_metrics.json", test_metrics)
    _write_status(output_dir, "wrote_core_metrics", test_metrics_path=str(output_dir / "test_metrics.json"))

    print(f"[linear_probe] running {int(args.shuffle_repeats)} shuffled-label control(s)", flush=True)
    _write_status(output_dir, "running_shuffle_controls", repeats=int(args.shuffle_repeats))
    shuffle_controls = run_shuffle_controls(
        cache=cache,
        pairs=pairs,
        split_labels=split_labels,
        c_grid=c_grid,
        repeats=int(args.shuffle_repeats),
        seed=int(args.seed),
        progress_callback=lambda event: (
            print(
                "[linear_probe] shuffle "
                f"{event['repeat']}/{event['total_repeats']} {event['event']}",
                flush=True,
            ),
            _write_status(output_dir, "running_shuffle_controls", **event),
        ),
    )
    atomic_write_csv(
        output_dir / "shuffle_controls.csv",
        list(shuffle_controls),
        fieldnames=[
            "repeat",
            "selected_layer_index",
            "selected_layer_position",
            "selected_C",
            "val_row_strict_accuracy",
            "val_pair_accuracy",
            "test_row_strict_accuracy",
            "test_pair_accuracy",
            "test_k1_accuracy",
            "test_k2_accuracy",
        ],
    )
    print(f"[linear_probe] wrote artifacts to {output_dir}", flush=True)
    _write_status(output_dir, "wrote_artifacts", test_metrics_path=str(output_dir / "test_metrics.json"))
    if not args.no_plots:
        _write_status(output_dir, "plotting", plot_output_dir=args.plot_output_dir or str(output_dir / "plots"))
        created_plots = plot_run(
            output_dir,
            output_dir=args.plot_output_dir,
            label=args.plot_label or output_dir.name,
            formats=args.plot_formats,
            dpi=int(args.plot_dpi),
            layer_score_split=args.plot_layer_score_split,
            layer_score_metric=args.plot_layer_score_metric,
            skip_layer_domain=bool(args.skip_layer_domain_plots),
            seed=int(args.seed),
        )
        plot_dir = Path(args.plot_output_dir).expanduser().resolve() if args.plot_output_dir else output_dir / "plots"
        print(f"[linear_probe] wrote {len(created_plots)} plot files to {plot_dir}", flush=True)
        _write_status(
            output_dir,
            "complete",
            test_metrics_path=str(output_dir / "test_metrics.json"),
            plot_dir=str(plot_dir),
            n_plot_files=len(created_plots),
        )
    else:
        _write_status(output_dir, "complete", test_metrics_path=str(output_dir / "test_metrics.json"))
    return test_metrics


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
