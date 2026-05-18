"""Projection-geometry audit for TrackStar-style scores.

This diagnostic isolates the random-projection stage from the one-step update
sanity checks.  For sampled candidate examples, it compares pre-projection
raw/Adam dot products against projected dot products across a sweep of
projection ranks.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .one_step_sanity import (
    DEFAULT_DEVICE,
    DEFAULT_EWOK_VARIANT,
    DEFAULT_GROUP_SIZE,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_REDUCTION,
    DEFAULT_SCORE_VIEW,
    DEFAULT_SEED,
    DEFAULT_TARGET_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    _build_tqdm,
    _infer_seq_len_from_scored_frame,
    _load_attribution_defaults,
    _resolve_candidate_kind,
    _resolve_device,
    build_candidate_groups,
)
from .paper_blocks import (
    MODULE_LAYOUT,
    PAPER_BLOCK_LAYOUT,
    build_gpt2_paper_block_layout,
)
from .raw_dot_audit import (
    DEFAULT_NUM_EXAMPLES_PER_GROUP,
    GROUP_NAMES,
    _candidate_gradient_modules,
    _clone_state_dict_to_cpu,
    _cosine_similarity,
    _dot_product,
    _pearson_corr,
    _spearman_corr,
    collect_candidate_grads,
    collect_mean_query_grads,
)
from .score_path_audit import (
    _apply_weight_normalizers_to_grads,
    _build_trackstar_config,
    _infer_checkpoint_kind,
    _project_grad_mapping_with_layout,
)
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame
from ..common.checkpoints import CheckpointRef, build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.ewok_targets import build_ewok_targets
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest
from ..run_trak import RunExecutionContext
from .backend import build_backend as build_trackstar_backend


DEFAULT_PROJECTION_RANKS = (16, 32, 64, 128)


def _status(*, enabled: bool, message: str) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [projection-geometry-audit] {message}", flush=True)


def _parse_projection_ranks(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_PROJECTION_RANKS
    ranks: list[int] = []
    for value in values:
        for part in str(value).split(","):
            stripped = part.strip()
            if not stripped:
                continue
            rank = int(stripped)
            if rank <= 0:
                raise ValueError(f"Projection ranks must be positive, got {rank}")
            ranks.append(rank)
    if not ranks:
        raise ValueError("At least one projection rank is required")
    return tuple(dict.fromkeys(ranks))


def _summary_stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _correlation_summary(frame: pd.DataFrame, *, left: str, right: str) -> dict[str, float | None]:
    if frame.empty or left not in frame.columns or right not in frame.columns:
        return {"pearson": None, "spearman": None}
    left_values = frame[left].astype(float).tolist()
    right_values = frame[right].astype(float).tolist()
    return {
        "pearson": _pearson_corr(left_values, right_values),
        "spearman": _spearman_corr(left_values, right_values),
    }


def _error_summary(frame: pd.DataFrame, *, target: str, estimate: str) -> dict[str, float | None]:
    if frame.empty or target not in frame.columns or estimate not in frame.columns:
        return {"rmse": None, "mae": None, "rmse_over_target_rms": None, "mae_over_target_abs_mean": None}
    target_values = frame[target].astype(float).to_numpy(dtype=np.float64)
    estimate_values = frame[estimate].astype(float).to_numpy(dtype=np.float64)
    error = estimate_values - target_values
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))
    target_rms = float(np.sqrt(np.mean(target_values * target_values)))
    target_abs_mean = float(np.mean(np.abs(target_values)))
    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_over_target_rms": None if np.isclose(target_rms, 0.0) else float(rmse / target_rms),
        "mae_over_target_abs_mean": None
        if np.isclose(target_abs_mean, 0.0)
        else float(mae / target_abs_mean),
    }


def _projected_dot_rescaled(
    query_features: Mapping[str, torch.Tensor],
    candidate_features: Mapping[str, torch.Tensor],
    *,
    scale_by_key: Mapping[str, float],
) -> float:
    total = 0.0
    for name, query_feature in query_features.items():
        candidate_feature = candidate_features.get(name)
        if candidate_feature is None:
            raise KeyError(f"Candidate projected features are missing key {name!r}")
        scale = float(scale_by_key[name])
        total += scale * float(torch.sum(query_feature * candidate_feature).item())
    return total


def _projection_scales(
    raw_grads: Mapping[str, torch.Tensor],
    *,
    projection_rank: int,
    projection_layout: str,
    paper_block_features: int,
) -> dict[str, float]:
    rank = int(projection_rank)
    if projection_layout == PAPER_BLOCK_LAYOUT:
        layout = build_gpt2_paper_block_layout(
            {name: tuple(int(dim) for dim in grad.shape) for name, grad in raw_grads.items()},
            feature_dim=int(paper_block_features),
        )
        return {
            block.name: float(block.row_dim * block.col_dim) / float(rank * rank)
            for block in layout.blocks
        }
    return {
        name: float(int(grad.shape[0]) * int(grad.shape[1])) / float(rank * rank)
        for name, grad in raw_grads.items()
    }


def _project_for_rank(
    grads: Mapping[str, torch.Tensor],
    *,
    projection_rank: int,
    projection_layout: str,
) -> dict[str, torch.Tensor]:
    return _project_grad_mapping_with_layout(
        grads,
        projection_dim=int(projection_rank),
        projection_layout=projection_layout,
        paper_block_features=int(projection_rank) * int(projection_rank),
    )


def _summarize_projection_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {"num_examples": 0}

    summary: dict[str, Any] = {"num_examples": int(len(frame))}
    metric_names = (
        "raw_dot",
        "raw_cosine",
        "adam_dot",
        "adam_cosine",
        "projected_raw_dot",
        "projected_raw_dot_rescaled",
        "projected_raw_cosine",
        "projected_adam_dot",
        "projected_adam_dot_rescaled",
        "projected_adam_cosine",
        "candidate_train_loss",
    )
    for metric_name in metric_names:
        if metric_name not in frame.columns:
            continue
        stats = _summary_stats(frame[metric_name].astype(float).tolist())
        if stats is not None:
            summary[metric_name] = stats
    return summary


def _summarize_by_rank(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {}

    by_rank: dict[str, Any] = {}
    for rank, rank_frame in frame.groupby("projection_rank", sort=True):
        rank_frame = rank_frame.reset_index(drop=True)
        by_rank[str(int(rank))] = {
            "num_examples": int(len(rank_frame)),
            "raw_dot_vs_projected_raw_dot": _correlation_summary(
                rank_frame,
                left="raw_dot",
                right="projected_raw_dot",
            ),
            "raw_dot_vs_projected_raw_dot_rescaled": _correlation_summary(
                rank_frame,
                left="raw_dot",
                right="projected_raw_dot_rescaled",
            ),
            "raw_cosine_vs_projected_raw_cosine": _correlation_summary(
                rank_frame,
                left="raw_cosine",
                right="projected_raw_cosine",
            ),
            "raw_dot_projected_error": _error_summary(
                rank_frame,
                target="raw_dot",
                estimate="projected_raw_dot",
            ),
            "raw_dot_projected_rescaled_error": _error_summary(
                rank_frame,
                target="raw_dot",
                estimate="projected_raw_dot_rescaled",
            ),
            "adam_dot_vs_projected_adam_dot": _correlation_summary(
                rank_frame,
                left="adam_dot",
                right="projected_adam_dot",
            ),
            "adam_dot_vs_projected_adam_dot_rescaled": _correlation_summary(
                rank_frame,
                left="adam_dot",
                right="projected_adam_dot_rescaled",
            ),
            "adam_cosine_vs_projected_adam_cosine": _correlation_summary(
                rank_frame,
                left="adam_cosine",
                right="projected_adam_cosine",
            ),
            "adam_dot_projected_error": _error_summary(
                rank_frame,
                target="adam_dot",
                estimate="projected_adam_dot",
            ),
            "adam_dot_projected_rescaled_error": _error_summary(
                rank_frame,
                target="adam_dot",
                estimate="projected_adam_dot_rescaled",
            ),
        }
    return by_rank


def run_projection_geometry_audit(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    output_dir: str | Path,
    score_mode: str = DEFAULT_SCORE_MODE,
    target_id: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    num_examples_per_group: int = DEFAULT_NUM_EXAMPLES_PER_GROUP,
    projection_ranks: Sequence[int] = DEFAULT_PROJECTION_RANKS,
    projection_layout: str | None = None,
    ewok_filter_spec: str | Path | None = None,
    ewok_variant: str = DEFAULT_EWOK_VARIANT,
    ewok_score_view: str = DEFAULT_SCORE_VIEW,
    score_reduction: str = DEFAULT_SCORE_REDUCTION,
    temperature: float = DEFAULT_TEMPERATURE,
    target_batch_size: int = DEFAULT_TARGET_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    show_progress: bool = True,
) -> dict[str, Path]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected one of {SUPPORTED_SCORE_MODES!r}")
    if num_examples_per_group <= 0:
        raise ValueError("num_examples_per_group must be > 0")
    if projection_layout is not None and projection_layout not in {MODULE_LAYOUT, PAPER_BLOCK_LAYOUT}:
        raise ValueError(
            f"Unsupported projection_layout={projection_layout!r}; "
            f"expected {MODULE_LAYOUT!r} or {PAPER_BLOCK_LAYOUT!r}"
        )

    ranks = tuple(int(rank) for rank in projection_ranks)
    if any(rank <= 0 for rank in ranks):
        raise ValueError(f"Projection ranks must be positive, got {ranks!r}")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    _status(
        enabled=show_progress,
        message=(
            f"starting step={int(step)} score_mode={score_mode} "
            f"ranks={','.join(str(int(rank)) for rank in ranks)}"
        ),
    )

    _status(enabled=show_progress, message="loading scored candidate frame")
    scored = load_candidate_score_frame(
        attribution_dir=attribution_dir,
        step=int(step),
        score_mode=score_mode,
        target_id=target_id,
    )
    candidate_kind = _resolve_candidate_kind(scored)
    seq_len = _infer_seq_len_from_scored_frame(scored)
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=candidate_kind,
        seq_len=seq_len,
    )
    _status(
        enabled=show_progress,
        message=f"building top/matched-random/bottom pools from {len(scored)} scored candidates",
    )

    groups = build_candidate_groups(
        scored,
        group_size=int(group_size),
        rng=rng,
        allow_relaxed_shard_match=True,
    )

    sampled_frames: dict[str, pd.DataFrame] = {}
    for name, group in groups.items():
        if len(group.frame) < int(num_examples_per_group):
            raise ValueError(
                f"Group {name!r} only has {len(group.frame)} rows, cannot sample {num_examples_per_group}"
            )
        sampled_indices = rng.choice(len(group.frame), size=int(num_examples_per_group), replace=False)
        sampled_frames[name] = group.frame.iloc[sampled_indices].reset_index(drop=True).copy()
        sampled_frames[name].to_csv(output_root / f"sampled_{name}_candidates.csv", index=False)
    _status(
        enabled=show_progress,
        message=f"sampled {len(GROUP_NAMES) * int(num_examples_per_group)} examples; loading checkpoint next",
    )

    attribution_defaults = _load_attribution_defaults(attribution_dir)
    effective_filter_spec = ewok_filter_spec
    if effective_filter_spec is None:
        effective_filter_spec = attribution_defaults.get("ewok_filter_spec")
    effective_projection_layout = projection_layout or str(attribution_defaults.get("projection_layout", MODULE_LAYOUT))
    if effective_projection_layout not in {MODULE_LAYOUT, PAPER_BLOCK_LAYOUT}:
        raise ValueError(
            f"Unsupported projection_layout={effective_projection_layout!r}; "
            f"expected {MODULE_LAYOUT!r} or {PAPER_BLOCK_LAYOUT!r}"
        )

    phase_progress = _build_tqdm(
        enabled=show_progress,
        total=4,
        desc="Projection-geometry audit setup",
        unit="phase",
    )

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    base_state = _clone_state_dict_to_cpu(model)
    raw_modules = _candidate_gradient_modules(model)
    if phase_progress is not None:
        phase_progress.set_postfix_str("model + modules")
        phase_progress.update(1)

    bundle = build_ewok_targets(
        score_view=str(ewok_score_view),
        target_scope="overall",
        score_reduction=str(score_reduction),
        variant=str(ewok_variant),
        filter_spec_path=effective_filter_spec,
        max_targets=0,
    )

    model.load_state_dict(base_state, strict=True)
    model.to(model_device)
    raw_query_grads = collect_mean_query_grads(
        model,
        tokenizer,
        bundle,
        modules=raw_modules,
        temperature=float(temperature),
        batch_size=int(target_batch_size),
        show_progress=show_progress,
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("raw query grads")
        phase_progress.update(1)

    checkpoint_ref = CheckpointRef(
        step=int(step),
        path=Path(base_ckpt).expanduser().resolve(),
        kind=_infer_checkpoint_kind(base_ckpt),
    )
    backend_defaults = dict(attribution_defaults)
    backend_defaults["projection_layout"] = effective_projection_layout
    trackstar_config = _build_trackstar_config(
        base_ckpt=base_ckpt,
        attribution_dir=attribution_dir,
        data_dir=data_dir,
        attribution_defaults=backend_defaults,
        device=model_device.type,
        show_progress=show_progress,
    )
    execution_context = RunExecutionContext.single_process(
        backend="trackstar",
        requested_device=str(device),
        resolved_device=str(model_device),
    )
    backend = build_trackstar_backend(
        config=trackstar_config,
        model=model,
        tokenizer=tokenizer,
        execution_context=execution_context,
    )
    backend._load_checkpoint_into_model(checkpoint_ref)

    query_weight_normalizers: dict[str, Any] = {}
    if backend._candidate_uses_adam_second_moment_correction(checkpoint_ref):
        query_weight_normalizers = backend._load_candidate_adam_normalizers(checkpoint_ref)
    raw_query_grads_adam = _apply_weight_normalizers_to_grads(
        raw_query_grads,
        weight_normalizers=query_weight_normalizers,
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("Adam normalizers")
        phase_progress.update(1)

    projected_query_by_rank: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    projection_scales_by_rank: dict[int, dict[str, float]] = {}
    for rank in ranks:
        projected_query_by_rank[int(rank)] = {
            "raw": _project_for_rank(
                raw_query_grads,
                projection_rank=int(rank),
                projection_layout=effective_projection_layout,
            ),
            "adam": _project_for_rank(
                raw_query_grads_adam,
                projection_rank=int(rank),
                projection_layout=effective_projection_layout,
            ),
        }
        projection_scales_by_rank[int(rank)] = _projection_scales(
            raw_query_grads,
            projection_rank=int(rank),
            projection_layout=effective_projection_layout,
            paper_block_features=int(rank) * int(rank),
        )
    if phase_progress is not None:
        phase_progress.set_postfix_str("query projections")
        phase_progress.update(1)
        phase_progress.set_postfix_str("done")
        phase_progress.close()
        phase_progress = None

    all_rows: list[dict[str, Any]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=int(num_examples_per_group) * len(GROUP_NAMES),
        desc=f"Projection-geometry audit ({score_mode})",
        unit="example",
    )

    try:
        for group_name in GROUP_NAMES:
            frame = sampled_frames[group_name]
            dataset = FiniteTrainingExampleDataset(
                manifest,
                tuple(int(value) for value in frame["candidate_id"].tolist()),
            )
            for local_index, row in enumerate(frame.itertuples(index=False)):
                sample = dataset[int(local_index)]
                candidate_id = int(row.candidate_id)
                model.load_state_dict(base_state, strict=True)
                model.to(model_device)
                raw_candidate_grads, candidate_train_loss = collect_candidate_grads(
                    model,
                    modules=raw_modules,
                    input_ids=sample["input_ids"].unsqueeze(0),
                    labels=sample["labels"].unsqueeze(0),
                    device=model_device,
                )
                adam_candidate_grads = _apply_weight_normalizers_to_grads(
                    raw_candidate_grads,
                    weight_normalizers=query_weight_normalizers,
                )

                raw_dot = float(_dot_product(raw_query_grads, raw_candidate_grads))
                raw_cosine = float(_cosine_similarity(raw_query_grads, raw_candidate_grads))
                adam_dot = float(_dot_product(raw_query_grads_adam, adam_candidate_grads))
                adam_cosine = float(_cosine_similarity(raw_query_grads_adam, adam_candidate_grads))

                for rank in ranks:
                    projected_raw_candidate = _project_for_rank(
                        raw_candidate_grads,
                        projection_rank=int(rank),
                        projection_layout=effective_projection_layout,
                    )
                    projected_adam_candidate = _project_for_rank(
                        adam_candidate_grads,
                        projection_rank=int(rank),
                        projection_layout=effective_projection_layout,
                    )
                    projected_raw_query = projected_query_by_rank[int(rank)]["raw"]
                    projected_adam_query = projected_query_by_rank[int(rank)]["adam"]
                    scale_by_key = projection_scales_by_rank[int(rank)]

                    record = {
                        "group": group_name,
                        "candidate_id": candidate_id,
                        "projection_rank": int(rank),
                        "projection_layout": effective_projection_layout,
                        "selection_score": float(row.selection_score),
                        "raw_dot": raw_dot,
                        "raw_cosine": raw_cosine,
                        "adam_dot": adam_dot,
                        "adam_cosine": adam_cosine,
                        "projected_raw_dot": float(_dot_product(projected_raw_query, projected_raw_candidate)),
                        "projected_raw_dot_rescaled": float(
                            _projected_dot_rescaled(
                                projected_raw_query,
                                projected_raw_candidate,
                                scale_by_key=scale_by_key,
                            )
                        ),
                        "projected_raw_cosine": float(
                            _cosine_similarity(projected_raw_query, projected_raw_candidate)
                        ),
                        "projected_adam_dot": float(_dot_product(projected_adam_query, projected_adam_candidate)),
                        "projected_adam_dot_rescaled": float(
                            _projected_dot_rescaled(
                                projected_adam_query,
                                projected_adam_candidate,
                                scale_by_key=scale_by_key,
                            )
                        ),
                        "projected_adam_cosine": float(
                            _cosine_similarity(projected_adam_query, projected_adam_candidate)
                        ),
                        "candidate_train_loss": float(candidate_train_loss),
                        "shard_path": str(row.shard_path),
                        "local_example_idx": int(row.local_example_idx),
                        "token_offset_start": int(row.token_offset_start),
                        "token_offset_end": int(row.token_offset_end),
                    }
                    all_rows.append(record)

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(group_name)
    finally:
        if progress is not None:
            progress.close()
        if phase_progress is not None:
            phase_progress.close()

    results_path = output_root / "audit_results.jsonl"
    write_jsonl(results_path, all_rows)

    summary_payload = {
        "score_mode": score_mode,
        "target_id": target_id,
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "num_targets": int(len(bundle.items)),
        "num_examples_per_group": int(num_examples_per_group),
        "projection_layout": effective_projection_layout,
        "projection_ranks": [int(rank) for rank in ranks],
        "module_count": int(len(raw_modules)),
        "adam_normalizer_count": int(len(query_weight_normalizers)),
        "global_projection_correlations": _summarize_by_rank(all_rows),
        "groups": {
            name: {
                "by_rank": _summarize_by_rank([row for row in all_rows if row["group"] == name]),
                "metric_summaries": {
                    str(int(rank)): _summarize_projection_rows(
                        [
                            row
                            for row in all_rows
                            if row["group"] == name and int(row["projection_rank"]) == int(rank)
                        ]
                    )
                    for rank in ranks
                },
            }
            for name in GROUP_NAMES
        },
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary_payload)

    config_path = output_root / "config.json"
    write_json(
        config_path,
        {
            "base_ckpt": str(Path(base_ckpt).expanduser().resolve()),
            "attribution_dir": str(Path(attribution_dir).expanduser().resolve()),
            "data_dir": str(Path(data_dir).expanduser().resolve()),
            "step": int(step),
            "score_mode": score_mode,
            "target_id": target_id,
            "group_size": int(group_size),
            "num_examples_per_group": int(num_examples_per_group),
            "projection_layout": effective_projection_layout,
            "projection_ranks": [int(rank) for rank in ranks],
            "ewok_filter_spec": None
            if effective_filter_spec is None
            else str(Path(effective_filter_spec).expanduser().resolve()),
            "ewok_variant": str(ewok_variant),
            "ewok_score_view": str(ewok_score_view),
            "score_reduction": str(score_reduction),
            "temperature": float(temperature),
            "target_batch_size": int(target_batch_size),
            "seed": int(seed),
            "device": str(model_device),
        },
    )

    return {
        "root": output_root,
        "config": config_path,
        "results": results_path,
        "summary": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep TrackStar projection ranks and measure how well projected "
            "raw/Adam dots preserve their pre-projection geometry."
        )
    )
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--attribution_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--score_mode",
        choices=SUPPORTED_SCORE_MODES,
        default=DEFAULT_SCORE_MODE,
    )
    parser.add_argument("--target_id", default=None)
    parser.add_argument("--group_size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--num_examples_per_group", type=int, default=DEFAULT_NUM_EXAMPLES_PER_GROUP)
    parser.add_argument(
        "--projection_ranks",
        nargs="+",
        default=None,
        help="Projection side ranks to sweep, e.g. `16 32 64 128` or `16,32,64,128`.",
    )
    parser.add_argument(
        "--projection_layout",
        choices=(MODULE_LAYOUT, PAPER_BLOCK_LAYOUT),
        default=None,
        help="Projection layout to audit. Defaults to the attribution artifact's config.",
    )
    parser.add_argument("--ewok_filter_spec", default=None)
    parser.add_argument("--ewok_variant", default=None)
    parser.add_argument("--ewok_score_view", default=None)
    parser.add_argument("--score_reduction", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--target_batch_size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--no_progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    projection_ranks = _parse_projection_ranks(args.projection_ranks)
    attribution_defaults = _load_attribution_defaults(args.attribution_dir)
    resolved_filter_spec = args.ewok_filter_spec or attribution_defaults.get("ewok_filter_spec")
    resolved_variant = args.ewok_variant or attribution_defaults.get("ewok_variant", DEFAULT_EWOK_VARIANT)
    resolved_score_view = args.ewok_score_view or attribution_defaults.get("ewok_score_view", DEFAULT_SCORE_VIEW)
    resolved_reduction = args.score_reduction or attribution_defaults.get("score_reduction", DEFAULT_SCORE_REDUCTION)
    resolved_temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(attribution_defaults.get("temperature", DEFAULT_TEMPERATURE))
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(args.attribution_dir).expanduser().resolve()
            / f"projection_geometry_audit_step{int(args.step):08d}_{args.score_mode}"
        )

    artifacts = run_projection_geometry_audit(
        base_ckpt=args.base_ckpt,
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=int(args.step),
        output_dir=output_dir,
        score_mode=args.score_mode,
        target_id=args.target_id,
        group_size=int(args.group_size),
        num_examples_per_group=int(args.num_examples_per_group),
        projection_ranks=projection_ranks,
        projection_layout=args.projection_layout,
        ewok_filter_spec=resolved_filter_spec,
        ewok_variant=str(resolved_variant),
        ewok_score_view=str(resolved_score_view),
        score_reduction=str(resolved_reduction),
        temperature=float(resolved_temperature),
        target_batch_size=int(args.target_batch_size),
        seed=int(args.seed),
        device=args.device,
        show_progress=not bool(args.no_progress),
    )
    print(f"wrote projection-geometry audit artifacts under {artifacts['root']}")
    print(f"results: {artifacts['results']}")
    print(f"summary: {artifacts['summary']}")
    return 0


__all__ = [
    "build_arg_parser",
    "main",
    "run_projection_geometry_audit",
]
