"""Per-block diagnostic for Adam-corrected TrackStar projections.

This audit is meant for the failure mode where raw projected dots preserve the
first-order geometry, but Adam-corrected projected dots do not.  It keeps the
same sampled top/matched-random/bottom examples as the other audits, then
breaks the pre- and post-projection geometry down by paper block.
"""

from __future__ import annotations

import argparse
import math
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
    PaperBlockLayoutSpec,
    build_gpt2_paper_block_layout,
)
from .projection_geometry_audit import (
    _correlation_summary,
    _error_summary,
    _parse_projection_ranks,
    _project_for_rank,
    _projection_scales,
    _summary_stats,
)
from .raw_dot_audit import (
    DEFAULT_NUM_EXAMPLES_PER_GROUP,
    GROUP_NAMES,
    _candidate_gradient_modules,
    _clone_state_dict_to_cpu,
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
)
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame
from ..common.checkpoints import CheckpointRef, build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.ewok_targets import build_ewok_targets
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest
from ..run_trak import RunExecutionContext
from .backend import build_backend as build_trackstar_backend


DEFAULT_PROJECTION_RANKS = (64,)


def _status(*, enabled: bool, message: str) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [adam-projection-diagnostic] {message}", flush=True)


def _module_dot(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    module_names: Sequence[str],
) -> float:
    total = 0.0
    for name in module_names:
        total += float(torch.sum(left[name] * right[name]).item())
    return total


def _module_norm_sq(
    values: Mapping[str, torch.Tensor],
    *,
    module_names: Sequence[str],
) -> float:
    total = 0.0
    for name in module_names:
        value = values[name]
        total += float(torch.sum(value * value).item())
    return total


def _cosine_from_dot(dot: float, left_norm: float, right_norm: float) -> float:
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def _feature_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left * right).item())


def _feature_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(dtype=torch.float32)).item())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if math.isclose(float(denominator), 0.0, rel_tol=0.0, abs_tol=1e-30):
        return None
    return float(numerator) / float(denominator)


def _tensor_distribution_stats(
    tensor: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float | int]:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    quantiles = torch.quantile(
        values,
        torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32),
    )
    return {
        f"{prefix}_numel": int(values.numel()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_q01": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_q99": float(quantiles[2].item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _normalizer_module_rows(
    *,
    layout: PaperBlockLayoutSpec,
    weight_normalizers: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in layout.blocks:
        for member in block.members:
            normalizer = weight_normalizers.get(member.module_name)
            if normalizer is None:
                rows.append(
                    {
                        "block_name": block.name,
                        "module_name": member.module_name,
                        "family": block.family,
                        "has_normalizer": False,
                    }
                )
                continue
            avg_sq = getattr(normalizer, "weight_avg_sq", None)
            if not isinstance(avg_sq, torch.Tensor):
                rows.append(
                    {
                        "block_name": block.name,
                        "module_name": member.module_name,
                        "family": block.family,
                        "has_normalizer": False,
                    }
                )
                continue
            avg_sq_stats = _tensor_distribution_stats(avg_sq, prefix="weight_avg_sq")
            scale = avg_sq.detach().to(device="cpu", dtype=torch.float32).sqrt().add_(1e-8).reciprocal_()
            scale_stats = _tensor_distribution_stats(scale, prefix="adam_scale")
            rows.append(
                {
                    "block_name": block.name,
                    "module_name": member.module_name,
                    "family": block.family,
                    "layer_index": int(member.layer_index),
                    "submodule_name": member.submodule_name,
                    "out_dim": int(member.out_dim),
                    "in_dim": int(member.in_dim),
                    "has_normalizer": True,
                    **avg_sq_stats,
                    **scale_stats,
                }
            )
    return rows


def _normalizer_block_summary(module_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame.from_records(list(module_rows))
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for block_name, block_frame in frame.groupby("block_name", sort=True):
        normalizer_frame = block_frame[block_frame["has_normalizer"].astype(bool)].copy()
        row: dict[str, Any] = {
            "block_name": str(block_name),
            "family": str(block_frame["family"].iloc[0]),
            "module_count": int(len(block_frame)),
            "normalizer_count": int(len(normalizer_frame)),
        }
        if not normalizer_frame.empty:
            weights = normalizer_frame["weight_avg_sq_numel"].astype(float).to_numpy(dtype=np.float64)
            weight_total = float(weights.sum())
            row.update(
                {
                    "weight_avg_sq_numel": int(weight_total),
                    "weight_avg_sq_min": float(normalizer_frame["weight_avg_sq_min"].astype(float).min()),
                    "weight_avg_sq_mean_weighted": float(
                        np.average(normalizer_frame["weight_avg_sq_mean"].astype(float), weights=weights)
                    ),
                    "weight_avg_sq_median_mean": float(normalizer_frame["weight_avg_sq_median"].astype(float).mean()),
                    "weight_avg_sq_q99_max_module": float(normalizer_frame["weight_avg_sq_q99"].astype(float).max()),
                    "weight_avg_sq_max": float(normalizer_frame["weight_avg_sq_max"].astype(float).max()),
                    "adam_scale_min": float(normalizer_frame["adam_scale_min"].astype(float).min()),
                    "adam_scale_mean_weighted": float(
                        np.average(normalizer_frame["adam_scale_mean"].astype(float), weights=weights)
                    ),
                    "adam_scale_median_mean": float(normalizer_frame["adam_scale_median"].astype(float).mean()),
                    "adam_scale_q99_max_module": float(normalizer_frame["adam_scale_q99"].astype(float).max()),
                    "adam_scale_max": float(normalizer_frame["adam_scale_max"].astype(float).max()),
                    "normalizer_coverage": float(len(normalizer_frame) / max(1, len(block_frame))),
                }
            )
            if math.isclose(weight_total, 0.0):
                row["weight_avg_sq_numel"] = 0
        rows.append(row)
    return rows


def _summarize_example_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {}

    summary: dict[str, Any] = {}
    for rank, rank_frame in frame.groupby("projection_rank", sort=True):
        rank_frame = rank_frame.reset_index(drop=True)
        summary[str(int(rank))] = {
            "num_examples": int(len(rank_frame)),
            "raw_dot_vs_projected_raw_dot_rescaled": _correlation_summary(
                rank_frame,
                left="raw_dot",
                right="projected_raw_dot_rescaled",
            ),
            "adam_dot_vs_projected_adam_dot_rescaled": _correlation_summary(
                rank_frame,
                left="adam_dot",
                right="projected_adam_dot_rescaled",
            ),
            "raw_projection_rescaled_error": _error_summary(
                rank_frame,
                target="raw_dot",
                estimate="projected_raw_dot_rescaled",
            ),
            "adam_projection_rescaled_error": _error_summary(
                rank_frame,
                target="adam_dot",
                estimate="projected_adam_dot_rescaled",
            ),
        }
    return summary


def _summarize_block_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {}

    by_rank: dict[str, Any] = {}
    for rank, rank_frame in frame.groupby("projection_rank", sort=True):
        rank_payload: dict[str, Any] = {
            "all_block_example_pairs": {
                "num_rows": int(len(rank_frame)),
                "raw_dot_vs_projected_raw_dot_rescaled": _correlation_summary(
                    rank_frame,
                    left="raw_dot",
                    right="projected_raw_dot_rescaled",
                ),
                "adam_dot_vs_projected_adam_dot_rescaled": _correlation_summary(
                    rank_frame,
                    left="adam_dot",
                    right="projected_adam_dot_rescaled",
                ),
                "raw_projection_rescaled_error": _error_summary(
                    rank_frame,
                    target="raw_dot",
                    estimate="projected_raw_dot_rescaled",
                ),
                "adam_projection_rescaled_error": _error_summary(
                    rank_frame,
                    target="adam_dot",
                    estimate="projected_adam_dot_rescaled",
                ),
            },
            "blocks": {},
        }
        block_summaries: list[dict[str, Any]] = []
        for block_name, block_frame in rank_frame.groupby("block_name", sort=True):
            block_frame = block_frame.reset_index(drop=True)
            raw_abs = block_frame["raw_dot"].astype(float).abs().tolist()
            adam_abs = block_frame["adam_dot"].astype(float).abs().tolist()
            block_payload = {
                "block_name": str(block_name),
                "family": str(block_frame["family"].iloc[0]),
                "num_examples": int(len(block_frame)),
                "raw_dot": _summary_stats(block_frame["raw_dot"].astype(float).tolist()),
                "adam_dot": _summary_stats(block_frame["adam_dot"].astype(float).tolist()),
                "raw_abs_dot": _summary_stats(raw_abs),
                "adam_abs_dot": _summary_stats(adam_abs),
                "query_adam_norm_gain": _summary_stats(
                    block_frame["query_adam_norm_gain"].dropna().astype(float).tolist()
                ),
                "candidate_adam_norm_gain": _summary_stats(
                    block_frame["candidate_adam_norm_gain"].dropna().astype(float).tolist()
                ),
                "raw_dot_vs_projected_raw_dot_rescaled": _correlation_summary(
                    block_frame,
                    left="raw_dot",
                    right="projected_raw_dot_rescaled",
                ),
                "adam_dot_vs_projected_adam_dot_rescaled": _correlation_summary(
                    block_frame,
                    left="adam_dot",
                    right="projected_adam_dot_rescaled",
                ),
                "raw_projection_rescaled_error": _error_summary(
                    block_frame,
                    target="raw_dot",
                    estimate="projected_raw_dot_rescaled",
                ),
                "adam_projection_rescaled_error": _error_summary(
                    block_frame,
                    target="adam_dot",
                    estimate="projected_adam_dot_rescaled",
                ),
            }
            rank_payload["blocks"][str(block_name)] = block_payload
            block_summaries.append(block_payload)

        problem_blocks = sorted(
            block_summaries,
            key=lambda item: (
                -1.0
                if item["adam_projection_rescaled_error"]["rmse_over_target_rms"] is None
                else float(item["adam_projection_rescaled_error"]["rmse_over_target_rms"])
            ),
            reverse=True,
        )
        dominant_blocks = sorted(
            block_summaries,
            key=lambda item: (
                0.0
                if item["adam_abs_dot"] is None
                else float(item["adam_abs_dot"]["mean"])
            ),
            reverse=True,
        )
        rank_payload["top_problem_blocks_by_adam_projection_error"] = [
            {
                "block_name": item["block_name"],
                "family": item["family"],
                "adam_rmse_over_target_rms": item["adam_projection_rescaled_error"]["rmse_over_target_rms"],
                "adam_pearson": item["adam_dot_vs_projected_adam_dot_rescaled"]["pearson"],
                "mean_abs_adam_dot": None if item["adam_abs_dot"] is None else item["adam_abs_dot"]["mean"],
            }
            for item in problem_blocks[:8]
        ]
        rank_payload["top_blocks_by_mean_abs_adam_dot"] = [
            {
                "block_name": item["block_name"],
                "family": item["family"],
                "mean_abs_adam_dot": None if item["adam_abs_dot"] is None else item["adam_abs_dot"]["mean"],
                "adam_rmse_over_target_rms": item["adam_projection_rescaled_error"]["rmse_over_target_rms"],
                "adam_pearson": item["adam_dot_vs_projected_adam_dot_rescaled"]["pearson"],
            }
            for item in dominant_blocks[:8]
        ]
        by_rank[str(int(rank))] = rank_payload
    return by_rank


def run_adam_projection_diagnostic(
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

    attribution_defaults = _load_attribution_defaults(attribution_dir)
    effective_filter_spec = ewok_filter_spec
    if effective_filter_spec is None:
        effective_filter_spec = attribution_defaults.get("ewok_filter_spec")
    effective_projection_layout = str(attribution_defaults.get("projection_layout", MODULE_LAYOUT))
    if effective_projection_layout != PAPER_BLOCK_LAYOUT:
        raise ValueError(
            "Adam projection diagnostic currently expects paper_blocks artifacts; "
            f"found projection_layout={effective_projection_layout!r}"
        )

    phase_progress = _build_tqdm(
        enabled=show_progress,
        total=5,
        desc="Adam-projection diagnostic setup",
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

    layout = build_gpt2_paper_block_layout(
        {name: tuple(int(dim) for dim in grad.shape) for name, grad in raw_query_grads.items()},
        feature_dim=int(attribution_defaults.get("paper_block_features", 4096)),
    )

    checkpoint_ref = CheckpointRef(
        step=int(step),
        path=Path(base_ckpt).expanduser().resolve(),
        kind=_infer_checkpoint_kind(base_ckpt),
    )
    trackstar_config = _build_trackstar_config(
        base_ckpt=base_ckpt,
        attribution_dir=attribution_dir,
        data_dir=data_dir,
        attribution_defaults=attribution_defaults,
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

    normalizer_module_rows = _normalizer_module_rows(
        layout=layout,
        weight_normalizers=query_weight_normalizers,
    )
    normalizer_block_rows = _normalizer_block_summary(normalizer_module_rows)
    pd.DataFrame.from_records(normalizer_module_rows).to_csv(
        output_root / "normalizer_module_summary.csv",
        index=False,
    )
    pd.DataFrame.from_records(normalizer_block_rows).to_csv(
        output_root / "normalizer_block_summary.csv",
        index=False,
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("normalizer summaries")
        phase_progress.update(1)

    projected_query_by_rank: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    projection_scales_by_rank: dict[int, dict[str, float]] = {}
    for rank in ranks:
        projected_query_by_rank[int(rank)] = {
            "raw": _project_for_rank(
                raw_query_grads,
                projection_rank=int(rank),
                projection_layout=PAPER_BLOCK_LAYOUT,
            ),
            "adam": _project_for_rank(
                raw_query_grads_adam,
                projection_rank=int(rank),
                projection_layout=PAPER_BLOCK_LAYOUT,
            ),
        }
        projection_scales_by_rank[int(rank)] = _projection_scales(
            raw_query_grads,
            projection_rank=int(rank),
            projection_layout=PAPER_BLOCK_LAYOUT,
            paper_block_features=int(rank) * int(rank),
        )
    if phase_progress is not None:
        phase_progress.set_postfix_str("query projections")
        phase_progress.update(1)
        phase_progress.set_postfix_str("done")
        phase_progress.close()
        phase_progress = None

    normalizer_by_block = {str(row["block_name"]): row for row in normalizer_block_rows}
    block_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=int(num_examples_per_group) * len(GROUP_NAMES),
        desc=f"Adam-projection diagnostic ({score_mode})",
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

                raw_dot_total = float(_dot_product(raw_query_grads, raw_candidate_grads))
                adam_dot_total = float(_dot_product(raw_query_grads_adam, adam_candidate_grads))

                projected_candidate_by_rank: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
                for rank in ranks:
                    projected_candidate_by_rank[int(rank)] = {
                        "raw": _project_for_rank(
                            raw_candidate_grads,
                            projection_rank=int(rank),
                            projection_layout=PAPER_BLOCK_LAYOUT,
                        ),
                        "adam": _project_for_rank(
                            adam_candidate_grads,
                            projection_rank=int(rank),
                            projection_layout=PAPER_BLOCK_LAYOUT,
                        ),
                    }

                for rank in ranks:
                    rank_int = int(rank)
                    example_projected_raw = 0.0
                    example_projected_adam = 0.0
                    example_projected_raw_rescaled = 0.0
                    example_projected_adam_rescaled = 0.0
                    for block in layout.blocks:
                        module_names = tuple(member.module_name for member in block.members)
                        raw_dot = _module_dot(raw_query_grads, raw_candidate_grads, module_names=module_names)
                        adam_dot = _module_dot(raw_query_grads_adam, adam_candidate_grads, module_names=module_names)
                        raw_query_norm = math.sqrt(
                            max(_module_norm_sq(raw_query_grads, module_names=module_names), 0.0)
                        )
                        raw_candidate_norm = math.sqrt(
                            max(_module_norm_sq(raw_candidate_grads, module_names=module_names), 0.0)
                        )
                        adam_query_norm = math.sqrt(
                            max(_module_norm_sq(raw_query_grads_adam, module_names=module_names), 0.0)
                        )
                        adam_candidate_norm = math.sqrt(
                            max(_module_norm_sq(adam_candidate_grads, module_names=module_names), 0.0)
                        )

                        projected_raw_query = projected_query_by_rank[rank_int]["raw"][block.name]
                        projected_adam_query = projected_query_by_rank[rank_int]["adam"][block.name]
                        projected_raw_candidate = projected_candidate_by_rank[rank_int]["raw"][block.name]
                        projected_adam_candidate = projected_candidate_by_rank[rank_int]["adam"][block.name]
                        projected_raw_dot = _feature_dot(projected_raw_query, projected_raw_candidate)
                        projected_adam_dot = _feature_dot(projected_adam_query, projected_adam_candidate)
                        projection_scale = float(projection_scales_by_rank[rank_int][block.name])
                        projected_raw_dot_rescaled = projection_scale * projected_raw_dot
                        projected_adam_dot_rescaled = projection_scale * projected_adam_dot
                        projected_raw_query_norm = _feature_norm(projected_raw_query)
                        projected_raw_candidate_norm = _feature_norm(projected_raw_candidate)
                        projected_adam_query_norm = _feature_norm(projected_adam_query)
                        projected_adam_candidate_norm = _feature_norm(projected_adam_candidate)

                        example_projected_raw += projected_raw_dot
                        example_projected_adam += projected_adam_dot
                        example_projected_raw_rescaled += projected_raw_dot_rescaled
                        example_projected_adam_rescaled += projected_adam_dot_rescaled

                        normalizer_summary = normalizer_by_block.get(block.name, {})
                        block_rows.append(
                            {
                                "group": group_name,
                                "candidate_id": candidate_id,
                                "projection_rank": rank_int,
                                "block_name": block.name,
                                "family": block.family,
                                "layer_start": int(block.layer_start),
                                "layer_end": int(block.layer_end),
                                "selection_score": float(row.selection_score),
                                "raw_dot": raw_dot,
                                "raw_cosine": _cosine_from_dot(raw_dot, raw_query_norm, raw_candidate_norm),
                                "raw_query_norm": raw_query_norm,
                                "raw_candidate_norm": raw_candidate_norm,
                                "adam_dot": adam_dot,
                                "adam_cosine": _cosine_from_dot(adam_dot, adam_query_norm, adam_candidate_norm),
                                "adam_query_norm": adam_query_norm,
                                "adam_candidate_norm": adam_candidate_norm,
                                "query_adam_norm_gain": _safe_ratio(adam_query_norm, raw_query_norm),
                                "candidate_adam_norm_gain": _safe_ratio(adam_candidate_norm, raw_candidate_norm),
                                "projected_raw_dot": projected_raw_dot,
                                "projected_raw_dot_rescaled": projected_raw_dot_rescaled,
                                "projected_raw_cosine": _cosine_from_dot(
                                    projected_raw_dot,
                                    projected_raw_query_norm,
                                    projected_raw_candidate_norm,
                                ),
                                "projected_raw_query_norm": projected_raw_query_norm,
                                "projected_raw_candidate_norm": projected_raw_candidate_norm,
                                "projected_adam_dot": projected_adam_dot,
                                "projected_adam_dot_rescaled": projected_adam_dot_rescaled,
                                "projected_adam_cosine": _cosine_from_dot(
                                    projected_adam_dot,
                                    projected_adam_query_norm,
                                    projected_adam_candidate_norm,
                                ),
                                "projected_adam_query_norm": projected_adam_query_norm,
                                "projected_adam_candidate_norm": projected_adam_candidate_norm,
                                "raw_projection_rescaled_error": projected_raw_dot_rescaled - raw_dot,
                                "adam_projection_rescaled_error": projected_adam_dot_rescaled - adam_dot,
                                "projection_scale": projection_scale,
                                "candidate_train_loss": float(candidate_train_loss),
                                "shard_path": str(row.shard_path),
                                "local_example_idx": int(row.local_example_idx),
                                "token_offset_start": int(row.token_offset_start),
                                "token_offset_end": int(row.token_offset_end),
                                "normalizer_adam_scale_mean_weighted": normalizer_summary.get(
                                    "adam_scale_mean_weighted"
                                ),
                                "normalizer_adam_scale_q99_max_module": normalizer_summary.get(
                                    "adam_scale_q99_max_module"
                                ),
                                "normalizer_adam_scale_max": normalizer_summary.get("adam_scale_max"),
                            }
                        )

                    example_rows.append(
                        {
                            "group": group_name,
                            "candidate_id": candidate_id,
                            "projection_rank": rank_int,
                            "selection_score": float(row.selection_score),
                            "raw_dot": raw_dot_total,
                            "adam_dot": adam_dot_total,
                            "projected_raw_dot": example_projected_raw,
                            "projected_raw_dot_rescaled": example_projected_raw_rescaled,
                            "projected_adam_dot": example_projected_adam,
                            "projected_adam_dot_rescaled": example_projected_adam_rescaled,
                            "candidate_train_loss": float(candidate_train_loss),
                            "shard_path": str(row.shard_path),
                            "local_example_idx": int(row.local_example_idx),
                            "token_offset_start": int(row.token_offset_start),
                            "token_offset_end": int(row.token_offset_end),
                        }
                    )

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(group_name)
    finally:
        if progress is not None:
            progress.close()
        if phase_progress is not None:
            phase_progress.close()

    block_results_path = output_root / "block_results.jsonl"
    example_results_path = output_root / "example_results.jsonl"
    write_jsonl(block_results_path, block_rows)
    write_jsonl(example_results_path, example_rows)

    example_frame = pd.DataFrame.from_records(example_rows)
    block_frame = pd.DataFrame.from_records(block_rows)
    summary_payload = {
        "score_mode": score_mode,
        "target_id": target_id,
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "num_targets": int(len(bundle.items)),
        "num_examples_per_group": int(num_examples_per_group),
        "projection_layout": PAPER_BLOCK_LAYOUT,
        "projection_ranks": [int(rank) for rank in ranks],
        "paper_block_features": int(attribution_defaults.get("paper_block_features", 4096)),
        "paper_block_count": int(len(layout.blocks)),
        "module_count": int(len(raw_modules)),
        "adam_normalizer_count": int(len(query_weight_normalizers)),
        "example_correlations_by_rank": _summarize_example_rows(example_rows),
        "block_correlations_by_rank": _summarize_block_rows(block_rows),
        "normalizer_blocks": {
            str(row["block_name"]): row
            for row in normalizer_block_rows
        },
    }
    if not example_frame.empty:
        summary_payload["adam_dot_vs_raw_dot"] = {
            "pearson": _pearson_corr(
                example_frame["adam_dot"].astype(float).tolist(),
                example_frame["raw_dot"].astype(float).tolist(),
            ),
            "spearman": _spearman_corr(
                example_frame["adam_dot"].astype(float).tolist(),
                example_frame["raw_dot"].astype(float).tolist(),
            ),
        }
    if not block_frame.empty:
        summary_payload["block_adam_dot_vs_raw_dot"] = {
            "pearson": _pearson_corr(
                block_frame["adam_dot"].astype(float).tolist(),
                block_frame["raw_dot"].astype(float).tolist(),
            ),
            "spearman": _spearman_corr(
                block_frame["adam_dot"].astype(float).tolist(),
                block_frame["raw_dot"].astype(float).tolist(),
            ),
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
        "block_results": block_results_path,
        "example_results": example_results_path,
        "summary": summary_path,
        "normalizer_module_summary": output_root / "normalizer_module_summary.csv",
        "normalizer_block_summary": output_root / "normalizer_block_summary.csv",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where Adam-corrected TrackStar paper-block projections lose "
            "pre-projection dot-product geometry."
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
        help="Projection side ranks to inspect. For paper_blocks, rank 64 means 16*4096=2^16 total dims.",
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
    if args.projection_ranks is None:
        projection_ranks = DEFAULT_PROJECTION_RANKS
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
            / f"adam_projection_diagnostic_step{int(args.step):08d}_{args.score_mode}"
        )

    artifacts = run_adam_projection_diagnostic(
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
    print(f"wrote Adam-projection diagnostic artifacts under {artifacts['root']}")
    print(f"block results: {artifacts['block_results']}")
    print(f"example results: {artifacts['example_results']}")
    print(f"summary: {artifacts['summary']}")
    return 0


__all__ = [
    "build_arg_parser",
    "main",
    "run_adam_projection_diagnostic",
]
