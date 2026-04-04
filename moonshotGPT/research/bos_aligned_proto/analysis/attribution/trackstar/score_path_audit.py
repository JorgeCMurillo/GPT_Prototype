"""Stage-by-stage audit for the TrackStar score path.

This helper is stricter than a CPT ablation and more diagnostic than the raw
dot audit alone. It answers:

Where does the useful first-order signal stop matching the exported TrackStar
selection score?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .bergson_datasets import BergsonCandidateDataset
from .bergson_queries import _apply_weight_normalizer, _project_query_grad, collect_query_module_grads
from .config import TrackstarConfig
from .paper_blocks import (
    PAPER_BLOCK_LAYOUT,
    apply_weight_normalizers_and_project_paper_blocks,
    build_gpt2_paper_block_layout,
    collect_query_paper_block_grads,
    project_paper_block_mapping,
)
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
    DEFAULT_UPDATE_LR,
    _build_tqdm,
    _infer_seq_len_from_scored_frame,
    _load_attribution_defaults,
    _resolve_candidate_kind,
    _resolve_device,
    build_candidate_groups,
    compute_query_loss_mean,
    measure_one_step_delta,
)
from .raw_dot_audit import (
    DEFAULT_NUM_EXAMPLES_PER_GROUP,
    GROUP_NAMES,
    _candidate_gradient_modules,
    _clone_state_dict_to_cpu,
    _collect_current_module_grads,
    _cosine_similarity,
    _dot_product,
    _pearson_corr,
    _spearman_corr,
    collect_candidate_grads,
    collect_mean_query_grads,
)
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame, resolve_step_artifacts
from ..common.candidates import CandidateSelection
from ..common.checkpoints import CheckpointRef, build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.ewok_targets import build_ewok_targets
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest
from ..run_trak import RunExecutionContext
from .backend import build_backend as build_trackstar_backend


def _infer_checkpoint_kind(base_ckpt: str | Path) -> str:
    name = Path(base_ckpt).expanduser().resolve().name
    if name.startswith("ckpt_final_"):
        return "final"
    if name.startswith("ckpt_periodic_"):
        return "periodic"
    return "periodic"


def _squeeze_feature_mapping(features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    squeezed: dict[str, torch.Tensor] = {}
    for name, value in features.items():
        tensor = value.detach().cpu().to(dtype=torch.float32)
        if tensor.ndim == 2:
            if int(tensor.shape[0]) != 1:
                raise ValueError(f"Expected one overall-query row for module {name!r}, got {tuple(tensor.shape)}")
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError(f"Expected a 1D feature vector for module {name!r}, got {tuple(tensor.shape)}")
        squeezed[str(name)] = tensor
    return squeezed


def _apply_weight_normalizers_to_grads(
    grads: Mapping[str, torch.Tensor],
    *,
    weight_normalizers: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    corrected: dict[str, torch.Tensor] = {}
    for name, grad in grads.items():
        value = grad.detach().to(dtype=torch.float32)
        normalizer = None if not weight_normalizers else weight_normalizers.get(str(name))
        if normalizer is not None:
            normalizer_device = None
            normalizer_dtype = None
            for attr in ("weight_avg_sq", "bias_avg_sq"):
                tensor = getattr(normalizer, attr, None)
                if isinstance(tensor, torch.Tensor):
                    normalizer_device = tensor.device
                    normalizer_dtype = tensor.dtype
                    break
            if normalizer_device is not None:
                target_dtype = normalizer_dtype if normalizer_dtype is not None else value.dtype
                value = value.to(device=normalizer_device, dtype=target_dtype)
        value = _apply_weight_normalizer(
            str(name),
            value,
            weight_normalizers=weight_normalizers,
        )
        corrected[str(name)] = value.detach().cpu().to(dtype=torch.float32)
    return corrected


def _project_grad_mapping(
    grads: Mapping[str, torch.Tensor],
    *,
    projection_dim: int | None,
    projection_type: str = "rademacher",
) -> dict[str, torch.Tensor]:
    projected: dict[str, torch.Tensor] = {}
    for name, grad in grads.items():
        value = grad.detach().cpu().to(dtype=torch.float32)
        if value.ndim != 2:
            raise ValueError(f"Expected a 2D module gradient for projection, got {tuple(value.shape)}")
        feature = _project_query_grad(
            str(name),
            value,
            projection_dim=projection_dim,
            projection_type=projection_type,
        )
        projected[str(name)] = feature.detach().cpu().to(dtype=torch.float32)
    return projected


def _project_grad_mapping_with_layout(
    grads: Mapping[str, torch.Tensor],
    *,
    projection_dim: int | None,
    projection_layout: str,
    paper_block_features: int,
    projection_type: str = "rademacher",
) -> dict[str, torch.Tensor]:
    if projection_layout == PAPER_BLOCK_LAYOUT:
        module_shapes = {name: tuple(int(dim) for dim in grad.shape) for name, grad in grads.items()}
        layout = build_gpt2_paper_block_layout(
            module_shapes,
            feature_dim=int(paper_block_features),
        )
        projected = project_paper_block_mapping(
            grads,
            layout=layout,
            projection_type=projection_type,
        )
        return {
            name: value.squeeze(0).detach().cpu().to(dtype=torch.float32)
            if value.ndim == 2 and int(value.shape[0]) == 1
            else value.detach().cpu().to(dtype=torch.float32)
            for name, value in projected.items()
        }
    return _project_grad_mapping(
        grads,
        projection_dim=projection_dim,
        projection_type=projection_type,
    )


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


def _summarize_rows(rows: Sequence[dict[str, Any]], *, metric_names: Sequence[str]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {"num_examples": 0}

    summary: dict[str, Any] = {"num_examples": int(len(frame))}
    for name in metric_names:
        if name not in frame.columns:
            continue
        values = frame[name].astype(float).tolist()
        stats = _summary_stats(values)
        if stats is not None:
            summary[name] = stats
    if "actual_delta_q" in frame.columns:
        summary["fraction_actual_negative_delta"] = float((frame["actual_delta_q"].astype(float) < 0.0).mean())
    return summary


def _build_trackstar_config(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    attribution_defaults: Mapping[str, Any],
    device: str,
    show_progress: bool,
) -> TrackstarConfig:
    checkpoint_path = Path(base_ckpt).expanduser().resolve()
    run_dir = checkpoint_path.parent
    root = Path(attribution_dir).expanduser().resolve()
    return TrackstarConfig(
        run_dir=run_dir,
        data_dir=Path(data_dir),
        exp_name=f"{root.name}_score_path_audit",
        output_dir=root,
        cache_dir=root / "cache",
        checkpoint_steps=(),
        candidate_strategy=str(attribution_defaults.get("candidate_strategy", "between_checkpoints")),
        candidate_from_step=attribution_defaults.get("candidate_from_step"),
        candidate_to_step=attribution_defaults.get("candidate_to_step"),
        max_candidate_rows=int(attribution_defaults.get("max_candidate_rows", 50_000)),
        candidate_seed=int(attribution_defaults.get("candidate_seed", 1337)),
        recent_window_steps=int(attribution_defaults.get("recent_window_steps", 2_000)),
        ewok_variant=str(attribution_defaults.get("ewok_variant", DEFAULT_EWOK_VARIANT)),
        ewok_filter_spec=attribution_defaults.get("ewok_filter_spec"),
        ewok_score_view=str(attribution_defaults.get("ewok_score_view", DEFAULT_SCORE_VIEW)),
        ewok_target_scope=str(attribution_defaults.get("ewok_target_scope", "overall")),
        score_reduction=str(attribution_defaults.get("score_reduction", DEFAULT_SCORE_REDUCTION)),
        temperature=float(attribution_defaults.get("temperature", DEFAULT_TEMPERATURE)),
        topk=int(attribution_defaults.get("topk", 100)),
        bottomk=int(attribution_defaults.get("bottomk", 0)),
        write_dense_scores=bool(attribution_defaults.get("write_dense_scores", True)),
        device=str(device),
        distributed="none",
        show_progress=bool(show_progress),
        batch_size=int(attribution_defaults.get("batch_size", 8)),
        proj_dim=int(attribution_defaults.get("proj_dim", 16)),
        use_fast_jl=bool(attribution_defaults.get("use_fast_jl", True)),
        projection_layout=str(attribution_defaults.get("projection_layout", "module")),
        paper_block_features=int(attribution_defaults.get("paper_block_features", 4096)),
        max_targets=int(attribution_defaults.get("max_targets", 0)),
        use_hessian_correction=bool(attribution_defaults.get("use_hessian_correction", True)),
        hessian_lambda=attribution_defaults.get("hessian_lambda"),
        hessian_target_components=int(attribution_defaults.get("hessian_target_components", 1000)),
    ).resolved()


def run_score_path_audit(
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
    update_lr: float = DEFAULT_UPDATE_LR,
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

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

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
    sampled_candidate_ids: set[int] = set()
    for name, group in groups.items():
        if len(group.frame) < int(num_examples_per_group):
            raise ValueError(
                f"Group {name!r} only has {len(group.frame)} rows, cannot sample {num_examples_per_group}"
            )
        sampled_indices = rng.choice(len(group.frame), size=int(num_examples_per_group), replace=False)
        sampled_frames[name] = group.frame.iloc[sampled_indices].reset_index(drop=True).copy()
        sampled_frames[name].to_csv(output_root / f"sampled_{name}_candidates.csv", index=False)
        sampled_candidate_ids.update(int(value) for value in sampled_frames[name]["candidate_id"].tolist())

    artifacts = resolve_step_artifacts(attribution_dir, int(step))
    dense_scores: np.ndarray | None = None
    if artifacts.dense_scores_path is not None:
        dense_scores = np.asarray(np.load(artifacts.dense_scores_path), dtype=np.float64)

    phase_progress = _build_tqdm(
        enabled=show_progress,
        total=7,
        desc="Score-path audit setup",
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

    attribution_defaults = _load_attribution_defaults(attribution_dir)
    effective_filter_spec = ewok_filter_spec
    if effective_filter_spec is None:
        effective_filter_spec = attribution_defaults.get("ewok_filter_spec")

    bundle = build_ewok_targets(
        score_view=str(ewok_score_view),
        target_scope="overall",
        score_reduction=str(score_reduction),
        variant=str(ewok_variant),
        filter_spec_path=effective_filter_spec,
        max_targets=0,
    )
    baseline_query_loss = compute_query_loss_mean(
        model,
        tokenizer,
        bundle,
        batch_size=int(target_batch_size),
        temperature=float(temperature),
        show_progress=show_progress,
        progress_desc="Baseline target loss",
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("baseline loss")
        phase_progress.update(1)

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

    ordered_candidate_ids = tuple(int(value) for value in scored["candidate_id"].astype(int).tolist())
    candidate_dataset = BergsonCandidateDataset(manifest, ordered_candidate_ids)
    index_dir, _ = backend._build_or_reuse_candidate_index(
        checkpoint=checkpoint_ref,
        candidate_dataset=candidate_dataset,
        candidate_ids=ordered_candidate_ids,
    )
    index_grads = backend._load_index_gradients(index_dir)
    if phase_progress is not None:
        phase_progress.set_postfix_str("candidate index")
        phase_progress.update(1)

    query_weight_normalizers: dict[str, Any] = {}
    if backend._candidate_uses_adam_second_moment_correction(checkpoint_ref):
        query_weight_normalizers = backend._load_candidate_adam_normalizers(checkpoint_ref)

    raw_query_grads_adam = _apply_weight_normalizers_to_grads(
        raw_query_grads,
        weight_normalizers=query_weight_normalizers,
    )
    projection_dim = int(trackstar_config.proj_dim) if trackstar_config.use_fast_jl else None
    if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT:
        paper_layout = build_gpt2_paper_block_layout(
            {name: tuple(int(dim) for dim in grad.shape) for name, grad in raw_query_grads.items()},
            feature_dim=int(trackstar_config.paper_block_features),
        )
        projected_raw_query_grads = collect_query_paper_block_grads(
            model,
            tokenizer,
            bundle,
            batch_size=int(target_batch_size),
            temperature=float(temperature),
            layout=paper_layout,
            reduction="overall",
            weight_normalizers=None,
            projection_type="rademacher",
            show_progress=False,
        )[1]
        projected_adam_query_grads = collect_query_paper_block_grads(
            model,
            tokenizer,
            bundle,
            batch_size=int(target_batch_size),
            temperature=float(temperature),
            layout=paper_layout,
            reduction="overall",
            weight_normalizers=query_weight_normalizers,
            projection_type="rademacher",
            show_progress=False,
        )[1]
    else:
        projected_raw_query_grads = collect_query_module_grads(
            model,
            tokenizer,
            bundle,
            batch_size=int(target_batch_size),
            temperature=float(temperature),
            module_names=tuple(index_grads),
            reduction="overall",
            projection_dim=projection_dim,
            projection_type="rademacher",
            weight_normalizers=None,
            show_progress=False,
        )[1]
        projected_adam_query_grads = collect_query_module_grads(
            model,
            tokenizer,
            bundle,
            batch_size=int(target_batch_size),
            temperature=float(temperature),
            module_names=tuple(index_grads),
            reduction="overall",
            projection_dim=projection_dim,
            projection_type="rademacher",
            weight_normalizers=query_weight_normalizers,
            show_progress=False,
        )[1]
    overall_no_hessian_scores = backend._score_queries_with_runtime(
        index_grads=index_grads,
        query_grads=projected_adam_query_grads,
        split_preconditioners=None,
        num_targets=1,
        num_candidates=len(ordered_candidate_ids),
    )[0]
    split_preconditioners = backend._build_mixed_hessian_preconditioners(
        index_grads=index_grads,
        query_grads=projected_adam_query_grads,
    )
    overall_hessian_scores = backend._score_queries_with_runtime(
        index_grads=index_grads,
        query_grads=projected_adam_query_grads,
        split_preconditioners=split_preconditioners,
        num_targets=1,
        num_candidates=len(ordered_candidate_ids),
    )[0]
    candidate_index_lookup = {candidate_id: idx for idx, candidate_id in enumerate(ordered_candidate_ids)}
    if phase_progress is not None:
        phase_progress.set_postfix_str("trackstar stages")
        phase_progress.update(1)

    projected_raw_query_features = _squeeze_feature_mapping(projected_raw_query_grads)
    projected_adam_query_features = _squeeze_feature_mapping(projected_adam_query_grads)

    metric_names = [
        "selection_score",
        "item_signed_sum_score",
        "item_signed_mean_score",
        "raw_dot",
        "raw_cosine",
        "adam_dot",
        "adam_cosine",
        "pooled_block_preprojection_raw_dot",
        "pooled_block_preprojection_adam_dot",
        "pooled_block_projected_raw_dot",
        "pooled_block_projected_raw_cosine",
        "pooled_block_projected_adam_dot",
        "pooled_block_projected_adam_cosine",
        "projected_raw_dot",
        "projected_raw_cosine",
        "projected_adam_dot",
        "projected_adam_cosine",
        "overall_no_hessian_score",
        "overall_hessian_score",
        "actual_delta_q",
    ]

    all_rows: list[dict[str, Any]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=int(num_examples_per_group) * len(GROUP_NAMES),
        desc=f"Score-path audit ({score_mode})",
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
                projected_raw_candidate_features = _project_grad_mapping_with_layout(
                    raw_candidate_grads,
                    projection_dim=projection_dim,
                    projection_layout=trackstar_config.projection_layout,
                    paper_block_features=int(trackstar_config.paper_block_features),
                )
                projected_adam_candidate_features = _project_grad_mapping_with_layout(
                    adam_candidate_grads,
                    projection_dim=projection_dim,
                    projection_layout=trackstar_config.projection_layout,
                    paper_block_features=int(trackstar_config.paper_block_features),
                )
                actual_metrics = measure_one_step_delta(
                    model,
                    base_state=base_state,
                    tokenizer=tokenizer,
                    bundle=bundle,
                    baseline_query_loss=baseline_query_loss,
                    input_ids=sample["input_ids"].unsqueeze(0),
                    labels=sample["labels"].unsqueeze(0),
                    update_lr=float(update_lr),
                    target_batch_size=int(target_batch_size),
                    temperature=float(temperature),
                    device=model_device,
                )

                dense_signed_sum_score = None
                dense_signed_mean_score = None
                if dense_scores is not None:
                    candidate_index = candidate_index_lookup[candidate_id]
                    dense_column = np.asarray(dense_scores[:, candidate_index], dtype=np.float64)
                    dense_signed_sum_score = float(dense_column.sum())
                    dense_signed_mean_score = float(dense_column.mean())

                candidate_index = candidate_index_lookup[candidate_id]
                record = {
                    "group": group_name,
                    "candidate_id": candidate_id,
                    "selection_score": float(row.selection_score),
                    "item_signed_sum_score": dense_signed_sum_score,
                    "item_signed_mean_score": dense_signed_mean_score,
                    "raw_dot": float(_dot_product(raw_query_grads, raw_candidate_grads)),
                    "raw_cosine": float(_cosine_similarity(raw_query_grads, raw_candidate_grads)),
                    "adam_dot": float(_dot_product(raw_query_grads_adam, adam_candidate_grads)),
                    "adam_cosine": float(_cosine_similarity(raw_query_grads_adam, adam_candidate_grads)),
                    "projected_raw_dot": float(_dot_product(projected_raw_query_features, projected_raw_candidate_features)),
                    "projected_raw_cosine": float(
                        _cosine_similarity(projected_raw_query_features, projected_raw_candidate_features)
                    ),
                    "projected_adam_dot": float(
                        _dot_product(projected_adam_query_features, projected_adam_candidate_features)
                    ),
                    "projected_adam_cosine": float(
                        _cosine_similarity(projected_adam_query_features, projected_adam_candidate_features)
                    ),
                    "pooled_block_preprojection_raw_dot": (
                        float(_dot_product(raw_query_grads, raw_candidate_grads))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "pooled_block_preprojection_adam_dot": (
                        float(_dot_product(raw_query_grads_adam, adam_candidate_grads))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "pooled_block_projected_raw_dot": (
                        float(_dot_product(projected_raw_query_features, projected_raw_candidate_features))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "pooled_block_projected_raw_cosine": (
                        float(_cosine_similarity(projected_raw_query_features, projected_raw_candidate_features))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "pooled_block_projected_adam_dot": (
                        float(_dot_product(projected_adam_query_features, projected_adam_candidate_features))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "pooled_block_projected_adam_cosine": (
                        float(_cosine_similarity(projected_adam_query_features, projected_adam_candidate_features))
                        if trackstar_config.projection_layout == PAPER_BLOCK_LAYOUT
                        else None
                    ),
                    "overall_no_hessian_score": float(overall_no_hessian_scores[candidate_index]),
                    "overall_hessian_score": float(overall_hessian_scores[candidate_index]),
                    "actual_delta_q": float(actual_metrics["delta_q"]),
                    "candidate_train_loss": float(candidate_train_loss),
                    "query_loss_before": float(actual_metrics["query_loss_before"]),
                    "query_loss_after": float(actual_metrics["query_loss_after"]),
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
            phase_progress.set_postfix_str("done")
            phase_progress.update(1)
            phase_progress.close()

    results_path = output_root / "audit_results.jsonl"
    write_jsonl(results_path, all_rows)

    all_frame = pd.DataFrame.from_records(all_rows)
    stage_metric_names = [
        "selection_score",
        "item_signed_sum_score",
        "item_signed_mean_score",
        "raw_dot",
        "raw_cosine",
        "adam_dot",
        "adam_cosine",
        "pooled_block_preprojection_raw_dot",
        "pooled_block_preprojection_adam_dot",
        "pooled_block_projected_raw_dot",
        "pooled_block_projected_raw_cosine",
        "pooled_block_projected_adam_dot",
        "pooled_block_projected_adam_cosine",
        "projected_raw_dot",
        "projected_raw_cosine",
        "projected_adam_dot",
        "projected_adam_cosine",
        "overall_no_hessian_score",
        "overall_hessian_score",
    ]
    target_values = (-all_frame["actual_delta_q"].astype(float)).tolist()
    global_correlations: dict[str, dict[str, float | None]] = {}
    for metric_name in stage_metric_names:
        if metric_name not in all_frame.columns:
            continue
        values = all_frame[metric_name]
        if values.isna().all():
            continue
        numeric = values.astype(float).tolist()
        global_correlations[metric_name] = {
            "pearson": _pearson_corr(numeric, target_values),
            "spearman": _spearman_corr(numeric, target_values),
        }

    summary_payload = {
        "baseline_query_loss": float(baseline_query_loss),
        "score_mode": score_mode,
        "target_id": target_id,
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "num_targets": int(len(bundle.items)),
        "num_examples_per_group": int(num_examples_per_group),
        "update_lr": float(update_lr),
        "projection_dim": None if projection_dim is None else int(projection_dim),
        "use_hessian_correction": bool(trackstar_config.use_hessian_correction),
        "use_fast_jl": bool(trackstar_config.use_fast_jl),
        "projection_layout": str(trackstar_config.projection_layout),
        "paper_block_features": int(trackstar_config.paper_block_features),
        "module_count": int(len(raw_modules)),
        "global_correlations": global_correlations,
        "groups": {
            name: _summarize_rows([row for row in all_rows if row["group"] == name], metric_names=metric_names)
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
            "update_lr": float(update_lr),
            "ewok_filter_spec": None if effective_filter_spec is None else str(Path(effective_filter_spec).expanduser().resolve()),
            "ewok_variant": str(ewok_variant),
            "ewok_score_view": str(ewok_score_view),
            "score_reduction": str(score_reduction),
            "temperature": float(temperature),
            "target_batch_size": int(target_batch_size),
            "seed": int(seed),
            "device": str(model_device),
            "projection_layout": str(trackstar_config.projection_layout),
            "paper_block_features": int(trackstar_config.paper_block_features),
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
            "Audit where TrackStar's score path diverges from direct tiny-step usefulness by "
            "comparing raw, corrected, projected, and final scores on sampled single examples."
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
    parser.add_argument("--update_lr", type=float, default=DEFAULT_UPDATE_LR)
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

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            Path(args.attribution_dir).expanduser().resolve()
            / f"score_path_audit_step{int(args.step):08d}_{args.score_mode}"
        )
    )

    artifacts = run_score_path_audit(
        base_ckpt=args.base_ckpt,
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=int(args.step),
        output_dir=output_dir,
        score_mode=str(args.score_mode),
        target_id=args.target_id,
        group_size=int(args.group_size),
        num_examples_per_group=int(args.num_examples_per_group),
        update_lr=float(args.update_lr),
        ewok_filter_spec=resolved_filter_spec,
        ewok_variant=str(resolved_variant),
        ewok_score_view=str(resolved_score_view),
        score_reduction=str(resolved_reduction),
        temperature=float(resolved_temperature),
        target_batch_size=int(args.target_batch_size),
        seed=int(args.seed),
        device=str(args.device),
        show_progress=not bool(args.no_progress),
    )
    print(f"wrote score-path audit artifacts under {artifacts['root']}")
    print(f"results: {artifacts['results']}")
    print(f"summary: {artifacts['summary']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
