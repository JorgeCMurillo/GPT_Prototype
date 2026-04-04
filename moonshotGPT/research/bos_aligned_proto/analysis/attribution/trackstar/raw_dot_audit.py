"""Direct raw-gradient audit for TrackStar attribution scores.

This module helps answer a sharper debugging question than the pooled CPT
ablation:

At a fixed checkpoint, do the exported attribution scores line up with the raw
first-order signal ``-<grad L_Q, grad loss_x>`` on the same parameter slice?
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from transformers.pytorch_utils import Conv1D as HFConv1D

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
    _resolve_bos_token_id,
    _resolve_candidate_kind,
    _resolve_device,
    build_candidate_groups,
    compute_candidate_batch_loss,
    compute_query_loss_mean,
    measure_one_step_delta,
)
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame
from ..common.checkpoints import build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.ewok_targets import build_ewok_targets, iter_target_batches, score_target_batch
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest


DEFAULT_NUM_EXAMPLES_PER_GROUP = 64
GROUP_NAMES = ("top", "matched_random", "bottom")
SUPPORTED_BERGSON_MODULES = (
    torch.nn.Linear,
    HFConv1D,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
)


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "base_model", model)


def _candidate_gradient_modules(model: torch.nn.Module) -> OrderedDict[str, torch.nn.Module]:
    modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for name, module in _base_model(model).named_modules():
        if not name:
            continue
        if not isinstance(module, SUPPORTED_BERGSON_MODULES):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        modules[str(name)] = module
    if not modules:
        raise ValueError("Model does not expose any Bergson-supported 2D weight modules")
    return modules


def _normalize_module_weight_grad(module: torch.nn.Module, grad: torch.Tensor) -> torch.Tensor:
    if not isinstance(grad, torch.Tensor) or grad.ndim != 2:
        raise RuntimeError(f"Expected a 2D weight gradient, got {type(grad)!r} with shape {getattr(grad, 'shape', None)}")
    if isinstance(module, HFConv1D):
        return grad.mT
    return grad


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _collect_current_module_grads(
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        grad = getattr(module, "weight").grad
        if grad is None:
            raise RuntimeError(f"Missing gradient for module {name!r}")
        grads[name] = _normalize_module_weight_grad(module, grad.detach()).cpu().to(dtype=torch.float32)
    return grads


def _dot_product(
    query_grads: Mapping[str, torch.Tensor],
    candidate_grads: Mapping[str, torch.Tensor],
) -> float:
    dot = 0.0
    for name, query_grad in query_grads.items():
        candidate_grad = candidate_grads.get(name)
        if candidate_grad is None:
            raise KeyError(f"Candidate gradients are missing module {name!r}")
        dot += float(torch.sum(query_grad * candidate_grad).item())
    return dot


def _norm_sq(grads: Mapping[str, torch.Tensor]) -> float:
    value = 0.0
    for grad in grads.values():
        value += float(torch.sum(grad * grad).item())
    return value


def _cosine_similarity(
    query_grads: Mapping[str, torch.Tensor],
    candidate_grads: Mapping[str, torch.Tensor],
) -> float:
    dot = _dot_product(query_grads, candidate_grads)
    q_norm = math.sqrt(max(_norm_sq(query_grads), 0.0))
    c_norm = math.sqrt(max(_norm_sq(candidate_grads), 0.0))
    if q_norm <= 0.0 or c_norm <= 0.0:
        return 0.0
    return float(dot / (q_norm * c_norm))


def collect_mean_query_grads(
    model: torch.nn.Module,
    tokenizer,
    bundle,
    *,
    modules: Mapping[str, torch.nn.Module],
    temperature: float,
    batch_size: int,
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    bos_token_id = _resolve_bos_token_id(tokenizer)
    total_targets = int(len(bundle.items))
    if total_targets <= 0:
        raise ValueError("Target bundle is empty")

    progress = _build_tqdm(
        enabled=show_progress,
        total=max(1, math.ceil(total_targets / int(batch_size))),
        desc="Raw query gradients",
        unit="batch",
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    try:
        for prepared in iter_target_batches(bundle, tokenizer, int(batch_size)):
            scores = score_target_batch(
                model,
                prepared.batch,
                score_view=bundle.score_view,
                score_reduction=bundle.score_reduction,
                temperature=float(temperature),
                bos_token_id=bos_token_id,
            )
            loss = scores["softplus_loss"].sum() / float(total_targets)
            loss.backward()
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    grads = _collect_current_module_grads(modules)
    model.zero_grad(set_to_none=True)
    return grads


def collect_candidate_grads(
    model: torch.nn.Module,
    *,
    modules: Mapping[str, torch.nn.Module],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    model.train()
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    loss = compute_candidate_batch_loss(model, input_ids=input_ids, labels=labels)
    loss.backward()
    grads = _collect_current_module_grads(modules)
    model.zero_grad(set_to_none=True)
    return grads, float(loss.detach().item())


def _pearson_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_rank = pd.Series(list(x), dtype=float).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(list(y), dtype=float).rank(method="average").to_numpy(dtype=np.float64)
    if np.allclose(x_rank.std(), 0.0) or np.allclose(y_rank.std(), 0.0):
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {"num_examples": 0}

    summary: dict[str, Any] = {"num_examples": int(len(frame))}
    for column in (
        "selection_score",
        "raw_dot",
        "raw_cosine",
        "predicted_delta_q",
        "actual_delta_q",
        "candidate_train_loss",
    ):
        values = frame[column].astype(float)
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary["fraction_actual_negative_delta"] = float((frame["actual_delta_q"].astype(float) < 0.0).mean())
    return summary


def run_raw_dot_audit(
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
    for name, group in groups.items():
        if len(group.frame) < int(num_examples_per_group):
            raise ValueError(
                f"Group {name!r} only has {len(group.frame)} rows, cannot sample {num_examples_per_group}"
            )
        sampled_indices = rng.choice(len(group.frame), size=int(num_examples_per_group), replace=False)
        sampled_frames[name] = group.frame.iloc[sampled_indices].reset_index(drop=True).copy()
        sampled_frames[name].to_csv(output_root / f"sampled_{name}_candidates.csv", index=False)

    phase_progress = _build_tqdm(
        enabled=show_progress,
        total=4,
        desc="Raw-dot audit setup",
        unit="phase",
    )

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    base_state = _clone_state_dict_to_cpu(model)
    modules = _candidate_gradient_modules(model)
    if phase_progress is not None:
        phase_progress.set_postfix_str("model + modules")
        phase_progress.update(1)

    bundle = build_ewok_targets(
        score_view=str(ewok_score_view),
        target_scope="overall",
        score_reduction=str(score_reduction),
        variant=str(ewok_variant),
        filter_spec_path=ewok_filter_spec,
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
    query_grads = collect_mean_query_grads(
        model,
        tokenizer,
        bundle,
        modules=modules,
        temperature=float(temperature),
        batch_size=int(target_batch_size),
        show_progress=show_progress,
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("query grads")
        phase_progress.update(1)

    all_rows: list[dict[str, Any]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=int(num_examples_per_group) * len(GROUP_NAMES),
        desc=f"Raw-dot audit ({score_mode})",
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
                model.load_state_dict(base_state, strict=True)
                model.to(model_device)
                candidate_grads, candidate_train_loss = collect_candidate_grads(
                    model,
                    modules=modules,
                    input_ids=sample["input_ids"].unsqueeze(0),
                    labels=sample["labels"].unsqueeze(0),
                    device=model_device,
                )
                raw_dot = _dot_product(query_grads, candidate_grads)
                raw_cosine = _cosine_similarity(query_grads, candidate_grads)
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
                record = {
                    "group": group_name,
                    "candidate_id": int(row.candidate_id),
                    "selection_score": float(row.selection_score),
                    "raw_dot": float(raw_dot),
                    "raw_cosine": float(raw_cosine),
                    "predicted_delta_q": float(-float(update_lr) * raw_dot),
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
    summary_payload = {
        "baseline_query_loss": float(baseline_query_loss),
        "score_mode": score_mode,
        "target_id": target_id,
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "num_targets": int(len(bundle.items)),
        "num_examples_per_group": int(num_examples_per_group),
        "update_lr": float(update_lr),
        "module_count": int(len(modules)),
        "global_correlations": {
            "selection_vs_raw_dot_pearson": _pearson_corr(
                all_frame["selection_score"].astype(float).tolist(),
                all_frame["raw_dot"].astype(float).tolist(),
            ),
            "selection_vs_raw_dot_spearman": _spearman_corr(
                all_frame["selection_score"].astype(float).tolist(),
                all_frame["raw_dot"].astype(float).tolist(),
            ),
            "selection_vs_negative_actual_delta_pearson": _pearson_corr(
                all_frame["selection_score"].astype(float).tolist(),
                (-all_frame["actual_delta_q"].astype(float)).tolist(),
            ),
            "selection_vs_negative_actual_delta_spearman": _spearman_corr(
                all_frame["selection_score"].astype(float).tolist(),
                (-all_frame["actual_delta_q"].astype(float)).tolist(),
            ),
            "raw_dot_vs_negative_actual_delta_pearson": _pearson_corr(
                all_frame["raw_dot"].astype(float).tolist(),
                (-all_frame["actual_delta_q"].astype(float)).tolist(),
            ),
            "raw_dot_vs_negative_actual_delta_spearman": _spearman_corr(
                all_frame["raw_dot"].astype(float).tolist(),
                (-all_frame["actual_delta_q"].astype(float)).tolist(),
            ),
        },
        "groups": {
            name: _summarize_rows([row for row in all_rows if row["group"] == name])
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
            "ewok_filter_spec": None if ewok_filter_spec is None else str(Path(ewok_filter_spec).expanduser().resolve()),
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
            "Audit TrackStar scores against raw -<grad L_Q, grad loss_x> and actual tiny-step query-loss deltas "
            "for sampled single examples."
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
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(args.attribution_dir).expanduser().resolve()
            / f"raw_dot_audit_step{int(args.step):08d}_{args.score_mode}"
        )

    artifacts = run_raw_dot_audit(
        base_ckpt=args.base_ckpt,
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=int(args.step),
        output_dir=output_dir,
        score_mode=args.score_mode,
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
        device=args.device,
        show_progress=not bool(args.no_progress),
    )
    print(f"wrote raw-dot audit artifacts under {artifacts['root']}")
    print(f"results: {artifacts['results']}")
    print(f"summary: {artifacts['summary']}")
    return 0


__all__ = [
    "build_arg_parser",
    "collect_candidate_grads",
    "collect_mean_query_grads",
    "main",
    "run_raw_dot_audit",
]
