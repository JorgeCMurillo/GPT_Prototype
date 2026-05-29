"""Export per-query top candidates ranked by exact raw-gradient geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .one_step_sanity import (
    DEFAULT_DEVICE,
    DEFAULT_SCORE_MODE,
    DEFAULT_TARGET_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    _build_tqdm,
    _infer_seq_len_from_scored_frame,
    _resolve_candidate_kind,
    _resolve_device,
)
from .per_query_top_examples import select_target_ids
from .raw_dot_audit import (
    _candidate_gradient_modules,
    _dot_product,
    _norm_sq,
    collect_candidate_grads,
    collect_mean_query_grads,
)
from .score_path_audit import _project_grad_mapping_with_layout
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame, resolve_step_artifacts
from ..common.checkpoints import build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.ewok_targets import CheckpointScores, EWOKTargetBundle, EWOKTargetItem, TargetDiagnostics
from ..common.export import export_target_items, write_checkpoint_outputs, write_json, write_jsonl
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest


DEFAULT_QUERY_SELECTION = "mixed"
DEFAULT_NUM_QUERIES = 10
DEFAULT_TOPK = 5
DEFAULT_BOTTOMK = 0
SUPPORTED_RANKING_METRICS = ("raw_dot", "raw_cosine", "projected_raw_dot", "projected_raw_cosine")
ALL_RANKING_METRICS = "all"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _build_selected_bundle(
    *,
    attribution_dir: str | Path,
    step: int,
    target_ids: Sequence[str],
    query_selection: str,
    num_queries: int,
) -> tuple[EWOKTargetBundle, tuple[TargetDiagnostics, ...], list[str]]:
    artifacts = resolve_step_artifacts(attribution_dir, int(step))
    if artifacts.target_items_path is None:
        raise FileNotFoundError(f"Missing target_items.jsonl under {Path(attribution_dir).expanduser().resolve()}")
    target_diagnostics_path = Path(attribution_dir).expanduser().resolve() / f"target_diagnostics_step{int(step):08d}.jsonl"
    if not target_diagnostics_path.exists():
        raise FileNotFoundError(f"Missing target diagnostics file: {target_diagnostics_path}")

    target_item_rows = _load_jsonl(artifacts.target_items_path)
    diagnostic_rows = _load_jsonl(target_diagnostics_path)
    diagnostics = pd.DataFrame.from_records(diagnostic_rows)
    selected_ids = select_target_ids(
        diagnostics,
        target_ids=tuple(str(value) for value in target_ids),
        query_selection=str(query_selection),
        num_queries=int(num_queries),
    )

    item_by_id = {str(row["target_id"]): EWOKTargetItem(**row) for row in target_item_rows}
    diag_by_id = {str(row["target_id"]): TargetDiagnostics(**row) for row in diagnostic_rows}
    missing_items = [target_id for target_id in selected_ids if target_id not in item_by_id]
    missing_diags = [target_id for target_id in selected_ids if target_id not in diag_by_id]
    if missing_items:
        raise ValueError(f"Selected target_id(s) missing from target_items.jsonl: {missing_items}")
    if missing_diags:
        raise ValueError(f"Selected target_id(s) missing from target diagnostics: {missing_diags}")

    items = tuple(item_by_id[target_id] for target_id in selected_ids)
    groups: dict[str, tuple[str, ...]] = {"overall": tuple(selected_ids)}
    for domain in sorted({item.domain for item in items}):
        groups[f"domain:{domain}"] = tuple(item.target_id for item in items if item.domain == domain)
    bundle = EWOKTargetBundle(
        items=items,
        groups=groups,
        source_path=Path(artifacts.target_items_path),
        score_view=items[0].score_view if items else "",
        score_reduction=diagnostic_rows[0].get("score_reduction", "mean") if diagnostic_rows else "mean",
    )
    return bundle, tuple(diag_by_id[target_id] for target_id in selected_ids), selected_ids


def _subset_candidates(
    frame: pd.DataFrame,
    *,
    max_candidates: int,
    candidate_subset: str,
    seed: int,
) -> pd.DataFrame:
    if max_candidates <= 0 or max_candidates >= len(frame):
        return frame.reset_index(drop=True).copy()
    if candidate_subset == "first":
        return frame.head(int(max_candidates)).reset_index(drop=True).copy()
    if candidate_subset == "top_selection":
        return (
            frame.sort_values(["selection_score", "candidate_id"], ascending=[False, True])
            .head(int(max_candidates))
            .sort_index()
            .reset_index(drop=True)
            .copy()
        )
    if candidate_subset == "random":
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(frame), size=int(max_candidates), replace=False)
        return frame.iloc[np.sort(chosen)].reset_index(drop=True).copy()
    raise ValueError("candidate_subset must be one of: first, top_selection, random")


def _candidate_metadata(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "candidate_id",
        "candidate_kind",
        "shard_path",
        "local_example_idx",
        "token_offset_start",
        "token_offset_end",
        "row_id",
        "local_row_idx",
        "document_token_offset_start",
        "document_token_offset_end",
        "selection_score",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].to_dict(orient="records")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_attribution_config(attribution_dir: str | Path) -> dict[str, Any]:
    config_path = Path(attribution_dir).expanduser().resolve() / "config.json"
    if not config_path.exists():
        return {}
    return _read_json(config_path)


def _resolve_projection_settings(
    *,
    attribution_dir: str | Path,
    projection_layout: str | None,
    projection_dim: int | None,
    paper_block_features: int | None,
) -> dict[str, Any]:
    config = _load_attribution_config(attribution_dir)
    resolved_layout = str(projection_layout or config.get("projection_layout") or "module")
    if resolved_layout not in {"module", "paper_blocks"}:
        raise ValueError(f"Unsupported projection_layout={resolved_layout!r}")
    resolved_dim = projection_dim
    if resolved_dim is None:
        if bool(config.get("use_fast_jl", True)):
            resolved_dim = int(config.get("proj_dim", 16))
    resolved_features = int(paper_block_features or config.get("paper_block_features", 4096))
    return {
        "projection_layout": resolved_layout,
        "projection_dim": None if resolved_dim is None else int(resolved_dim),
        "paper_block_features": int(resolved_features),
        "source_config": str(Path(attribution_dir).expanduser().resolve() / "config.json")
        if (Path(attribution_dir).expanduser().resolve() / "config.json").exists()
        else None,
    }


def _maybe_project_query_grads(
    query_grads: Sequence[Mapping[str, torch.Tensor]],
    *,
    ranking_metric: str,
    projection_settings: Mapping[str, Any],
) -> list[dict[str, torch.Tensor]]:
    if not ranking_metric.startswith("projected_"):
        return [dict(grads) for grads in query_grads]
    return [
        _project_grad_mapping_with_layout(
            grads,
            projection_dim=projection_settings["projection_dim"],
            projection_layout=str(projection_settings["projection_layout"]),
            paper_block_features=int(projection_settings["paper_block_features"]),
            projection_type="rademacher",
        )
        for grads in query_grads
    ]


def _normalize_ranking_metrics(
    *,
    ranking_metric: str | None,
    ranking_metrics: Sequence[str] | None,
) -> tuple[str, ...]:
    raw_metrics = tuple(str(metric) for metric in (ranking_metrics or (ranking_metric or "raw_dot",)))
    expanded: list[str] = []
    for metric in raw_metrics:
        if metric == ALL_RANKING_METRICS:
            expanded.extend(SUPPORTED_RANKING_METRICS)
        else:
            expanded.append(metric)
    metrics = tuple(dict.fromkeys(expanded))
    unsupported = [metric for metric in metrics if metric not in SUPPORTED_RANKING_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported ranking_metric(s)={unsupported!r}; expected one of {SUPPORTED_RANKING_METRICS}")
    if not metrics:
        raise ValueError("At least one ranking metric is required")
    return metrics


def _collect_query_grad_cache(
    *,
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    modules: Mapping[str, torch.nn.Module],
    temperature: float,
    target_batch_size: int,
    show_progress: bool,
) -> list[dict[str, torch.Tensor]]:
    query_grads: list[dict[str, torch.Tensor]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=len(bundle.items),
        desc="Raw-dot query gradients",
        unit="query",
    )
    try:
        for item in bundle.items:
            single = EWOKTargetBundle(
                items=(item,),
                groups={"overall": (item.target_id,), f"domain:{item.domain}": (item.target_id,)},
                source_path=bundle.source_path,
                score_view=bundle.score_view,
                score_reduction=bundle.score_reduction,
            )
            query_grads.append(
                collect_mean_query_grads(
                    model,
                    tokenizer,
                    single,
                    modules=modules,
                    temperature=float(temperature),
                    batch_size=int(target_batch_size),
                    show_progress=False,
                )
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return query_grads


def _score_metrics_with_cached_query_grads(
    *,
    model: torch.nn.Module,
    manifest,
    candidate_ids: Sequence[int],
    query_grads: Sequence[Mapping[str, torch.Tensor]],
    modules: Mapping[str, torch.nn.Module],
    device: torch.device,
    ranking_metrics: Sequence[str],
    projection_settings: Mapping[str, Any],
    show_progress: bool,
) -> dict[str, np.ndarray]:
    metrics = _normalize_ranking_metrics(ranking_metric=None, ranking_metrics=ranking_metrics)
    dataset = FiniteTrainingExampleDataset(manifest, tuple(int(value) for value in candidate_ids))
    score_matrices = {
        metric: np.zeros((len(query_grads), len(candidate_ids)), dtype=np.float32)
        for metric in metrics
    }
    needs_projected = any(metric.startswith("projected_") for metric in metrics)
    query_grads_by_space: dict[str, list[dict[str, torch.Tensor]]] = {
        "raw": [dict(grads) for grads in query_grads],
    }
    if needs_projected:
        query_grads_by_space["projected"] = _maybe_project_query_grads(
            query_grads,
            ranking_metric="projected_raw_dot",
            projection_settings=projection_settings,
        )
    query_norms_by_space = {
        name: np.asarray([max(float(_norm_sq(grads)), 0.0) ** 0.5 for grads in grads_list], dtype=np.float64)
        for name, grads_list in query_grads_by_space.items()
    }
    desc = "multi-metric candidate gradients" if len(metrics) > 1 else f"{metrics[0]} candidate gradients"
    progress = _build_tqdm(
        enabled=show_progress,
        total=len(candidate_ids),
        desc=desc,
        unit="candidate",
    )
    try:
        for candidate_index, _candidate_id in enumerate(candidate_ids):
            sample = dataset[candidate_index]
            candidate_grads, _loss = collect_candidate_grads(
                model,
                modules=modules,
                input_ids=sample["input_ids"].unsqueeze(0),
                labels=sample["labels"].unsqueeze(0),
                device=device,
            )
            candidate_grads_by_space: dict[str, Mapping[str, torch.Tensor]] = {"raw": candidate_grads}
            if needs_projected:
                candidate_grads_by_space["projected"] = _project_grad_mapping_with_layout(
                    candidate_grads,
                    projection_dim=projection_settings["projection_dim"],
                    projection_layout=str(projection_settings["projection_layout"]),
                    paper_block_features=int(projection_settings["paper_block_features"]),
                    projection_type="rademacher",
                )
            candidate_norms_by_space = {
                name: max(float(_norm_sq(grads)), 0.0) ** 0.5
                for name, grads in candidate_grads_by_space.items()
            }
            for metric in metrics:
                space = "projected" if metric.startswith("projected_") else "raw"
                effective_candidate_grads = candidate_grads_by_space[space]
                candidate_norm = candidate_norms_by_space[space]
                query_norms = query_norms_by_space[space]
                for target_index, query_grad in enumerate(query_grads_by_space[space]):
                    raw_dot = float(_dot_product(query_grad, effective_candidate_grads))
                    if metric in {"raw_dot", "projected_raw_dot"}:
                        score = raw_dot
                    else:
                        denom = float(query_norms[target_index] * candidate_norm)
                        score = 0.0 if denom <= 0.0 else raw_dot / denom
                    score_matrices[metric][target_index, candidate_index] = float(score)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return score_matrices


def _common_summary_payload(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    score_mode: str,
    candidate_kind: str,
    seq_len: int,
    candidate_ids: Sequence[int],
    bundle: EWOKTargetBundle,
    selected_ids: Sequence[str],
    query_selection: str,
    num_queries: int,
    max_candidates: int,
    candidate_subset: str,
    topk: int,
    bottomk: int,
    projection_settings: Mapping[str, Any],
    write_dense_scores: bool,
    temperature: float,
    target_batch_size: int,
    model_device: torch.device,
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, Any]:
    return {
        "base_ckpt": str(Path(base_ckpt).expanduser().resolve()),
        "source_attribution_dir": str(Path(attribution_dir).expanduser().resolve()),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "step": int(step),
        "score_mode_source": str(score_mode),
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "candidate_count": int(len(candidate_ids)),
        "target_count": int(len(bundle.items)),
        "target_ids": list(selected_ids),
        "query_selection": str(query_selection),
        "num_queries": int(num_queries),
        "max_candidates": int(max_candidates),
        "candidate_subset": str(candidate_subset),
        "topk": int(topk),
        "bottomk": int(bottomk),
        "projection_settings": dict(projection_settings),
        "write_dense_scores": bool(write_dense_scores),
        "temperature": float(temperature),
        "target_batch_size": int(target_batch_size),
        "device": str(model_device),
        "module_count": int(len(modules)),
        "modules": list(modules.keys()),
    }


def _write_metric_outputs(
    *,
    metric_output_dir: Path,
    metric: str,
    score_matrix: np.ndarray,
    base_ckpt: str | Path,
    step: int,
    candidate_ids: Sequence[int],
    selected_ids: Sequence[str],
    diagnostics: Sequence[TargetDiagnostics],
    bundle: EWOKTargetBundle,
    manifest,
    scored: pd.DataFrame,
    topk: int,
    bottomk: int,
    write_dense_scores: bool,
    common_summary: Mapping[str, Any],
) -> dict[str, Path | None]:
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    result = CheckpointScores(
        checkpoint_step=int(step),
        checkpoint_path=str(Path(base_ckpt).expanduser().resolve()),
        candidate_ids=tuple(int(value) for value in candidate_ids),
        target_ids=tuple(selected_ids),
        score_matrix=score_matrix,
        target_diagnostics=tuple(diagnostics),
    )
    export_target_items(bundle, metric_output_dir / "target_items.jsonl")
    paths = write_checkpoint_outputs(
        output_dir=metric_output_dir,
        result=result,
        bundle=bundle,
        manifest=manifest,
        topk=int(topk),
        bottomk=int(bottomk),
        write_dense_scores=bool(write_dense_scores),
    )
    write_jsonl(metric_output_dir / f"candidate_metadata_step{int(step):08d}.jsonl", _candidate_metadata(scored))
    write_json(
        metric_output_dir / "raw_dot_ranking_summary.json",
        {
            **dict(common_summary),
            "ranking_metric": str(metric),
            "artifacts": {name: None if path is None else str(path) for name, path in paths.items()},
        },
    )
    return paths


def run_raw_dot_rankings(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    output_dir: str | Path,
    score_mode: str = DEFAULT_SCORE_MODE,
    target_ids: Sequence[str] = (),
    query_selection: str = DEFAULT_QUERY_SELECTION,
    num_queries: int = DEFAULT_NUM_QUERIES,
    max_candidates: int = 0,
    candidate_subset: str = "first",
    candidate_seed: int = 42,
    topk: int = DEFAULT_TOPK,
    bottomk: int = DEFAULT_BOTTOMK,
    ranking_metric: str | None = "raw_dot",
    ranking_metrics: Sequence[str] | None = None,
    projection_layout: str | None = None,
    projection_dim: int | None = None,
    paper_block_features: int | None = None,
    write_dense_scores: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    target_batch_size: int = DEFAULT_TARGET_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    show_progress: bool = True,
) -> dict[str, Path | None]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected one of {SUPPORTED_SCORE_MODES!r}")
    metrics = _normalize_ranking_metrics(ranking_metric=ranking_metric, ranking_metrics=ranking_metrics)
    projection_settings = _resolve_projection_settings(
        attribution_dir=attribution_dir,
        projection_layout=projection_layout,
        projection_dim=projection_dim,
        paper_block_features=paper_block_features,
    )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scored = load_candidate_score_frame(
        attribution_dir=attribution_dir,
        step=int(step),
        score_mode=score_mode,
    )
    scored = _subset_candidates(
        scored,
        max_candidates=int(max_candidates),
        candidate_subset=str(candidate_subset),
        seed=int(candidate_seed),
    )
    candidate_ids = tuple(int(value) for value in scored["candidate_id"].tolist())
    if not candidate_ids:
        raise ValueError("No candidates selected for raw-dot ranking")

    candidate_kind = _resolve_candidate_kind(scored)
    seq_len = _infer_seq_len_from_scored_frame(scored)
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=candidate_kind,
        seq_len=seq_len,
    )

    bundle, diagnostics, selected_ids = _build_selected_bundle(
        attribution_dir=attribution_dir,
        step=int(step),
        target_ids=tuple(target_ids),
        query_selection=query_selection,
        num_queries=int(num_queries),
    )
    if not bundle.items:
        raise ValueError("No EWoK targets selected for raw-dot ranking")

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    modules = _candidate_gradient_modules(model)

    query_grads = _collect_query_grad_cache(
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        modules=modules,
        temperature=float(temperature),
        target_batch_size=int(target_batch_size),
        show_progress=show_progress,
    )
    score_matrices = _score_metrics_with_cached_query_grads(
        model=model,
        manifest=manifest,
        candidate_ids=candidate_ids,
        query_grads=query_grads,
        modules=modules,
        device=model_device,
        ranking_metrics=metrics,
        projection_settings=projection_settings,
        show_progress=show_progress,
    )

    common_summary = _common_summary_payload(
        base_ckpt=base_ckpt,
        attribution_dir=attribution_dir,
        data_dir=data_dir,
        step=int(step),
        score_mode=score_mode,
        candidate_kind=str(candidate_kind),
        seq_len=int(seq_len),
        candidate_ids=candidate_ids,
        bundle=bundle,
        selected_ids=selected_ids,
        query_selection=query_selection,
        num_queries=int(num_queries),
        max_candidates=int(max_candidates),
        candidate_subset=candidate_subset,
        topk=int(topk),
        bottomk=int(bottomk),
        projection_settings=projection_settings,
        write_dense_scores=bool(write_dense_scores),
        temperature=float(temperature),
        target_batch_size=int(target_batch_size),
        model_device=model_device,
        modules=modules,
    )
    metric_paths: dict[str, dict[str, Path | None]] = {}
    single_metric = len(metrics) == 1
    for metric in metrics:
        metric_output_dir = output_root if single_metric else output_root / metric
        metric_paths[metric] = _write_metric_outputs(
            metric_output_dir=metric_output_dir,
            metric=metric,
            score_matrix=score_matrices[metric],
            base_ckpt=base_ckpt,
            step=int(step),
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            diagnostics=diagnostics,
            bundle=bundle,
            manifest=manifest,
            scored=scored,
            topk=int(topk),
            bottomk=int(bottomk),
            write_dense_scores=bool(write_dense_scores),
            common_summary=common_summary,
        )

    if not single_metric:
        write_json(
            output_root / "multi_metric_ranking_summary.json",
            {
                **dict(common_summary),
                "ranking_metrics": list(metrics),
                "metric_dirs": {metric: str(output_root / metric) for metric in metrics},
                "artifacts": {
                    metric: {name: None if path is None else str(path) for name, path in paths.items()}
                    for metric, paths in metric_paths.items()
                },
            },
        )
        return {
            metric: (paths.get("top_rows") if isinstance(paths, dict) else None)
            for metric, paths in metric_paths.items()
        }
    return metric_paths[metrics[0]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--attribution_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_mode", choices=SUPPORTED_SCORE_MODES, default=DEFAULT_SCORE_MODE)
    parser.add_argument("--target_ids", nargs="*", default=())
    parser.add_argument(
        "--query_selection",
        choices=("mixed", "lowest_margin", "highest_margin", "median_margin", "all"),
        default=DEFAULT_QUERY_SELECTION,
    )
    parser.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--candidate_subset", choices=("first", "top_selection", "random"), default="first")
    parser.add_argument("--candidate_seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--bottomk", type=int, default=DEFAULT_BOTTOMK)
    parser.add_argument(
        "--ranking_metric",
        choices=SUPPORTED_RANKING_METRICS,
        default=None,
        help="Score candidates by raw or projected raw-gradient dot/cosine.",
    )
    parser.add_argument(
        "--ranking_metrics",
        nargs="+",
        choices=SUPPORTED_RANKING_METRICS + (ALL_RANKING_METRICS,),
        default=None,
        help=(
            "Score candidates once and export one subdirectory per metric. "
            "Use `all` for every supported metric. Overrides --ranking_metric."
        ),
    )
    parser.add_argument("--projection_layout", choices=("module", "paper_blocks"), default=None)
    parser.add_argument("--projection_dim", type=int, default=None)
    parser.add_argument("--paper_block_features", type=int, default=None)
    parser.add_argument("--no_dense_scores", action="store_false", dest="write_dense_scores")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--target_batch_size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = run_raw_dot_rankings(
        base_ckpt=args.base_ckpt,
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=int(args.step),
        output_dir=args.output_dir,
        score_mode=str(args.score_mode),
        target_ids=tuple(args.target_ids),
        query_selection=str(args.query_selection),
        num_queries=int(args.num_queries),
        max_candidates=int(args.max_candidates),
        candidate_subset=str(args.candidate_subset),
        candidate_seed=int(args.candidate_seed),
        topk=int(args.topk),
        bottomk=int(args.bottomk),
        ranking_metric=None if args.ranking_metric is None else str(args.ranking_metric),
        ranking_metrics=None if args.ranking_metrics is None else tuple(str(value) for value in args.ranking_metrics),
        projection_layout=args.projection_layout,
        projection_dim=args.projection_dim,
        paper_block_features=args.paper_block_features,
        write_dense_scores=bool(args.write_dense_scores),
        temperature=float(args.temperature),
        target_batch_size=int(args.target_batch_size),
        device=str(args.device),
        show_progress=bool(args.show_progress),
    )
    print(f"wrote raw-dot ranking artifacts under {Path(args.output_dir).expanduser().resolve()}")
    for name, path in paths.items():
        if path is not None:
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
