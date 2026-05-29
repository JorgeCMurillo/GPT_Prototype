"""Build and score reusable projected TrackStar feature banks.

This module is intentionally separate from :mod:`raw_dot_rankings`: the raw
ranking script is good for small one-off checks, while this path stores the
expensive candidate projections once so later query/metric sweeps can reuse
them without another candidate-gradient pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from transformers.pytorch_utils import Conv1D as HFConv1D

from .backend import build_backend
from .config import TrackstarConfig
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
from .paper_blocks import (
    PAPER_BLOCK_LAYOUT,
    apply_weight_normalizers_and_project_paper_blocks,
    build_gpt2_paper_block_layout,
    paper_block_layout_metadata,
    project_paper_block_mapping,
    validate_paper_block_features,
)
from .raw_dot_audit import _candidate_gradient_modules, collect_candidate_grads
from .raw_dot_rankings import (
    _build_selected_bundle,
    _candidate_metadata,
    _collect_query_grad_cache,
    _common_summary_payload,
    _subset_candidates,
    _write_metric_outputs,
)
from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame
from ..common.checkpoints import CheckpointRef, build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest
from ..run_trak import RunExecutionContext


FEATURE_BANK_VERSION = 1
DEFAULT_BANKS = ("raw", "adam")
SUPPORTED_BANKS = ("raw", "adam")
SUPPORTED_STORAGE_DTYPES = ("float16", "float32")
SUPPORTED_CANDIDATE_SOURCES = ("attribution", "random_document_rows", "candidate_ids_path")
SUPPORTED_FEATURE_METRICS = (
    "projected_raw_dot",
    "projected_raw_cosine",
    "projected_adam_dot",
    "projected_adam_cosine",
    "trackstar_no_hessian",
)
ALL_FEATURE_METRICS = "all"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _normalize_banks(banks: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(str(value) for value in (banks or DEFAULT_BANKS))
    unsupported = [value for value in values if value not in SUPPORTED_BANKS]
    if unsupported:
        raise ValueError(f"Unsupported bank(s)={unsupported!r}; expected one of {SUPPORTED_BANKS!r}")
    normalized = tuple(dict.fromkeys(values))
    if not normalized:
        raise ValueError("At least one feature bank is required")
    return normalized


def _normalize_feature_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    raw_metrics = tuple(str(metric) for metric in (metrics or SUPPORTED_FEATURE_METRICS))
    expanded: list[str] = []
    for metric in raw_metrics:
        if metric == ALL_FEATURE_METRICS:
            expanded.extend(SUPPORTED_FEATURE_METRICS)
        else:
            expanded.append(metric)
    normalized = tuple(dict.fromkeys(expanded))
    unsupported = [metric for metric in normalized if metric not in SUPPORTED_FEATURE_METRICS]
    if unsupported:
        raise ValueError(
            f"Unsupported metric(s)={unsupported!r}; expected one of {SUPPORTED_FEATURE_METRICS!r}"
        )
    if not normalized:
        raise ValueError("At least one feature metric is required")
    return normalized


def _candidate_metadata_from_manifest(
    manifest,
    candidate_ids: Sequence[int],
    *,
    selection_score: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        ref = manifest.example_ref(int(candidate_id))
        rows.append(
            {
                "candidate_id": int(ref.global_example_id),
                "candidate_kind": str(ref.candidate_kind),
                "shard_path": str(ref.shard_path),
                "local_example_idx": int(ref.local_example_idx),
                "token_offset_start": int(ref.token_offset_start),
                "token_offset_end": int(ref.token_offset_end),
                "row_id": int(ref.global_example_id),
                "local_row_idx": int(ref.local_example_idx),
                "document_token_offset_start": (
                    None if ref.document_token_offset_start is None else int(ref.document_token_offset_start)
                ),
                "document_token_offset_end": (
                    None if ref.document_token_offset_end is None else int(ref.document_token_offset_end)
                ),
                "selection_score": selection_score,
            }
        )
    return rows


def _candidate_metadata_frame_from_manifest(
    manifest,
    candidate_ids: Sequence[int],
    *,
    selection_score: float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        _candidate_metadata_from_manifest(
            manifest,
            candidate_ids,
            selection_score=selection_score,
        )
    )


def _sample_random_candidate_ids(
    *,
    manifest,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("random_candidate_count must be > 0 for random_document_rows candidate source")
    if count > len(manifest):
        raise ValueError(
            f"random_candidate_count={count} exceeds manifest size {len(manifest)} for {manifest.candidate_kind}"
        )
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(manifest), size=int(count), replace=False)
    return tuple(int(value) for value in np.sort(chosen))


def _load_candidate_ids_path(path: str | Path) -> tuple[int, ...]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"candidate_ids_path not found: {resolved}")
    if resolved.suffix == ".npy":
        values = np.load(resolved)
    else:
        values = np.asarray(
            [int(line.strip()) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()],
            dtype=np.int64,
        )
    ids = tuple(int(value) for value in np.asarray(values).reshape(-1).tolist())
    if not ids:
        raise ValueError(f"candidate_ids_path is empty: {resolved}")
    return ids


def _validate_candidate_ids(manifest, candidate_ids: Sequence[int]) -> None:
    invalid = [int(candidate_id) for candidate_id in candidate_ids if int(candidate_id) < 0 or int(candidate_id) >= len(manifest)]
    if invalid:
        preview = invalid[:10]
        raise ValueError(
            f"Candidate id(s) out of bounds for manifest size {len(manifest)}: {preview}"
            + (" ..." if len(invalid) > len(preview) else "")
        )


def _module_gradient_shapes(modules: Mapping[str, torch.nn.Module]) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in modules.items():
        if isinstance(module, HFConv1D):
            shapes[str(name)] = (int(module.nf), int(module.nx))
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"Expected a 2D weight module for {name!r}, got {getattr(weight, 'shape', None)!r}")
        shapes[str(name)] = tuple(int(dim) for dim in weight.shape)
    return shapes


def _checkpoint_ref(base_ckpt: str | Path, step: int) -> CheckpointRef:
    path = Path(base_ckpt).expanduser().resolve()
    name = path.name
    kind = "final" if "final" in name else "periodic"
    return CheckpointRef(step=int(step), path=path, kind=kind)


def _load_adam_normalizers(
    *,
    base_ckpt: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    tokenizer,
    paper_block_features: int,
    device: torch.device,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {}

    checkpoint = _checkpoint_ref(base_ckpt, int(step))
    config = TrackstarConfig(
        run_dir=checkpoint.path.parent,
        data_dir=Path(data_dir).expanduser().resolve(),
        exp_name=f"{Path(output_dir).expanduser().resolve().name}_feature_bank",
        output_dir=Path(output_dir).expanduser().resolve(),
        cache_dir=Path(output_dir).expanduser().resolve() / "cache",
        checkpoint_steps=(int(step),),
        max_candidate_rows=1,
        use_fast_jl=True,
        projection_layout=PAPER_BLOCK_LAYOUT,
        paper_block_features=int(paper_block_features),
        device=str(device),
    ).resolved()
    context = RunExecutionContext.single_process(
        backend="trackstar",
        requested_device=str(device),
        resolved_device=str(device),
    )
    backend = build_backend(
        config=config,
        model=model,
        tokenizer=tokenizer,
        execution_context=context,
    )
    if not backend._candidate_uses_adam_second_moment_correction(checkpoint):
        optimizer_state = checkpoint.path / "optimizer.pt"
        raise RuntimeError(
            "Adam feature bank requested, but TrackStar cannot load candidate-side "
            f"Adam normalizers from {optimizer_state}"
        )
    normalizers = backend._load_candidate_adam_normalizers(checkpoint)
    if not normalizers:
        raise RuntimeError("Adam feature bank requested, but no Adam normalizers were loaded")
    return normalizers


def _feature_block_offsets(block_names: Sequence[str], block_dim: int) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    cursor = 0
    for name in block_names:
        end = cursor + int(block_dim)
        offsets.append(
            {
                "name": str(name),
                "offset_start": int(cursor),
                "offset_end": int(end),
                "feature_dim": int(block_dim),
            }
        )
        cursor = end
    return offsets


def _project_features(
    grads: Mapping[str, torch.Tensor],
    *,
    layout,
    block_names: Sequence[str],
    device: torch.device,
    weight_normalizers: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    device_grads = {str(name): value.detach().to(device=device, dtype=torch.float32) for name, value in grads.items()}
    if weight_normalizers:
        projected = apply_weight_normalizers_and_project_paper_blocks(
            device_grads,
            layout=layout,
            weight_normalizers=weight_normalizers,
            projection_type="rademacher",
        )
    else:
        projected = project_paper_block_mapping(
            device_grads,
            layout=layout,
            projection_type="rademacher",
        )
    missing = [name for name in block_names if name not in projected]
    if missing:
        raise KeyError(f"Projected feature mapping is missing block(s): {missing}")
    return {str(name): projected[str(name)] for name in block_names}


def _flatten_projected_features(
    projected: Mapping[str, torch.Tensor],
    *,
    block_names: Sequence[str],
    total_dim: int,
) -> np.ndarray:
    flat = np.empty((int(total_dim),), dtype=np.float32)
    cursor = 0
    for name in block_names:
        value = projected[str(name)].detach()
        if value.ndim == 2 and int(value.shape[0]) == 1:
            value = value.squeeze(0)
        if value.ndim != 1:
            raise ValueError(f"Expected a 1D projected feature vector for block {name!r}, got {tuple(value.shape)}")
        width = int(value.numel())
        flat[cursor : cursor + width] = value.cpu().to(dtype=torch.float32).numpy()
        cursor += width
    if cursor != int(total_dim):
        raise ValueError(f"Flattened feature width mismatch: wrote {cursor}, expected {total_dim}")
    return flat


def _write_bank_manifest(
    output_root: Path,
    payload: Mapping[str, Any],
) -> Path:
    manifest_path = output_root / "feature_bank_manifest.json"
    write_json(manifest_path, dict(payload))
    return manifest_path


def build_projected_feature_bank(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    output_dir: str | Path,
    score_mode: str = DEFAULT_SCORE_MODE,
    candidate_source: str = "attribution",
    candidate_kind: str | None = None,
    candidate_ids_path: str | Path | None = None,
    random_candidate_count: int = 0,
    seq_len: int | None = None,
    max_candidates: int = 0,
    candidate_subset: str = "first",
    candidate_seed: int = 42,
    banks: Sequence[str] | None = DEFAULT_BANKS,
    paper_block_features: int = 16_384,
    storage_dtype: str = "float16",
    device: str = DEFAULT_DEVICE,
    show_progress: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build mmap-backed projected raw/Adam candidate feature banks."""

    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected one of {SUPPORTED_SCORE_MODES!r}")
    if candidate_source not in SUPPORTED_CANDIDATE_SOURCES:
        raise ValueError(
            f"Unsupported candidate_source={candidate_source!r}; expected one of {SUPPORTED_CANDIDATE_SOURCES!r}"
        )
    selected_banks = _normalize_banks(banks)
    if storage_dtype not in SUPPORTED_STORAGE_DTYPES:
        raise ValueError(f"storage_dtype must be one of {SUPPORTED_STORAGE_DTYPES!r}")
    validate_paper_block_features(int(paper_block_features))

    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Feature bank output directory already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_scored = load_candidate_score_frame(
        attribution_dir=attribution_dir,
        step=int(step),
        score_mode=score_mode,
    )
    inferred_candidate_kind = _resolve_candidate_kind(source_scored)
    resolved_candidate_kind = str(candidate_kind or inferred_candidate_kind)
    resolved_seq_len = int(seq_len) if seq_len is not None else _infer_seq_len_from_scored_frame(source_scored)
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=resolved_candidate_kind,
        seq_len=resolved_seq_len,
        show_progress=bool(show_progress and candidate_source != "attribution"),
    )
    candidate_selection_metadata: list[dict[str, Any]]
    candidate_ids_input_path = None if candidate_ids_path is None else Path(candidate_ids_path).expanduser().resolve()
    if candidate_source == "attribution":
        scored = _subset_candidates(
            source_scored,
            max_candidates=int(max_candidates),
            candidate_subset=str(candidate_subset),
            seed=int(candidate_seed),
        )
        candidate_ids = tuple(int(value) for value in scored["candidate_id"].tolist())
        candidate_selection_metadata = _candidate_metadata(scored)
    elif candidate_source == "candidate_ids_path":
        if candidate_ids_input_path is None:
            raise ValueError("candidate_ids_path is required when candidate_source='candidate_ids_path'")
        candidate_ids = _load_candidate_ids_path(candidate_ids_input_path)
        if max_candidates > 0 and len(candidate_ids) > int(max_candidates):
            candidate_ids = candidate_ids[: int(max_candidates)]
        candidate_selection_metadata = _candidate_metadata_from_manifest(
            manifest,
            candidate_ids,
            selection_score=None,
        )
    else:
        count = int(random_candidate_count or max_candidates)
        candidate_ids = _sample_random_candidate_ids(
            manifest=manifest,
            count=count,
            seed=int(candidate_seed),
        )
        candidate_selection_metadata = _candidate_metadata_from_manifest(
            manifest,
            candidate_ids,
            selection_score=None,
        )
    if not candidate_ids:
        raise ValueError("No candidates selected for projected feature bank")
    _validate_candidate_ids(manifest, candidate_ids)
    dataset = FiniteTrainingExampleDataset(manifest, candidate_ids)

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    modules = _candidate_gradient_modules(model)
    layout = build_gpt2_paper_block_layout(
        _module_gradient_shapes(modules),
        feature_dim=int(paper_block_features),
    )
    block_names = tuple(block.name for block in layout.blocks)
    total_dim = int(len(block_names) * int(paper_block_features))
    block_offsets = _feature_block_offsets(block_names, int(paper_block_features))

    adam_normalizers = _load_adam_normalizers(
        base_ckpt=base_ckpt,
        data_dir=data_dir,
        output_dir=output_root,
        step=int(step),
        model=model,
        tokenizer=tokenizer,
        paper_block_features=int(paper_block_features),
        device=model_device,
        required="adam" in selected_banks,
    )

    np_dtype = np.dtype(storage_dtype)
    bank_paths: dict[str, Path] = {}
    norm_paths: dict[str, Path] = {}
    bank_arrays: dict[str, np.memmap] = {}
    norm_arrays: dict[str, np.memmap] = {}
    for bank in selected_banks:
        bank_path = output_root / f"{bank}_features.npy"
        norm_path = output_root / f"{bank}_feature_norms.npy"
        bank_arrays[bank] = np.lib.format.open_memmap(
            bank_path,
            mode="w+",
            dtype=np_dtype,
            shape=(len(candidate_ids), total_dim),
        )
        norm_arrays[bank] = np.lib.format.open_memmap(
            norm_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(candidate_ids),),
        )
        bank_paths[bank] = bank_path
        norm_paths[bank] = norm_path

    candidate_ids_output_path = output_root / f"candidate_ids_step{int(step):08d}.npy"
    np.save(candidate_ids_output_path, np.asarray(candidate_ids, dtype=np.int64))
    candidate_metadata_path = output_root / f"candidate_metadata_step{int(step):08d}.jsonl"
    write_jsonl(candidate_metadata_path, candidate_selection_metadata)
    write_json(output_root / "paper_block_layout.json", paper_block_layout_metadata(layout))

    progress = _build_tqdm(
        enabled=show_progress,
        total=len(candidate_ids),
        desc="Projected feature bank candidates",
        unit="candidate",
    )
    try:
        for candidate_index, _candidate_id in enumerate(candidate_ids):
            sample = dataset[candidate_index]
            raw_grads, _loss = collect_candidate_grads(
                model,
                modules=modules,
                input_ids=sample["input_ids"].unsqueeze(0),
                labels=sample["labels"].unsqueeze(0),
                device=model_device,
            )
            if "raw" in selected_banks:
                raw_projected = _project_features(
                    raw_grads,
                    layout=layout,
                    block_names=block_names,
                    device=model_device,
                    weight_normalizers=None,
                )
                raw_flat = _flatten_projected_features(
                    raw_projected,
                    block_names=block_names,
                    total_dim=total_dim,
                )
                bank_arrays["raw"][candidate_index, :] = raw_flat.astype(np_dtype, copy=False)
                norm_arrays["raw"][candidate_index] = float(np.linalg.norm(raw_flat.astype(np.float64)))
            if "adam" in selected_banks:
                adam_projected = _project_features(
                    raw_grads,
                    layout=layout,
                    block_names=block_names,
                    device=model_device,
                    weight_normalizers=adam_normalizers,
                )
                adam_flat = _flatten_projected_features(
                    adam_projected,
                    block_names=block_names,
                    total_dim=total_dim,
                )
                bank_arrays["adam"][candidate_index, :] = adam_flat.astype(np_dtype, copy=False)
                norm_arrays["adam"][candidate_index] = float(np.linalg.norm(adam_flat.astype(np.float64)))
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    for array in list(bank_arrays.values()) + list(norm_arrays.values()):
        array.flush()

    manifest_payload = {
        "format_version": FEATURE_BANK_VERSION,
        "base_ckpt": str(Path(base_ckpt).expanduser().resolve()),
        "source_attribution_dir": str(Path(attribution_dir).expanduser().resolve()),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "step": int(step),
        "score_mode_source": str(score_mode),
        "candidate_kind": str(resolved_candidate_kind),
        "seq_len": int(resolved_seq_len),
        "candidate_count": int(len(candidate_ids)),
        "candidate_ids_path": str(candidate_ids_output_path),
        "candidate_metadata_path": str(candidate_metadata_path),
        "candidate_source": str(candidate_source),
        "source_candidate_count": int(len(source_scored)),
        "candidate_ids_input_path": None if candidate_ids_input_path is None else str(candidate_ids_input_path),
        "random_candidate_count": int(random_candidate_count),
        "candidate_subset": str(candidate_subset),
        "candidate_seed": int(candidate_seed),
        "max_candidates": int(max_candidates),
        "projection_layout": PAPER_BLOCK_LAYOUT,
        "projection_type": "rademacher",
        "paper_block_features": int(paper_block_features),
        "paper_block_side": int(validate_paper_block_features(int(paper_block_features))),
        "feature_dim_total": int(total_dim),
        "block_offsets": block_offsets,
        "storage_dtype": str(np_dtype),
        "banks": {
            bank: {
                "features_path": str(bank_paths[bank]),
                "norms_path": str(norm_paths[bank]),
                "dtype": str(np_dtype),
                "shape": [int(len(candidate_ids)), int(total_dim)],
            }
            for bank in selected_banks
        },
        "adam_normalizer_count": int(len(adam_normalizers)),
    }
    manifest_path = _write_bank_manifest(output_root, manifest_payload)
    return {
        "manifest": manifest_path,
        "candidate_ids": candidate_ids_output_path,
        "candidate_metadata": candidate_metadata_path,
        **{f"{bank}_features": bank_paths[bank] for bank in selected_banks},
        **{f"{bank}_norms": norm_paths[bank] for bank in selected_banks},
    }


def _load_bank_array(bank_payload: Mapping[str, Any]) -> np.memmap:
    path = Path(str(bank_payload["features_path"])).expanduser().resolve()
    return np.load(path, mmap_mode="r")


def _load_norm_array(bank_payload: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(bank_payload["norms_path"])).expanduser().resolve()
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


def _query_features_for_banks(
    *,
    query_grads: Sequence[Mapping[str, torch.Tensor]],
    layout,
    block_names: Sequence[str],
    total_dim: int,
    device: torch.device,
    needs_raw: bool,
    needs_adam: bool,
    adam_normalizers: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    if needs_raw:
        rows = []
        for grads in query_grads:
            projected = _project_features(
                grads,
                layout=layout,
                block_names=block_names,
                device=device,
                weight_normalizers=None,
            )
            rows.append(_flatten_projected_features(projected, block_names=block_names, total_dim=total_dim))
        features["raw"] = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if needs_adam:
        if not adam_normalizers:
            raise RuntimeError("Adam-scored feature metrics require Adam normalizers")
        rows = []
        for grads in query_grads:
            projected = _project_features(
                grads,
                layout=layout,
                block_names=block_names,
                device=device,
                weight_normalizers=adam_normalizers,
            )
            rows.append(_flatten_projected_features(projected, block_names=block_names, total_dim=total_dim))
        features["adam"] = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return features


def _score_feature_matrix(
    *,
    candidate_features: np.ndarray,
    candidate_norms: np.ndarray,
    query_features: np.ndarray,
    cosine: bool,
    chunk_size: int,
    show_progress: bool,
    desc: str,
) -> np.ndarray:
    num_targets = int(query_features.shape[0])
    num_candidates = int(candidate_features.shape[0])
    scores = np.zeros((num_targets, num_candidates), dtype=np.float32)
    query = np.asarray(query_features, dtype=np.float32)
    query_norms = np.linalg.norm(query.astype(np.float64), axis=1).astype(np.float32)
    progress = _build_tqdm(
        enabled=show_progress,
        total=max(1, (num_candidates + int(chunk_size) - 1) // int(chunk_size)),
        desc=desc,
        unit="chunk",
    )
    try:
        for start in range(0, num_candidates, int(chunk_size)):
            end = min(num_candidates, start + int(chunk_size))
            candidates = np.asarray(candidate_features[start:end], dtype=np.float32)
            block = query @ candidates.T
            if cosine:
                denom = query_norms[:, None] * np.asarray(candidate_norms[start:end], dtype=np.float32)[None, :]
                block = np.divide(block, denom, out=np.zeros_like(block), where=denom > 0.0)
            scores[:, start:end] = block.astype(np.float32, copy=False)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return scores


def _cosine_scores_from_dot_scores(
    dot_scores: np.ndarray,
    *,
    query_features: np.ndarray,
    candidate_norms: np.ndarray,
) -> np.ndarray:
    query_norms = np.linalg.norm(np.asarray(query_features, dtype=np.float64), axis=1).astype(np.float32)
    denom = query_norms[:, None] * np.asarray(candidate_norms, dtype=np.float32)[None, :]
    return np.divide(
        np.asarray(dot_scores, dtype=np.float32),
        denom,
        out=np.zeros_like(dot_scores, dtype=np.float32),
        where=denom > 0.0,
    )


def score_projected_feature_bank(
    *,
    feature_bank_dir: str | Path,
    output_dir: str | Path,
    target_ids: Sequence[str] = (),
    query_selection: str = "mixed",
    num_queries: int = 10,
    topk: int = 5,
    bottomk: int = 0,
    metrics: Sequence[str] | None = (ALL_FEATURE_METRICS,),
    write_dense_scores: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    target_batch_size: int = DEFAULT_TARGET_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    chunk_size: int = 128,
    show_progress: bool = True,
) -> dict[str, Path | None]:
    """Score selected EWoK queries against a previously built feature bank."""

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bank_root = Path(feature_bank_dir).expanduser().resolve()
    manifest_path = bank_root / "feature_bank_manifest.json"
    manifest_payload = _read_json(manifest_path)
    if int(manifest_payload.get("format_version", 0)) != FEATURE_BANK_VERSION:
        raise ValueError(f"Unsupported feature bank version in {manifest_path}")
    if str(manifest_payload.get("projection_layout")) != PAPER_BLOCK_LAYOUT:
        raise ValueError("Only paper_blocks feature banks are supported")

    selected_metrics = _normalize_feature_metrics(metrics)
    needs_raw = any(metric.startswith("projected_raw_") for metric in selected_metrics)
    needs_adam = any(metric.startswith("projected_adam_") or metric == "trackstar_no_hessian" for metric in selected_metrics)
    banks = dict(manifest_payload.get("banks", {}))
    if needs_raw and "raw" not in banks:
        raise FileNotFoundError("Requested raw metrics, but this feature bank has no `raw` bank")
    if needs_adam and "adam" not in banks:
        raise FileNotFoundError("Requested Adam/TrackStar metrics, but this feature bank has no `adam` bank")

    base_ckpt = Path(str(manifest_payload["base_ckpt"])).expanduser().resolve()
    attribution_dir = Path(str(manifest_payload["source_attribution_dir"])).expanduser().resolve()
    data_dir = Path(str(manifest_payload["data_dir"])).expanduser().resolve()
    step = int(manifest_payload["step"])
    candidate_ids = tuple(int(value) for value in np.load(Path(str(manifest_payload["candidate_ids_path"]))))
    candidate_metadata = pd.DataFrame.from_records(_load_jsonl(str(manifest_payload["candidate_metadata_path"])))

    candidate_kind = str(manifest_payload["candidate_kind"])
    seq_len = int(manifest_payload["seq_len"])
    example_manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=candidate_kind,
        seq_len=seq_len,
    )

    bundle, diagnostics, selected_ids = _build_selected_bundle(
        attribution_dir=attribution_dir,
        step=step,
        target_ids=tuple(target_ids),
        query_selection=query_selection,
        num_queries=int(num_queries),
    )
    if not bundle.items:
        raise ValueError("No EWoK targets selected for feature-bank scoring")

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    modules = _candidate_gradient_modules(model)
    paper_block_features = int(manifest_payload["paper_block_features"])
    layout = build_gpt2_paper_block_layout(
        _module_gradient_shapes(modules),
        feature_dim=paper_block_features,
    )
    block_names = tuple(block.name for block in layout.blocks)
    total_dim = int(manifest_payload["feature_dim_total"])
    if total_dim != len(block_names) * paper_block_features:
        raise ValueError(
            "Feature bank dimensionality does not match the live model layout: "
            f"{total_dim} vs {len(block_names) * paper_block_features}"
        )

    adam_normalizers = _load_adam_normalizers(
        base_ckpt=base_ckpt,
        data_dir=data_dir,
        output_dir=output_root,
        step=step,
        model=model,
        tokenizer=tokenizer,
        paper_block_features=paper_block_features,
        device=model_device,
        required=needs_adam,
    )

    query_grads = _collect_query_grad_cache(
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        modules=modules,
        temperature=float(temperature),
        target_batch_size=int(target_batch_size),
        show_progress=show_progress,
    )
    query_features = _query_features_for_banks(
        query_grads=query_grads,
        layout=layout,
        block_names=block_names,
        total_dim=total_dim,
        device=model_device,
        needs_raw=needs_raw,
        needs_adam=needs_adam,
        adam_normalizers=adam_normalizers,
    )

    bank_arrays: dict[str, np.memmap] = {}
    norm_arrays: dict[str, np.ndarray] = {}
    if needs_raw:
        bank_arrays["raw"] = _load_bank_array(banks["raw"])
        norm_arrays["raw"] = _load_norm_array(banks["raw"])
    if needs_adam:
        bank_arrays["adam"] = _load_bank_array(banks["adam"])
        norm_arrays["adam"] = _load_norm_array(banks["adam"])

    score_matrices: dict[str, np.ndarray] = {}
    bank_score_cache: dict[str, dict[str, np.ndarray]] = {}
    for bank_name in ("raw", "adam"):
        bank_metrics = [
            metric
            for metric in selected_metrics
            if (
                (bank_name == "raw" and metric.startswith("projected_raw_"))
                or (
                    bank_name == "adam"
                    and (metric.startswith("projected_adam_") or metric == "trackstar_no_hessian")
                )
            )
        ]
        if not bank_metrics:
            continue
        dot_scores = _score_feature_matrix(
            candidate_features=bank_arrays[bank_name],
            candidate_norms=norm_arrays[bank_name],
            query_features=query_features[bank_name],
            cosine=False,
            chunk_size=int(chunk_size),
            show_progress=show_progress,
            desc=f"{bank_name} feature-bank dot scoring",
        )
        bank_score_cache[bank_name] = {"dot": dot_scores}
        if any(metric.endswith("_cosine") or metric == "trackstar_no_hessian" for metric in bank_metrics):
            bank_score_cache[bank_name]["cosine"] = _cosine_scores_from_dot_scores(
                dot_scores,
                query_features=query_features[bank_name],
                candidate_norms=norm_arrays[bank_name],
            )

    for metric in selected_metrics:
        if metric == "projected_raw_dot":
            score_matrices[metric] = bank_score_cache["raw"]["dot"]
        elif metric == "projected_raw_cosine":
            score_matrices[metric] = bank_score_cache["raw"]["cosine"]
        elif metric == "projected_adam_dot":
            score_matrices[metric] = bank_score_cache["adam"]["dot"]
        elif metric in {"projected_adam_cosine", "trackstar_no_hessian"}:
            score_matrices[metric] = bank_score_cache["adam"]["cosine"]
        else:  # pragma: no cover - guarded by _normalize_feature_metrics
            raise ValueError(f"Unsupported feature-bank metric: {metric!r}")

    common_summary = _common_summary_payload(
        base_ckpt=base_ckpt,
        attribution_dir=attribution_dir,
        data_dir=data_dir,
        step=step,
        score_mode=str(manifest_payload["score_mode_source"]),
        candidate_kind=candidate_kind,
        seq_len=seq_len,
        candidate_ids=candidate_ids,
        bundle=bundle,
        selected_ids=selected_ids,
        query_selection=query_selection,
        num_queries=int(num_queries),
        max_candidates=int(manifest_payload.get("max_candidates", 0)),
        candidate_subset=str(manifest_payload.get("candidate_subset", "feature_bank")),
        topk=int(topk),
        bottomk=int(bottomk),
        projection_settings={
            "projection_layout": PAPER_BLOCK_LAYOUT,
            "projection_dim": None,
            "paper_block_features": paper_block_features,
            "feature_dim_total": total_dim,
            "feature_bank_manifest": str(manifest_path),
        },
        write_dense_scores=bool(write_dense_scores),
        temperature=float(temperature),
        target_batch_size=int(target_batch_size),
        model_device=model_device,
        modules=modules,
    )

    metric_paths: dict[str, dict[str, Path | None]] = {}
    single_metric = len(selected_metrics) == 1
    for metric in selected_metrics:
        metric_output_dir = output_root if single_metric else output_root / metric
        metric_paths[metric] = _write_metric_outputs(
            metric_output_dir=metric_output_dir,
            metric=metric,
            score_matrix=score_matrices[metric],
            base_ckpt=base_ckpt,
            step=step,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            diagnostics=diagnostics,
            bundle=bundle,
            manifest=example_manifest,
            scored=candidate_metadata,
            topk=int(topk),
            bottomk=int(bottomk),
            write_dense_scores=bool(write_dense_scores),
            common_summary={
                **dict(common_summary),
                "feature_bank_dir": str(bank_root),
            },
        )

    if not single_metric:
        write_json(
            output_root / "feature_bank_score_summary.json",
            {
                **dict(common_summary),
                "feature_bank_dir": str(bank_root),
                "metrics": list(selected_metrics),
                "metric_dirs": {metric: str(output_root / metric) for metric in selected_metrics},
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
    return metric_paths[selected_metrics[0]]


def _add_build_args(subparsers) -> None:
    parser = subparsers.add_parser("build", help="Build projected candidate feature banks")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--attribution_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_mode", choices=SUPPORTED_SCORE_MODES, default=DEFAULT_SCORE_MODE)
    parser.add_argument(
        "--candidate_source",
        choices=SUPPORTED_CANDIDATE_SOURCES,
        default="attribution",
        help=(
            "`attribution` reuses candidate ids from --attribution_dir; "
            "`random_document_rows` samples candidate ids from the training manifest; "
            "`candidate_ids_path` loads explicit ids from .npy or newline text."
        ),
    )
    parser.add_argument("--candidate_kind", default=None)
    parser.add_argument("--candidate_ids_path", default=None)
    parser.add_argument("--random_candidate_count", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--candidate_subset", choices=("first", "top_selection", "random"), default="first")
    parser.add_argument("--candidate_seed", type=int, default=42)
    parser.add_argument("--banks", nargs="+", choices=SUPPORTED_BANKS, default=list(DEFAULT_BANKS))
    parser.add_argument("--paper_block_features", type=int, default=16_384)
    parser.add_argument("--storage_dtype", choices=SUPPORTED_STORAGE_DTYPES, default="float16")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(command="build")


def _add_score_args(subparsers) -> None:
    parser = subparsers.add_parser("score", help="Score queries against projected feature banks")
    parser.add_argument("--feature_bank_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_ids", nargs="*", default=())
    parser.add_argument(
        "--query_selection",
        choices=("mixed", "lowest_margin", "highest_margin", "median_margin", "all"),
        default="mixed",
    )
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--bottomk", type=int, default=0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_FEATURE_METRICS + (ALL_FEATURE_METRICS,),
        default=[ALL_FEATURE_METRICS],
    )
    parser.add_argument("--no_dense_scores", action="store_false", dest="write_dense_scores")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--target_batch_size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    parser.set_defaults(command="score")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build_args(subparsers)
    _add_score_args(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "build":
        paths = build_projected_feature_bank(
            base_ckpt=args.base_ckpt,
            attribution_dir=args.attribution_dir,
            data_dir=args.data_dir,
            step=int(args.step),
            output_dir=args.output_dir,
            score_mode=str(args.score_mode),
            candidate_source=str(args.candidate_source),
            candidate_kind=args.candidate_kind,
            candidate_ids_path=args.candidate_ids_path,
            random_candidate_count=int(args.random_candidate_count),
            seq_len=args.seq_len,
            max_candidates=int(args.max_candidates),
            candidate_subset=str(args.candidate_subset),
            candidate_seed=int(args.candidate_seed),
            banks=tuple(str(value) for value in args.banks),
            paper_block_features=int(args.paper_block_features),
            storage_dtype=str(args.storage_dtype),
            device=str(args.device),
            show_progress=bool(args.show_progress),
            overwrite=bool(args.overwrite),
        )
        print(f"wrote projected feature bank under {Path(args.output_dir).expanduser().resolve()}")
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    if args.command == "score":
        paths = score_projected_feature_bank(
            feature_bank_dir=args.feature_bank_dir,
            output_dir=args.output_dir,
            target_ids=tuple(args.target_ids),
            query_selection=str(args.query_selection),
            num_queries=int(args.num_queries),
            topk=int(args.topk),
            bottomk=int(args.bottomk),
            metrics=tuple(str(value) for value in args.metrics),
            write_dense_scores=bool(args.write_dense_scores),
            temperature=float(args.temperature),
            target_batch_size=int(args.target_batch_size),
            device=str(args.device),
            chunk_size=int(args.chunk_size),
            show_progress=bool(args.show_progress),
        )
        print(f"wrote feature-bank score artifacts under {Path(args.output_dir).expanduser().resolve()}")
        for name, path in paths.items():
            if path is not None:
                print(f"{name}: {path}")
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
