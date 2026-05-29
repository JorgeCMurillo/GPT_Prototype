"""Artifact export helpers for attribution runs over faithful training examples.

This module converts checkpoint-level attribution results into the concrete
files that downstream analysis consumes. It handles JSON and CSV serialization,
candidate-level and domain-level summaries, target diagnostics, and the
optional dense score dump without mixing that reporting logic into the runner
itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ewok_targets import CheckpointScores, EWOKTargetBundle, TargetDiagnostics
from .training_examples import ExampleManifest, ExampleRef


def _to_jsonable(value: Any):
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row)) + "\n")


def export_target_items(bundle: EWOKTargetBundle, path: str | Path) -> None:
    write_jsonl(path, [item.to_json() for item in bundle.items])


def _diagnostics_by_target_id(target_diagnostics: tuple[TargetDiagnostics, ...]) -> dict[str, TargetDiagnostics]:
    return {diag.target_id: diag for diag in target_diagnostics}


def _candidate_metadata(example_ref: ExampleRef) -> dict[str, Any]:
    """Return stable export metadata for one candidate example.

    We keep the legacy `row_*` aliases so existing notebooks continue to work,
    but the canonical fields are the more general `candidate_*` / `local_example_idx`
    / token-offset columns.
    """

    metadata = {
        "candidate_id": int(example_ref.global_example_id),
        "candidate_kind": str(example_ref.candidate_kind),
        "shard_path": example_ref.shard_path,
        "local_example_idx": int(example_ref.local_example_idx),
        "token_offset_start": int(example_ref.token_offset_start),
        "token_offset_end": int(example_ref.token_offset_end),
        "row_id": int(example_ref.global_example_id),
        "local_row_idx": int(example_ref.local_example_idx),
    }
    if example_ref.document_token_offset_start is not None:
        metadata["document_token_offset_start"] = int(example_ref.document_token_offset_start)
    if example_ref.document_token_offset_end is not None:
        metadata["document_token_offset_end"] = int(example_ref.document_token_offset_end)
    return metadata


def build_top_rows_frame(
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
    *,
    topk: int,
) -> pd.DataFrame:
    return _build_ranked_rows_frame(
        result,
        bundle,
        manifest,
        k=topk,
        descending=True,
    )


def build_bottom_rows_frame(
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
    *,
    bottomk: int,
) -> pd.DataFrame:
    return _build_ranked_rows_frame(
        result,
        bundle,
        manifest,
        k=bottomk,
        descending=False,
    )


def _build_ranked_rows_frame(
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
    *,
    k: int,
    descending: bool,
) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame()

    diagnostics = _diagnostics_by_target_id(result.target_diagnostics)
    records: list[dict[str, Any]] = []
    candidate_ids = np.asarray(result.candidate_ids, dtype=np.int64)

    for target_idx, item in enumerate(bundle.items):
        target_scores = result.score_matrix[target_idx]
        ranked_indices = np.argsort(target_scores)
        if descending:
            ranked_indices = ranked_indices[::-1]
        top_indices = ranked_indices[:k]
        diag = diagnostics[item.target_id]
        for rank, local_idx in enumerate(top_indices, start=1):
            example_id = int(candidate_ids[int(local_idx)])
            example_ref = manifest.example_ref(example_id)
            records.append(
                {
                    "checkpoint_step": result.checkpoint_step,
                    "target_id": item.target_id,
                    "domain": item.domain,
                    "score": float(target_scores[int(local_idx)]),
                    "rank": rank,
                    "m1": diag.margin_1,
                    "m2": diag.margin_2,
                    "softplus_loss": diag.softplus_loss,
                    "combined_margin": diag.combined_margin,
                    **_candidate_metadata(example_ref),
                }
            )

    return pd.DataFrame.from_records(records)


def _summarize_rows(
    *,
    score_matrix: np.ndarray,
    candidate_ids: tuple[int, ...],
    manifest: ExampleManifest,
    checkpoint_step: int,
    group_name: str,
) -> pd.DataFrame:
    mean_score = score_matrix.mean(axis=0)
    mean_abs_score = np.abs(score_matrix).mean(axis=0)
    positive_score_sum = np.clip(score_matrix, 0.0, None).sum(axis=0)
    negative_score_sum = np.clip(score_matrix, None, 0.0).sum(axis=0)
    max_abs_score = np.abs(score_matrix).max(axis=0)

    records: list[dict[str, Any]] = []
    for idx, candidate_id in enumerate(candidate_ids):
        example_ref = manifest.example_ref(int(candidate_id))
        records.append(
            {
                "checkpoint_step": checkpoint_step,
                "group": group_name,
                "mean_score": float(mean_score[idx]),
                "mean_abs_score": float(mean_abs_score[idx]),
                "positive_score_sum": float(positive_score_sum[idx]),
                "negative_score_sum": float(negative_score_sum[idx]),
                "max_abs_score": float(max_abs_score[idx]),
                "target_count": int(score_matrix.shape[0]),
                **_candidate_metadata(example_ref),
            }
        )
    return pd.DataFrame.from_records(records)


def build_row_summary_frame(
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
) -> pd.DataFrame:
    return _summarize_rows(
        score_matrix=result.score_matrix,
        candidate_ids=result.candidate_ids,
        manifest=manifest,
        checkpoint_step=result.checkpoint_step,
        group_name="overall",
    )


def build_domain_summary_frame(
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
) -> pd.DataFrame:
    target_index = bundle.index_by_target_id()
    frames: list[pd.DataFrame] = []
    for group_name, target_ids in sorted(bundle.groups.items()):
        if not group_name.startswith("domain:"):
            continue
        indices = [target_index[target_id] for target_id in target_ids]
        if not indices:
            continue
        frames.append(
            _summarize_rows(
                score_matrix=result.score_matrix[indices, :],
                candidate_ids=result.candidate_ids,
                manifest=manifest,
                checkpoint_step=result.checkpoint_step,
                group_name=group_name,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_checkpoint_outputs(
    *,
    output_dir: str | Path,
    result: CheckpointScores,
    bundle: EWOKTargetBundle,
    manifest: ExampleManifest,
    topk: int,
    bottomk: int,
    write_dense_scores: bool,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_tag = f"step{result.checkpoint_step:08d}"

    top_rows_df = build_top_rows_frame(result, bundle, manifest, topk=topk)
    top_rows_path = out_dir / f"top_rows_{step_tag}.csv"
    top_rows_df.to_csv(top_rows_path, index=False)

    bottom_rows_path = out_dir / f"bottom_rows_{step_tag}.csv"
    bottom_rows_df = build_bottom_rows_frame(result, bundle, manifest, bottomk=bottomk)
    if not bottom_rows_df.empty:
        bottom_rows_df.to_csv(bottom_rows_path, index=False)

    row_summary_df = build_row_summary_frame(result, bundle, manifest)
    row_summary_path = out_dir / f"row_summary_{step_tag}.csv"
    row_summary_df.to_csv(row_summary_path, index=False)

    domain_summary_df = build_domain_summary_frame(result, bundle, manifest)
    domain_summary_path = out_dir / f"domain_summary_{step_tag}.csv"
    if not domain_summary_df.empty:
        domain_summary_df.to_csv(domain_summary_path, index=False)

    diagnostics_path = out_dir / f"target_diagnostics_{step_tag}.jsonl"
    write_jsonl(diagnostics_path, [diag.to_json() for diag in result.target_diagnostics])

    dense_scores_path = out_dir / f"dense_scores_{step_tag}.npy"
    if write_dense_scores:
        np.save(dense_scores_path, result.score_matrix)

    return {
        "top_rows": top_rows_path,
        "bottom_rows": bottom_rows_path if not bottom_rows_df.empty else None,
        "row_summary": row_summary_path,
        "domain_summary": domain_summary_path,
        "target_diagnostics": diagnostics_path,
        "dense_scores": dense_scores_path if write_dense_scores else None,
    }


__all__ = [
    "build_bottom_rows_frame",
    "build_domain_summary_frame",
    "build_row_summary_frame",
    "build_top_rows_frame",
    "export_target_items",
    "write_checkpoint_outputs",
    "write_json",
    "write_jsonl",
]
