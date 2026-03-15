"""Top-level orchestration for BOS-row TRAK attribution runs.

This module wires the package together at the pipeline level. It resolves the
configured checkpoints, builds the row manifest and exposure index, constructs
EWoK targets, delegates scoring to the backend, and writes checkpoint-by-
checkpoint artifacts plus final comparison summaries.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .candidates import CandidateSelection, select_candidate_rows
from .checkpoints import (
    CheckpointRef,
    build_model_from_checkpoint,
    discover_checkpoints,
    load_tokenizer_from_checkpoint,
    select_checkpoints,
)
from .config import TRAKConfig, parse_args
from .ewok_targets import EWOKTargetBundle, build_ewok_targets
from .export import export_target_items, write_checkpoint_outputs, write_json
from .exposures import ExposureIndex, build_exposure_index
from .model_output import CheckpointScores, TrakAttributionBackend, build_backend
from .row_dataset import RowManifest, build_row_manifest
from .compare import write_checkpoint_compare_csv


class AttributionBackend(Protocol):
    def score_checkpoint(
        self,
        *,
        checkpoint: CheckpointRef,
        manifest: RowManifest,
        candidate_selection: CandidateSelection,
        target_bundle: EWOKTargetBundle,
    ) -> CheckpointScores:
        ...


def execute_trak_run(
    *,
    config: TRAKConfig,
    checkpoints: list[CheckpointRef],
    manifest: RowManifest,
    exposure_index: ExposureIndex,
    target_bundle: EWOKTargetBundle,
    backend: AttributionBackend,
) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "config.json", asdict(config))
    export_target_items(target_bundle, output_dir / "target_items.jsonl")
    write_json(
        output_dir / "checkpoint_manifest.json",
        [asdict(checkpoint) for checkpoint in checkpoints],
    )

    row_summaries: dict[int, object] = {}
    checkpoint_runs: list[dict] = []
    for idx, checkpoint in enumerate(checkpoints):
        previous_step = checkpoints[idx - 1].step if idx > 0 else None
        candidate_selection = select_candidate_rows(
            exposure_index,
            strategy=config.candidate_strategy,
            checkpoint_step=checkpoint.step,
            previous_step=previous_step,
            max_candidate_rows=config.max_candidate_rows,
            seed=config.candidate_seed,
            recent_window_steps=config.recent_window_steps,
        )
        result = backend.score_checkpoint(
            checkpoint=checkpoint,
            manifest=manifest,
            candidate_selection=candidate_selection,
            target_bundle=target_bundle,
        )
        paths = write_checkpoint_outputs(
            output_dir=output_dir,
            result=result,
            bundle=target_bundle,
            manifest=manifest,
            topk=config.topk,
            write_dense_scores=config.write_dense_scores,
        )
        row_summary_path = paths["row_summary"]
        import pandas as pd

        row_summaries[checkpoint.step] = pd.read_csv(row_summary_path)
        checkpoint_runs.append(
            {
                "checkpoint_step": checkpoint.step,
                "checkpoint_path": str(checkpoint.path),
                "candidate_strategy": candidate_selection.strategy,
                "source_candidate_count": candidate_selection.source_count,
                "selected_candidate_count": candidate_selection.selected_count,
                "artifacts": {name: (None if path is None else str(path)) for name, path in paths.items()},
            }
        )

    compare_path = write_checkpoint_compare_csv(
        output_dir / "checkpoint_compare.csv",
        row_summaries,
        topk=config.topk,
    )
    run_summary = {
        "config": asdict(config),
        "checkpoints": checkpoint_runs,
        "checkpoint_compare_csv": str(compare_path),
        "target_count": len(target_bundle.items),
        "groups": {name: len(ids) for name, ids in target_bundle.groups.items()},
    }
    write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def run(config: TRAKConfig) -> dict:
    resolved = config.resolved()
    checkpoints = select_checkpoints(
        discover_checkpoints(resolved.run_dir),
        requested_steps=resolved.checkpoint_steps,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints discovered under {resolved.run_dir}")

    manifest = build_row_manifest(resolved.data_dir, split="train")
    exposure_index = build_exposure_index(resolved.run_dir, manifest)
    target_bundle = build_ewok_targets(
        score_view=resolved.ewok_score_view,
        target_scope=resolved.ewok_target_scope,
        score_reduction=resolved.score_reduction,
        max_targets=resolved.max_targets,
    )

    tokenizer = load_tokenizer_from_checkpoint(checkpoints[0].path)
    model = build_model_from_checkpoint(checkpoints[0].path, device=resolved.device)
    backend = build_backend(config=resolved, model=model, tokenizer=tokenizer)
    return execute_trak_run(
        config=resolved,
        checkpoints=checkpoints,
        manifest=manifest,
        exposure_index=exposure_index,
        target_bundle=target_bundle,
        backend=backend,
    )


def main(argv: list[str] | None = None) -> dict:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    main()
