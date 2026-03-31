"""Top-level orchestration for attribution runs over faithful training examples.

The runner owns the shared workflow around candidate/example selection,
checkpoint discovery, EWoK target construction, and artifact export. Backend
implementations only need to score one checkpoint against an ordered set of
candidate training examples.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed as dist

from .common.candidates import CandidateSelection, select_candidate_rows
from .common.checkpoints import (
    CheckpointRef,
    build_model_from_checkpoint,
    discover_checkpoints,
    load_tokenizer_from_checkpoint,
    select_checkpoints,
)
from .common.compare import write_checkpoint_compare_csv
from .common.config_base import AttributionConfigBase
from .common.ewok_targets import CheckpointScores, EWOKTargetBundle, build_ewok_targets
from .common.export import export_target_items, write_checkpoint_outputs, write_json
from .common.exposures import ExposureIndex, build_exposure_index
from .common.training_examples import ExampleManifest, build_example_manifest
from .common.training_metadata import resolve_training_example_spec
from .trackstar.backend import build_backend as build_trackstar_backend
from .trak.backend import build_backend as build_trak_backend
from .trak.config import parse_args


@dataclass(frozen=True)
class RunExecutionContext:
    backend: str
    launcher: str
    requested_device: str
    resolved_device: str
    distributed_mode: str
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.distributed_mode != "none"

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.resolved_device)

    @classmethod
    def single_process(
        cls,
        *,
        backend: str,
        requested_device: str,
        resolved_device: str,
    ) -> "RunExecutionContext":
        return cls(
            backend=backend,
            launcher="python",
            requested_device=requested_device,
            resolved_device=resolved_device,
            distributed_mode="none",
            rank=0,
            world_size=1,
            local_rank=0,
        )


class AttributionBackend(Protocol):
    def score_checkpoint(
        self,
        *,
        checkpoint: CheckpointRef,
        manifest: ExampleManifest,
        candidate_selection: CandidateSelection,
        target_bundle: EWOKTargetBundle,
    ) -> CheckpointScores | None:
        ...


def _status(message: str, *, context: RunExecutionContext | None = None, root_only: bool = False) -> None:
    if context is not None and root_only and not context.is_root:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "[attribution]"
    if context is not None:
        prefix = f"{prefix}[{context.backend}]"
        if context.world_size > 1:
            prefix = f"{prefix}[rank {context.rank}/{context.world_size}]"
    print(f"{timestamp} {prefix} {message}", flush=True)


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str, leave: bool = True):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is not available in this Python environment. "
                "If you normally run attribution from the repo root, try launching with "
                "`conda run -n babylm python -m research.bos_aligned_proto.analysis.attribution.run_trak ... --device cuda`, "
                "and verify that the host NVIDIA driver is available."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_env_int(keys: tuple[str, ...], default: int) -> int:
    import os

    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return int(default)


def _detect_launcher() -> str:
    import os

    if any(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return "torchrun"
    if any(
        name in os.environ
        for name in (
            "ACCELERATE_PROCESS_INDEX",
            "ACCELERATE_NUM_PROCESSES",
            "ACCELERATE_LOCAL_PROCESS_INDEX",
        )
    ):
        return "accelerate"
    return "python"


def resolve_execution_context(config: AttributionConfigBase) -> RunExecutionContext:
    launcher = _detect_launcher()
    rank = _get_env_int(("RANK", "ACCELERATE_PROCESS_INDEX"), 0)
    world_size = _get_env_int(("WORLD_SIZE", "ACCELERATE_NUM_PROCESSES", "ACCELERATE_PROCESS_COUNT"), 1)
    local_rank = _get_env_int(("LOCAL_RANK", "ACCELERATE_LOCAL_PROCESS_INDEX"), rank)

    if world_size > 1 and config.backend != "trackstar":
        raise RuntimeError(
            f"Detected a multi-process {launcher} launch, but backend={config.backend!r} is only supported "
            "in single-process mode. Use `run_trackstar.py` for distributed execution."
        )
    if world_size > 1 and config.distributed == "none":
        raise RuntimeError(
            f"Detected a multi-process {launcher} launch, but --distributed none was requested. "
            "Use --distributed ddp or --distributed fsdp."
        )
    if config.backend != "trackstar" and config.distributed != "none":
        raise ValueError(
            f"backend={config.backend!r} does not support --distributed {config.distributed!r}. "
            "Leave TRAK on --distributed none."
        )

    requested_device = str(config.device)
    device = _get_device(requested_device)
    if config.distributed != "none" and device.type != "cuda":
        raise RuntimeError(
            f"--distributed {config.distributed!r} requires CUDA, but --device {requested_device!r} resolved "
            f"to {device.type!r}."
        )
    if config.distributed != "none" and world_size <= 1:
        raise RuntimeError(
            f"--distributed {config.distributed!r} requires a multi-process launcher such as `torchrun` "
            "or `accelerate launch`."
        )

    if device.type == "cuda":
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA was selected for attribution, but torch.cuda.device_count() returned 0.")
        if int(local_rank) >= int(device_count):
            raise RuntimeError(
                f"Resolved local_rank={local_rank}, but only {device_count} CUDA device(s) are visible."
            )
        resolved_device = f"cuda:{int(local_rank) if world_size > 1 else 0}"
    else:
        resolved_device = "cpu"

    distributed_mode = config.distributed if world_size > 1 else "none"
    return RunExecutionContext(
        backend=config.backend,
        launcher=launcher,
        requested_device=requested_device,
        resolved_device=resolved_device,
        distributed_mode=distributed_mode,
        rank=int(rank),
        world_size=int(world_size),
        local_rank=int(local_rank),
    )


def initialize_execution_context(context: RunExecutionContext) -> None:
    if context.torch_device.type == "cuda":
        torch.cuda.set_device(context.torch_device)
    if not context.is_distributed or dist.is_initialized():
        return
    backend = "nccl" if context.torch_device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def shutdown_execution_context(context: RunExecutionContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_attribution_backend(
    *,
    config: AttributionConfigBase,
    model: torch.nn.Module,
    tokenizer,
    execution_context: RunExecutionContext,
) -> AttributionBackend:
    if config.backend == "traker":
        return build_trak_backend(config=config, model=model, tokenizer=tokenizer)
    if config.backend == "trackstar":
        return build_trackstar_backend(
            config=config,
            model=model,
            tokenizer=tokenizer,
            execution_context=execution_context,
        )
    raise ValueError(f"Unsupported backend={config.backend!r}")


def execute_attribution_run(
    *,
    config: AttributionConfigBase,
    checkpoints: list[CheckpointRef],
    manifest: ExampleManifest,
    exposure_index: ExposureIndex,
    target_bundle: EWOKTargetBundle,
    backend: AttributionBackend,
    execution_context: RunExecutionContext | None = None,
) -> dict:
    context = execution_context or RunExecutionContext.single_process(
        backend=config.backend,
        requested_device=config.device,
        resolved_device=config.device,
    )
    output_dir = Path(config.output_dir)

    if context.is_root:
        _status(
            f"writing run metadata under {output_dir}",
            context=context,
            root_only=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "config.json", asdict(config))
        export_target_items(target_bundle, output_dir / "target_items.jsonl")
        write_json(
            output_dir / "checkpoint_manifest.json",
            [asdict(checkpoint) for checkpoint in checkpoints],
        )

    row_summaries: dict[int, object] = {}
    checkpoint_runs: list[dict] = []
    checkpoint_progress = _build_tqdm(
        enabled=context.is_root and config.show_progress and len(checkpoints) > 0,
        total=len(checkpoints),
        desc=f"{config.backend} checkpoints",
        unit="ckpt",
        leave=True,
    )

    try:
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
            if checkpoint_progress is not None:
                checkpoint_progress.set_postfix_str(
                    f"step={checkpoint.step} candidates={candidate_selection.selected_count}"
                )
            _status(
                "checkpoint "
                f"step={checkpoint.step}: selected {candidate_selection.selected_count}/"
                f"{candidate_selection.source_count} candidate example(s) "
                f"with strategy={candidate_selection.strategy}",
                context=context,
                root_only=True,
            )
            result = backend.score_checkpoint(
                checkpoint=checkpoint,
                manifest=manifest,
                candidate_selection=candidate_selection,
                target_bundle=target_bundle,
            )
            if result is None:
                if checkpoint_progress is not None:
                    checkpoint_progress.update(1)
                continue
            if not context.is_root:
                raise RuntimeError("Only rank 0 may return assembled checkpoint scores for export")

            paths = write_checkpoint_outputs(
                output_dir=output_dir,
                result=result,
                bundle=target_bundle,
                manifest=manifest,
                topk=config.topk,
                bottomk=config.bottomk,
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
            _status(
                "checkpoint "
                f"step={checkpoint.step}: wrote artifacts top_rows={paths['top_rows']} "
                f"row_summary={paths['row_summary']}",
                context=context,
                root_only=True,
            )
            if checkpoint_progress is not None:
                checkpoint_progress.update(1)
    finally:
        if checkpoint_progress is not None:
            checkpoint_progress.close()

    if not context.is_root:
        return {
            "backend": config.backend,
            "distributed_mode": context.distributed_mode,
            "rank": context.rank,
            "world_size": context.world_size,
            "exported": False,
        }

    compare_path = write_checkpoint_compare_csv(
        output_dir / "checkpoint_compare.csv",
        row_summaries,
        topk=config.topk,
    )
    run_summary = {
        "config": asdict(config),
        "backend": config.backend,
        "distributed_mode": context.distributed_mode,
        "candidate_kind": manifest.candidate_kind,
        "seq_len": int(manifest.seq_len),
        "example_tokens": int(manifest.example_tokens),
        "checkpoints": checkpoint_runs,
        "checkpoint_compare_csv": str(compare_path),
        "target_count": len(target_bundle.items),
        "groups": {name: len(ids) for name, ids in target_bundle.groups.items()},
    }
    write_json(output_dir / "run_summary.json", run_summary)
    _status(
        f"finished run: wrote {compare_path} and {output_dir / 'run_summary.json'}",
        context=context,
        root_only=True,
    )
    return run_summary


execute_trak_run = execute_attribution_run


def run(config: AttributionConfigBase) -> dict:
    resolved = config.resolved()
    context = resolve_execution_context(resolved)
    runtime_config = replace(resolved, device=context.resolved_device, distributed=context.distributed_mode)
    _status(
        "starting run "
        f"launcher={context.launcher} device={context.resolved_device} "
        f"distributed={context.distributed_mode} exp_name={runtime_config.exp_name}",
        context=context,
    )

    checkpoints = select_checkpoints(
        discover_checkpoints(runtime_config.run_dir),
        requested_steps=runtime_config.checkpoint_steps,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints discovered under {runtime_config.run_dir}")
    _status(
        f"resolved {len(checkpoints)} checkpoint(s): {[checkpoint.step for checkpoint in checkpoints]}",
        context=context,
        root_only=True,
    )

    initialize_execution_context(context)
    try:
        example_spec = resolve_training_example_spec(
            run_dir=runtime_config.run_dir,
            data_dir=runtime_config.data_dir,
        )
        _status(
            "resolved training-example semantics "
            f"candidate_kind={example_spec.candidate_kind} seq_len={example_spec.seq_len}",
            context=context,
            root_only=True,
        )
        manifest = build_example_manifest(
            runtime_config.data_dir,
            split="train",
            candidate_kind=example_spec.candidate_kind,
            seq_len=example_spec.seq_len,
            show_progress=context.is_root and runtime_config.show_progress,
        )
        _status(
            "loaded training-example manifest with "
            f"{len(manifest.examples)} example(s) across {len(manifest.shard_paths)} shard(s) "
            f"for candidate_kind={manifest.candidate_kind}",
            context=context,
            root_only=True,
        )
        exposure_index = build_exposure_index(
            runtime_config.run_dir,
            manifest,
            show_progress=context.is_root and runtime_config.show_progress,
        )
        _status(
            f"built exposure index across {len(exposure_index.step_to_example_ids)} logged step(s)",
            context=context,
            root_only=True,
        )
        target_bundle = build_ewok_targets(
            variant=runtime_config.ewok_variant,
            filter_spec_path=runtime_config.ewok_filter_spec,
            score_view=runtime_config.ewok_score_view,
            target_scope=runtime_config.ewok_target_scope,
            score_reduction=runtime_config.score_reduction,
            max_targets=runtime_config.max_targets,
        )
        _status(
            "built target bundle with "
            f"{len(target_bundle.items)} item(s) and groups={list(target_bundle.groups)} "
            f"(variant={runtime_config.ewok_variant}, filter_spec={runtime_config.ewok_filter_spec})",
            context=context,
            root_only=True,
        )

        tokenizer = load_tokenizer_from_checkpoint(checkpoints[0].path)
        model = build_model_from_checkpoint(checkpoints[0].path, device=runtime_config.device)
        _status(
            f"loaded tokenizer and model from {checkpoints[0].path}",
            context=context,
            root_only=True,
        )
        backend = build_attribution_backend(
            config=runtime_config,
            model=model,
            tokenizer=tokenizer,
            execution_context=context,
        )
        _status(
            f"constructed backend={runtime_config.backend}",
            context=context,
            root_only=True,
        )
        return execute_attribution_run(
            config=runtime_config,
            checkpoints=checkpoints,
            manifest=manifest,
            exposure_index=exposure_index,
            target_bundle=target_bundle,
            backend=backend,
            execution_context=context,
        )
    finally:
        shutdown_execution_context(context)


def main(argv: list[str] | None = None) -> dict:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    main()
