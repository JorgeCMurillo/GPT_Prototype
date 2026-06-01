"""Shared configuration primitives for BOS attribution runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .ewok_filters import EWOK_VARIANTS


EWOK_SCORE_VIEWS = (
    "babylm_completion_choice",
    "ewok_paper_context_sensitivity",
    "ewok_context_sensitivity",
)
EWOK_TARGET_SCOPES = ("overall", "per_domain", "both")
SCORE_REDUCTIONS = ("mean", "sum")
DEVICE_CHOICES = ("auto", "cuda", "cpu")
DISTRIBUTED_MODES = ("none", "ddp", "fsdp")
CANDIDATE_STRATEGIES = (
    "between_checkpoints",
    "up_to_step",
    "recent_window",
    "new_since_prev",
    "raw_window_range",
    "raw_window_random",
)
RAW_WINDOW_CANDIDATE_STRATEGIES = (
    "raw_window_range",
    "raw_window_random",
)
CANDIDATE_KIND_CHOICES = (
    "auto",
    "bos_packed_row",
    "stream_window",
    "document_aligned_row",
)


def _normalize_steps(steps: Sequence[int] | None) -> tuple[int, ...]:
    if not steps:
        return ()
    return tuple(sorted({int(step) for step in steps}))


@dataclass(frozen=True)
class AttributionConfigBase:
    run_dir: Path
    data_dir: Path
    backend: str
    exp_name: str = "default"
    output_dir: Path | None = None
    cache_dir: Path | None = None
    checkpoint_steps: tuple[int, ...] = ()
    candidate_strategy: str = "between_checkpoints"
    candidate_from_step: int | None = None
    candidate_to_step: int | None = None
    max_candidate_rows: int = 50_000
    candidate_kind: str = "auto"
    candidate_seed: int = 1337
    recent_window_steps: int = 2_000
    raw_window_start_id: int = 0
    raw_window_end_id: int | None = None
    ewok_variant: str = "fast"
    ewok_filter_spec: Path | None = None
    ewok_score_view: str = "babylm_completion_choice"
    ewok_target_scope: str = "both"
    score_reduction: str = "mean"
    temperature: float = 1.0
    topk: int = 100
    bottomk: int = 0
    write_dense_scores: bool = False
    device: str = "cuda"
    distributed: str = "none"
    show_progress: bool = True
    batch_size: int = 8
    proj_dim: int = 2048
    use_fast_jl: bool = False
    score_candidate_chunk_size: int = 4096
    max_targets: int = 0

    def resolved(self) -> "AttributionConfigBase":
        run_dir = Path(self.run_dir).expanduser().resolve()
        data_dir = Path(self.data_dir).expanduser().resolve()
        output_dir = (
            Path(self.output_dir).expanduser().resolve()
            if self.output_dir is not None
            else run_dir / "analysis" / "attribution" / self.exp_name
        )
        cache_dir = (
            Path(self.cache_dir).expanduser().resolve()
            if self.cache_dir is not None
            else output_dir / "cache"
        )
        ewok_filter_spec = (
            Path(self.ewok_filter_spec).expanduser().resolve()
            if self.ewok_filter_spec is not None
            else None
        )

        if self.candidate_strategy not in CANDIDATE_STRATEGIES:
            raise ValueError(
                f"Unknown candidate_strategy={self.candidate_strategy!r}; "
                f"expected one of {CANDIDATE_STRATEGIES!r}"
            )
        if self.ewok_variant not in EWOK_VARIANTS:
            raise ValueError(
                f"Unknown ewok_variant={self.ewok_variant!r}; "
                f"expected one of {EWOK_VARIANTS!r}"
            )
        if self.ewok_score_view not in EWOK_SCORE_VIEWS:
            raise ValueError(
                f"Unknown ewok_score_view={self.ewok_score_view!r}; "
                f"expected one of {EWOK_SCORE_VIEWS!r}"
            )
        if self.ewok_target_scope not in EWOK_TARGET_SCOPES:
            raise ValueError(
                f"Unknown ewok_target_scope={self.ewok_target_scope!r}; "
                f"expected one of {EWOK_TARGET_SCOPES!r}"
            )
        if self.score_reduction not in SCORE_REDUCTIONS:
            raise ValueError(
                f"Unknown score_reduction={self.score_reduction!r}; "
                f"expected one of {SCORE_REDUCTIONS!r}"
            )
        if self.device not in DEVICE_CHOICES:
            raise ValueError(
                f"Unknown device={self.device!r}; expected one of {DEVICE_CHOICES!r}"
            )
        if self.distributed not in DISTRIBUTED_MODES:
            raise ValueError(
                f"Unknown distributed={self.distributed!r}; "
                f"expected one of {DISTRIBUTED_MODES!r}"
            )
        if self.max_candidate_rows <= 0:
            raise ValueError("max_candidate_rows must be > 0")
        if self.candidate_kind not in CANDIDATE_KIND_CHOICES:
            raise ValueError(
                f"Unknown candidate_kind={self.candidate_kind!r}; expected one of {CANDIDATE_KIND_CHOICES!r}"
            )
        if self.candidate_from_step is not None and int(self.candidate_from_step) < 0:
            raise ValueError("candidate_from_step must be >= 0 when provided")
        if self.candidate_to_step is not None and int(self.candidate_to_step) < 0:
            raise ValueError("candidate_to_step must be >= 0 when provided")
        if int(self.raw_window_start_id) < 0:
            raise ValueError("raw_window_start_id must be >= 0")
        if (
            self.raw_window_end_id is not None
            and int(self.raw_window_end_id) <= int(self.raw_window_start_id)
        ):
            raise ValueError("raw_window_end_id must be greater than raw_window_start_id")
        if self.candidate_strategy in RAW_WINDOW_CANDIDATE_STRATEGIES:
            if self.candidate_from_step is not None or self.candidate_to_step is not None:
                raise ValueError(
                    "candidate_from_step/candidate_to_step are exposure-window options; "
                    "use raw_window_start_id/raw_window_end_id with raw_window_* strategies"
                )
        elif self.raw_window_start_id != 0 or self.raw_window_end_id is not None:
            raise ValueError(
                "raw_window_start_id/raw_window_end_id are only supported with "
                "candidate_strategy='raw_window_range' or 'raw_window_random'"
            )
        if (
            self.candidate_from_step is not None
            and self.candidate_strategy not in {"between_checkpoints", "new_since_prev"}
        ):
            raise ValueError(
                "candidate_from_step is only supported with candidate_strategy "
                "'between_checkpoints' or 'new_since_prev'"
            )
        if (
            self.candidate_from_step is not None
            and self.candidate_to_step is not None
            and int(self.candidate_to_step) <= int(self.candidate_from_step)
        ):
            raise ValueError("candidate_to_step must be greater than candidate_from_step")
        if self.recent_window_steps <= 0:
            raise ValueError("recent_window_steps must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.topk <= 0:
            raise ValueError("topk must be > 0")
        if self.bottomk < 0:
            raise ValueError("bottomk must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")
        if self.score_candidate_chunk_size <= 0:
            raise ValueError("score_candidate_chunk_size must be > 0")
        if self.max_targets < 0:
            raise ValueError("max_targets must be >= 0")

        return replace(
            self,
            run_dir=run_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            ewok_filter_spec=ewok_filter_spec,
            checkpoint_steps=_normalize_steps(self.checkpoint_steps),
            candidate_from_step=(
                None if self.candidate_from_step is None else int(self.candidate_from_step)
            ),
            candidate_to_step=None if self.candidate_to_step is None else int(self.candidate_to_step),
            raw_window_start_id=int(self.raw_window_start_id),
            raw_window_end_id=None if self.raw_window_end_id is None else int(self.raw_window_end_id),
        )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to a finished training run directory")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the training-data view that matches the run (BOS-packed rows or raw token stream)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="Experiment name under analysis/attribution/",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Override cache directory")
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        nargs="*",
        default=(),
        help="Optional explicit checkpoint steps; default is all discovered checkpoints",
    )
    parser.add_argument(
        "--candidate_strategy",
        type=str,
        choices=CANDIDATE_STRATEGIES,
        default="between_checkpoints",
    )
    parser.add_argument(
        "--candidate_from_step",
        type=int,
        default=None,
        help=(
            "Optional explicit lower bound for candidate selection. This lets you score one "
            "checkpoint against a different exposure window, e.g. checkpoint 16000 on candidates "
            "from 16000->20000."
        ),
    )
    parser.add_argument(
        "--candidate_to_step",
        type=int,
        default=None,
        help=(
            "Optional explicit upper bound for candidate selection. Defaults to the scored "
            "checkpoint step when omitted."
        ),
    )
    parser.add_argument("--max_candidate_rows", type=int, default=50_000)
    parser.add_argument(
        "--candidate_kind",
        type=str,
        choices=CANDIDATE_KIND_CHOICES,
        default="auto",
        help=(
            "Override the candidate example unit. The default 'auto' reconstructs the exact training "
            "example surface; 'document_aligned_row' scores one full model-context row starting at each "
            "document boundary in raw stream data."
        ),
    )
    parser.add_argument("--candidate_seed", type=int, default=1337)
    parser.add_argument("--recent_window_steps", type=int, default=2_000)
    parser.add_argument(
        "--raw_window_start_id",
        type=int,
        default=0,
        help=(
            "First manifest example id for raw-window candidate strategies. "
            "Only valid with --candidate_strategy raw_window_range/raw_window_random."
        ),
    )
    parser.add_argument(
        "--raw_window_end_id",
        type=int,
        default=None,
        help=(
            "Exclusive manifest example-id upper bound for raw-window candidate strategies. "
            "Defaults to the end of the manifest."
        ),
    )
    parser.add_argument(
        "--ewok_variant",
        type=str,
        choices=EWOK_VARIANTS,
        default="fast",
        help="Which stored EWoK split to use when building query targets.",
    )
    parser.add_argument(
        "--ewok_filter_spec",
        type=Path,
        default=None,
        help="Optional JSON file that filters EWoK targets by fields such as domain or context_diff.",
    )
    parser.add_argument(
        "--ewok_score_view",
        type=str,
        choices=EWOK_SCORE_VIEWS,
        default="babylm_completion_choice",
    )
    parser.add_argument(
        "--ewok_target_scope",
        type=str,
        choices=EWOK_TARGET_SCOPES,
        default="both",
    )
    parser.add_argument(
        "--score_reduction",
        type=str,
        choices=SCORE_REDUCTIONS,
        default="mean",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--bottomk",
        type=int,
        default=0,
        help="Optional number of lowest-scoring candidate examples to export per target. 0 disables bottom export.",
    )
    parser.add_argument("--write_dense_scores", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=DEVICE_CHOICES,
        help="Execution device (default: cuda). Use --device auto to allow CPU fallback.",
    )
    parser.add_argument(
        "--distributed",
        type=str,
        default="none",
        choices=DISTRIBUTED_MODES,
        help="Distributed execution mode for multi-process attribution.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_false",
        dest="show_progress",
        help="Disable tqdm progress bars and periodic long-step heartbeats.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=2048)
    parser.add_argument("--use_fast_jl", action="store_true")
    parser.add_argument(
        "--score_candidate_chunk_size",
        type=int,
        default=4096,
        help=(
            "Candidate rows per chunk during local TrackStar scoring. Lower this to reduce "
            "CPU RAM at the cost of more matrix-multiply calls."
        ),
    )
    parser.add_argument("--max_targets", type=int, default=0)
    return parser


__all__ = [
    "AttributionConfigBase",
    "CANDIDATE_STRATEGIES",
    "DEVICE_CHOICES",
    "DISTRIBUTED_MODES",
    "EWOK_SCORE_VIEWS",
    "EWOK_TARGET_SCOPES",
    "EWOK_VARIANTS",
    "RAW_WINDOW_CANDIDATE_STRATEGIES",
    "SCORE_REDUCTIONS",
    "add_common_args",
]
