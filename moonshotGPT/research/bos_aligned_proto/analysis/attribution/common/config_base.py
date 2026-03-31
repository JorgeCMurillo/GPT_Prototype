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
    max_candidate_rows: int = 50_000
    candidate_seed: int = 1337
    recent_window_steps: int = 2_000
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
    parser.add_argument("--max_candidate_rows", type=int, default=50_000)
    parser.add_argument("--candidate_seed", type=int, default=1337)
    parser.add_argument("--recent_window_steps", type=int, default=2_000)
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
    "SCORE_REDUCTIONS",
    "add_common_args",
]
