"""Configuration primitives and CLI parsing for BOS-row TRAK runs.

This module is the single source of truth for runtime defaults, path
resolution, and validation of user-supplied settings. The rest of the package
expects to receive a resolved ``TRAKConfig`` so orchestration code can stay
focused on the pipeline rather than on argument handling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import torch


EWOK_SCORE_VIEWS = (
    "babylm_completion_choice",
    "ewok_paper_context_sensitivity",
)
EWOK_TARGET_SCOPES = ("overall", "per_domain", "both")
SCORE_REDUCTIONS = ("mean", "sum")
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
class TRAKConfig:
    run_dir: Path
    data_dir: Path
    exp_name: str = "default"
    output_dir: Path | None = None
    cache_dir: Path | None = None
    checkpoint_steps: tuple[int, ...] = ()
    candidate_strategy: str = "between_checkpoints"
    max_candidate_rows: int = 50_000
    candidate_seed: int = 1337
    recent_window_steps: int = 2_000
    ewok_score_view: str = "babylm_completion_choice"
    ewok_target_scope: str = "both"
    score_reduction: str = "mean"
    temperature: float = 1.0
    topk: int = 100
    write_dense_scores: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    proj_dim: int = 2048
    use_fast_jl: bool = False
    max_targets: int = 0

    def resolved(self) -> "TRAKConfig":
        run_dir = Path(self.run_dir).expanduser().resolve()
        data_dir = Path(self.data_dir).expanduser().resolve()
        output_dir = (
            Path(self.output_dir).expanduser().resolve()
            if self.output_dir is not None
            else run_dir / "analysis" / "trak" / self.exp_name
        )
        cache_dir = (
            Path(self.cache_dir).expanduser().resolve()
            if self.cache_dir is not None
            else output_dir / "cache"
        )

        if self.candidate_strategy not in CANDIDATE_STRATEGIES:
            raise ValueError(
                f"Unknown candidate_strategy={self.candidate_strategy!r}; "
                f"expected one of {CANDIDATE_STRATEGIES!r}"
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
        if self.max_candidate_rows <= 0:
            raise ValueError("max_candidate_rows must be > 0")
        if self.recent_window_steps <= 0:
            raise ValueError("recent_window_steps must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.topk <= 0:
            raise ValueError("topk must be > 0")
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
            checkpoint_steps=_normalize_steps(self.checkpoint_steps),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TRAK analysis for BOS-row EWoK attribution")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to a BOS run directory")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to BOS row-packed data")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name under analysis/trak/")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Override TRAK cache directory")
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
    parser.add_argument("--write_dense_scores", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=2048)
    parser.add_argument("--use_fast_jl", action="store_true")
    parser.add_argument("--max_targets", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TRAKConfig:
    ns = build_arg_parser().parse_args(argv)
    return TRAKConfig(
        run_dir=ns.run_dir,
        data_dir=ns.data_dir,
        exp_name=ns.exp_name,
        output_dir=ns.output_dir,
        cache_dir=ns.cache_dir,
        checkpoint_steps=tuple(ns.checkpoint_steps),
        candidate_strategy=ns.candidate_strategy,
        max_candidate_rows=ns.max_candidate_rows,
        candidate_seed=ns.candidate_seed,
        recent_window_steps=ns.recent_window_steps,
        ewok_score_view=ns.ewok_score_view,
        ewok_target_scope=ns.ewok_target_scope,
        score_reduction=ns.score_reduction,
        temperature=ns.temperature,
        topk=ns.topk,
        write_dense_scores=ns.write_dense_scores,
        device=ns.device,
        batch_size=ns.batch_size,
        proj_dim=ns.proj_dim,
        use_fast_jl=ns.use_fast_jl,
        max_targets=ns.max_targets,
    ).resolved()


__all__ = [
    "CANDIDATE_STRATEGIES",
    "EWOK_SCORE_VIEWS",
    "EWOK_TARGET_SCOPES",
    "SCORE_REDUCTIONS",
    "TRAKConfig",
    "build_arg_parser",
    "parse_args",
]
