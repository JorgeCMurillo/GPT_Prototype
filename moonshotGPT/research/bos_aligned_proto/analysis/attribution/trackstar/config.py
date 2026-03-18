"""Configuration and CLI parsing for the TrackStar/Bergson runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from ..common.config_base import AttributionConfigBase, add_common_args


@dataclass(frozen=True)
class TrackstarConfig(AttributionConfigBase):
    backend: str = "trackstar"
    use_hessian_correction: bool = True
    hessian_lambda: float = 0.9


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TrackStar attribution for BOS-row EWoK analysis")
    parser = add_common_args(parser)
    # Bergson's projection_dim is per-module and per-side, not the same
    # overall JL dimension used by the TRAK backend. A small projected index is
    # the practical default for GPT-2-medium-scale models.
    parser.set_defaults(use_fast_jl=True, proj_dim=16)
    parser.add_argument(
        "--no_fast_jl",
        action="store_false",
        dest="use_fast_jl",
        help="Disable Bergson's per-module random projection. Not recommended for large models.",
    )
    parser.add_argument(
        "--no_hessian_correction",
        action="store_false",
        dest="use_hessian_correction",
        help="Disable the TrackStar-style mixed Hessian correction during scoring.",
    )
    parser.add_argument(
        "--hessian_lambda",
        type=float,
        default=0.9,
        help=(
            "Paper-style lambda for the mixed Hessian correction: "
            "H_mix = lambda * H_query + (1 - lambda) * H_index."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TrackstarConfig:
    ns = build_arg_parser().parse_args(argv)
    return TrackstarConfig(
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
        bottomk=ns.bottomk,
        write_dense_scores=ns.write_dense_scores,
        device=ns.device,
        distributed=ns.distributed,
        batch_size=ns.batch_size,
        proj_dim=ns.proj_dim,
        use_fast_jl=ns.use_fast_jl,
        use_hessian_correction=ns.use_hessian_correction,
        hessian_lambda=ns.hessian_lambda,
        max_targets=ns.max_targets,
    ).resolved()


__all__ = ["TrackstarConfig", "build_arg_parser", "parse_args"]
