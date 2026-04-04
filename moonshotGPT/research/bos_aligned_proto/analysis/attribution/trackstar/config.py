"""Configuration and CLI parsing for the TrackStar/Bergson runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from ..common.config_base import AttributionConfigBase, add_common_args
from .paper_blocks import MODULE_LAYOUT, PAPER_BLOCK_LAYOUT, validate_paper_block_features


@dataclass(frozen=True)
class TrackstarConfig(AttributionConfigBase):
    backend: str = "trackstar"
    use_hessian_correction: bool = True
    hessian_lambda: float | None = None
    hessian_target_components: int = 1000
    projection_layout: str = MODULE_LAYOUT
    paper_block_features: int = 4096
    paper_block_side: int = 64

    def resolved(self) -> "TrackstarConfig":
        base = super().resolved()
        projection_layout = str(getattr(base, "projection_layout", self.projection_layout))
        if projection_layout not in {MODULE_LAYOUT, PAPER_BLOCK_LAYOUT}:
            raise ValueError(
                f"Unknown projection_layout={projection_layout!r}; "
                f"expected one of {(MODULE_LAYOUT, PAPER_BLOCK_LAYOUT)!r}"
            )
        paper_block_features = int(getattr(base, "paper_block_features", self.paper_block_features))
        paper_block_side = validate_paper_block_features(paper_block_features)
        if projection_layout == PAPER_BLOCK_LAYOUT and not bool(base.use_fast_jl):
            raise ValueError("paper_blocks mode requires random projection; omit --no_fast_jl")
        payload = dict(base.__dict__)
        payload["projection_layout"] = projection_layout
        payload["paper_block_features"] = paper_block_features
        payload["paper_block_side"] = paper_block_side
        return TrackstarConfig(**payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TrackStar attribution for EWoK analysis")
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
        "--projection_layout",
        choices=(MODULE_LAYOUT, PAPER_BLOCK_LAYOUT),
        default=MODULE_LAYOUT,
        help=(
            "Projection layout to use. `module` keeps Bergson's existing per-module projection; "
            "`paper_blocks` pools GPT-2 gradients into the TrackStar paper's 8 layer blocks "
            "with separate attention/MLP projections."
        ),
    )
    parser.add_argument(
        "--paper_block_features",
        type=int,
        default=4096,
        help="Projected feature count per pooled paper block. Must be a perfect square.",
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
        default=None,
        help=(
            "Optional fixed override for the mixed Hessian coefficient. When omitted, "
            "TrackStar uses Bergson-style compute_lambda on the pooled query/index spectra."
        ),
    )
    parser.add_argument(
        "--hessian_target_components",
        type=int,
        default=1000,
        help=(
            "Target spectral component k for Bergson-style compute_lambda. "
            "Ignored when --hessian_lambda is provided."
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
        candidate_from_step=ns.candidate_from_step,
        candidate_to_step=ns.candidate_to_step,
        max_candidate_rows=ns.max_candidate_rows,
        candidate_seed=ns.candidate_seed,
        recent_window_steps=ns.recent_window_steps,
        ewok_variant=ns.ewok_variant,
        ewok_filter_spec=ns.ewok_filter_spec,
        ewok_score_view=ns.ewok_score_view,
        ewok_target_scope=ns.ewok_target_scope,
        score_reduction=ns.score_reduction,
        temperature=ns.temperature,
        topk=ns.topk,
        bottomk=ns.bottomk,
        write_dense_scores=ns.write_dense_scores,
        device=ns.device,
        distributed=ns.distributed,
        show_progress=ns.show_progress,
        batch_size=ns.batch_size,
        proj_dim=ns.proj_dim,
        use_fast_jl=ns.use_fast_jl,
        use_hessian_correction=ns.use_hessian_correction,
        hessian_lambda=ns.hessian_lambda,
        hessian_target_components=ns.hessian_target_components,
        projection_layout=ns.projection_layout,
        paper_block_features=ns.paper_block_features,
        max_targets=ns.max_targets,
    ).resolved()


__all__ = ["TrackstarConfig", "build_arg_parser", "parse_args"]
