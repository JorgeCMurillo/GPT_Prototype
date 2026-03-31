"""Configuration and CLI parsing for the TRAK attribution runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from ..common.config_base import AttributionConfigBase, add_common_args


@dataclass(frozen=True)
class TRAKConfig(AttributionConfigBase):
    backend: str = "traker"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TRAK attribution for EWoK analysis")
    return add_common_args(parser)


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
        max_targets=ns.max_targets,
    ).resolved()


__all__ = ["TRAKConfig", "build_arg_parser", "parse_args"]
