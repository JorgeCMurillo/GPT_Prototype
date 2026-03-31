"""Evaluation package for training-time and standalone benchmarks."""

from .ewok import evaluate, ewok_df
from .runner import (
    build_ewok_row_category_lookup,
    run_ewok_eval_step,
    run_final_ewok_eval_main_process,
    run_hellaswag_eval_step,
    run_parallel_validation,
)

__all__ = [
    "evaluate",
    "ewok_df",
    "build_ewok_row_category_lookup",
    "run_ewok_eval_step",
    "run_final_ewok_eval_main_process",
    "run_hellaswag_eval_step",
    "run_parallel_validation",
]
