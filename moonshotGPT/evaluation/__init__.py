"""Evaluation package for training-time and standalone benchmarks."""

from .core import (
    CORE_BUNDLE_URL,
    DEFAULT_EVAL_BUNDLE_DIR,
    ensure_eval_bundle,
    evaluate_core,
    load_core_bundle,
    resolve_bos_token_id,
)
from .ewok import (
    BABYLM_COMPLETION_CHOICE,
    BABYLM_COMPLETION_CHOICE_SCORING,
    EWOK_CONTEXT_SENSITIVITY,
    EWOK_PAPER_CONTEXT_SENSITIVITY,
    evaluate,
    evaluate_all_ewok_scoring_methods,
    evaluate_babylm_completion_choice,
    ewok_df,
    ewok_per_item_records,
    ewok_score_records_all_methods,
)
from .runner import (
    build_ewok_row_category_lookup,
    run_core_eval_step,
    run_ewok_eval_step,
    run_final_ewok_eval_main_process,
    run_hellaswag_eval_step,
    run_parallel_validation,
)

__all__ = [
    "CORE_BUNDLE_URL",
    "DEFAULT_EVAL_BUNDLE_DIR",
    "ensure_eval_bundle",
    "evaluate_core",
    "load_core_bundle",
    "resolve_bos_token_id",
    "BABYLM_COMPLETION_CHOICE",
    "BABYLM_COMPLETION_CHOICE_SCORING",
    "EWOK_CONTEXT_SENSITIVITY",
    "EWOK_PAPER_CONTEXT_SENSITIVITY",
    "evaluate",
    "evaluate_all_ewok_scoring_methods",
    "evaluate_babylm_completion_choice",
    "ewok_df",
    "ewok_per_item_records",
    "ewok_score_records_all_methods",
    "build_ewok_row_category_lookup",
    "run_core_eval_step",
    "run_ewok_eval_step",
    "run_final_ewok_eval_main_process",
    "run_hellaswag_eval_step",
    "run_parallel_validation",
]
