"""Evaluation helpers for the BOS-aligned research prototype."""

from . import core
from .core import (
    CORE_BUNDLE_URL,
    DEFAULT_EVAL_BUNDLE_DIR,
    ensure_eval_bundle,
    evaluate_core,
    load_core_bundle,
    resolve_bos_token_id,
)
from .ewok_category import (
    aggregate_eval_full_by_category,
    build_ewok_row_category_lookup,
    plot_ewok_category_subplots,
)

__all__ = [
    "core",
    "CORE_BUNDLE_URL",
    "DEFAULT_EVAL_BUNDLE_DIR",
    "ensure_eval_bundle",
    "evaluate_core",
    "load_core_bundle",
    "resolve_bos_token_id",
    "aggregate_eval_full_by_category",
    "build_ewok_row_category_lookup",
    "plot_ewok_category_subplots",
]
