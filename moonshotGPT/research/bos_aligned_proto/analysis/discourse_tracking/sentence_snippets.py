"""Compatibility exports for discourse sentence-snippet mining.

Implementation now lives in smaller modules:
- sentence_windows.py: sentence splitting and contiguous windows
- snippet_features.py: snippet feature extraction
- selector_recipes.py: selector gates, scores, and non-overlap helpers
- pool_selection.py: dataframe pool assignment and clustering
"""

from __future__ import annotations

from .features import preview_text
from .sentence_windows import (
    SentenceSpan,
    generate_sentence_windows,
    split_sentences,
    split_sentences_regex,
    split_sentences_spacy,
)
from .snippet_features import compute_snippet_features
from .selector_recipes import (
    SELECTOR_NAMES,
    ensure_selector_record,
    is_valid_snippet_record,
    select_non_overlapping_snippets,
    selector_gate_features,
    selector_passes_gate,
    selector_score_features,
    selector_score_values,
    selector_sort_columns,
    selector_sort_key,
    sentence_intervals_overlap,
    snippet_feature_columns,
    snippet_sentence_interval,
)
from .pool_selection import assign_selector_pools, cluster_selector_pool

__all__ = [
    "SELECTOR_NAMES",
    "SentenceSpan",
    "assign_selector_pools",
    "cluster_selector_pool",
    "compute_snippet_features",
    "ensure_selector_record",
    "generate_sentence_windows",
    "is_valid_snippet_record",
    "select_non_overlapping_snippets",
    "selector_gate_features",
    "selector_passes_gate",
    "selector_score_features",
    "selector_score_values",
    "selector_sort_columns",
    "selector_sort_key",
    "sentence_intervals_overlap",
    "snippet_feature_columns",
    "snippet_sentence_interval",
    "split_sentences",
    "split_sentences_regex",
    "split_sentences_spacy",
    "preview_text",
]
