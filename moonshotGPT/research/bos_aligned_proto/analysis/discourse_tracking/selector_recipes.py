"""Selector gates, ranking recipes, and non-overlap helpers."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .snippet_features import (
    DUPLICATE_SENTENCE_GATE_MAX,
    HEAVY_LIST_NOISE_GATE_MAX,
    LIST_NOISE_GATE_MAX,
    REPEATED_3GRAM_GATE_MAX,
)


SELECTOR_NAMES = (
    "relation_role",
    "entity_persistence",
    "attribute_rich",
    "role_alternation",
    "state_update",
    "internal_state",
    "mixed_structural",
)


def selector_sort_columns(selector: str) -> tuple[list[str], list[bool]]:
    if selector == "relation_role":
        return ["relation_role_score", "directed_relation_count", "two_entity_relation_sentence_fraction", "window_id"], [False, False, False, True]
    if selector == "entity_persistence":
        return ["entity_recurrence", "entity_persistence", "window_id"], [False, False, True]
    if selector == "attribute_rich":
        return ["attribute_rich_score", "property_word_count", "attribute_density", "window_id"], [False, False, False, True]
    if selector == "role_alternation":
        return ["role_alternation_score", "role_alternating_pair_count", "same_pair_multi_relation_count", "window_id"], [False, False, False, True]
    if selector == "state_update":
        return ["state_update_score", "change_verb_density", "change_verb_count", "window_id"], [False, False, False, True]
    if selector == "internal_state":
        return ["internal_state_score", "mental_state_density", "agent_state_edge_count", "window_id"], [False, False, False, True]
    if selector == "mixed_structural":
        return ["mixed_structural_score", "active_selector_type_count", "mixed_primary_signal", "window_id"], [False, False, False, True]
    raise ValueError(f"Unknown selector {selector!r}; expected one of {SELECTOR_NAMES!r}")


SELECTOR_DEFAULT_COLUMNS = (
    "relation_density",
    "unique_entity_count",
    "entity_recurrence",
    "entity_persistence",
    "attribute_density",
    "property_word_count",
    "role_alternating_pair_count",
    "same_pair_multi_relation_count",
    "two_entity_relation_sentence_fraction",
    "change_verb_density",
    "change_verb_count",
    "temporal_marker_density",
    "result_state_pattern_count",
    "same_entity_event_chain_count",
    "mental_state_density",
    "mental_verb_count",
    "agent_state_edge_count",
    "state_complement_count",
    "preference_goal_intent_count",
    "unique_attribute_count",
    "layout_noise_score",
    "repeated_3gram_ratio",
    "duplicate_sentence_fraction",
    "bos_contamination_penalty",
    "token_count_text",
    "active_selector_type_count",
    "mixed_primary_signal",
    "relation_role_score",
    "attribute_rich_score",
    "role_alternation_score",
    "state_update_score",
    "internal_state_score",
    "mixed_structural_score",
)


def _record_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = record.get(key, default)
    except AttributeError:
        value = default
    if value is None or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _record_int(record: dict[str, Any], key: str, default: int = 0) -> int:
    return int(_record_float(record, key, float(default)))


def _selector_noise_penalty(record: dict[str, Any]) -> float:
    return float(
        _record_float(record, "layout_noise_score")
        + 3.0 * _record_float(record, "repeated_3gram_ratio")
        + 2.0 * _record_float(record, "duplicate_sentence_fraction")
    )


def selector_score_features(record: dict[str, Any]) -> dict[str, float]:
    noise_penalty = _selector_noise_penalty(record)
    relation_role_score = float(
        2.0 * _record_float(record, "directed_relation_count")
        + 3.0 * _record_float(record, "two_entity_relation_sentence_fraction")
        + 0.25 * min(_record_float(record, "relation_density"), 6.0)
        - 2.0 * noise_penalty
    )
    attribute_rich_score = float(
        min(_record_float(record, "attribute_density"), 5.0)
        + 0.75 * min(_record_float(record, "property_word_count"), 8.0)
        + 0.50 * min(_record_float(record, "unique_attribute_count"), 6.0)
        - 2.0 * noise_penalty
    )
    role_alternation_score = float(
        2.0 * _record_float(record, "role_alternating_pair_count")
        + min(_record_float(record, "same_pair_multi_relation_count"), 12.0)
        - 2.0 * noise_penalty
    )
    state_update_score = float(
        min(_record_float(record, "change_verb_density"), 5.0)
        + 0.50 * _record_float(record, "same_entity_event_chain_count")
        + 0.25 * _record_float(record, "temporal_marker_density")
        + 0.25 * _record_float(record, "result_state_pattern_count")
        - 2.0 * noise_penalty
    )
    internal_state_score = float(
        min(_record_float(record, "mental_state_density"), 5.0)
        + 0.75 * _record_float(record, "agent_state_edge_count")
        + 0.50 * _record_float(record, "state_complement_count")
        + 0.25 * _record_float(record, "preference_goal_intent_count")
        - 1.5 * noise_penalty
    )
    mixed_structural_score = float(
        _record_float(record, "active_selector_type_count")
        + min(_record_float(record, "mixed_primary_signal"), 10.0)
        - 2.0 * noise_penalty
    )
    return {
        "relation_role_score": relation_role_score,
        "attribute_rich_score": attribute_rich_score,
        "role_alternation_score": role_alternation_score,
        "state_update_score": state_update_score,
        "internal_state_score": internal_state_score,
        "mixed_structural_score": mixed_structural_score,
    }


def is_valid_snippet_record(record: dict[str, Any]) -> bool:
    return (
        _record_int(record, "snippet_sentence_count", _record_int(record, "sentence_count", 0)) > 0
        and _record_int(record, "token_count_text", 0) > 0
        and _record_float(record, "bos_contamination_penalty", 0.0) <= 0.0
        and len(str(record.get("snippet_text_preview", ""))) > 0
    )


def selector_gate_features(record: dict[str, Any]) -> dict[str, float | int | bool]:
    valid = is_valid_snippet_record(record)
    scores = selector_score_features(record)
    repeat_ok = (
        _record_float(record, "repeated_3gram_ratio") <= REPEATED_3GRAM_GATE_MAX
        and _record_float(record, "duplicate_sentence_fraction") <= DUPLICATE_SENTENCE_GATE_MAX
    )
    list_noise = _record_float(record, "layout_noise_score")
    list_ok = list_noise <= LIST_NOISE_GATE_MAX
    heavy_list_ok = list_noise <= HEAVY_LIST_NOISE_GATE_MAX
    gates = {
        "relation_role": (
            valid
            and repeat_ok
            and heavy_list_ok
            and _record_int(record, "unique_entity_count", 0) >= 2
            and _record_int(record, "directed_relation_count", 0) > 0
            and scores["relation_role_score"] > 0.0
        ),
        "entity_persistence": (
            valid
            and _record_int(record, "unique_entity_count", 0) >= 2
            and _record_float(record, "entity_recurrence") > 0.0
        ),
        "attribute_rich": (
            valid
            and repeat_ok
            and list_ok
            and _record_float(record, "attribute_density") > 0.0
            and scores["attribute_rich_score"] > 0.0
        ),
        "role_alternation": (
            valid
            and repeat_ok
            and list_ok
            and _record_float(record, "role_alternating_pair_count") > 0.0
            and scores["role_alternation_score"] > 0.0
        ),
        "state_update": (
            valid
            and repeat_ok
            and list_ok
            and _record_float(record, "change_verb_density") > 0.0
            and scores["state_update_score"] > 0.0
        ),
        "internal_state": (
            valid
            and repeat_ok
            and list_ok
            and _record_float(record, "mental_state_density") > 0.0
            and (
                _record_int(record, "agent_state_edge_count", 0) > 0
                or _record_int(record, "state_complement_count", 0) > 0
                or _record_int(record, "preference_goal_intent_count", 0) > 0
            )
            and scores["internal_state_score"] > 0.0
        ),
    }
    active_selector_type_count = int(sum(1 for passed in gates.values() if passed))
    mixed_primary_signal = float(
        _record_float(record, "relation_density")
        + _record_float(record, "entity_recurrence")
        + _record_float(record, "attribute_density")
        + _record_float(record, "change_verb_density")
        + _record_float(record, "mental_state_density")
    )
    scores["mixed_structural_score"] = float(
        active_selector_type_count + min(mixed_primary_signal, 10.0) - 2.0 * _selector_noise_penalty(record)
    )
    gates["mixed_structural"] = valid and repeat_ok and heavy_list_ok and active_selector_type_count >= 2 and scores["mixed_structural_score"] > 0.0
    return {
        "is_valid_snippet": bool(valid),
        "active_selector_type_count": int(active_selector_type_count),
        "mixed_primary_signal": float(mixed_primary_signal),
        **scores,
        **{f"passes_{selector}_gate": bool(passed) for selector, passed in gates.items()},
    }


def ensure_selector_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    for column in SELECTOR_DEFAULT_COLUMNS:
        enriched.setdefault(column, 0.0)
    if "snippet_sentence_count" not in enriched:
        enriched["snippet_sentence_count"] = enriched.get("sentence_count", 0)
    enriched.update(selector_gate_features(enriched))
    return enriched


def selector_passes_gate(record: dict[str, Any], selector: str) -> bool:
    if selector not in SELECTOR_NAMES:
        raise ValueError(f"Unknown selector {selector!r}; expected one of {SELECTOR_NAMES!r}")
    enriched = record if f"passes_{selector}_gate" in record else ensure_selector_record(record)
    return bool(enriched.get(f"passes_{selector}_gate", False))


def selector_score_values(record: dict[str, Any], selector: str) -> tuple[float, ...]:
    columns, ascending = selector_sort_columns(selector)
    values: list[float] = []
    for column, is_ascending in zip(columns, ascending):
        if column == "window_id":
            continue
        value = _record_float(record, column)
        values.append(-value if is_ascending else value)
    return tuple(values)


def selector_sort_key(record: dict[str, Any], selector: str) -> tuple[Any, ...]:
    columns, ascending = selector_sort_columns(selector)
    values: list[Any] = []
    for column, is_ascending in zip(columns, ascending):
        value = record.get(column, "")
        if column == "window_id":
            values.append(str(value))
            continue
        numeric = _record_float(record, column)
        values.append(numeric if is_ascending else -numeric)
    return tuple(values)


def snippet_sentence_interval(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["sentence_start_idx"]), int(record["sentence_end_idx"])


def sentence_intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(int(left[0]), int(right[0])) < min(int(left[1]), int(right[1]))


def select_non_overlapping_snippets(
    rows: Sequence[dict[str, Any]],
    *,
    selector: str,
    max_count: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_intervals: list[tuple[int, int]] = []
    for row in sorted(rows, key=lambda item: selector_sort_key(item, selector)):
        interval = snippet_sentence_interval(row)
        if any(sentence_intervals_overlap(interval, existing) for existing in selected_intervals):
            continue
        selected.append(dict(row))
        selected_intervals.append(interval)
        if max_count is not None and len(selected) >= int(max_count):
            break
    return selected


def _ensure_selector_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in SELECTOR_DEFAULT_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
    if "snippet_sentence_count" not in working.columns:
        working["snippet_sentence_count"] = working.get("sentence_count", 0)
    return working

def snippet_feature_columns() -> tuple[str, ...]:
    return (
        "sentence_count",
        "token_count_text",
        "unique_entity_count",
        "entity_recurrence",
        "entity_persistence",
        "relation_density",
        "directed_relation_count",
        "two_entity_relation_sentence_fraction",
        "pair_recurrence",
        "top_pair_sentence_share",
        "attribute_density",
        "property_word_count",
        "unique_attribute_count",
        "entity_attribute_edge_count",
        "role_alternating_pair_count",
        "same_pair_multi_relation_count",
        "change_verb_density",
        "change_verb_count",
        "temporal_marker_density",
        "result_state_pattern_count",
        "same_entity_event_chain_count",
        "mental_state_density",
        "mental_verb_count",
        "agent_state_edge_count",
        "agent_state_edge_density",
        "state_complement_count",
        "preference_goal_intent_count",
        "layout_noise_score",
        "list_marker_count",
        "bullet_line_fraction",
        "dense_separator_density",
        "relation_role_score",
        "attribute_rich_score",
        "role_alternation_score",
        "state_update_score",
        "internal_state_score",
        "mixed_structural_score",
        "repeated_3gram_ratio",
        "duplicate_sentence_fraction",
    )
