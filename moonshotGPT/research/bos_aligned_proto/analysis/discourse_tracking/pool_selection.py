"""Pool assignment and clustering for discourse sentence snippets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import build_text_embeddings
from .selector_recipes import (
    ATTRIBUTE_RICH_EDGE_COUNT_CAP,
    ATTRIBUTE_RICH_ENTITY_MIN,
    MIXED_ACTIVE_SELECTOR_WEIGHT,
    MIXED_ATTRIBUTE_DENSITY_CAP,
    MIXED_BIBLIOGRAPHY_PENALTY_WEIGHT,
    MIXED_CHANGE_DENSITY_CAP,
    MIXED_CORE_SELECTOR_WEIGHT,
    MIXED_ENTITY_RECURRENCE_CAP,
    MIXED_INTERNAL_STATE_DENSITY_CAP,
    MIXED_NOISE_PENALTY_WEIGHT,
    MIXED_RELATION_DENSITY_CAP,
    MIXED_TABLE_CATALOG_PENALTY_WEIGHT,
    PERSISTENT_RELATION_ROLE_CAST_SIZE_SOFT_CAP,
    PERSISTENT_RELATION_ROLE_DIRECTED_COUNT_CAP,
    RELATION_ROLE_DIRECTED_COUNT_CAP,
    SELECTOR_NAMES,
    _ensure_selector_columns,
    selector_sort_columns,
)
from .snippet_features import (
    BIBLIOGRAPHY_NOISE_GATE_MAX,
    DUPLICATE_SENTENCE_GATE_MAX,
    HEAVY_LIST_NOISE_GATE_MAX,
    LIST_NOISE_GATE_MAX,
    REPEATED_3GRAM_GATE_MAX,
)


def assign_selector_pools(
    frame: pd.DataFrame,
    *,
    selectors: Sequence[str],
    num_treated_snippets: int,
    random_seed: int = 42,
    random_control_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selectors = tuple(str(selector).strip() for selector in selectors if str(selector).strip())
    unknown = sorted(set(selectors) - set(SELECTOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown selectors {unknown!r}; expected subset of {SELECTOR_NAMES!r}")

    working = _ensure_selector_columns(frame)
    working["is_valid_snippet"] = (
        (working["snippet_sentence_count"].astype(int) > 0)
        & (working["token_count_text"].astype(int) > 0)
        & (working["bos_contamination_penalty"].astype(float) <= 0.0)
        & working["snippet_text_preview"].astype(str).str.len().gt(0)
    )
    noise_penalty = (
        working["layout_noise_score"].astype(float)
        + 3.0 * working["repeated_3gram_ratio"].astype(float)
        + 2.0 * working["duplicate_sentence_fraction"].astype(float)
    )
    working["relation_role_score"] = (
        2.0 * working["directed_relation_count"].astype(float).clip(upper=RELATION_ROLE_DIRECTED_COUNT_CAP)
        + 3.0 * working["two_entity_relation_sentence_fraction"].astype(float)
        + 0.25 * working["relation_density"].astype(float).clip(upper=6.0)
        - 2.0 * noise_penalty
    )
    working["persistent_relation_role_score"] = (
        1.5 * working["directed_relation_count"].astype(float).clip(upper=12.0)
        + 2.0 * working["two_entity_relation_sentence_fraction"].astype(float)
        + 5.0 * working["mean_entity_persistence"].astype(float)
        + 3.0 * working["entity_sentence_coverage"].astype(float)
        + 1.5 * working["entity_recurrence"].astype(float)
        + 4.0 * working["pair_recurrence"].astype(float)
        + 3.0 * working["adjacent_entity_overlap"].astype(float)
        + 0.75 * working["same_pair_multi_relation_count"].astype(float).clip(upper=6.0)
        - 0.75
        * (
            working["directed_relation_count"].astype(float) - PERSISTENT_RELATION_ROLE_DIRECTED_COUNT_CAP
        ).clip(lower=0.0)
        - 0.50
        * (working["effective_cast_size"].astype(float) - PERSISTENT_RELATION_ROLE_CAST_SIZE_SOFT_CAP).clip(
            lower=0.0
        )
        - 2.5 * noise_penalty
        - 1.5 * working["bibliography_noise_score"].astype(float)
        - 1.5 * working["table_catalog_symptom_noise_score"].astype(float)
        - 0.75 * working["dense_separator_density"].astype(float)
    )
    working["attribute_rich_score"] = (
        0.85 * working["attribute_density"].astype(float).clip(upper=5.0)
        + 0.65 * working["entity_attribute_edge_count"].astype(float).clip(upper=ATTRIBUTE_RICH_EDGE_COUNT_CAP)
        + 0.55 * working["property_word_count"].astype(float).clip(upper=8.0)
        + 0.45 * working["unique_attribute_count"].astype(float).clip(upper=6.0)
        - 2.25 * noise_penalty
        - 1.0 * working["bibliography_noise_score"].astype(float)
        - 0.25 * working["inline_list_glyph_count"].astype(float).clip(upper=4.0)
        - 0.80 * working["table_catalog_symptom_noise_score"].astype(float)
    )
    working["role_alternation_score"] = (
        2.0 * working["role_alternating_pair_count"].astype(float)
        + working["same_pair_multi_relation_count"].astype(float).clip(upper=12.0)
        - 2.0 * noise_penalty
    )
    capped_event_chain_count = working["same_entity_event_chain_count"].astype(float).clip(upper=3.0)
    working["state_update_score"] = (
        0.85 * working["change_verb_density"].astype(float).clip(upper=5.0)
        + 0.65 * capped_event_chain_count
        + 0.35 * working["temporal_marker_density"].astype(float)
        + 0.35 * working["result_state_pattern_count"].astype(float)
        - 2.10 * noise_penalty
        - 1.50 * working["bibliography_noise_score"].astype(float)
    )
    working["persistent_state_update_score"] = (
        working["change_verb_count"].astype(float).clip(upper=6.0)
        + working["change_verb_density"].astype(float).clip(upper=3.0)
        + 3.0 * working["mean_entity_persistence"].astype(float)
        + 2.0 * working["entity_sentence_coverage"].astype(float)
        + 1.5 * working["entity_recurrence"].astype(float)
        + 2.0 * working["same_entity_event_chain_count"].astype(float).clip(upper=3.0)
        + 0.75 * working["result_state_pattern_count"].astype(float).clip(upper=3.0)
        + 0.50 * working["temporal_marker_count"].astype(float).clip(upper=4.0)
        + 0.50 * working["before_after_marker_count"].astype(float).clip(upper=2.0)
        - 2.25 * noise_penalty
        - 1.50 * working["bibliography_noise_score"].astype(float)
        - 1.25 * working["table_catalog_symptom_noise_score"].astype(float)
        - 0.50 * working["dense_separator_density"].astype(float)
    )
    working["persistent_relation_state_update_score"] = (
        2.5 * working["mean_entity_persistence"].astype(float)
        + 2.0 * working["entity_sentence_coverage"].astype(float)
        + working["entity_recurrence"].astype(float)
        + working["directed_relation_count"].astype(float).clip(upper=8.0)
        + 1.25 * working["two_entity_relation_sentence_fraction"].astype(float).clip(upper=0.75)
        + 0.75 * working["same_pair_multi_relation_count"].astype(float).clip(upper=3.0)
        + 0.75 * working["pair_recurrence"].astype(float).clip(upper=1.0)
        + 0.75 * working["adjacent_entity_overlap"].astype(float).clip(upper=1.0)
        + working["change_verb_count"].astype(float).clip(upper=4.0)
        + 0.75 * working["change_verb_density"].astype(float).clip(upper=2.0)
        + 1.50 * working["same_entity_event_chain_count"].astype(float).clip(upper=3.0)
        + 0.50 * working["result_state_pattern_count"].astype(float).clip(upper=3.0)
        + 0.50 * working["temporal_marker_count"].astype(float).clip(upper=4.0)
        + 0.25 * working["before_after_marker_count"].astype(float).clip(upper=2.0)
        - 0.35 * (working["effective_cast_size"].astype(float) - 12.0).clip(lower=0.0)
        - 2.25 * noise_penalty
        - 1.25 * working["bibliography_noise_score"].astype(float)
        - 1.25 * working["table_catalog_symptom_noise_score"].astype(float)
        - 0.50 * working["dense_separator_density"].astype(float)
    )
    working["internal_state_score"] = (
        working["strong_internal_state_density"].astype(float).clip(upper=5.0)
        + working["strong_agent_state_edge_count"].astype(float).clip(upper=4.0)
        + working["state_complement_count"].astype(float).clip(upper=4.0)
        + working["preference_goal_intent_count"].astype(float).clip(upper=4.0)
        + 0.25 * working["weak_internal_state_density"].astype(float).clip(upper=2.0)
        + 0.25 * working["weak_agent_state_edge_count"].astype(float).clip(upper=2.0)
        - 1.5 * noise_penalty
        - 1.50 * working["internal_state_instructional_noise_score"].astype(float)
        - 1.75 * working["patent_intent_noise_score"].astype(float)
    )
    repeat_ok = (
        working["repeated_3gram_ratio"].astype(float).le(REPEATED_3GRAM_GATE_MAX)
        & working["duplicate_sentence_fraction"].astype(float).le(DUPLICATE_SENTENCE_GATE_MAX)
    )
    list_ok = working["layout_noise_score"].astype(float).le(LIST_NOISE_GATE_MAX)
    heavy_list_ok = working["layout_noise_score"].astype(float).le(HEAVY_LIST_NOISE_GATE_MAX)
    bibliography_ok = working["bibliography_noise_score"].astype(float).le(BIBLIOGRAPHY_NOISE_GATE_MAX)
    state_update_anchor = (
        (working["same_entity_event_chain_count"].astype(int) > 0)
        | (working["result_state_pattern_count"].astype(int) > 0)
        | (working["temporal_marker_count"].astype(int) > 0)
    )
    internal_state_anchor = (
        (working["strong_internal_state_count"].astype(int) > 0)
        | (working["state_complement_count"].astype(int) > 0)
        | (working["preference_goal_intent_count"].astype(int) > 0)
    )

    gate_masks = {
        "relation_role": (
            working["is_valid_snippet"]
            & repeat_ok
            & heavy_list_ok
            & (working["unique_entity_count"].astype(int) >= 2)
            & (working["directed_relation_count"].astype(int) > 0)
            & working["relation_role_score"].astype(float).gt(0.0)
        ),
        "entity_persistence": (
            working["is_valid_snippet"]
            & (working["unique_entity_count"].astype(int) >= 2)
            & (working["entity_recurrence"].astype(float) > 0.0)
        ),
        "persistent_relation_role": (
            working["is_valid_snippet"]
            & repeat_ok
            & heavy_list_ok
            & bibliography_ok
            & (working["dense_separator_density"].astype(float) <= 1.00)
            & (working["table_catalog_symptom_noise_score"].astype(float) <= 0.25)
            & (working["snippet_sentence_count"].astype(int) >= 2)
            & (working["unique_entity_count"].astype(int) >= 2)
            & (working["entity_sentence_coverage"].astype(float) >= 0.60)
            & (working["mean_entity_persistence"].astype(float) >= 0.25)
            & (working["entity_recurrence"].astype(float) > 0.0)
            & (working["directed_relation_count"].astype(int) >= 2)
            & (working["two_entity_relation_sentence_fraction"].astype(float) >= 0.20)
            & (
                (working["pair_recurrence"].astype(float) > 0.0)
                | (working["adjacent_entity_overlap"].astype(float) > 0.0)
                | (working["same_pair_multi_relation_count"].astype(int) > 0)
            )
            & working["persistent_relation_role_score"].astype(float).gt(0.0)
        ),
        "attribute_rich": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & (working["unique_entity_count"].astype(int) >= ATTRIBUTE_RICH_ENTITY_MIN)
            & (working["attribute_density"].astype(float) > 0.0)
            & working["attribute_rich_score"].astype(float).gt(0.0)
        ),
        "role_alternation": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & (working["role_alternating_pair_count"].astype(float) > 0.0)
            & working["role_alternation_score"].astype(float).gt(0.0)
        ),
        "state_update": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & bibliography_ok
            & state_update_anchor
            & (working["change_verb_density"].astype(float) > 0.0)
            & working["state_update_score"].astype(float).gt(0.0)
        ),
        "persistent_state_update": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & bibliography_ok
            & (working["dense_separator_density"].astype(float) <= 1.00)
            & (working["table_catalog_symptom_noise_score"].astype(float) <= 0.25)
            & (working["snippet_sentence_count"].astype(int) >= 2)
            & (working["unique_entity_count"].astype(int) >= 1)
            & (working["entity_sentence_coverage"].astype(float) >= 0.60)
            & (working["mean_entity_persistence"].astype(float) >= 0.25)
            & (working["entity_recurrence"].astype(float) > 0.0)
            & (working["change_verb_count"].astype(int) >= 1)
            & state_update_anchor
            & working["persistent_state_update_score"].astype(float).gt(0.0)
        ),
        "persistent_relation_state_update": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & bibliography_ok
            & (working["dense_separator_density"].astype(float) <= 1.00)
            & (working["table_catalog_symptom_noise_score"].astype(float) <= 0.25)
            & (working["snippet_sentence_count"].astype(int) >= 2)
            & (working["unique_entity_count"].astype(int) >= 2)
            & (working["entity_sentence_coverage"].astype(float) >= 0.60)
            & (working["mean_entity_persistence"].astype(float) >= 0.25)
            & (working["entity_recurrence"].astype(float) > 0.0)
            & (working["directed_relation_count"].astype(int) >= 2)
            & (working["two_entity_relation_sentence_fraction"].astype(float) >= 0.20)
            & (working["change_verb_count"].astype(int) >= 1)
            & state_update_anchor
            & working["persistent_relation_state_update_score"].astype(float).gt(0.0)
        ),
        "internal_state": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & (working["mental_state_density"].astype(float) > 0.0)
            & internal_state_anchor
            & (
                (working["agent_state_edge_count"].astype(int) > 0)
                | (working["state_complement_count"].astype(int) > 0)
                | (working["preference_goal_intent_count"].astype(int) > 0)
            )
            & working["internal_state_score"].astype(float).gt(0.0)
        ),
    }
    for selector_name, mask in gate_masks.items():
        working[f"passes_{selector_name}_gate"] = mask.astype(bool)

    active_counts = np.zeros(len(working), dtype=np.int64)
    for selector_name in (
        "relation_role",
        "persistent_relation_role",
        "entity_persistence",
        "attribute_rich",
        "role_alternation",
        "state_update",
        "persistent_state_update",
        "persistent_relation_state_update",
        "internal_state",
    ):
        active_counts += working[f"passes_{selector_name}_gate"].astype(bool).to_numpy(dtype=np.int64)
    working["active_selector_type_count"] = active_counts
    core_counts = np.zeros(len(working), dtype=np.int64)
    for selector_name in (
        "relation_role",
        "persistent_relation_role",
        "persistent_relation_state_update",
        "entity_persistence",
        "internal_state",
    ):
        core_counts += working[f"passes_{selector_name}_gate"].astype(bool).to_numpy(dtype=np.int64)
    working["mixed_core_selector_count"] = core_counts
    mixed_internal_state_density = (
        working["strong_internal_state_density"].astype(float)
        + 0.25 * working["weak_internal_state_density"].astype(float)
    )
    working["mixed_primary_signal"] = (
        working["relation_density"].astype(float).clip(upper=MIXED_RELATION_DENSITY_CAP)
        + working["entity_recurrence"].astype(float).clip(upper=MIXED_ENTITY_RECURRENCE_CAP)
        + working["attribute_density"].astype(float).clip(upper=MIXED_ATTRIBUTE_DENSITY_CAP)
        + working["change_verb_density"].astype(float).clip(upper=MIXED_CHANGE_DENSITY_CAP)
        + mixed_internal_state_density.clip(upper=MIXED_INTERNAL_STATE_DENSITY_CAP)
    )
    mixed_noise_penalty = (
        MIXED_NOISE_PENALTY_WEIGHT * noise_penalty
        + MIXED_TABLE_CATALOG_PENALTY_WEIGHT * working["table_catalog_symptom_noise_score"].astype(float)
        + MIXED_BIBLIOGRAPHY_PENALTY_WEIGHT * working["bibliography_noise_score"].astype(float)
    )
    working["mixed_structural_score"] = (
        MIXED_ACTIVE_SELECTOR_WEIGHT * working["active_selector_type_count"].astype(float)
        + MIXED_CORE_SELECTOR_WEIGHT * working["mixed_core_selector_count"].astype(float)
        + working["mixed_primary_signal"].astype(float)
        - mixed_noise_penalty
    )
    working["passes_mixed_structural_gate"] = (
        working["is_valid_snippet"]
        & repeat_ok
        & heavy_list_ok
        & (working["active_selector_type_count"].astype(int) >= 2)
        & (working["mixed_core_selector_count"].astype(int) >= 1)
        & working["mixed_structural_score"].astype(float).gt(0.0)
    )

    selected_ids_by_selector: dict[str, list[str]] = {}
    selector_summary: dict[str, Any] = {}
    treated_union: set[str] = set()
    for selector in selectors:
        column = f"is_treated_{selector}"
        working[column] = False
        gate = working[f"passes_{selector}_gate"].astype(bool)
        sort_columns, ascending = selector_sort_columns(selector)
        eligible = working.loc[gate].sort_values(sort_columns, ascending=ascending).copy()
        selected = eligible.head(max(0, int(num_treated_snippets)))
        selected_ids = [str(value) for value in selected["window_id"].tolist()]
        selected_ids_by_selector[selector] = selected_ids
        treated_union.update(selected_ids)
        working.loc[working["window_id"].astype(str).isin(selected_ids), column] = True
        selector_summary[selector] = {
            "eligible_count": int(gate.sum()),
            "treated_count": int(len(selected)),
            "primary_sort_columns": list(sort_columns),
            "sparse": bool(len(selected) < int(num_treated_snippets)),
        }

    working["is_random_control_pool"] = False
    random_candidates = working.loc[
        working["is_valid_snippet"] & ~working["window_id"].astype(str).isin(treated_union)
    ].copy()
    if random_control_size is None:
        random_control_size = int(num_treated_snippets)
    random_control_size = min(max(0, int(random_control_size)), len(random_candidates))
    if random_control_size > 0:
        sampled = random_candidates.sample(n=random_control_size, replace=False, random_state=int(random_seed))
        sampled_ids = set(str(value) for value in sampled["window_id"].tolist())
        working.loc[working["window_id"].astype(str).isin(sampled_ids), "is_random_control_pool"] = True

    summary = {
        "selectors": selector_summary,
        "counts": {
            "total_snippets": int(len(working)),
            "valid_snippets": int(working["is_valid_snippet"].sum()),
            "random_control": int(working["is_random_control_pool"].sum()),
        },
        "random_control": {
            "seed": int(random_seed),
            "requested": int(random_control_size),
            "excluded_treated_union_count": int(len(treated_union)),
        },
    }
    return working, summary


def cluster_selector_pool(
    frame: pd.DataFrame,
    *,
    selector: str,
    text_lookup: dict[str, str],
    embedding_backend: str,
    embedding_model: str,
    num_clusters: int,
    min_cluster_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    column = f"is_treated_{selector}"
    selected = frame.loc[frame[column].astype(bool)].copy()
    if selected.empty:
        return selected.assign(cluster_id=-1), pd.DataFrame(), {"ran": False, "reason": "treated pool is empty"}
    if len(selected) < max(4, int(min_cluster_size)):
        return (
            selected.assign(cluster_id=-1),
            pd.DataFrame(),
            {"ran": False, "reason": "treated pool is too small", "treated_count": int(len(selected))},
        )

    texts = [text_lookup[str(window_id)] for window_id in selected["window_id"].astype(str).tolist()]
    embeddings, embedding_info = build_text_embeddings(
        texts,
        backend=embedding_backend,
        embedding_model=embedding_model,
        seed=seed,
    )

    from sklearn.cluster import KMeans

    resolved_clusters = max(2, min(int(num_clusters), len(selected) // 2))
    kmeans = KMeans(n_clusters=resolved_clusters, random_state=int(seed), n_init=10)
    selected["cluster_id"] = kmeans.fit_predict(embeddings)
    summary = (
        selected.groupby("cluster_id", dropna=False)
        .agg(
            cluster_size=("window_id", "size"),
            mean_relation_density=("relation_density", "mean"),
            mean_entity_recurrence=("entity_recurrence", "mean"),
            mean_attribute_density=("attribute_density", "mean"),
            mean_change_verb_density=("change_verb_density", "mean"),
            mean_mental_state_density=("mental_state_density", "mean"),
            mean_active_selector_type_count=("active_selector_type_count", "mean"),
        )
        .reset_index()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return selected, summary, {
        "ran": True,
        "selector": str(selector),
        "embedding": embedding_info,
        "num_clusters": int(resolved_clusters),
        "min_cluster_size": int(min_cluster_size),
    }
