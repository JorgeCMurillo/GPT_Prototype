"""Pool assignment and clustering for discourse sentence snippets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import build_text_embeddings
from .selector_recipes import SELECTOR_NAMES, _ensure_selector_columns, selector_sort_columns
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
        2.0 * working["directed_relation_count"].astype(float)
        + 3.0 * working["two_entity_relation_sentence_fraction"].astype(float)
        + 0.25 * working["relation_density"].astype(float).clip(upper=6.0)
        - 2.0 * noise_penalty
    )
    working["attribute_rich_score"] = (
        working["attribute_density"].astype(float).clip(upper=5.0)
        + 0.75 * working["property_word_count"].astype(float).clip(upper=8.0)
        + 0.50 * working["unique_attribute_count"].astype(float).clip(upper=6.0)
        - 2.0 * noise_penalty
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
    working["internal_state_score"] = (
        working["mental_state_density"].astype(float).clip(upper=5.0)
        + 0.75 * working["agent_state_edge_count"].astype(float)
        + 0.50 * working["state_complement_count"].astype(float)
        + 0.25 * working["preference_goal_intent_count"].astype(float)
        - 1.5 * noise_penalty
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
        "attribute_rich": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
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
        "internal_state": (
            working["is_valid_snippet"]
            & repeat_ok
            & list_ok
            & (working["mental_state_density"].astype(float) > 0.0)
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
        "entity_persistence",
        "attribute_rich",
        "role_alternation",
        "state_update",
        "internal_state",
    ):
        active_counts += working[f"passes_{selector_name}_gate"].astype(bool).to_numpy(dtype=np.int64)
    working["active_selector_type_count"] = active_counts
    working["mixed_primary_signal"] = (
        working["relation_density"].astype(float)
        + working["entity_recurrence"].astype(float)
        + working["attribute_density"].astype(float)
        + working["change_verb_density"].astype(float)
        + working["mental_state_density"].astype(float)
    )
    working["mixed_structural_score"] = (
        working["active_selector_type_count"].astype(float)
        + working["mixed_primary_signal"].astype(float).clip(upper=10.0)
        - 2.0 * noise_penalty
    )
    working["passes_mixed_structural_gate"] = (
        working["is_valid_snippet"]
        & repeat_ok
        & heavy_list_ok
        & (working["active_selector_type_count"].astype(int) >= 2)
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
