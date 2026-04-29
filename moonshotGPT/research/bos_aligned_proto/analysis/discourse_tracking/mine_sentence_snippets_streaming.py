#!/usr/bin/env python3
"""Two-pass streaming top-K miner for sentence-snippet selector ablations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..attribution.common.checkpoints import load_tokenizer_from_checkpoint
from ..attribution.common.training_examples import (
    FiniteTrainingExampleDataset,
    PACKED_INDEX_FORMAT,
    PackedIndexView,
    build_example_manifest,
)
from .features import preview_text, resolve_backend
from .mine_candidate_pools import (
    _clean_decoded_text,
    _decode_candidate_tokens_sparse,
    _full_tokens_from_sample,
    _infer_seq_len,
    _load_candidate_frame,
    _load_data_meta,
)
from .sentence_windows import generate_sentence_windows
from .snippet_features import compute_snippet_features
from .selector_recipes import (
    SELECTOR_NAMES,
    ensure_selector_record,
    is_valid_snippet_record,
    select_non_overlapping_snippets,
    selector_passes_gate,
    selector_score_values,
    selector_sort_columns,
    selector_sort_key,
    sentence_intervals_overlap,
    snippet_feature_columns,
    snippet_sentence_interval,
)


JSON_PREVIEW_FEATURES = {
    "relation_role": (
        "relation_role_score",
        "directed_relation_count",
        "two_entity_relation_sentence_fraction",
        "relation_density",
        "layout_noise_score",
    ),
    "entity_persistence": ("entity_recurrence", "entity_persistence", "unique_entity_count"),
    "attribute_rich": (
        "attribute_rich_score",
        "attribute_density",
        "property_word_count",
        "entity_attribute_edge_count",
        "layout_noise_score",
    ),
    "role_alternation": (
        "role_alternation_score",
        "role_alternating_pair_count",
        "same_pair_multi_relation_count",
        "reversal_fraction",
        "layout_noise_score",
    ),
    "state_update": (
        "state_update_score",
        "change_verb_density",
        "change_verb_count",
        "temporal_marker_count",
        "same_entity_event_chain_count",
        "layout_noise_score",
    ),
    "internal_state": (
        "internal_state_score",
        "mental_state_density",
        "mental_verb_count",
        "agent_state_edge_count",
        "layout_noise_score",
    ),
    "mixed_structural": (
        "mixed_structural_score",
        "active_selector_type_count",
        "mixed_primary_signal",
        "relation_density",
        "entity_recurrence",
        "layout_noise_score",
    ),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_progress(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _parse_selectors(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() in {"all", "*"}:
        return SELECTOR_NAMES
    selectors = tuple(piece.strip() for piece in str(raw).split(",") if piece.strip())
    unknown = sorted(set(selectors) - set(SELECTOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown selectors {unknown!r}; expected comma-separated subset of {SELECTOR_NAMES!r}")
    if not selectors:
        raise ValueError("--selectors must not be empty")
    return selectors


def _stable_random_unit(*parts: Any) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def _stratum_key(row: dict[str, Any], *, token_bucket_width: int) -> tuple[str, int, int]:
    shard_name = Path(str(row.get("shard_path", ""))).name
    sentence_count = int(row.get("snippet_sentence_count", row.get("sentence_count", 0)))
    token_bucket = int(row.get("token_count_text", 0)) // max(1, int(token_bucket_width))
    return shard_name, sentence_count, token_bucket


def _pool_row(row: dict[str, Any], text: str, *, selector: str, rank: int, pool_role: str) -> dict[str, Any]:
    sort_columns, _ascending = selector_sort_columns(selector)
    return {
        "rank": int(rank),
        "selector": str(selector),
        "pool_role": str(pool_role),
        "sort_columns": list(sort_columns),
        "sort_key": list(selector_score_values(row, selector)),
        "window_id": str(row["window_id"]),
        "parent_candidate_id": int(row["parent_candidate_id"]),
        "sentence_range": [int(row["sentence_start_idx"]), int(row["sentence_end_idx"])],
        "snippet_sentence_count": int(row["snippet_sentence_count"]),
        "token_count_text": int(row.get("token_count_text", 0)),
        "text_preview": str(row.get("snippet_text_preview", preview_text(text))),
        "features": {
            key: row.get(key)
            for key in JSON_PREVIEW_FEATURES.get(selector, ())
            if key in row
        },
        "text": text,
    }


@dataclass
class CandidateTextDecoder:
    frame: pd.DataFrame
    data_dir: Path
    checkpoint_dir: Path
    seq_len: int | None
    decode_strategy: str

    def __post_init__(self) -> None:
        self.tokenizer = load_tokenizer_from_checkpoint(self.checkpoint_dir)
        self.resolved_strategy = str(self.decode_strategy).strip().lower()
        if self.resolved_strategy not in {"auto", "sparse", "manifest"}:
            raise ValueError("--decode_strategy must be auto, sparse, or manifest")
        if self.resolved_strategy == "auto":
            self.resolved_strategy = "sparse"

        self.resolved_seq_len = _infer_seq_len(self.frame, self.seq_len)
        self.manifest = None
        self.dataset = None
        self.meta: dict[str, Any] | None = None
        self.packed_index_view: PackedIndexView | None = None

        if self.resolved_strategy == "manifest":
            self.manifest = build_example_manifest(
                self.data_dir,
                split="train",
                candidate_kind=str(self.frame["candidate_kind"].iloc[0]),
                seq_len=int(self.resolved_seq_len),
            )
            candidate_ids = tuple(int(value) for value in self.frame["candidate_id"].tolist())
            self.dataset = FiniteTrainingExampleDataset(self.manifest, candidate_ids)
        else:
            self.meta = _load_data_meta(str(self.data_dir))
            if str(self.meta.get("format", "")) == PACKED_INDEX_FORMAT:
                self.packed_index_view = PackedIndexView(self.data_dir)

    def decode(self, record: dict[str, Any], row_index: int) -> str:
        if self.resolved_strategy == "manifest":
            if self.dataset is None:
                raise RuntimeError("manifest decoder was not initialized")
            tokens = _full_tokens_from_sample(self.dataset[int(row_index)]).tolist()
        else:
            if self.meta is None:
                raise RuntimeError("sparse decoder was not initialized")
            tokens = _decode_candidate_tokens_sparse(
                record=record,
                data_dir=self.data_dir,
                meta=self.meta,
                packed_index_view=self.packed_index_view,
            ).tolist()
        decoded = self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        return _clean_decoded_text(
            decoded,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
        )

    def info(self) -> dict[str, Any]:
        return {
            "decode_strategy": self.resolved_strategy,
            "seq_len": int(self.resolved_seq_len),
            "row_tokens": int(self.resolved_seq_len) + 1,
            "data_dir": str(self.data_dir),
        }


@dataclass
class RankedSnippet:
    selector: str
    scores: tuple[float, ...]
    window_id: str
    row: dict[str, Any]
    text: str

    def __lt__(self, other: "RankedSnippet") -> bool:
        if self.scores != other.scores:
            return self.scores < other.scores
        # For current selectors, smaller window_id is the deterministic winner.
        return self.window_id > other.window_id


class FixedTopKHeap:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self._heap: list[RankedSnippet] = []

    def push(self, entry: RankedSnippet) -> None:
        if self.capacity <= 0:
            return
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            return
        if self._heap[0] < entry:
            heapq.heapreplace(self._heap, entry)

    def entries(self) -> list[RankedSnippet]:
        return sorted(self._heap, reverse=True)


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = max(0, int(capacity))
        self.rng = random.Random(int(seed))
        self.seen = 0
        self.items: list[RankedSnippet] = []

    def add(self, item: RankedSnippet) -> None:
        self.seen += 1
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replace_idx = self.rng.randrange(self.seen)
        if replace_idx < self.capacity:
            self.items[replace_idx] = item


class StratifiedControlSampler:
    def __init__(
        self,
        *,
        selector: str,
        quotas: Counter[tuple[str, int, int]],
        seed: int,
        fallback_capacity: int,
    ) -> None:
        self.selector = selector
        self.quotas = Counter(quotas)
        self.reservoirs = {
            key: Reservoir(capacity, seed + idx * 997)
            for idx, (key, capacity) in enumerate(sorted(self.quotas.items()))
            if int(capacity) > 0
        }
        self.fallback = Reservoir(fallback_capacity, seed + 7919)
        self.total_seen = 0
        self.exact_seen = Counter()

    def add(self, row: dict[str, Any], text: str, *, stratum: tuple[str, int, int]) -> None:
        self.total_seen += 1
        entry = RankedSnippet(
            selector=self.selector,
            scores=(1.0 - _stable_random_unit("control", self.selector, row["window_id"]),),
            window_id=str(row["window_id"]),
            row=dict(row),
            text=text,
        )
        if stratum in self.reservoirs:
            self.exact_seen[stratum] += 1
            self.reservoirs[stratum].add(entry)
        self.fallback.add(entry)

    def finalize(self, target_count: int) -> tuple[list[RankedSnippet], dict[str, Any]]:
        selected: list[RankedSnippet] = []
        selected_ids: set[str] = set()
        for key in sorted(self.reservoirs):
            for item in self.reservoirs[key].items:
                if item.window_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.window_id)
        exact_count = len(selected)
        for item in self.fallback.items:
            if len(selected) >= int(target_count):
                break
            if item.window_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.window_id)
        selected = selected[: int(target_count)]
        return selected, {
            "target_count": int(target_count),
            "selected_count": int(len(selected)),
            "exact_stratified_count": int(exact_count),
            "fallback_count": int(max(0, len(selected) - exact_count)),
            "candidate_seen_count": int(self.total_seen),
            "quota_strata": int(len(self.quotas)),
            "filled_exact_strata": int(sum(1 for key, reservoir in self.reservoirs.items() if reservoir.items)),
        }


def _length_quotas(
    counts: Counter[int],
    *,
    top_k: int,
    mode: str,
    floor: int,
) -> Counter[int]:
    top_k = max(0, int(top_k))
    present = [length for length, count in sorted(counts.items()) if int(count) > 0]
    if top_k <= 0 or not present:
        return Counter()
    if mode == "none":
        return Counter({-1: top_k})
    if mode == "equal":
        base = top_k // len(present)
        remainder = top_k - base * len(present)
        return Counter({length: base + (1 if idx < remainder else 0) for idx, length in enumerate(present)})

    quotas = Counter()
    remaining = top_k
    floor = max(0, int(floor))
    if floor > 0 and floor * len(present) <= top_k:
        for length in present:
            quota = min(int(counts[length]), floor)
            quotas[length] = quota
            remaining -= quota

    available = {length: max(0, int(counts[length]) - int(quotas[length])) for length in present}
    total_available = sum(available.values())
    if remaining <= 0 or total_available <= 0:
        return quotas

    raw = {length: remaining * (available[length] / total_available) for length in present}
    adds = {length: min(available[length], int(raw[length])) for length in present}
    used = sum(adds.values())
    for length, value in adds.items():
        quotas[length] += value
    leftovers = remaining - used
    fractions = sorted(
        present,
        key=lambda length: (raw[length] - int(raw[length]), available[length]),
        reverse=True,
    )
    while leftovers > 0 and fractions:
        progressed = False
        for length in fractions:
            if leftovers <= 0:
                break
            if quotas[length] >= int(counts[length]):
                continue
            quotas[length] += 1
            leftovers -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _finalize_selector_entries(
    entries: Sequence[RankedSnippet],
    *,
    selector: str,
    eligible_length_counts: Counter[int],
    top_k: int,
    length_balance: str,
    length_quota_floor: int,
) -> list[RankedSnippet]:
    if length_balance == "none":
        return sorted(entries, reverse=True)[: int(top_k)]

    quotas = _length_quotas(
        eligible_length_counts,
        top_k=int(top_k),
        mode=length_balance,
        floor=int(length_quota_floor),
    )
    by_length: dict[int, list[RankedSnippet]] = defaultdict(list)
    for entry in entries:
        by_length[int(entry.row["snippet_sentence_count"])].append(entry)
    selected: list[RankedSnippet] = []
    selected_ids: set[str] = set()
    for length, quota in sorted(quotas.items()):
        for entry in sorted(by_length.get(length, []), reverse=True)[: int(quota)]:
            if entry.window_id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry.window_id)
    if len(selected) < int(top_k):
        for entry in sorted(entries, reverse=True):
            if len(selected) >= int(top_k):
                break
            if entry.window_id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry.window_id)
    return sorted(selected, reverse=True)[: int(top_k)]


def _make_snippet_rows_for_parent(
    *,
    record: dict[str, Any],
    text: str,
    parser_backend: str,
    spacy_model: str,
    nlp,
    min_sentences: int,
    max_sentences: int,
) -> list[tuple[dict[str, Any], str]]:
    windows = generate_sentence_windows(
        record,
        text=text,
        min_sentences=int(min_sentences),
        max_sentences=int(max_sentences),
        nlp=nlp if parser_backend == "spacy" else None,
    )
    rows: list[tuple[dict[str, Any], str]] = []
    for window in windows:
        snippet_text = str(window.pop("snippet_text"))
        features = compute_snippet_features(
            snippet_text,
            parser_backend=parser_backend,
            spacy_model=spacy_model,
            nlp=nlp if parser_backend == "spacy" else None,
        )
        row = ensure_selector_record(
            {
                **window,
                **features,
                "snippet_sha1": hashlib.sha1(snippet_text.encode("utf-8")).hexdigest(),
            }
        )
        rows.append((row, snippet_text))
    return rows


def _ranked_entry(selector: str, row: dict[str, Any], text: str) -> RankedSnippet:
    return RankedSnippet(
        selector=selector,
        scores=selector_score_values(row, selector),
        window_id=str(row["window_id"]),
        row=dict(row),
        text=text,
    )


def _select_local_controls(
    rows: Sequence[tuple[dict[str, Any], str]],
    *,
    selector: str,
    max_count: int,
    seed: int,
) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    selected_intervals: list[tuple[int, int]] = []
    ordered = sorted(
        rows,
        key=lambda pair: _stable_random_unit(seed, selector, pair[0]["window_id"]),
    )
    for row, text in ordered:
        interval = snippet_sentence_interval(row)
        if any(sentence_intervals_overlap(interval, existing) for existing in selected_intervals):
            continue
        selected.append((row, text))
        selected_intervals.append(interval)
        if len(selected) >= int(max_count):
            break
    return selected


def _rerank_with_spacy(
    entries_by_selector: dict[str, list[RankedSnippet]],
    *,
    spacy_model: str,
    show_progress: bool,
    nlp=None,
    backend_info: dict[str, Any] | None = None,
) -> tuple[dict[str, list[RankedSnippet]], dict[str, Any]]:
    if nlp is None:
        parser_backend, nlp, resolved_info = resolve_backend(parser_backend="spacy", spacy_model=spacy_model)
        if parser_backend != "spacy" or nlp is None:
            raise RuntimeError(f"spaCy rerank requested but unavailable: {resolved_info.detail}")
        backend_info = resolved_info.to_json()
    if backend_info is None:
        backend_info = {"requested": "spacy", "used": "spacy", "detail": str(spacy_model)}
    total = sum(len(entries) for entries in entries_by_selector.values())
    progress = _build_progress(enabled=show_progress, total=total, desc="spaCy reranking frontiers", unit="snippet")
    reranked: dict[str, list[RankedSnippet]] = {}
    try:
        for selector, entries in entries_by_selector.items():
            reranked_entries: list[RankedSnippet] = []
            for entry in entries:
                features = compute_snippet_features(
                    entry.text,
                    parser_backend="spacy",
                    spacy_model=spacy_model,
                    nlp=nlp,
                )
                row = ensure_selector_record({**entry.row, **features})
                if selector_passes_gate(row, selector):
                    reranked_entries.append(_ranked_entry(selector, row, entry.text))
                if progress is not None:
                    progress.update(1)
            reranked[selector] = reranked_entries
    finally:
        if progress is not None:
            progress.close()
    return reranked, backend_info


def _preview_entries(entries: Sequence[RankedSnippet], *, selector: str, pool_role: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    for rank, entry in enumerate(entries[: max(0, int(limit))], start=1):
        rows.append(_pool_row(entry.row, entry.text, selector=selector, rank=rank, pool_role=pool_role))
    return rows


def _stratified_entries(entries: Sequence[RankedSnippet], *, band_size: int) -> list[tuple[int, RankedSnippet]]:
    n = len(entries)
    if n <= 0 or int(band_size) <= 0:
        return []
    indices: list[int] = []
    for anchor in (0.0, 0.025, 0.10, 0.50, 0.90, 1.0):
        if anchor <= 0.0:
            start = 0
        elif anchor >= 1.0:
            start = max(0, n - int(band_size))
        else:
            center = int(round(anchor * (n - 1)))
            start = max(0, min(n - int(band_size), center - int(band_size) // 2))
        indices.extend(range(start, min(n, start + int(band_size))))
    seen: set[int] = set()
    result: list[tuple[int, RankedSnippet]] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        result.append((idx + 1, entries[idx]))
    return result


def _write_markdown_preview(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"## Rank {row['rank']} | {row['selector']} | parent {row['parent_candidate_id']} | "
                f"s{row['sentence_range'][0]}-s{row['sentence_range'][1]}\n\n"
            )
            handle.write(f"pool: {row['pool_role']}\n\n")
            handle.write(f"sort_key: {row['sort_key']}\n\n")
            handle.write(f"features: {json.dumps(row['features'], sort_keys=True)}\n\n")
            handle.write("> " + str(row["text"]).replace("\n", "\n> ") + "\n\n")


def _entries_to_frame_rows(
    *,
    selectors: Sequence[str],
    treated: dict[str, list[RankedSnippet]],
    controls: dict[str, list[RankedSnippet]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    text_by_id: dict[str, str] = {}
    flag_columns = [f"is_treated_{selector}" for selector in selectors] + [f"is_control_{selector}" for selector in selectors]

    def merge(entry: RankedSnippet, *, flag: str, pool_label: str) -> None:
        window_id = str(entry.window_id)
        if window_id not in rows_by_id:
            row = dict(entry.row)
            for column in flag_columns:
                row[column] = False
            row["is_random_control_pool"] = False
            row["pool_label"] = pool_label
            rows_by_id[window_id] = row
            text_by_id[window_id] = entry.text
        rows_by_id[window_id][flag] = True
        if flag.startswith("is_control_"):
            rows_by_id[window_id]["is_random_control_pool"] = True

    for selector, entries in treated.items():
        for entry in entries:
            merge(entry, flag=f"is_treated_{selector}", pool_label=f"treated_{selector}")
    for selector, entries in controls.items():
        for entry in entries:
            merge(entry, flag=f"is_control_{selector}", pool_label=f"control_{selector}")
    rows = sorted(rows_by_id.values(), key=lambda row: str(row["window_id"]))
    return rows, text_by_id


def _write_pool_files(
    output_dir: Path,
    *,
    selector: str,
    pool_name: str,
    entries: Sequence[RankedSnippet],
) -> tuple[str, str]:
    rows = []
    for rank, entry in enumerate(entries, start=1):
        row = dict(entry.row)
        row["rank"] = int(rank)
        row[f"is_{pool_name}_{selector}"] = True
        rows.append(row)
    csv_path = output_dir / f"{pool_name}_{selector}.csv"
    jsonl_path = output_dir / f"{pool_name}_{selector}.jsonl"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    _write_jsonl(
        jsonl_path,
        [
            {
                **row,
                "text": entry.text,
            }
            for row, entry in zip(rows, entries)
        ],
    )
    return str(csv_path), str(jsonl_path)


def _write_review_artifacts(
    output_dir: Path,
    *,
    selector: str,
    treated: Sequence[RankedSnippet],
    controls: Sequence[RankedSnippet],
    preview_top_n: int,
    stratified_preview_band_size: int,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    preview_dir = output_dir / "previews"
    review_dir = output_dir / "review"

    top_rows = _preview_entries(treated, selector=selector, pool_role="treated", limit=preview_top_n)
    top_jsonl = preview_dir / f"{selector}_top{int(preview_top_n)}.jsonl"
    top_md = review_dir / f"{selector}_top{int(preview_top_n)}.md"
    _write_jsonl(top_jsonl, top_rows)
    _write_markdown_preview(top_md, top_rows)
    artifacts[f"{selector}_top_preview_jsonl"] = str(top_jsonl)
    artifacts[f"{selector}_top_preview_md"] = str(top_md)

    stratified_rows = [
        _pool_row(entry.row, entry.text, selector=selector, rank=rank, pool_role="treated")
        for rank, entry in _stratified_entries(treated, band_size=stratified_preview_band_size)
    ]
    stratified_jsonl = preview_dir / f"{selector}_stratified.jsonl"
    stratified_md = review_dir / f"{selector}_stratified.md"
    _write_jsonl(stratified_jsonl, stratified_rows)
    _write_markdown_preview(stratified_md, stratified_rows)
    artifacts[f"{selector}_stratified_jsonl"] = str(stratified_jsonl)
    artifacts[f"{selector}_stratified_md"] = str(stratified_md)

    control_rows = _preview_entries(controls, selector=selector, pool_role="control", limit=preview_top_n)
    control_jsonl = preview_dir / f"{selector}_control_sample{int(preview_top_n)}.jsonl"
    control_md = review_dir / f"{selector}_control_sample{int(preview_top_n)}.md"
    _write_jsonl(control_jsonl, control_rows)
    _write_markdown_preview(control_md, control_rows)
    artifacts[f"{selector}_control_preview_jsonl"] = str(control_jsonl)
    artifacts[f"{selector}_control_preview_md"] = str(control_md)
    return artifacts


def _write_diagnostics(
    output_dir: Path,
    *,
    selectors: Sequence[str],
    treated: dict[str, list[RankedSnippet]],
    controls: dict[str, list[RankedSnippet]],
) -> dict[str, str]:
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    length_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for selector in selectors:
        for pool_name, entries in (("treated", treated.get(selector, [])), ("control", controls.get(selector, []))):
            length_counts = Counter(int(entry.row.get("snippet_sentence_count", 0)) for entry in entries)
            for length, count in sorted(length_counts.items()):
                length_rows.append({"selector": selector, "pool": pool_name, "snippet_sentence_count": length, "count": count})

            parent_counts = Counter(int(entry.row.get("parent_candidate_id", -1)) for entry in entries)
            for parent_candidate_id, count in parent_counts.most_common(200):
                parent_rows.append({"selector": selector, "pool": pool_name, "parent_candidate_id": parent_candidate_id, "count": count})

            shard_counts = Counter(Path(str(entry.row.get("shard_path", ""))).name for entry in entries)
            for shard, count in shard_counts.most_common(500):
                shard_rows.append({"selector": selector, "pool": pool_name, "shard": shard, "count": count})

            if entries:
                frame = pd.DataFrame.from_records([entry.row for entry in entries])
                for column in snippet_feature_columns():
                    if column not in frame.columns:
                        continue
                    values = pd.to_numeric(frame[column], errors="coerce").dropna()
                    if values.empty:
                        continue
                    feature_rows.append(
                        {
                            "selector": selector,
                            "pool": pool_name,
                            "feature": column,
                            "q00": float(values.quantile(0.0)),
                            "q25": float(values.quantile(0.25)),
                            "q50": float(values.quantile(0.50)),
                            "q75": float(values.quantile(0.75)),
                            "q100": float(values.quantile(1.0)),
                            "mean": float(values.mean()),
                        }
                    )

    paths = {
        "selector_length_distribution": str(diag_dir / "selector_length_distribution.csv"),
        "selector_parent_concentration": str(diag_dir / "selector_parent_concentration.csv"),
        "selector_shard_distribution": str(diag_dir / "selector_shard_distribution.csv"),
        "selector_feature_quantiles": str(diag_dir / "selector_feature_quantiles.csv"),
    }
    pd.DataFrame.from_records(length_rows).to_csv(paths["selector_length_distribution"], index=False)
    pd.DataFrame.from_records(parent_rows).to_csv(paths["selector_parent_concentration"], index=False)
    pd.DataFrame.from_records(shard_rows).to_csv(paths["selector_shard_distribution"], index=False)
    pd.DataFrame.from_records(feature_rows).to_csv(paths["selector_feature_quantiles"], index=False)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream decoded training candidates twice to mine non-overlapping top-K "
            "sentence snippets per selector and selector-specific matched controls."
        )
    )
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--decode_strategy", choices=("auto", "sparse", "manifest"), default="auto")
    parser.add_argument("--max_candidates", type=int, default=100_000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--selectors", type=str, default="all")
    parser.add_argument("--min_sentences", type=int, default=3)
    parser.add_argument("--max_sentences", type=int, default=6)
    parser.add_argument("--top_k", "--num_treated_snippets", type=int, default=10_000)
    parser.add_argument("--num_control_snippets", type=int, default=None)
    parser.add_argument("--max_snippets_per_parent_per_selector", type=int, default=3)
    parser.add_argument("--parser_backend", choices=("regex", "spacy"), default="regex")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--rerank_backend", choices=("none", "spacy"), default="none")
    parser.add_argument("--frontier_multiplier", type=float, default=1.0)
    parser.add_argument("--length_balance", choices=("none", "equal", "proportional"), default="proportional")
    parser.add_argument("--length_quota_floor", type=int, default=0)
    parser.add_argument("--control_token_bucket_width", type=int, default=16)
    parser.add_argument("--control_fallback_pool_size", type=int, default=50_000)
    parser.add_argument("--preview_top_n", type=int, default=200)
    parser.add_argument("--stratified_preview_band_size", type=int, default=25)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selectors = _parse_selectors(args.selectors)
    if int(args.top_k) <= 0:
        raise ValueError("--top_k must be > 0")
    if int(args.max_snippets_per_parent_per_selector) <= 0:
        raise ValueError("--max_snippets_per_parent_per_selector must be > 0")
    if int(args.min_sentences) <= 0 or int(args.max_sentences) < int(args.min_sentences):
        raise ValueError("Require 0 < --min_sentences <= --max_sentences")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_candidate_frame(args.candidate_csv)
    if int(args.max_candidates) > 0 and len(frame) > int(args.max_candidates):
        frame = (
            frame.sample(n=int(args.max_candidates), random_state=int(args.sample_seed), replace=False)
            .sort_values("candidate_id")
            .reset_index(drop=True)
        )

    parser_backend, nlp, backend_info = resolve_backend(parser_backend=args.parser_backend, spacy_model=args.spacy_model)
    print(
        "parser backend: "
        f"{backend_info.used} "
        f"(requested={backend_info.requested}; detail={backend_info.detail})"
    )
    rerank_nlp = None
    rerank_backend_info: dict[str, Any] | None = None
    if args.rerank_backend == "spacy":
        rerank_parser_backend, rerank_nlp, rerank_info_obj = resolve_backend(
            parser_backend="spacy",
            spacy_model=args.spacy_model,
        )
        if rerank_parser_backend != "spacy" or rerank_nlp is None:
            raise RuntimeError(f"spaCy rerank requested but unavailable: {rerank_info_obj.detail}")
        rerank_backend_info = rerank_info_obj.to_json()
        print(
            "rerank backend: "
            f"{rerank_info_obj.used} "
            f"(requested={rerank_info_obj.requested}; detail={rerank_info_obj.detail})"
        )
    decoder = CandidateTextDecoder(
        frame=frame,
        data_dir=Path(args.data_dir).expanduser().resolve(),
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        seq_len=args.seq_len,
        decode_strategy=str(args.decode_strategy),
    )

    frontier_capacity = max(int(args.top_k), int(np.ceil(float(args.top_k) * float(args.frontier_multiplier))))
    heaps: dict[str, dict[int, FixedTopKHeap]] = {
        selector: defaultdict(lambda capacity=frontier_capacity: FixedTopKHeap(capacity))
        for selector in selectors
    }
    eligible_length_counts: dict[str, Counter[int]] = {selector: Counter() for selector in selectors}
    pass1_counts: dict[str, Any] = {
        "parents_seen": 0,
        "snippets_seen": 0,
        "valid_snippets_seen": 0,
        "gate_pass_counts": {selector: 0 for selector in selectors},
        "local_nonoverlap_counts": {selector: 0 for selector in selectors},
    }

    records = frame.to_dict(orient="records")
    progress = _build_progress(enabled=bool(args.show_progress), total=len(records), desc="Pass 1 top-K mining", unit="parent")
    try:
        for row_index, record in enumerate(records):
            text = decoder.decode(record, row_index)
            snippets = _make_snippet_rows_for_parent(
                record=record,
                text=text,
                parser_backend=parser_backend,
                spacy_model=str(args.spacy_model),
                nlp=nlp,
                min_sentences=int(args.min_sentences),
                max_sentences=int(args.max_sentences),
            )
            pass1_counts["parents_seen"] += 1
            pass1_counts["snippets_seen"] += len(snippets)
            pass1_counts["valid_snippets_seen"] += sum(1 for row, _text in snippets if is_valid_snippet_record(row))

            for selector in selectors:
                local_rows: list[dict[str, Any]] = []
                text_by_window: dict[str, str] = {}
                for row, snippet_text in snippets:
                    if not selector_passes_gate(row, selector):
                        continue
                    pass1_counts["gate_pass_counts"][selector] += 1
                    eligible_length_counts[selector][int(row["snippet_sentence_count"])] += 1
                    local_rows.append(row)
                    text_by_window[str(row["window_id"])] = snippet_text
                selected_local = select_non_overlapping_snippets(
                    local_rows,
                    selector=selector,
                    max_count=int(args.max_snippets_per_parent_per_selector),
                )
                pass1_counts["local_nonoverlap_counts"][selector] += len(selected_local)
                for row in selected_local:
                    entry = _ranked_entry(selector, row, text_by_window[str(row["window_id"])])
                    length_bucket = int(row["snippet_sentence_count"]) if args.length_balance != "none" else -1
                    heaps[selector][length_bucket].push(entry)

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    frontier_entries = {
        selector: [
            entry
            for heap_by_length in heaps[selector].values()
            for entry in heap_by_length.entries()
        ]
        for selector in selectors
    }
    rerank_info: dict[str, Any] = {"ran": False, "backend": "none"}
    if args.rerank_backend == "spacy":
        frontier_entries, rerank_backend_info = _rerank_with_spacy(
            frontier_entries,
            spacy_model=str(args.spacy_model),
            show_progress=bool(args.show_progress),
            nlp=rerank_nlp,
            backend_info=rerank_backend_info,
        )
        rerank_info = {"ran": True, "backend": "spacy", "parser_backend": rerank_backend_info}

    treated_entries: dict[str, list[RankedSnippet]] = {}
    for selector in selectors:
        treated_entries[selector] = _finalize_selector_entries(
            frontier_entries.get(selector, []),
            selector=selector,
            eligible_length_counts=eligible_length_counts[selector],
            top_k=int(args.top_k),
            length_balance=str(args.length_balance),
            length_quota_floor=int(args.length_quota_floor),
        )

    treated_intervals: dict[str, dict[int, list[tuple[int, int]]]] = {
        selector: defaultdict(list) for selector in selectors
    }
    treated_window_ids: dict[str, set[str]] = {selector: set() for selector in selectors}
    treated_quotas: dict[str, Counter[tuple[str, int, int]]] = {selector: Counter() for selector in selectors}
    for selector, entries in treated_entries.items():
        for entry in entries:
            parent_id = int(entry.row["parent_candidate_id"])
            treated_intervals[selector][parent_id].append(snippet_sentence_interval(entry.row))
            treated_window_ids[selector].add(entry.window_id)
            treated_quotas[selector][_stratum_key(entry.row, token_bucket_width=int(args.control_token_bucket_width))] += 1

    control_target = int(args.num_control_snippets or args.top_k)
    control_samplers = {
        selector: StratifiedControlSampler(
            selector=selector,
            quotas=treated_quotas[selector],
            seed=int(args.sample_seed) + 1009 * idx,
            fallback_capacity=int(args.control_fallback_pool_size),
        )
        for idx, selector in enumerate(selectors)
    }
    pass2_counts: dict[str, Any] = {
        "parents_seen": 0,
        "candidate_control_counts": {selector: 0 for selector in selectors},
        "local_nonoverlap_control_counts": {selector: 0 for selector in selectors},
    }

    progress = _build_progress(enabled=bool(args.show_progress), total=len(records), desc="Pass 2 controls", unit="parent")
    try:
        for row_index, record in enumerate(records):
            text = decoder.decode(record, row_index)
            snippets = _make_snippet_rows_for_parent(
                record=record,
                text=text,
                parser_backend=parser_backend,
                spacy_model=str(args.spacy_model),
                nlp=nlp,
                min_sentences=int(args.min_sentences),
                max_sentences=int(args.max_sentences),
            )
            pass2_counts["parents_seen"] += 1
            parent_id = int(record["candidate_id"])
            for selector in selectors:
                local_control: list[tuple[dict[str, Any], str]] = []
                blocked_intervals = treated_intervals[selector].get(parent_id, [])
                for row, snippet_text in snippets:
                    if not is_valid_snippet_record(row):
                        continue
                    if str(row["window_id"]) in treated_window_ids[selector]:
                        continue
                    interval = snippet_sentence_interval(row)
                    if any(sentence_intervals_overlap(interval, blocked) for blocked in blocked_intervals):
                        continue
                    local_control.append((row, snippet_text))
                pass2_counts["candidate_control_counts"][selector] += len(local_control)
                selected_local = _select_local_controls(
                    local_control,
                    selector=selector,
                    max_count=int(args.max_snippets_per_parent_per_selector),
                    seed=int(args.sample_seed),
                )
                pass2_counts["local_nonoverlap_control_counts"][selector] += len(selected_local)
                for row, snippet_text in selected_local:
                    control_samplers[selector].add(
                        row,
                        snippet_text,
                        stratum=_stratum_key(row, token_bucket_width=int(args.control_token_bucket_width)),
                    )

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    control_entries: dict[str, list[RankedSnippet]] = {}
    control_summary: dict[str, Any] = {}
    for selector, sampler in control_samplers.items():
        entries, summary = sampler.finalize(control_target)
        control_entries[selector] = entries
        control_summary[selector] = summary

    feature_rows, text_by_id = _entries_to_frame_rows(
        selectors=selectors,
        treated=treated_entries,
        controls=control_entries,
    )
    feature_frame = pd.DataFrame.from_records(feature_rows)
    feature_frame.to_csv(output_dir / "snippet_features.csv", index=False)
    _write_jsonl(
        output_dir / "snippet_text.jsonl",
        [
            {
                "window_id": window_id,
                "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                "text_preview": preview_text(text),
                "text": text,
            }
            for window_id, text in sorted(text_by_id.items())
        ],
    )

    artifacts: dict[str, Any] = {
        "snippet_features": str(output_dir / "snippet_features.csv"),
        "snippet_text": str(output_dir / "snippet_text.jsonl"),
        "pools": {},
        "review": {},
    }
    for selector in selectors:
        treated_csv, treated_jsonl = _write_pool_files(output_dir, selector=selector, pool_name="treated", entries=treated_entries[selector])
        control_csv, control_jsonl = _write_pool_files(output_dir, selector=selector, pool_name="control", entries=control_entries[selector])
        artifacts["pools"][selector] = {
            "treated_csv": treated_csv,
            "treated_jsonl": treated_jsonl,
            "control_csv": control_csv,
            "control_jsonl": control_jsonl,
        }
        artifacts["review"].update(
            _write_review_artifacts(
                output_dir,
                selector=selector,
                treated=treated_entries[selector],
                controls=control_entries[selector],
                preview_top_n=int(args.preview_top_n),
                stratified_preview_band_size=int(args.stratified_preview_band_size),
            )
        )

    artifacts["diagnostics"] = _write_diagnostics(
        output_dir,
        selectors=selectors,
        treated=treated_entries,
        controls=control_entries,
    )

    summary = {
        "candidate_csv": str(Path(args.candidate_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
        "selectors": list(selectors),
        "top_k": int(args.top_k),
        "control_target": int(control_target),
        "max_candidates": int(args.max_candidates),
        "actual_candidates": int(len(frame)),
        "snippet_window": {
            "min_sentences": int(args.min_sentences),
            "max_sentences": int(args.max_sentences),
        },
        "nonoverlap": {
            "scope": "per_parent_per_selector",
            "max_snippets_per_parent_per_selector": int(args.max_snippets_per_parent_per_selector),
        },
        "length_balance": {
            "mode": str(args.length_balance),
            "length_quota_floor": int(args.length_quota_floor),
            "eligible_length_counts": {
                selector: {str(length): int(count) for length, count in sorted(eligible_length_counts[selector].items())}
                for selector in selectors
            },
        },
        "matching": {
            "control_pool": "per_selector",
            "stratum": ["parent_shard_basename", "snippet_sentence_count", "token_count_text_bucket"],
            "token_bucket_width": int(args.control_token_bucket_width),
        },
        "parser_backend": backend_info.to_json(),
        "rerank": rerank_info,
        "decode": decoder.info(),
        "pass1_counts": pass1_counts,
        "pass2_counts": pass2_counts,
        "selector_summary": {
            selector: {
                "frontier_count": int(len(frontier_entries.get(selector, []))),
                "treated_count": int(len(treated_entries.get(selector, []))),
                "control_count": int(len(control_entries.get(selector, []))),
                "control_summary": control_summary.get(selector, {}),
            }
            for selector in selectors
        },
        "feature_columns": list(snippet_feature_columns()),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
