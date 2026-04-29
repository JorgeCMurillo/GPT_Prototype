#!/usr/bin/env python3
"""Rerank cached sentence-snippet mining outputs without re-decoding data.

This is the fast iteration path for selector recipes. It consumes an existing
`snippet_features.csv` plus `snippet_text.jsonl`, recomputes current snippet
features from the cached text, applies current selector gates/scores, rebuilds
treated/control pools, and writes the same pool/review/diagnostic artifacts as
the streaming miner.

Important limitation: this only reranks snippets present in the cache. It cannot
recover sentence windows discarded by the original streaming top-K pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .features import preview_text, resolve_backend
from .mine_sentence_snippets_streaming import (
    RankedSnippet,
    StratifiedControlSampler,
    _build_progress,
    _entries_to_frame_rows,
    _finalize_selector_entries,
    _parse_selectors,
    _ranked_entry,
    _select_local_controls,
    _stratum_key,
    _write_diagnostics,
    _write_json,
    _write_jsonl,
    _write_pool_files,
    _write_review_artifacts,
)
from .selector_recipes import (
    SELECTOR_NAMES,
    ensure_selector_record,
    is_valid_snippet_record,
    select_non_overlapping_snippets,
    selector_passes_gate,
    sentence_intervals_overlap,
    snippet_feature_columns,
    snippet_sentence_interval,
)
from .snippet_features import _layout_noise_features, compute_snippet_features


POOL_PREFIXES = ("is_treated_", "is_control_")


def _load_text_cache(path: Path) -> dict[str, str]:
    text_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            window_id = str(row["window_id"])
            text_by_id[window_id] = str(row.get("text", ""))
    return text_by_id


def _strip_pool_columns(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in row.items():
        if any(str(key).startswith(prefix) for prefix in POOL_PREFIXES):
            continue
        if key in {"is_random_control_pool", "pool_label", "rank"}:
            continue
        cleaned[key] = value
    return cleaned


def _load_cached_rows(
    *,
    snippet_features_csv: Path,
    snippet_text_jsonl: Path,
    parser_backend: str,
    spacy_model: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    feature_frame = pd.read_csv(snippet_features_csv)
    text_by_id = _load_text_cache(snippet_text_jsonl)

    nlp = None
    backend_info = {
        "requested": parser_backend,
        "used": parser_backend,
        "detail": "cached scalar features reused; layout features recomputed from cached text",
    }
    if parser_backend != "cache":
        resolved_backend, nlp, resolved_info = resolve_backend(
            parser_backend=parser_backend,
            spacy_model=spacy_model,
        )
        backend_info = resolved_info.to_json()
        if resolved_backend != parser_backend:
            backend_info["note"] = "requested backend fell back during cache rerank"

    progress = _build_progress(
        enabled=show_progress,
        total=len(feature_frame),
        desc="Reranking cached snippets",
        unit="snippet",
    )
    rows_by_id: dict[str, dict[str, Any]] = {}
    try:
        for raw_row in feature_frame.to_dict(orient="records"):
            window_id = str(raw_row["window_id"])
            text = text_by_id.get(window_id, "")
            if not text:
                if progress is not None:
                    progress.update(1)
                continue
            row = _strip_pool_columns(dict(raw_row))
            sentence_count = int(row.get("sentence_count") or row.get("snippet_sentence_count") or 1)
            if parser_backend != "cache":
                features = compute_snippet_features(
                    text,
                    parser_backend=parser_backend,
                    spacy_model=spacy_model,
                    nlp=nlp,
                )
                row.update(features)
            else:
                row.update(_layout_noise_features(text, sentence_count))
            row["snippet_text_preview"] = preview_text(text)
            row["snippet_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
            rows_by_id[window_id] = ensure_selector_record(row)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    rows = sorted(rows_by_id.values(), key=lambda row: str(row["window_id"]))
    return rows, text_by_id, backend_info


def _eligible_entries_by_selector(
    rows: Sequence[dict[str, Any]],
    text_by_id: dict[str, str],
    *,
    selectors: Sequence[str],
    max_snippets_per_parent_per_selector: int,
) -> tuple[dict[str, list[RankedSnippet]], dict[str, Counter[int]], dict[str, Any]]:
    entries_by_selector: dict[str, list[RankedSnippet]] = {selector: [] for selector in selectors}
    eligible_length_counts: dict[str, Counter[int]] = {selector: Counter() for selector in selectors}
    gate_pass_counts = {selector: 0 for selector in selectors}
    local_nonoverlap_counts = {selector: 0 for selector in selectors}

    rows_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_parent[int(row["parent_candidate_id"])].append(row)

    for parent_rows in rows_by_parent.values():
        for selector in selectors:
            local_rows: list[dict[str, Any]] = []
            for row in parent_rows:
                if not selector_passes_gate(row, selector):
                    continue
                gate_pass_counts[selector] += 1
                eligible_length_counts[selector][int(row["snippet_sentence_count"])] += 1
                local_rows.append(row)
            selected_local = select_non_overlapping_snippets(
                local_rows,
                selector=selector,
                max_count=max_snippets_per_parent_per_selector,
            )
            local_nonoverlap_counts[selector] += len(selected_local)
            for row in selected_local:
                window_id = str(row["window_id"])
                entries_by_selector[selector].append(_ranked_entry(selector, row, text_by_id[window_id]))

    return entries_by_selector, eligible_length_counts, {
        "gate_pass_counts": gate_pass_counts,
        "local_nonoverlap_counts": local_nonoverlap_counts,
    }


def _build_control_entries(
    rows: Sequence[dict[str, Any]],
    text_by_id: dict[str, str],
    *,
    selectors: Sequence[str],
    treated_entries: dict[str, list[RankedSnippet]],
    control_target: int,
    token_bucket_width: int,
    max_snippets_per_parent_per_selector: int,
    control_fallback_pool_size: int,
    sample_seed: int,
) -> tuple[dict[str, list[RankedSnippet]], dict[str, Any], dict[str, Any]]:
    treated_intervals: dict[str, dict[int, list[tuple[int, int]]]] = {
        selector: defaultdict(list) for selector in selectors
    }
    treated_window_ids: dict[str, set[str]] = {selector: set() for selector in selectors}
    treated_quotas: dict[str, Counter[tuple[str, int, int]]] = {selector: Counter() for selector in selectors}
    for selector, entries in treated_entries.items():
        for entry in entries:
            parent_id = int(entry.row["parent_candidate_id"])
            treated_intervals[selector][parent_id].append(snippet_sentence_interval(entry.row))
            treated_window_ids[selector].add(str(entry.window_id))
            treated_quotas[selector][_stratum_key(entry.row, token_bucket_width=token_bucket_width)] += 1

    samplers = {
        selector: StratifiedControlSampler(
            selector=selector,
            quotas=treated_quotas[selector],
            seed=int(sample_seed) + 1009 * idx,
            fallback_capacity=int(control_fallback_pool_size),
        )
        for idx, selector in enumerate(selectors)
    }
    candidate_control_counts = {selector: 0 for selector in selectors}
    local_nonoverlap_control_counts = {selector: 0 for selector in selectors}
    rows_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_parent[int(row["parent_candidate_id"])].append(row)

    for parent_id, parent_rows in rows_by_parent.items():
        for selector in selectors:
            local_control: list[tuple[dict[str, Any], str]] = []
            blocked_intervals = treated_intervals[selector].get(parent_id, [])
            for row in parent_rows:
                if not is_valid_snippet_record(row):
                    continue
                if str(row["window_id"]) in treated_window_ids[selector]:
                    continue
                interval = snippet_sentence_interval(row)
                if any(sentence_intervals_overlap(interval, blocked) for blocked in blocked_intervals):
                    continue
                text = text_by_id.get(str(row["window_id"]), "")
                if not text:
                    continue
                local_control.append((row, text))
            candidate_control_counts[selector] += len(local_control)
            selected_local = _select_local_controls(
                local_control,
                selector=selector,
                max_count=max_snippets_per_parent_per_selector,
                seed=sample_seed,
            )
            local_nonoverlap_control_counts[selector] += len(selected_local)
            for row, text in selected_local:
                samplers[selector].add(
                    row,
                    text,
                    stratum=_stratum_key(row, token_bucket_width=token_bucket_width),
                )

    control_entries: dict[str, list[RankedSnippet]] = {}
    control_summary: dict[str, Any] = {}
    for selector, sampler in samplers.items():
        entries, summary = sampler.finalize(control_target)
        control_entries[selector] = entries
        control_summary[selector] = summary
    return control_entries, control_summary, {
        "candidate_control_counts": candidate_control_counts,
        "local_nonoverlap_control_counts": local_nonoverlap_control_counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank cached sentence-snippet mining outputs and rebuild treated/control "
            "JSONL/CSV pools without re-decoding the parent dataset."
        )
    )
    parser.add_argument("--input_dir", default=None, help="Directory containing snippet_features.csv and snippet_text.jsonl")
    parser.add_argument("--snippet_features_csv", default=None)
    parser.add_argument("--snippet_text_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selectors", type=str, default="all")
    parser.add_argument("--top_k", "--num_treated_snippets", type=int, default=10_000)
    parser.add_argument("--num_control_snippets", type=int, default=None)
    parser.add_argument("--max_snippets_per_parent_per_selector", type=int, default=3)
    parser.add_argument(
        "--parser_backend",
        choices=("cache", "regex", "spacy"),
        default="cache",
        help=(
            "cache reuses cached scalar features and recomputes layout features only; "
            "regex/spacy recompute all snippet features from cached text."
        ),
    )
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--length_balance", choices=("none", "equal", "proportional"), default="proportional")
    parser.add_argument("--length_quota_floor", type=int, default=0)
    parser.add_argument("--control_token_bucket_width", type=int, default=16)
    parser.add_argument("--control_fallback_pool_size", type=int, default=50_000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--preview_top_n", type=int, default=200)
    parser.add_argument("--stratified_preview_band_size", type=int, default=25)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    parser.set_defaults(show_progress=True)
    return parser


def _resolve_cache_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        features_path = input_dir / "snippet_features.csv"
        text_path = input_dir / "snippet_text.jsonl"
    else:
        if not args.snippet_features_csv or not args.snippet_text_jsonl:
            raise ValueError("Provide either --input_dir or both --snippet_features_csv and --snippet_text_jsonl")
        features_path = Path(args.snippet_features_csv).expanduser().resolve()
        text_path = Path(args.snippet_text_jsonl).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    return features_path, text_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selectors = _parse_selectors(args.selectors)
    if int(args.top_k) <= 0:
        raise ValueError("--top_k must be > 0")
    if int(args.max_snippets_per_parent_per_selector) <= 0:
        raise ValueError("--max_snippets_per_parent_per_selector must be > 0")

    snippet_features_csv, snippet_text_jsonl = _resolve_cache_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, text_by_id, backend_info = _load_cached_rows(
        snippet_features_csv=snippet_features_csv,
        snippet_text_jsonl=snippet_text_jsonl,
        parser_backend=str(args.parser_backend),
        spacy_model=str(args.spacy_model),
        show_progress=bool(args.show_progress),
    )
    entries_by_selector, eligible_length_counts, candidate_summary = _eligible_entries_by_selector(
        rows,
        text_by_id,
        selectors=selectors,
        max_snippets_per_parent_per_selector=int(args.max_snippets_per_parent_per_selector),
    )

    treated_entries: dict[str, list[RankedSnippet]] = {}
    for selector in selectors:
        treated_entries[selector] = _finalize_selector_entries(
            entries_by_selector.get(selector, []),
            selector=selector,
            eligible_length_counts=eligible_length_counts[selector],
            top_k=int(args.top_k),
            length_balance=str(args.length_balance),
            length_quota_floor=int(args.length_quota_floor),
        )

    control_target = int(args.num_control_snippets or args.top_k)
    control_entries, control_summary, control_candidate_summary = _build_control_entries(
        rows,
        text_by_id,
        selectors=selectors,
        treated_entries=treated_entries,
        control_target=control_target,
        token_bucket_width=int(args.control_token_bucket_width),
        max_snippets_per_parent_per_selector=int(args.max_snippets_per_parent_per_selector),
        control_fallback_pool_size=int(args.control_fallback_pool_size),
        sample_seed=int(args.sample_seed),
    )

    feature_rows, output_text_by_id = _entries_to_frame_rows(
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
            for window_id, text in sorted(output_text_by_id.items())
        ],
    )

    artifacts: dict[str, Any] = {
        "snippet_features": str(output_dir / "snippet_features.csv"),
        "snippet_text": str(output_dir / "snippet_text.jsonl"),
        "pools": {},
        "review": {},
    }
    for selector in selectors:
        treated_csv, treated_jsonl = _write_pool_files(
            output_dir,
            selector=selector,
            pool_name="treated",
            entries=treated_entries[selector],
        )
        control_csv, control_jsonl = _write_pool_files(
            output_dir,
            selector=selector,
            pool_name="control",
            entries=control_entries[selector],
        )
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
        "input": {
            "snippet_features_csv": str(snippet_features_csv),
            "snippet_text_jsonl": str(snippet_text_jsonl),
        },
        "output_dir": str(output_dir),
        "mode": "cached_rerank",
        "limitation": "Only snippets present in the input cache were eligible; discarded original windows cannot be recovered.",
        "selectors": list(selectors),
        "top_k": int(args.top_k),
        "control_target": int(control_target),
        "cache_rows_loaded": int(len(rows)),
        "text_rows_loaded": int(len(text_by_id)),
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
            "control_pool": "per_selector_from_cached_snippets",
            "stratum": ["parent_shard_basename", "snippet_sentence_count", "token_count_text_bucket"],
            "token_bucket_width": int(args.control_token_bucket_width),
        },
        "parser_backend": backend_info,
        "candidate_counts": candidate_summary,
        "control_candidate_counts": control_candidate_summary,
        "selector_summary": {
            selector: {
                "eligible_after_gate_count": int(candidate_summary["gate_pass_counts"].get(selector, 0)),
                "local_nonoverlap_count": int(candidate_summary["local_nonoverlap_counts"].get(selector, 0)),
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
