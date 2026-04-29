#!/usr/bin/env python3
"""Mine selector-scored 3-6 sentence snippets from decoded training candidates."""

from __future__ import annotations

import argparse
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .features import resolve_backend
from .mine_candidate_pools import (
    _decode_candidate_texts,
    _load_candidate_frame,
    _write_jsonl,
)
from .pool_selection import assign_selector_pools, cluster_selector_pool
from .sentence_windows import generate_sentence_windows
from .snippet_features import compute_snippet_features
from .selector_recipes import (
    SELECTOR_NAMES,
    selector_sort_columns,
    snippet_feature_columns,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def _build_progress(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode candidate training spans, create contiguous sentence snippets, "
            "score selector features, and export treated/random-control snippet pools."
        )
    )
    parser.add_argument("--candidate_csv", required=True, help="CSV with candidate metadata, usually row_summary_stepXXXXXXXX.csv")
    parser.add_argument("--data_dir", required=True, help="Training data directory backing those candidates")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint/tokenizer directory used to decode tokens")
    parser.add_argument("--output_dir", required=True, help="Where to write snippet features and selector pools")
    parser.add_argument("--seq_len", type=int, default=None, help="Optional explicit seq_len override")
    parser.add_argument(
        "--decode_strategy",
        choices=("auto", "sparse", "manifest"),
        default="auto",
        help="How to decode candidates before sentence splitting.",
    )
    parser.add_argument("--max_candidates", type=int, default=0, help="Deterministically subsample candidates before decoding")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for candidate subsampling and random controls")
    parser.add_argument(
        "--selectors",
        type=str,
        default="all",
        help=f"Comma-separated selectors or 'all'. Available: {','.join(SELECTOR_NAMES)}",
    )
    parser.add_argument("--min_sentences", type=int, default=3)
    parser.add_argument("--max_sentences", type=int, default=6)
    parser.add_argument("--num_treated_snippets", type=int, default=1024)
    parser.add_argument(
        "--random_control_size",
        type=int,
        default=None,
        help="Number of random normal snippets to export. Defaults to --num_treated_snippets.",
    )
    parser.add_argument("--parser_backend", choices=("auto", "spacy", "regex"), default="auto")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--embedding_backend", choices=("none", "tfidf_svd", "sentence_transformers"), default="none")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--min_cluster_size", type=int, default=8)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    return parser


def _build_snippet_rows(
    *,
    frame: pd.DataFrame,
    texts: dict[int, str],
    parser_backend: str,
    spacy_model: str,
    nlp,
    min_sentences: int,
    max_sentences: int,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    records = frame.to_dict(orient="records")
    total_candidates = len(records)
    progress = _build_progress(
        enabled=show_progress,
        total=total_candidates,
        desc="Generating sentence snippets",
        unit="candidate",
    )
    rows: list[dict[str, Any]] = []
    text_lookup: dict[str, str] = {}
    preview_lookup: dict[str, str] = {}
    try:
        for record in records:
            candidate_id = int(record["candidate_id"])
            text = texts[candidate_id]
            windows = generate_sentence_windows(
                record,
                text=text,
                min_sentences=int(min_sentences),
                max_sentences=int(max_sentences),
                nlp=nlp if parser_backend == "spacy" else None,
            )
            for window in windows:
                snippet_text = str(window.pop("snippet_text"))
                window_id = str(window["window_id"])
                text_lookup[window_id] = snippet_text
                preview_lookup[window_id] = str(window["snippet_text_preview"])
                features = compute_snippet_features(
                    snippet_text,
                    parser_backend=parser_backend,
                    spacy_model=spacy_model,
                    nlp=nlp if parser_backend == "spacy" else None,
                )
                rows.append(
                    {
                        **window,
                        **features,
                        "snippet_sha1": sha1(snippet_text.encode("utf-8")).hexdigest(),
                    }
                )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return rows, text_lookup, preview_lookup


def _pool_json_rows(feature_frame: pd.DataFrame, *, text_lookup: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in feature_frame.to_dict(orient="records"):
        window_id = str(record["window_id"])
        rows.append({**record, "text": text_lookup[window_id]})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selectors = _parse_selectors(args.selectors)
    if int(args.min_sentences) <= 0:
        raise ValueError("--min_sentences must be > 0")
    if int(args.max_sentences) < int(args.min_sentences):
        raise ValueError("--max_sentences must be >= --min_sentences")
    if int(args.num_treated_snippets) <= 0:
        raise ValueError("--num_treated_snippets must be > 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_candidate_frame(args.candidate_csv)
    if int(args.max_candidates) > 0 and len(frame) > int(args.max_candidates):
        frame = (
            frame.sample(n=int(args.max_candidates), random_state=int(args.sample_seed), replace=False)
            .sort_values("candidate_id")
            .reset_index(drop=True)
        )

    texts, decode_info = _decode_candidate_texts(
        frame,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        show_progress=bool(args.show_progress),
        decode_strategy=str(args.decode_strategy),
    )
    parser_backend, nlp, backend_info = resolve_backend(
        parser_backend=args.parser_backend,
        spacy_model=args.spacy_model,
    )
    print(
        "parser backend: "
        f"{backend_info.used} "
        f"(requested={backend_info.requested}; detail={backend_info.detail})"
    )

    snippet_rows, text_lookup, preview_lookup = _build_snippet_rows(
        frame=frame,
        texts=texts,
        parser_backend=parser_backend,
        spacy_model=str(args.spacy_model),
        nlp=nlp,
        min_sentences=int(args.min_sentences),
        max_sentences=int(args.max_sentences),
        show_progress=bool(args.show_progress),
    )
    feature_frame = pd.DataFrame.from_records(snippet_rows)
    if feature_frame.empty:
        raise ValueError("No sentence snippets were generated from the supplied candidates")

    feature_frame, selector_summary = assign_selector_pools(
        feature_frame,
        selectors=selectors,
        num_treated_snippets=int(args.num_treated_snippets),
        random_seed=int(args.sample_seed),
        random_control_size=args.random_control_size,
    )
    feature_frame = feature_frame.sort_values(["parent_candidate_id", "sentence_start_idx", "sentence_end_idx"]).reset_index(drop=True)
    feature_frame.to_csv(output_dir / "snippet_features.csv", index=False)

    text_rows = [
        {
            "window_id": str(window_id),
            "text_sha1": sha1(text.encode("utf-8")).hexdigest(),
            "text_preview": preview_lookup[str(window_id)],
            "text": text,
        }
        for window_id, text in sorted(text_lookup.items())
    ]
    _write_jsonl(output_dir / "snippet_text.jsonl", text_rows)

    artifacts: dict[str, str] = {
        "snippet_features": str(output_dir / "snippet_features.csv"),
        "snippet_text": str(output_dir / "snippet_text.jsonl"),
    }
    for selector in selectors:
        pool = feature_frame.loc[feature_frame[f"is_treated_{selector}"].astype(bool)].copy()
        sort_columns, ascending = selector_sort_columns(selector)
        pool = pool.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        csv_path = output_dir / f"treated_{selector}.csv"
        jsonl_path = output_dir / f"treated_{selector}.jsonl"
        pool.to_csv(csv_path, index=False)
        _write_jsonl(jsonl_path, _pool_json_rows(pool, text_lookup=text_lookup))
        artifacts[f"treated_{selector}"] = str(csv_path)
        artifacts[f"treated_{selector}_jsonl"] = str(jsonl_path)

    random_pool = feature_frame.loc[feature_frame["is_random_control_pool"].astype(bool)].copy()
    random_pool = random_pool.sort_values(["parent_candidate_id", "sentence_start_idx", "sentence_end_idx"]).reset_index(drop=True)
    random_pool.to_csv(output_dir / "random_control_pool.csv", index=False)
    _write_jsonl(output_dir / "random_control_pool.jsonl", _pool_json_rows(random_pool, text_lookup=text_lookup))
    artifacts["random_control_pool"] = str(output_dir / "random_control_pool.csv")
    artifacts["random_control_pool_jsonl"] = str(output_dir / "random_control_pool.jsonl")

    cluster_summary: dict[str, Any] = {"ran": False, "reason": "embedding_backend=none"}
    if args.embedding_backend != "none":
        cluster_summary = {}
        for selector in selectors:
            clustered, cluster_summary_frame, payload = cluster_selector_pool(
                feature_frame,
                selector=selector,
                text_lookup=text_lookup,
                embedding_backend=str(args.embedding_backend),
                embedding_model=str(args.embedding_model),
                num_clusters=int(args.num_clusters),
                min_cluster_size=int(args.min_cluster_size),
                seed=int(args.sample_seed),
            )
            cluster_summary[selector] = payload
            if payload.get("ran"):
                assignments_path = output_dir / f"cluster_assignments_{selector}.csv"
                summary_path = output_dir / f"cluster_summary_{selector}.csv"
                clustered[["window_id", "cluster_id", "snippet_text_preview"]].to_csv(assignments_path, index=False)
                cluster_summary_frame.to_csv(summary_path, index=False)
                artifacts[f"cluster_assignments_{selector}"] = str(assignments_path)
                artifacts[f"cluster_summary_{selector}"] = str(summary_path)

    summary = {
        "candidate_csv": str(Path(args.candidate_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
        "decode": decode_info,
        "parser_backend": backend_info.to_json(),
        "selectors": list(selectors),
        "snippet_window": {
            "min_sentences": int(args.min_sentences),
            "max_sentences": int(args.max_sentences),
        },
        "feature_columns": list(snippet_feature_columns()),
        "selector_summary": selector_summary,
        "cluster_summary": cluster_summary,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
