#!/usr/bin/env python3
"""Batch materialize multiple selector snippet pools into CPT datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..attribution.common.checkpoints import load_tokenizer_from_checkpoint
from ..attribution.trackstar.cpt_ablation import load_checkpoint_model_config
from .materialize_sentence_snippet_pools import materialize_selector_snippet_pool
from .selector_recipes import SELECTOR_NAMES


def _read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_selectors(raw: str | None, available: Sequence[str]) -> list[str]:
    if raw is None or str(raw).strip().lower() == "all":
        return list(available)
    requested = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Requested selectors are missing from manifest: {unknown!r}")
    return requested


def _coerce_manifest_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "selectors" in payload:
        payload = payload["selectors"]

    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for selector, spec in payload.items():
            if isinstance(spec, str):
                records.append({"selector": str(selector), "input_dir": spec})
            elif isinstance(spec, dict):
                records.append({"selector": str(spec.get("selector", selector)), **spec})
            else:
                raise TypeError(
                    f"Manifest entry for selector {selector!r} must be a directory string or object, got {type(spec)}"
                )
    elif isinstance(payload, list):
        for index, spec in enumerate(payload):
            if not isinstance(spec, dict):
                raise TypeError(f"Manifest list entry {index} must be an object, got {type(spec)}")
            if "selector" not in spec:
                raise ValueError(f"Manifest list entry {index} is missing 'selector'")
            records.append(dict(spec))
    else:
        raise TypeError("Manifest must be a selector mapping, a {'selectors': ...} object, or a list of selector objects")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        selector = str(record.get("selector", "")).strip()
        if selector not in SELECTOR_NAMES:
            raise ValueError(f"Unknown selector {selector!r}; expected one of {SELECTOR_NAMES!r}")
        if selector in seen:
            raise ValueError(f"Duplicate selector in manifest: {selector!r}")
        seen.add(selector)
        normalized.append({**record, "selector": selector})
    return normalized


def _resolve_selector_inputs(record: dict[str, Any]) -> tuple[Path, Path]:
    input_dir = record.get("input_dir")
    features = record.get("snippet_features_csv")
    text = record.get("snippet_text_jsonl")
    if input_dir:
        input_path = Path(str(input_dir)).expanduser().resolve()
        features = features or input_path / "snippet_features.csv"
        text = text or input_path / "snippet_text.jsonl"
    if not features or not text:
        raise ValueError(
            f"Selector {record['selector']!r} needs either 'input_dir' or both "
            "'snippet_features_csv' and 'snippet_text_jsonl'"
        )
    features_path = Path(features).expanduser().resolve()
    text_path = Path(text).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(f"Missing snippet_features_csv for {record['selector']}: {features_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing snippet_text_jsonl for {record['selector']}: {text_path}")
    return features_path, text_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize multiple selector-scored sentence-snippet pools into "
            "child directories consumable by run_cpt_ablation.py."
        )
    )
    parser.add_argument("--manifest", required=True, help="JSON selector manifest")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--selectors",
        default="all",
        help="Comma-separated selector subset from the manifest, or 'all'",
    )
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--rows_per_shard", type=int, default=50_000)
    parser.add_argument("--num_treated_snippets", type=int, default=None)
    parser.add_argument("--num_control_snippets", type=int, default=None)
    parser.add_argument("--selection_seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    records = _coerce_manifest_records(_read_json(manifest_path))
    record_by_selector = {str(record["selector"]): record for record in records}
    selectors = _parse_selectors(args.selectors, list(record_by_selector))

    model_config = load_checkpoint_model_config(checkpoint_dir)
    seq_len = int(args.seq_len or model_config.seq_len)
    if seq_len <= 0:
        raise ValueError("--seq_len must be > 0")
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)

    summaries: dict[str, Any] = {}
    for selector in selectors:
        record = record_by_selector[selector]
        features_path, text_path = _resolve_selector_inputs(record)
        selector_output_dir = output_dir / str(record.get("output_name", selector))
        print(f"materializing {selector} -> {selector_output_dir}")
        summaries[selector] = materialize_selector_snippet_pool(
            snippet_features_csv=features_path,
            snippet_text_jsonl=text_path,
            selector=selector,
            checkpoint_dir=checkpoint_dir,
            output_dir=selector_output_dir,
            seq_len=seq_len,
            rows_per_shard=int(args.rows_per_shard),
            num_treated_snippets=record.get("num_treated_snippets", args.num_treated_snippets),
            num_control_snippets=record.get("num_control_snippets", args.num_control_snippets),
            selection_seed=int(record.get("selection_seed", args.selection_seed)),
            tokenizer=tokenizer,
        )

    summary = {
        "created_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "selectors": selectors,
        "seq_len": int(seq_len),
        "rows_per_shard": int(args.rows_per_shard),
        "num_treated_snippets": args.num_treated_snippets,
        "num_control_snippets": args.num_control_snippets,
        "selection_seed": int(args.selection_seed),
        "selector_summaries": summaries,
    }
    _write_json(output_dir / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
