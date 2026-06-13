#!/usr/bin/env python3
"""Build token-budgeted natural/synthetic text mixtures for spatial fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_utils.raw_bos_docs import RawBOSDocumentStream, ordered_raw_shard_paths


EXTRA_FIELDS = [
    "mix_source",
    "mix_token_estimate",
    "mix_synthetic_ratio",
    "mix_target_tokens",
    "mix_seed",
    "natural_doc_index",
    "natural_source_segments",
]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) + 1


def read_synthetic_rows(path: Path, text_column: str, tokenizer) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if text_column not in fieldnames:
            raise ValueError(f"{path} does not contain text column {text_column!r}")
        rows = []
        for row in reader:
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue
            out = dict(row)
            out["mix_source"] = "synthetic"
            out["mix_token_estimate"] = str(token_count(tokenizer, text))
            rows.append(out)
    if not rows:
        raise ValueError(f"No synthetic rows loaded from {path}")
    return rows, fieldnames


def select_token_prefix(
    rows: Sequence[Dict[str, str]],
    *,
    target_tokens: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, str]], int, Dict[str, object]]:
    if target_tokens <= 0:
        return [], 0, {"synthetic_grouped_selection": False}

    has_contrast_sets = any(str(row.get("contrast_set_id", "")).strip() for row in rows)
    if has_contrast_sets:
        groups: List[Tuple[str, List[Dict[str, str]]]] = []
        grouped: Dict[str, List[Dict[str, str]]] = {}
        singleton_idx = 0
        for row in rows:
            contrast_set_id = str(row.get("contrast_set_id", "")).strip()
            if contrast_set_id:
                key = f"contrast:{contrast_set_id}"
            else:
                key = f"row:{singleton_idx}"
                singleton_idx += 1
            if key not in grouped:
                grouped[key] = []
                groups.append((key, grouped[key]))
            grouped[key].append(row)

        rng.shuffle(groups)
        selected: List[Dict[str, str]] = []
        selected_keys: List[str] = []
        total = 0
        for key, group_rows in groups:
            selected.extend(dict(row) for row in group_rows)
            selected_keys.append(key)
            total += sum(int(row["mix_token_estimate"]) for row in group_rows)
            if total >= target_tokens:
                role_counts = Counter(
                    str(row.get("contrast_role", "")).strip()
                    for row in selected
                    if str(row.get("contrast_role", "")).strip()
                )
                family_counts = Counter(
                    str(row.get("contrast_family", "")).strip()
                    for row in selected
                    if str(row.get("contrast_family", "")).strip()
                )
                stats = {
                    "synthetic_grouped_selection": True,
                    "synthetic_groups_available": len(groups),
                    "synthetic_groups_selected": len(selected_keys),
                    "synthetic_contrastive_groups_available": sum(
                        1 for group_key, _ in groups if group_key.startswith("contrast:")
                    ),
                    "synthetic_contrastive_groups_selected": sum(
                        1 for group_key in selected_keys if group_key.startswith("contrast:")
                    ),
                    "synthetic_singleton_groups_available": sum(
                        1 for group_key, _ in groups if group_key.startswith("row:")
                    ),
                    "synthetic_singleton_groups_selected": sum(
                        1 for group_key in selected_keys if group_key.startswith("row:")
                    ),
                    "synthetic_contrast_role_counts_selected": dict(role_counts),
                    "synthetic_contrast_family_counts_selected": dict(family_counts),
                }
                return selected, total, stats

        raise ValueError(
            f"Synthetic rows only provide {total} grouped tokens, below requested target {target_tokens}."
        )

    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: List[Dict[str, str]] = []
    total = 0
    for row in shuffled:
        selected.append(dict(row))
        total += int(row["mix_token_estimate"])
        if total >= target_tokens:
            return selected, total, {
                "synthetic_grouped_selection": False,
                "synthetic_groups_available": len(rows),
                "synthetic_groups_selected": len(selected),
            }
    raise ValueError(
        f"Synthetic rows only provide {total} tokens, below requested target {target_tokens}."
    )


def natural_text_rows(
    *,
    data_dir: Path,
    tokenizer,
    target_tokens: int,
    seed: int,
    split: str,
    min_doc_tokens: int,
    max_doc_tokens: int,
    natural_skip_docs: int,
) -> Tuple[List[Dict[str, str]], int]:
    if target_tokens <= 0:
        return [], 0

    shard_paths = ordered_raw_shard_paths(data_dir, split=split)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise RuntimeError("Tokenizer must expose bos_token_id or eos_token_id.")

    stream = RawBOSDocumentStream(
        shard_paths,
        bos_token_id=int(bos_token_id),
        skip_leading_non_bos=True,
    )
    for _ in range(max(0, natural_skip_docs)):
        next(stream)

    rows: List[Dict[str, str]] = []
    total = 0
    docs_seen = 0
    while total < target_tokens:
        try:
            doc = next(stream)
        except StopIteration as exc:
            raise ValueError(
                f"Natural source exhausted at {total} tokens, below requested target {target_tokens}."
            ) from exc

        docs_seen += 1
        ids = [int(x) for x in doc.tokens.tolist()]
        if ids and ids[0] == int(bos_token_id):
            ids = ids[1:]
        if not ids:
            continue
        if max_doc_tokens > 0:
            ids = ids[:max_doc_tokens]
        if len(ids) < min_doc_tokens:
            continue

        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = " ".join(text.strip().split())
        if not text:
            continue

        count = token_count(tokenizer, text)
        if count < min_doc_tokens:
            continue

        rows.append(
            {
                "text": text,
                "context": "",
                "completion": "",
                "difficulty": "",
                "difficulty_label": "",
                "domain": "natural",
                "mix_source": "natural",
                "mix_token_estimate": str(count),
                "natural_doc_index": str(doc.docs_consumed),
                "natural_source_segments": json.dumps(
                    [
                        {
                            "shard_idx": seg.source_shard_idx,
                            "token_start": seg.source_token_start,
                            "token_end": seg.source_token_end,
                        }
                        for seg in doc.segments
                    ],
                    separators=(",", ":"),
                ),
            }
        )
        total += count

    return rows, total


def fieldnames_for(synthetic_fields: Sequence[str], rows: Iterable[Dict[str, str]]) -> List[str]:
    ordered = []
    for field in synthetic_fields:
        if field not in ordered:
            ordered.append(field)
    for field in ("text", "context", "completion", "difficulty", "difficulty_label", "domain"):
        if field not in ordered:
            ordered.append(field)
    for field in EXTRA_FIELDS:
        if field not in ordered:
            ordered.append(field)
    for row in rows:
        for field in row:
            if field not in ordered:
                ordered.append(field)
    return ordered


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-csv", type=Path, required=True)
    parser.add_argument("--natural-data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--target-tokens", type=int, default=400_000)
    parser.add_argument("--synthetic-token-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--natural-split", default="train")
    parser.add_argument("--natural-skip-docs", type=int, default=0)
    parser.add_argument("--min-natural-doc-tokens", type=int, default=16)
    parser.add_argument("--max-natural-doc-tokens", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be positive")
    if not 0.0 <= args.synthetic_token_ratio <= 1.0:
        raise ValueError("--synthetic-token-ratio must be in [0, 1]")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    synthetic_rows_all, synthetic_fields = read_synthetic_rows(
        args.synthetic_csv,
        args.text_column,
        tokenizer,
    )

    synthetic_target = int(round(args.target_tokens * args.synthetic_token_ratio))
    natural_target = max(0, int(args.target_tokens) - synthetic_target)

    synthetic_rows, synthetic_tokens, synthetic_selection_stats = select_token_prefix(
        synthetic_rows_all,
        target_tokens=synthetic_target,
        rng=random.Random(args.seed + 101),
    )
    natural_rows, natural_tokens = natural_text_rows(
        data_dir=args.natural_data_dir,
        tokenizer=tokenizer,
        target_tokens=natural_target,
        seed=args.seed,
        split=args.natural_split,
        min_doc_tokens=args.min_natural_doc_tokens,
        max_doc_tokens=args.max_natural_doc_tokens,
        natural_skip_docs=args.natural_skip_docs,
    )

    for row in synthetic_rows:
        row["mix_synthetic_ratio"] = f"{args.synthetic_token_ratio:.6g}"
        row["mix_target_tokens"] = str(args.target_tokens)
        row["mix_seed"] = str(args.seed)
    for row in natural_rows:
        row["mix_synthetic_ratio"] = f"{args.synthetic_token_ratio:.6g}"
        row["mix_target_tokens"] = str(args.target_tokens)
        row["mix_seed"] = str(args.seed)

    mixed_rows = [*natural_rows, *synthetic_rows]
    random.Random(args.seed + 303).shuffle(mixed_rows)
    fieldnames = fieldnames_for(synthetic_fields, mixed_rows)
    write_csv(args.out, mixed_rows, fieldnames)

    manifest = {
        "out": str(args.out),
        "synthetic_csv": str(args.synthetic_csv),
        "natural_data_dir": str(args.natural_data_dir),
        "tokenizer_name": str(args.tokenizer_name),
        "target_tokens": args.target_tokens,
        "synthetic_token_ratio": args.synthetic_token_ratio,
        "synthetic_target_tokens": synthetic_target,
        "natural_target_tokens": natural_target,
        "synthetic_actual_tokens": synthetic_tokens,
        "natural_actual_tokens": natural_tokens,
        "total_actual_tokens": synthetic_tokens + natural_tokens,
        "synthetic_rows": len(synthetic_rows),
        "natural_rows": len(natural_rows),
        "total_rows": len(mixed_rows),
        "synthetic_selection": synthetic_selection_stats,
        "seed": args.seed,
        "natural_split": args.natural_split,
        "natural_skip_docs": args.natural_skip_docs,
        "min_natural_doc_tokens": args.min_natural_doc_tokens,
        "max_natural_doc_tokens": args.max_natural_doc_tokens,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "Wrote mixed data: "
        f"{args.out} ({len(mixed_rows)} rows, "
        f"natural={natural_tokens} tokens, synthetic={synthetic_tokens} tokens, "
        f"total={synthetic_tokens + natural_tokens})"
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
