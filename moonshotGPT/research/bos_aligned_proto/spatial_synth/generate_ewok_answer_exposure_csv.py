#!/usr/bin/env python3
"""Build an intentional EWoK answer-exposure dataset.

Each EWoK item contributes the correct BabyLM completion-choice pairs:

- C_1, T_1: Context1 followed by Target1
- C_2, T_2: Context2 followed by Target2

This is deliberately not a fair generalization dataset. It is a memorization
probe: after fine-tuning on the correct answers, the usual EWoK evaluator can
show how quickly scores and margins move at each checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ewok_data import load_ewok_df


FIELDNAMES = [
    "text",
    "context",
    "completion",
    "difficulty",
    "domain",
    "ewok_row_index",
    "answer_side",
    "context_column",
    "target_column",
    "concept_a",
    "concept_b",
    "context_type",
    "context_diff",
    "target_diff",
    "ewok_variant",
]


def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def selected_domains(domains: Sequence[str] | None) -> set[str] | None:
    if not domains:
        return None
    out = {domain.strip() for domain in domains if domain.strip()}
    return out or None


def iter_answer_rows(
    df: pd.DataFrame,
    *,
    ewok_variant: str,
    sides: Sequence[str],
    domains: Sequence[str] | None = None,
) -> Iterable[dict[str, str]]:
    domain_filter = selected_domains(domains)
    valid_sides = {"official", "symmetric"}
    requested_sides = tuple(sides)
    unknown = set(requested_sides).difference(valid_sides)
    if unknown:
        raise ValueError(f"Unknown sides {sorted(unknown)}; expected any of {sorted(valid_sides)}")

    for row_index, row in df.iterrows():
        domain = normalize_text(row.get("Domain", ""))
        if domain_filter is not None and domain not in domain_filter:
            continue
        pairs = []
        if "official" in requested_sides:
            pairs.append(("C1_T1", "Context1", "Target1"))
        if "symmetric" in requested_sides:
            pairs.append(("C2_T2", "Context2", "Target2"))

        for answer_side, context_col, target_col in pairs:
            context = normalize_text(row.get(context_col, ""))
            completion = normalize_text(row.get(target_col, ""))
            if not context or not completion:
                continue
            yield {
                "text": f"{context} {completion}",
                "context": context,
                "completion": completion,
                "difficulty": "",
                "domain": domain,
                "ewok_row_index": str(int(row_index)),
                "answer_side": answer_side,
                "context_column": context_col,
                "target_column": target_col,
                "concept_a": normalize_text(row.get("ConceptA", "")),
                "concept_b": normalize_text(row.get("ConceptB", "")),
                "context_type": normalize_text(row.get("ContextType", "")),
                "context_diff": normalize_text(row.get("ContextDiff", "")),
                "target_diff": normalize_text(row.get("TargetDiff", "")),
                "ewok_variant": ewok_variant,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewok-variant", choices=("fast", "full"), default="fast")
    parser.add_argument(
        "--sides",
        choices=("both", "official", "symmetric"),
        default="both",
        help="Which correct answer pairs to expose. both means C_1,T_1 and C_2,T_2.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional EWoK domain filter, e.g. spatial-relations. Default uses all domains.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on original EWoK rows before side expansion.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/research/bos_aligned_proto/spatial_synth/ewok_answer_exposure.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, src = load_ewok_df(args.ewok_variant)
    domain_filter = selected_domains(args.domains)
    if domain_filter is not None:
        domain_series = df["Domain"].map(normalize_text)
        df = df[domain_series.isin(domain_filter)].copy()
    if args.max_items is not None:
        df = df.head(max(0, int(args.max_items))).copy()

    sides = ("official", "symmetric") if args.sides == "both" else (args.sides,)
    rows = list(iter_answer_rows(df, ewok_variant=args.ewok_variant, sides=sides))
    if not rows:
        raise ValueError("No EWoK answer-exposure rows were produced.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    domain_counts = pd.Series([row["domain"] for row in rows]).value_counts().sort_index().to_dict()
    print(f"Loaded EWoK {args.ewok_variant} from {src}")
    print(f"Wrote {len(rows)} correct-answer rows to {args.out}")
    print(f"Domains: {domain_counts}")


if __name__ == "__main__":
    main()
