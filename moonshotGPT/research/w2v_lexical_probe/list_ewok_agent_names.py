"""List likely recurring agent names from the EWoK benchmark text.

This is a lightweight heuristic:
- scan the EWoK text fields for title-cased word tokens
- drop obvious sentence starters and non-name words
- count recurring candidates
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from evaluation.ewok_data import load_ewok_df

_NAME_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_DEFAULT_TEXT_COLUMNS = ("Context1", "Context2")
_STOPWORDS = {
    "A",
    "All",
    "An",
    "Any",
    "As",
    "At",
    "Because",
    "Before",
    "But",
    "Each",
    "Every",
    "Getting",
    "Heat",
    "How",
    "If",
    "In",
    "It",
    "Most",
    "Nobody",
    "Nothing",
    "One",
    "Our",
    "Over",
    "Speaking",
    "That",
    "The",
    "There",
    "These",
    "This",
    "Those",
    "Through",
    "To",
    "Today",
    "Toothpaste",
    "Two",
    "Upon",
    "When",
    "After",
    "And",
    "Another",
    "Around",
    "Along",
    "During",
    "Why",
    "While",
    "Then",
    "Rings",
}


def extract_likely_agent_names(
    *,
    text_columns: tuple[str, ...] = _DEFAULT_TEXT_COLUMNS,
    min_count: int = 2,
) -> tuple[Counter, str]:
    ewok_df, source = load_ewok_df()
    counts: Counter[str] = Counter()

    for column in text_columns:
        for text in ewok_df[column].astype(str):
            for candidate in _NAME_RE.findall(text):
                if candidate in _STOPWORDS:
                    continue
                counts[candidate] += 1

    filtered = Counter({name: count for name, count in counts.items() if count >= min_count})
    return filtered, str(source)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List likely recurring agent names from EWoK text.")
    parser.add_argument("--min_count", type=int, default=2, help="Only show names with at least this many mentions")
    parser.add_argument("--top_k", type=int, default=30, help="How many names to print")
    parser.add_argument(
        "--include_targets",
        action="store_true",
        help="Also scan Target1/Target2 in addition to Context1/Context2",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    columns = _DEFAULT_TEXT_COLUMNS + (("Target1", "Target2") if args.include_targets else ())
    counts, source = extract_likely_agent_names(
        text_columns=columns,
        min_count=args.min_count,
    )
    print(f"source\t{source}")
    print("name\tcount")
    for name, count in counts.most_common(args.top_k):
        print(f"{name}\t{count}")


if __name__ == "__main__":
    main()
