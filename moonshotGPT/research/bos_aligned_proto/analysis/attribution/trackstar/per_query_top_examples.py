"""Decode top attribution candidates for selected individual EWoK queries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_KEYWORDS = (
    "absorb",
    "bend",
    "bends",
    "biodiesel",
    "break",
    "breaks",
    "broke",
    "broken",
    "carbon",
    "chemical",
    "chemicals",
    "concrete",
    "conductive",
    "corrosion",
    "deform",
    "deflect",
    "drip",
    "drips",
    "dissolve",
    "fiber",
    "fibers",
    "fibre",
    "fibres",
    "flow",
    "fluid",
    "fold",
    "folds",
    "friction",
    "fuel",
    "gas",
    "gases",
    "hydro",
    "landslide",
    "liquid",
    "load",
    "material",
    "materials",
    "metal",
    "melt",
    "melts",
    "oil",
    "plastic",
    "pour",
    "pours",
    "rigid",
    "rip",
    "rips",
    "soil",
    "squeeze",
    "squeezes",
    "steel",
    "stir",
    "stirs",
    "strain",
    "stress",
    "surface",
    "tensile",
    "texture",
    "vapor",
    "water",
    "wrinkle",
    "wrinkles",
)

QUERY_TERM_STOPWORDS = {
    "about",
    "ali",
    "and",
    "are",
    "because",
    "been",
    "being",
    "chao",
    "does",
    "for",
    "from",
    "has",
    "have",
    "is",
    "its",
    "it",
    "into",
    "jesse",
    "li",
    "like",
    "mohammed",
    "not",
    "onto",
    "over",
    "sees",
    "she",
    "some",
    "something",
    "that",
    "the",
    "then",
    "there",
    "they",
    "this",
    "through",
    "under",
    "when",
    "where",
    "while",
    "with",
}


def _step_tag(step: int) -> str:
    return f"step{int(step):08d}"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _resolve_step_file(attribution_dir: Path, name: str, step: int, suffix: str) -> Path:
    path = attribution_dir / f"{name}_{_step_tag(step)}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def _select_mixed_targets(frame: pd.DataFrame, num_queries: int) -> list[str]:
    if frame.empty:
        return []
    ordered = frame.sort_values(["combined_margin", "target_id"]).reset_index(drop=True)
    median_value = float(ordered["combined_margin"].median())
    median_ordered = (
        ordered.assign(_median_distance=(ordered["combined_margin"] - median_value).abs())
        .sort_values(["_median_distance", "target_id"])
        .reset_index(drop=True)
    )
    candidate_ids: list[str] = []
    for source in (ordered, ordered.iloc[::-1].reset_index(drop=True), median_ordered):
        for target_id in source["target_id"].astype(str).tolist():
            if target_id not in candidate_ids:
                candidate_ids.append(target_id)
            if len(candidate_ids) >= int(num_queries):
                return candidate_ids
    return candidate_ids


def select_target_ids(
    diagnostics: pd.DataFrame,
    *,
    target_ids: Sequence[str] = (),
    query_selection: str,
    num_queries: int,
) -> list[str]:
    if target_ids:
        available = set(diagnostics["target_id"].astype(str).tolist())
        missing = [target_id for target_id in target_ids if str(target_id) not in available]
        if missing:
            raise ValueError(f"Requested target_id(s) not found in diagnostics: {missing}")
        return [str(target_id) for target_id in target_ids]

    if num_queries <= 0:
        raise ValueError("num_queries must be > 0 unless explicit --target_ids are provided")

    frame = diagnostics.copy()
    if "combined_margin" not in frame.columns:
        raise ValueError("Target diagnostics must contain combined_margin for automatic query selection")

    if query_selection == "all":
        return frame.sort_values("target_id")["target_id"].astype(str).tolist()
    if query_selection == "lowest_margin":
        selected = frame.sort_values(["combined_margin", "target_id"]).head(int(num_queries))
    elif query_selection == "highest_margin":
        selected = frame.sort_values(["combined_margin", "target_id"], ascending=[False, True]).head(int(num_queries))
    elif query_selection == "median_margin":
        median_value = float(frame["combined_margin"].median())
        selected = (
            frame.assign(_median_distance=(frame["combined_margin"] - median_value).abs())
            .sort_values(["_median_distance", "target_id"])
            .head(int(num_queries))
        )
    elif query_selection == "mixed":
        return _select_mixed_targets(frame, int(num_queries))
    else:
        raise ValueError(f"Unsupported query_selection={query_selection!r}")
    return selected["target_id"].astype(str).tolist()


def _normalize_keyword(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _split_keyword_args(values: Sequence[str]) -> tuple[str, ...]:
    keywords: list[str] = []
    for value in values:
        for part in str(value).split(","):
            normalized = _normalize_keyword(part)
            if normalized:
                keywords.append(normalized)
    return tuple(dict.fromkeys(keywords))


def _query_keywords(item: dict[str, Any]) -> tuple[str, ...]:
    fields = (
        "concept_a",
        "concept_b",
        "context1",
        "context2",
        "target1",
        "target2",
    )
    terms: list[str] = []
    for field in fields:
        text = _normalize_keyword(str(item.get(field, "")))
        for term in re.findall(r"[a-z][a-z-]{2,}", text):
            if term in QUERY_TERM_STOPWORDS:
                continue
            terms.append(term)
    return tuple(dict.fromkeys(terms))


def query_action_terms_for_item(item: dict[str, Any]) -> tuple[str, ...]:
    """Terms from the specific EWoK context/target contrast.

    This intentionally excludes concept names and the broad default material
    lexicon. It is an audit view for whether a top candidate contains words
    tied to the particular queried affordance/state contrast.
    """

    fields = (
        "context1",
        "context2",
        "target1",
        "target2",
    )
    terms: list[str] = []
    for field in fields:
        text = _normalize_keyword(str(item.get(field, "")))
        for term in re.findall(r"[a-z][a-z-]{1,}", text):
            if term in QUERY_TERM_STOPWORDS:
                continue
            terms.append(term)
    return tuple(dict.fromkeys(terms))


def keywords_for_item(
    item: dict[str, Any],
    *,
    extra_keywords: Sequence[str] = (),
    include_query_keywords: bool = True,
) -> tuple[str, ...]:
    terms: list[str] = list(DEFAULT_KEYWORDS)
    if include_query_keywords:
        terms.extend(_query_keywords(item))
    terms.extend(_split_keyword_args(extra_keywords))
    return tuple(dict.fromkeys(_normalize_keyword(term) for term in terms if _normalize_keyword(term)))


def _keyword_pattern(term: str) -> re.Pattern[str]:
    escaped_parts = [re.escape(part) for part in _normalize_keyword(term).split()]
    body = r"\s+".join(escaped_parts)
    return re.compile(rf"(?<!\w){body}(?!\w)", flags=re.IGNORECASE)


def _highlight_terms(text: str, terms: Sequence[str]) -> str:
    highlighted = str(text)
    for term in sorted(set(terms), key=len, reverse=True):
        pattern = _keyword_pattern(term)
        highlighted = pattern.sub(lambda match: f"**{match.group(0)}**", highlighted)
    return highlighted


def keyword_hits_in_text(text: str, keywords: Sequence[str]) -> tuple[str, ...]:
    normalized_keywords = tuple(dict.fromkeys(_normalize_keyword(term) for term in keywords if _normalize_keyword(term)))
    hits = [term for term in normalized_keywords if _keyword_pattern(term).search(text)]
    return tuple(hits)


def extract_keyword_snippets(
    text: str,
    keywords: Sequence[str],
    *,
    window_chars: int,
    max_snippets: int,
) -> list[dict[str, Any]]:
    if max_snippets <= 0:
        return []

    normalized_keywords = tuple(dict.fromkeys(_normalize_keyword(term) for term in keywords if _normalize_keyword(term)))
    matches: list[tuple[int, int, str]] = []
    for term in normalized_keywords:
        for match in _keyword_pattern(term).finditer(text):
            matches.append((int(match.start()), int(match.end()), term))
    if not matches:
        return []

    half_window = max(80, int(window_chars) // 2)
    # Prefer dense windows that contain many distinct keyword hits, with a
    # stable left-to-right tie break.
    candidate_windows: list[tuple[int, int, int, tuple[str, ...]]] = []
    for start, end, _term in matches:
        center = (start + end) // 2
        left = max(0, center - half_window)
        right = min(len(text), center + half_window)
        terms_in_window = tuple(
            sorted(
                {
                    term
                    for match_start, match_end, term in matches
                    if match_start < right and match_end > left
                }
            )
        )
        candidate_windows.append((left, right, len(terms_in_window), terms_in_window))

    selected: list[tuple[int, int, tuple[str, ...]]] = []
    for left, right, _score, terms_in_window in sorted(
        candidate_windows,
        key=lambda item: (-item[2], item[0], item[1]),
    ):
        overlaps_existing = any(not (right <= prev_left or left >= prev_right) for prev_left, prev_right, _ in selected)
        if overlaps_existing:
            continue
        selected.append((left, right, terms_in_window))
        if len(selected) >= int(max_snippets):
            break

    snippets: list[dict[str, Any]] = []
    for left, right, terms_in_window in sorted(selected, key=lambda item: item[0]):
        snippet_text = text[left:right].strip()
        if left > 0:
            snippet_text = "..." + snippet_text
        if right < len(text):
            snippet_text = snippet_text + "..."
        snippets.append(
            {
                "char_start": int(left),
                "char_end": int(right),
                "matched_terms": list(terms_in_window),
                "text": snippet_text,
                "highlighted_text": _highlight_terms(snippet_text, terms_in_window),
            }
        )
    return snippets


class TokenDecoder:
    def __init__(
        self,
        tokenizer_path: str | Path,
        *,
        preview_tokens: int,
        preview_chars: int,
        snippet_tokens: int,
        snippet_window_chars: int,
        max_snippets: int,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(tokenizer_path).expanduser().resolve()))
        self.preview_tokens = int(preview_tokens)
        self.preview_chars = int(preview_chars)
        self.snippet_tokens = int(snippet_tokens)
        self.snippet_window_chars = int(snippet_window_chars)
        self.max_snippets = int(max_snippets)
        self._memmaps: dict[str, np.memmap] = {}

    def _memmap_for_path(self, path: str | Path) -> np.memmap:
        key = str(path)
        if key not in self._memmaps:
            self._memmaps[key] = np.memmap(key, dtype=np.uint16, mode="r")
        return self._memmaps[key]

    def _decode_row(self, row: pd.Series, *, max_tokens: int) -> str:
        memmap = self._memmap_for_path(str(row["shard_path"]))
        start = int(row["token_offset_start"])
        token_end = int(row["token_offset_end"])
        end = min(token_end, start + int(max_tokens))
        tokens = np.asarray(memmap[start:end], dtype=np.int64).tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(text).split())

    def preview(self, row: pd.Series) -> str:
        text = self._decode_row(row, max_tokens=self.preview_tokens)
        if len(text) > self.preview_chars:
            return text[: self.preview_chars].rstrip() + "..."
        return text

    def snippets(self, row: pd.Series, keywords: Sequence[str]) -> list[dict[str, Any]]:
        text = self._decode_row(row, max_tokens=self.snippet_tokens)
        return extract_keyword_snippets(
            text,
            keywords,
            window_chars=self.snippet_window_chars,
            max_snippets=self.max_snippets,
        )

    def keyword_hits(self, row: pd.Series, keywords: Sequence[str]) -> tuple[str, ...]:
        text = self._decode_row(row, max_tokens=self.snippet_tokens)
        return keyword_hits_in_text(text, keywords)


def _target_context_text(item: dict[str, Any]) -> str:
    return (
        f"{item.get('context1')} -> {item.get('target1')} | "
        f"{item.get('context2')} -> {item.get('target2')}"
    )


def build_records(
    *,
    attribution_dir: str | Path,
    step: int,
    tokenizer_path: str | Path,
    target_ids: Sequence[str] = (),
    query_selection: str = "mixed",
    num_queries: int = 6,
    topk: int = 5,
    preview_tokens: int = 180,
    preview_chars: int = 700,
    snippet_mode: str = "preview",
    snippet_tokens: int = 1025,
    snippet_window_chars: int = 700,
    max_snippets: int = 3,
    keywords: Sequence[str] = (),
    include_query_keywords: bool = True,
    include_action_overlap: bool = True,
) -> list[dict[str, Any]]:
    if snippet_mode not in {"preview", "keyword", "both"}:
        raise ValueError("snippet_mode must be one of: preview, keyword, both")
    root = Path(attribution_dir).expanduser().resolve()
    top_rows_path = _resolve_step_file(root, "top_rows", step, ".csv")
    diagnostics_path = _resolve_step_file(root, "target_diagnostics", step, ".jsonl")
    target_items_path = root / "target_items.jsonl"
    if not target_items_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {target_items_path}")

    top_rows = pd.read_csv(top_rows_path)
    diagnostics = pd.DataFrame.from_records(_load_jsonl(diagnostics_path))
    target_items = {str(row["target_id"]): row for row in _load_jsonl(target_items_path)}
    diagnostics_by_id = {str(row["target_id"]): row for row in diagnostics.to_dict(orient="records")}

    selected_ids = select_target_ids(
        diagnostics,
        target_ids=target_ids,
        query_selection=query_selection,
        num_queries=num_queries,
    )
    decoder = TokenDecoder(
        tokenizer_path,
        preview_tokens=preview_tokens,
        preview_chars=preview_chars,
        snippet_tokens=snippet_tokens,
        snippet_window_chars=snippet_window_chars,
        max_snippets=max_snippets,
    )

    records: list[dict[str, Any]] = []
    for target_id in selected_ids:
        item = target_items.get(target_id)
        diag = diagnostics_by_id.get(target_id)
        if item is None or diag is None:
            raise ValueError(f"Missing target metadata for target_id={target_id!r}")
        item_keywords = keywords_for_item(
            item,
            extra_keywords=keywords,
            include_query_keywords=include_query_keywords,
        )
        query_action_terms = query_action_terms_for_item(item)
        rows = (
            top_rows.loc[top_rows["target_id"].astype(str) == target_id]
            .sort_values(["rank", "score"], ascending=[True, False])
            .head(int(topk))
        )
        for row in rows.to_dict(orient="records"):
            row_series = pd.Series(row)
            snippets = decoder.snippets(row_series, item_keywords) if snippet_mode in {"keyword", "both"} else []
            action_hits: tuple[str, ...] = ()
            if include_action_overlap and query_action_terms:
                action_hits = decoder.keyword_hits(row_series, query_action_terms)
            records.append(
                {
                    "target_id": target_id,
                    "row_index": int(item["row_index"]),
                    "domain": item["domain"],
                    "context_type": item["context_type"],
                    "context_diff": item["context_diff"],
                    "target_diff": item["target_diff"],
                    "concept_a": item["concept_a"],
                    "concept_b": item["concept_b"],
                    "query_text": _target_context_text(item),
                    "combined_margin": float(diag["combined_margin"]),
                    "softplus_loss": float(diag["softplus_loss"]),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "candidate_id": int(row["candidate_id"]),
                    "candidate_kind": str(row["candidate_kind"]),
                    "shard_path": str(row["shard_path"]),
                    "token_offset_start": int(row["token_offset_start"]),
                    "token_offset_end": int(row["token_offset_end"]),
                    "preview": decoder.preview(row_series),
                    "query_action_terms": list(query_action_terms),
                    "query_action_hits": list(action_hits),
                    "query_action_overlap_count": int(len(action_hits)),
                    "query_action_overlap_fraction": (
                        float(len(action_hits) / len(query_action_terms)) if query_action_terms else 0.0
                    ),
                    "snippet_mode": snippet_mode,
                    "keyword_count": int(len(item_keywords)),
                    "keyword_hits": sorted({term for snippet in snippets for term in snippet["matched_terms"]}),
                    "snippets": snippets,
                }
            )
    return records


def write_markdown(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Per-Query Top Examples", ""]
    current_target: str | None = None
    for record in records:
        target_id = str(record["target_id"])
        if target_id != current_target:
            current_target = target_id
            lines.extend(
                [
                    f"## {target_id}",
                    "",
                    f"- row_index: `{record['row_index']}`",
                    f"- metadata: `{record['domain']}`, `{record['context_type']}`, "
                    f"`{record['context_diff']}`, `{record['target_diff']}`",
                    f"- concepts: `{record['concept_a']}` / `{record['concept_b']}`",
                    f"- combined_margin: `{record['combined_margin']:.6g}`",
                    f"- softplus_loss: `{record['softplus_loss']:.6g}`",
                    f"- query: {record['query_text']}",
                    "",
                ]
            )
        lines.extend(
            [
                f"### Rank {record['rank']}: candidate `{record['candidate_id']}`",
                "",
                f"- score: `{record['score']:.6g}`",
                f"- shard: `{Path(str(record['shard_path'])).name}`",
                f"- offsets: `{record['token_offset_start']}..{record['token_offset_end']}`",
                f"- query-action overlap: `{record.get('query_action_overlap_count', 0)}`/"
                f"`{len(record.get('query_action_terms') or [])}` "
                f"({float(record.get('query_action_overlap_fraction', 0.0)):.3f})",
                f"- query-action hits: "
                f"{', '.join(f'`{term}`' for term in record.get('query_action_hits', [])) or '_none_'}",
                "",
            ]
        )
        snippet_mode = str(record.get("snippet_mode", "preview"))
        if snippet_mode in {"preview", "both"}:
            lines.extend(["Preview:", "", str(record["preview"]), ""])
        snippets = list(record.get("snippets") or [])
        if snippet_mode in {"keyword", "both"}:
            if snippets:
                lines.extend(["Keyword snippets:", ""])
                for index, snippet in enumerate(snippets, start=1):
                    terms = ", ".join(f"`{term}`" for term in snippet.get("matched_terms", []))
                    lines.extend(
                        [
                            f"{index}. matched: {terms}",
                            "",
                            str(snippet.get("highlighted_text") or snippet.get("text") or ""),
                            "",
                        ]
                    )
            else:
                lines.extend(
                    [
                        "Keyword snippets: no configured keyword hits found; showing fallback preview.",
                        "",
                        str(record["preview"]),
                        "",
                    ]
                )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tokenizer_path", required=True, help="Checkpoint/tokenizer directory used for decoding")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_ids", nargs="*", default=())
    parser.add_argument(
        "--query_selection",
        choices=("mixed", "lowest_margin", "highest_margin", "median_margin", "all"),
        default="mixed",
    )
    parser.add_argument("--num_queries", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--preview_tokens", type=int, default=180)
    parser.add_argument("--preview_chars", type=int, default=700)
    parser.add_argument(
        "--snippet_mode",
        choices=("preview", "keyword", "both"),
        default="preview",
        help="Whether Markdown shows the opening preview, keyword hit snippets, or both.",
    )
    parser.add_argument(
        "--snippet_tokens",
        type=int,
        default=1025,
        help="Number of candidate-row tokens to decode when searching keyword snippets.",
    )
    parser.add_argument("--snippet_window_chars", type=int, default=700)
    parser.add_argument("--max_snippets", type=int, default=3)
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=(),
        help="Extra comma-separated or space-separated keyword terms to search in decoded rows.",
    )
    parser.add_argument(
        "--no_query_keywords",
        action="store_false",
        dest="include_query_keywords",
        help="Do not add terms from the EWoK query text/concepts to the default keyword lexicon.",
    )
    parser.add_argument(
        "--no_action_overlap",
        action="store_false",
        dest="include_action_overlap",
        help="Do not compute query-action overlap metadata for each decoded top candidate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    attribution_dir = Path(args.attribution_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else attribution_dir / f"per_query_top_examples_{_step_tag(args.step)}"
    )

    records = build_records(
        attribution_dir=attribution_dir,
        step=int(args.step),
        tokenizer_path=args.tokenizer_path,
        target_ids=tuple(args.target_ids),
        query_selection=str(args.query_selection),
        num_queries=int(args.num_queries),
        topk=int(args.topk),
        preview_tokens=int(args.preview_tokens),
        preview_chars=int(args.preview_chars),
        snippet_mode=str(args.snippet_mode),
        snippet_tokens=int(args.snippet_tokens),
        snippet_window_chars=int(args.snippet_window_chars),
        max_snippets=int(args.max_snippets),
        keywords=tuple(args.keywords),
        include_query_keywords=bool(args.include_query_keywords),
        include_action_overlap=bool(args.include_action_overlap),
    )
    markdown_path = output_dir / "top_examples.md"
    jsonl_path = output_dir / "top_examples.jsonl"
    write_markdown(markdown_path, records)
    write_jsonl(jsonl_path, records)
    print(f"wrote {len(records)} per-query top-example records")
    print(f"markdown: {markdown_path}")
    print(f"jsonl: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
