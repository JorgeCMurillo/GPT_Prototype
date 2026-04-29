"""Sentence splitting and contiguous sentence-window generation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .features import preview_text


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    char_start: int
    char_end: int


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences_regex(text: str) -> list[SentenceSpan]:
    """Split text into sentence spans while preserving original char offsets."""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    boundary_re = re.compile(r"(?<=[.!?])\s+|\n{2,}")
    spans: list[SentenceSpan] = []
    start = 0

    def append_piece(piece_start: int, piece_end: int) -> None:
        piece = raw[piece_start:piece_end]
        leading = len(piece) - len(piece.lstrip())
        trailing = len(piece.rstrip())
        adjusted_start = piece_start + leading
        adjusted_end = piece_start + trailing
        if adjusted_end <= adjusted_start:
            return
        spans.append(
            SentenceSpan(
                text=raw[adjusted_start:adjusted_end],
                char_start=int(adjusted_start),
                char_end=int(adjusted_end),
            )
        )

    for match in boundary_re.finditer(raw):
        append_piece(start, match.start())
        start = match.end()
    append_piece(start, len(raw))

    if spans:
        return spans
    cleaned = raw.strip()
    if not cleaned:
        return []
    return [SentenceSpan(text=cleaned, char_start=raw.find(cleaned), char_end=raw.find(cleaned) + len(cleaned))]


def split_sentences_spacy(text: str, nlp: Any) -> list[SentenceSpan]:
    doc = nlp(str(text))
    spans = [
        SentenceSpan(
            text=str(sent.text).strip(),
            char_start=int(sent.start_char),
            char_end=int(sent.end_char),
        )
        for sent in doc.sents
        if str(sent.text).strip()
    ]
    return spans or split_sentences_regex(text)


def split_sentences(text: str, *, nlp: Any | None = None) -> list[SentenceSpan]:
    if nlp is None:
        return split_sentences_regex(text)
    return split_sentences_spacy(text, nlp)


def generate_sentence_windows(
    record: dict[str, Any],
    *,
    text: str,
    min_sentences: int = 3,
    max_sentences: int = 6,
    nlp: Any | None = None,
) -> list[dict[str, Any]]:
    if int(min_sentences) <= 0:
        raise ValueError("min_sentences must be > 0")
    if int(max_sentences) < int(min_sentences):
        raise ValueError("max_sentences must be >= min_sentences")

    sentences = split_sentences(text, nlp=nlp)
    rows: list[dict[str, Any]] = []
    candidate_id = int(record["candidate_id"])
    for start_idx in range(len(sentences)):
        for length in range(int(min_sentences), int(max_sentences) + 1):
            end_idx = start_idx + length
            if end_idx > len(sentences):
                continue
            start_char = int(sentences[start_idx].char_start)
            end_char = int(sentences[end_idx - 1].char_end)
            snippet_text = str(text)[start_char:end_char].strip()
            if not snippet_text:
                continue
            rows.append(
                {
                    **record,
                    "window_id": f"{candidate_id}:s{start_idx:03d}-{end_idx:03d}",
                    "parent_candidate_id": candidate_id,
                    "sentence_start_idx": int(start_idx),
                    "sentence_end_idx": int(end_idx),
                    "snippet_sentence_count": int(length),
                    "char_start": start_char,
                    "char_end": end_char,
                    "snippet_text": snippet_text,
                    "snippet_text_preview": preview_text(snippet_text),
                }
            )
    return rows
