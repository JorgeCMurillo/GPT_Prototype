"""Helpers for resolving tokenizer and vocab settings from shard metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any


@dataclass(frozen=True)
class ResolvedTokenizerSpec:
    name_or_path: str
    meta: dict[str, Any]
    meta_tokenizer: str


def load_data_dir_meta(data_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_tokenizer_source(source: str) -> str:
    value = str(source or "").strip()
    if not value:
        return ""
    expanded = os.path.expanduser(value)
    if os.path.exists(expanded):
        return os.path.abspath(expanded)
    return value


def tokenizer_sources_match(lhs: str, rhs: str) -> bool:
    return normalize_tokenizer_source(lhs) == normalize_tokenizer_source(rhs)


def resolve_tokenizer_name_or_path(
    data_dir: str,
    explicit_tokenizer: str = "",
    *,
    explicit_arg_name: str,
    consumer_name: str,
) -> ResolvedTokenizerSpec:
    meta = load_data_dir_meta(data_dir)
    meta_tokenizer = str(meta.get("tokenizer") or "").strip()
    explicit = str(explicit_tokenizer or "").strip()

    if explicit:
        if meta_tokenizer and not tokenizer_sources_match(explicit, meta_tokenizer):
            raise ValueError(
                f"{consumer_name} tokenizer mismatch: {explicit_arg_name}={explicit!r} "
                f"but {os.path.join(data_dir, 'meta.json')} records tokenizer={meta_tokenizer!r}."
            )
        return ResolvedTokenizerSpec(name_or_path=explicit, meta=meta, meta_tokenizer=meta_tokenizer)

    if meta_tokenizer:
        return ResolvedTokenizerSpec(name_or_path=meta_tokenizer, meta=meta, meta_tokenizer=meta_tokenizer)

    raise ValueError(
        f"{consumer_name} could not resolve a tokenizer. Pass {explicit_arg_name} explicitly "
        f"or add a 'tokenizer' field to {os.path.join(data_dir, 'meta.json')}."
    )


def resolve_model_vocab_size(requested_vocab_size: int, tokenizer_vocab_size: int) -> int:
    requested = int(requested_vocab_size)
    tok_vocab = int(tokenizer_vocab_size)
    if requested == 0:
        return tok_vocab
    if requested < tok_vocab:
        raise ValueError(f"vocab_size={requested} must be >= tokenizer vocab ({tok_vocab}).")
    return requested


def effective_tokenizer_vocab_size(tokenizer) -> int:
    try:
        resolved = int(len(tokenizer))
    except Exception:
        resolved = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if resolved <= 0:
        raise ValueError(f"Could not resolve a positive effective tokenizer vocab size from {tokenizer!r}.")
    return resolved


def tokenizers_match(lhs, rhs) -> bool:
    attrs = ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    for attr in attrs:
        if getattr(lhs, attr, None) != getattr(rhs, attr, None):
            return False
    try:
        if int(len(lhs)) != int(len(rhs)):
            return False
    except Exception:
        pass
    try:
        return lhs.get_vocab() == rhs.get_vocab()
    except Exception:
        return False


__all__ = [
    "ResolvedTokenizerSpec",
    "effective_tokenizer_vocab_size",
    "load_data_dir_meta",
    "normalize_tokenizer_source",
    "resolve_model_vocab_size",
    "resolve_tokenizer_name_or_path",
    "tokenizer_sources_match",
    "tokenizers_match",
]
