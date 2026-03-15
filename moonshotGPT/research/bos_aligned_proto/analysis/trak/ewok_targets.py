"""EWoK target records and tokenization for TRAK scoring.

This module converts the fast EWoK evaluation bundle into stable item-level
targets that the attribution pipeline can score. It owns metadata
normalization, grouping for overall or per-domain summaries, and the tokenized
pair construction needed by the EWoK margin computations in the TRAK backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from evaluation.ewok import BABYLM_COMPLETION_CHOICE, EWOK_PAPER_CONTEXT_SENSITIVITY
from evaluation.ewok_data import load_ewok_df


def _normalize_context_diff(value) -> str:
    raw = str(value).strip()
    if raw == "variable_swap":
        return "variable swap"
    return raw


def _normalize_string(value) -> str:
    return str(value).strip()


@dataclass(frozen=True)
class EWOKTargetItem:
    target_id: str
    domain: str
    row_index: int
    score_view: str
    context1: str
    context2: str
    target1: str
    target2: str
    context_type_raw: str
    context_type: str
    context_diff_raw: str
    context_diff: str
    target_diff_raw: str
    target_diff: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PreparedEWOKBatch:
    batch: tuple[torch.Tensor, ...]
    items: tuple[EWOKTargetItem, ...]


@dataclass(frozen=True)
class EWOKTargetBundle:
    items: tuple[EWOKTargetItem, ...]
    groups: dict[str, tuple[str, ...]]
    source_path: Path
    score_view: str
    score_reduction: str

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(item.target_id for item in self.items)

    def index_by_target_id(self) -> dict[str, int]:
        return {item.target_id: idx for idx, item in enumerate(self.items)}


def build_ewok_targets(
    *,
    score_view: str,
    target_scope: str,
    score_reduction: str,
    max_targets: int = 0,
) -> EWOKTargetBundle:
    if score_view not in {BABYLM_COMPLETION_CHOICE, EWOK_PAPER_CONTEXT_SENSITIVITY}:
        raise ValueError(f"Unsupported score_view: {score_view!r}")
    if target_scope not in {"overall", "per_domain", "both"}:
        raise ValueError(f"Unsupported target_scope: {target_scope!r}")

    df, src = load_ewok_df("fast")
    df = df.convert_dtypes()
    df = df.reset_index()

    items: list[EWOKTargetItem] = []
    for row in df.itertuples(index=False):
        domain = _normalize_string(row.Domain)
        item = EWOKTargetItem(
            target_id=f"ewok-fast:{score_view}:{score_reduction}:{domain}:{int(row.index)}",
            domain=domain,
            row_index=int(row.index),
            score_view=score_view,
            context1=str(row.Context1),
            context2=str(row.Context2),
            target1=str(row.Target1),
            target2=str(row.Target2),
            context_type_raw=str(row.ContextType),
            context_type=_normalize_string(row.ContextType),
            context_diff_raw=str(row.ContextDiff),
            context_diff=_normalize_context_diff(row.ContextDiff),
            target_diff_raw=str(row.TargetDiff),
            target_diff=_normalize_string(row.TargetDiff),
        )
        items.append(item)

    if max_targets > 0:
        items = items[: int(max_targets)]

    groups: dict[str, tuple[str, ...]] = {}
    if target_scope in {"overall", "both"}:
        groups["overall"] = tuple(item.target_id for item in items)
    if target_scope in {"per_domain", "both"}:
        domains = sorted({item.domain for item in items})
        for domain in domains:
            groups[f"domain:{domain}"] = tuple(item.target_id for item in items if item.domain == domain)

    return EWOKTargetBundle(
        items=tuple(items),
        groups=groups,
        source_path=Path(src),
        score_view=score_view,
        score_reduction=score_reduction,
    )


def _encode_pair(context: str, target: str, tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    text = context + " " + target
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=True, return_tensors="pt")
    context_len = len(tokenizer.encode(context, add_special_tokens=False))
    return (
        enc["input_ids"][0].to(dtype=torch.long),
        enc["attention_mask"][0].to(dtype=torch.long),
        int(context_len),
    )


def _pad_tensors(tensors: Sequence[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(int(tensor.numel()) for tensor in tensors)
    out = torch.full((len(tensors), max_len), fill_value=pad_value, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        out[idx, : tensor.numel()] = tensor
    return out


def iter_target_batches(
    bundle: EWOKTargetBundle,
    tokenizer,
    batch_size: int,
) -> Iterator[PreparedEWOKBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    items = list(bundle.items)
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        c1t1 = [_encode_pair(item.context1, item.target1, tokenizer) for item in batch_items]
        c1t2 = [_encode_pair(item.context1, item.target2, tokenizer) for item in batch_items]
        c2t2 = [_encode_pair(item.context2, item.target2, tokenizer) for item in batch_items]
        c2t1 = [_encode_pair(item.context2, item.target1, tokenizer) for item in batch_items]

        batch = (
            _pad_tensors([x[0] for x in c1t1]),
            _pad_tensors([x[1] for x in c1t1]),
            torch.tensor([x[2] for x in c1t1], dtype=torch.long),
            _pad_tensors([x[0] for x in c1t2]),
            _pad_tensors([x[1] for x in c1t2]),
            torch.tensor([x[2] for x in c1t2], dtype=torch.long),
            _pad_tensors([x[0] for x in c2t2]),
            _pad_tensors([x[1] for x in c2t2]),
            torch.tensor([x[2] for x in c2t2], dtype=torch.long),
            _pad_tensors([x[0] for x in c2t1]),
            _pad_tensors([x[1] for x in c2t1]),
            torch.tensor([x[2] for x in c2t1], dtype=torch.long),
        )
        yield PreparedEWOKBatch(batch=batch, items=tuple(batch_items))


__all__ = [
    "EWOKTargetBundle",
    "EWOKTargetItem",
    "PreparedEWOKBatch",
    "build_ewok_targets",
    "iter_target_batches",
]
