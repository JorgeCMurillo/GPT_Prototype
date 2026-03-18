"""Shared EWoK targets, diagnostics, and scalar query scoring helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

from evaluation.ewok import BABYLM_COMPLETION_CHOICE, EWOK_PAPER_CONTEXT_SENSITIVITY
from evaluation.ewok_data import load_ewok_df
import numpy as np


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
    concept_a: str
    concept_b: str
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


@dataclass(frozen=True)
class TargetDiagnostics:
    target_id: str
    domain: str
    score_view: str
    score_reduction: str
    s11_mean: float
    s12_mean: float
    s22_mean: float
    s21_mean: float
    s11_sum: float
    s12_sum: float
    s22_sum: float
    s21_sum: float
    margin_1: float
    margin_2: float
    combined_margin: float
    softplus_loss: float
    score: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointScores:
    checkpoint_step: int
    checkpoint_path: str
    candidate_row_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    score_matrix: np.ndarray
    target_diagnostics: tuple[TargetDiagnostics, ...]


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
            concept_a=str(row.ConceptA),
            concept_b=str(row.ConceptB),
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


def _resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define bos_token_id, eos_token_id, or pad_token_id")


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


def reduce_masked_token_logprobs(
    token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    token_mask = token_mask.to(dtype=token_logprobs.dtype)
    summed = (token_logprobs * token_mask).sum(dim=1)
    if reduction == "sum":
        return summed
    counts = token_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def compute_view_margins(
    *,
    score_view: str,
    s11: torch.Tensor,
    s12: torch.Tensor,
    s22: torch.Tensor,
    s21: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_view == BABYLM_COMPLETION_CHOICE:
        return s11 - s12, s22 - s21
    if score_view == EWOK_PAPER_CONTEXT_SENSITIVITY:
        return s11 - s21, s22 - s12
    raise ValueError(f"Unsupported score_view: {score_view!r}")


def compute_softplus_score(
    *,
    score_view: str,
    s11: torch.Tensor,
    s12: torch.Tensor,
    s22: torch.Tensor,
    s21: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    margin_1, margin_2 = compute_view_margins(
        score_view=score_view,
        s11=s11,
        s12=s12,
        s22=s22,
        s21=s21,
    )
    combined_margin = 0.5 * (margin_1 + margin_2)
    softplus_loss = 0.5 * (
        F.softplus((-margin_1) / temperature) + F.softplus((-margin_2) / temperature)
    )
    score = -softplus_loss
    return score, margin_1, margin_2, combined_margin


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _conditional_target_token_logprobs(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start: torch.Tensor,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _resolve_model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_start = target_start.to(device)

    batch_size = input_ids.shape[0]
    bos_column = torch.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype, device=device)
    bos_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)

    model_input_ids = torch.cat([bos_column, input_ids], dim=1)
    model_attention_mask = torch.cat([bos_mask, attention_mask], dim=1)

    logits = model(input_ids=model_input_ids, attention_mask=model_attention_mask).logits[:, :-1, :]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    valid_lengths = attention_mask.sum(dim=1, keepdim=True)
    target_mask = (positions >= target_start.unsqueeze(1)) & (positions < valid_lengths)
    return token_logprobs, target_mask


def _reduce_conditional_scores(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start: torch.Tensor,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logprobs, target_mask = _conditional_target_token_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_start=target_start,
        bos_token_id=bos_token_id,
    )
    mean_scores = reduce_masked_token_logprobs(token_logprobs, target_mask, reduction="mean")
    sum_scores = reduce_masked_token_logprobs(token_logprobs, target_mask, reduction="sum")
    return mean_scores, sum_scores


def score_target_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    score_view: str,
    score_reduction: str,
    temperature: float,
    bos_token_id: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(batch, tuple) or len(batch) != 12:
        raise TypeError("Expected an EWoK target batch with 12 tensor fields")

    c1t1_ids, c1t1_mask, c1t1_start = batch[0], batch[1], batch[2]
    c1t2_ids, c1t2_mask, c1t2_start = batch[3], batch[4], batch[5]
    c2t2_ids, c2t2_mask, c2t2_start = batch[6], batch[7], batch[8]
    c2t1_ids, c2t1_mask, c2t1_start = batch[9], batch[10], batch[11]

    s11_mean, s11_sum = _reduce_conditional_scores(
        model,
        input_ids=c1t1_ids,
        attention_mask=c1t1_mask,
        target_start=c1t1_start,
        bos_token_id=bos_token_id,
    )
    s12_mean, s12_sum = _reduce_conditional_scores(
        model,
        input_ids=c1t2_ids,
        attention_mask=c1t2_mask,
        target_start=c1t2_start,
        bos_token_id=bos_token_id,
    )
    s22_mean, s22_sum = _reduce_conditional_scores(
        model,
        input_ids=c2t2_ids,
        attention_mask=c2t2_mask,
        target_start=c2t2_start,
        bos_token_id=bos_token_id,
    )
    s21_mean, s21_sum = _reduce_conditional_scores(
        model,
        input_ids=c2t1_ids,
        attention_mask=c2t1_mask,
        target_start=c2t1_start,
        bos_token_id=bos_token_id,
    )

    if score_reduction == "mean":
        active_s11, active_s12, active_s22, active_s21 = s11_mean, s12_mean, s22_mean, s21_mean
    elif score_reduction == "sum":
        active_s11, active_s12, active_s22, active_s21 = s11_sum, s12_sum, s22_sum, s21_sum
    else:
        raise ValueError(f"Unsupported score_reduction: {score_reduction!r}")

    score, margin_1, margin_2, combined_margin = compute_softplus_score(
        score_view=score_view,
        s11=active_s11,
        s12=active_s12,
        s22=active_s22,
        s21=active_s21,
        temperature=temperature,
    )

    return {
        "score": score,
        "softplus_loss": -score,
        "margin_1": margin_1,
        "margin_2": margin_2,
        "combined_margin": combined_margin,
        "s11_mean": s11_mean,
        "s12_mean": s12_mean,
        "s22_mean": s22_mean,
        "s21_mean": s21_mean,
        "s11_sum": s11_sum,
        "s12_sum": s12_sum,
        "s22_sum": s22_sum,
        "s21_sum": s21_sum,
    }


def score_target_bundle(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
) -> tuple[TargetDiagnostics, ...]:
    bos_token_id = _resolve_bos_token_id(tokenizer)
    diagnostics: list[TargetDiagnostics] = []
    model.eval()
    with torch.no_grad():
        for prepared in iter_target_batches(bundle, tokenizer, batch_size):
            scores = score_target_batch(
                model,
                prepared.batch,
                score_view=bundle.score_view,
                score_reduction=bundle.score_reduction,
                temperature=temperature,
                bos_token_id=bos_token_id,
            )
            for idx, item in enumerate(prepared.items):
                diagnostics.append(
                    TargetDiagnostics(
                        target_id=item.target_id,
                        domain=item.domain,
                        score_view=bundle.score_view,
                        score_reduction=bundle.score_reduction,
                        s11_mean=float(scores["s11_mean"][idx].item()),
                        s12_mean=float(scores["s12_mean"][idx].item()),
                        s22_mean=float(scores["s22_mean"][idx].item()),
                        s21_mean=float(scores["s21_mean"][idx].item()),
                        s11_sum=float(scores["s11_sum"][idx].item()),
                        s12_sum=float(scores["s12_sum"][idx].item()),
                        s22_sum=float(scores["s22_sum"][idx].item()),
                        s21_sum=float(scores["s21_sum"][idx].item()),
                        margin_1=float(scores["margin_1"][idx].item()),
                        margin_2=float(scores["margin_2"][idx].item()),
                        combined_margin=float(scores["combined_margin"][idx].item()),
                        softplus_loss=float(scores["softplus_loss"][idx].item()),
                        score=float(scores["score"][idx].item()),
                    )
                )
    return tuple(diagnostics)


__all__ = [
    "CheckpointScores",
    "EWOKTargetBundle",
    "EWOKTargetItem",
    "PreparedEWOKBatch",
    "TargetDiagnostics",
    "build_ewok_targets",
    "compute_softplus_score",
    "compute_view_margins",
    "iter_target_batches",
    "reduce_masked_token_logprobs",
    "score_target_batch",
    "score_target_bundle",
]
