"""Activation extraction and cache I/O for EWoK linear probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.ewok import resolve_bos_token_id

from .artifacts import atomic_save_npz
from .data import EWOKProbePair, validate_pair_alignment


@dataclass(frozen=True)
class ActivationCache:
    features: np.ndarray
    layer_indices: tuple[int, ...]
    pair_ids: tuple[str, ...]
    row_indices: np.ndarray
    roles: tuple[str, ...]
    labels: np.ndarray
    lm_score_mean: np.ndarray
    lm_score_sum: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.features.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.features.shape[2])


def _cache_dtype(name: str):
    normalized = str(name).lower()
    if normalized in {"fp16", "float16"}:
        return np.float16
    if normalized in {"fp32", "float32"}:
        return np.float32
    raise ValueError(f"Unsupported activation cache dtype: {name!r}")


def parse_layer_arg(value: str | None) -> tuple[int, ...] | None:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return None
    layers = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        return None
    return tuple(dict.fromkeys(layers))


def _encode_pair(tokenizer, pair: EWOKProbePair) -> tuple[list[int], int]:
    input_ids = tokenizer.encode(pair.text, add_special_tokens=False)
    context_len = len(tokenizer.encode(pair.context, add_special_tokens=False))
    if not input_ids:
        raise ValueError(f"Tokenizer produced no tokens for pair_id={pair.pair_id}")
    return list(map(int, input_ids)), int(context_len)


def _pad_encoded_batch(
    encoded: Sequence[tuple[list[int], int]],
    *,
    bos_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    text_lengths = [len(ids) for ids, _ in encoded]
    max_text_len = max(text_lengths)
    batch_size = len(encoded)
    input_ids = torch.full(
        (batch_size, max_text_len + 1),
        fill_value=int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((batch_size, max_text_len + 1), dtype=torch.long)
    target_starts = torch.empty((batch_size,), dtype=torch.long)
    text_lengths_tensor = torch.tensor(text_lengths, dtype=torch.long)

    input_ids[:, 0] = int(bos_token_id)
    attention_mask[:, 0] = 1
    for idx, (ids, target_start) in enumerate(encoded):
        length = len(ids)
        input_ids[idx, 1 : length + 1] = torch.tensor(ids, dtype=torch.long)
        attention_mask[idx, 1 : length + 1] = 1
        target_starts[idx] = int(target_start)
    return input_ids, attention_mask, target_starts, text_lengths_tensor


def _model_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    except TypeError:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )


def _reduce_target_logprobs(
    logits: torch.Tensor,
    model_input_ids: torch.Tensor,
    target_starts: torch.Tensor,
    text_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_token_logits = logits[:, :-1, :]
    text_token_ids = model_input_ids[:, 1:]
    token_logprobs = F.log_softmax(next_token_logits, dim=-1).gather(
        dim=-1,
        index=text_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    positions = torch.arange(text_token_ids.shape[1], device=text_token_ids.device).unsqueeze(0)
    target_mask = (positions >= target_starts.unsqueeze(1)) & (positions < text_lengths.unsqueeze(1))
    target_mask_f = target_mask.to(dtype=token_logprobs.dtype)
    sums = (token_logprobs * target_mask_f).sum(dim=1)
    counts = target_mask_f.sum(dim=1).clamp(min=1.0)
    return sums / counts, sums


def extract_activation_cache(
    model: torch.nn.Module,
    tokenizer,
    pairs: Sequence[EWOKProbePair],
    *,
    batch_size: int,
    layers: tuple[int, ...] | None,
    cache_dtype: str = "float16",
) -> ActivationCache:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    device = next(model.parameters()).device
    bos_token_id = resolve_bos_token_id(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = bos_token_id

    feature_batches: list[np.ndarray] = []
    lm_mean_batches: list[np.ndarray] = []
    lm_sum_batches: list[np.ndarray] = []
    resolved_layers: tuple[int, ...] | None = None
    dtype = _cache_dtype(cache_dtype)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = [_encode_pair(tokenizer, pair) for pair in batch_pairs]
            input_ids, attention_mask, target_starts, text_lengths = _pad_encoded_batch(
                encoded,
                bos_token_id=bos_token_id,
                pad_token_id=int(pad_token_id),
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_starts = target_starts.to(device)
            text_lengths = text_lengths.to(device)

            outputs = _model_forward(model, input_ids, attention_mask)
            hidden_states = tuple(outputs.hidden_states)
            if resolved_layers is None:
                if layers is None:
                    resolved_layers = tuple(range(len(hidden_states)))
                else:
                    bad = [layer for layer in layers if layer < 0 or layer >= len(hidden_states)]
                    if bad:
                        raise ValueError(
                            f"Requested layer(s) {bad!r}, but model returned {len(hidden_states)} hidden-state tensors."
                        )
                    resolved_layers = tuple(layers)

            final_positions = attention_mask.sum(dim=1) - 1
            batch_features = []
            gather_rows = torch.arange(input_ids.shape[0], device=device)
            for layer in resolved_layers:
                selected = hidden_states[layer][gather_rows, final_positions, :]
                batch_features.append(selected.detach().float().cpu().numpy())
            feature_batches.append(np.stack(batch_features, axis=1).astype(dtype, copy=False))

            lm_mean, lm_sum = _reduce_target_logprobs(
                outputs.logits,
                input_ids,
                target_starts,
                text_lengths,
            )
            lm_mean_batches.append(lm_mean.detach().float().cpu().numpy().astype(np.float32, copy=False))
            lm_sum_batches.append(lm_sum.detach().float().cpu().numpy().astype(np.float32, copy=False))

    assert resolved_layers is not None
    return ActivationCache(
        features=np.concatenate(feature_batches, axis=0),
        layer_indices=tuple(int(x) for x in resolved_layers),
        pair_ids=tuple(pair.pair_id for pair in pairs),
        row_indices=np.asarray([pair.row_index for pair in pairs], dtype=np.int64),
        roles=tuple(pair.role for pair in pairs),
        labels=np.asarray([pair.label for pair in pairs], dtype=np.int64),
        lm_score_mean=np.concatenate(lm_mean_batches, axis=0),
        lm_score_sum=np.concatenate(lm_sum_batches, axis=0),
    )


def save_activation_cache(path: str | Path, cache: ActivationCache) -> None:
    atomic_save_npz(
        path,
        features=cache.features,
        layer_indices=np.asarray(cache.layer_indices, dtype=np.int64),
        pair_ids=np.asarray(cache.pair_ids, dtype=object),
        row_indices=cache.row_indices,
        roles=np.asarray(cache.roles, dtype=object),
        labels=cache.labels,
        lm_score_mean=cache.lm_score_mean,
        lm_score_sum=cache.lm_score_sum,
    )


def load_activation_cache(path: str | Path, pairs: Sequence[EWOKProbePair] | None = None) -> ActivationCache:
    payload = np.load(Path(path), allow_pickle=True)
    pair_ids = tuple(str(x) for x in payload["pair_ids"].tolist())
    if pairs is not None:
        validate_pair_alignment(pair_ids, pairs)
    return ActivationCache(
        features=np.asarray(payload["features"]),
        layer_indices=tuple(int(x) for x in payload["layer_indices"].tolist()),
        pair_ids=pair_ids,
        row_indices=np.asarray(payload["row_indices"], dtype=np.int64),
        roles=tuple(str(x) for x in payload["roles"].tolist()),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        lm_score_mean=np.asarray(payload["lm_score_mean"], dtype=np.float32),
        lm_score_sum=np.asarray(payload["lm_score_sum"], dtype=np.float32),
    )


__all__ = [
    "ActivationCache",
    "extract_activation_cache",
    "load_activation_cache",
    "parse_layer_arg",
    "save_activation_cache",
]
