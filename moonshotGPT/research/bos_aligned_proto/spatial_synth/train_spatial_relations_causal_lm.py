#!/usr/bin/env python3
"""Fine-tune GPT-style causal LMs on synthetic spatial-relations text.

This script intentionally keeps the data path simple: read generated text,
tokenize with the default GPT-2 tokenizer, pack examples into fixed-length
causal-LM blocks, fine-tune one or more checkpoints, and periodically evaluate
using the repo's EWoK BabyLM completion-choice full-mean metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from research.bos_aligned_proto.spatial_synth.synthetic_spatial_eval import (
    CONCEPTS as SYNTHETIC_SPATIAL_CONCEPTS,
    TIERS as SYNTHETIC_SPATIAL_TIERS,
    evaluate_synthetic_spatial,
    generate_synthetic_spatial_eval_items,
    write_synthetic_spatial_eval_dataset,
)


BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
SPATIAL_DOMAIN = "spatial-relations"
DIFFICULTY_LABELS = ("easy", "medium", "hard")
DOMAIN_ORDER = (
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations",
)


@dataclass(frozen=True)
class TextRow:
    text: str
    difficulty: str | None
    source: Dict[str, object]
    context: str = ""
    completion: str = ""


@dataclass(frozen=True)
class PackStats:
    examples: int
    tokens_with_eos: int
    blocks: int
    padded_tokens: int
    loss_tokens: int
    loss_token_fraction: float
    loss_weight_sum: float
    full_loss_examples: int
    completion_loss_examples: int
    weighted_loss_examples: int


class PackedCausalDataset(Dataset):
    def __init__(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        labels: List[torch.Tensor],
        loss_weights: List[torch.Tensor],
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.loss_weights = loss_weights

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "loss_weights": self.loss_weights[idx],
        }


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.item()
    try:
        import numpy as np

        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    value = value.rstrip("/").split("/")[-1] or value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def normalize_difficulty(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in {"1", "1.0"}:
        return "easy"
    if raw in {"2", "2.0"}:
        return "medium"
    if raw in {"3", "3.0"}:
        return "hard"
    if raw in {"easy", "easier"}:
        return "easy"
    if raw in {"medium", "mid", "moderate"}:
        return "medium"
    if raw in {"hard", "difficult", "hardest"}:
        return "hard"
    try:
        number = float(raw)
    except ValueError:
        return raw if raw in DIFFICULTY_LABELS else None
    if number <= 1.5:
        return "easy"
    if number <= 2.5:
        return "medium"
    return "hard"


def infer_row_difficulty(row: Dict[str, object]) -> str | None:
    for key in (
        "difficulty_label",
        "difficulty",
        "difficulty_band",
        "both_right_difficulty_band",
        "difficulty_by_both_right",
    ):
        if key in row:
            label = normalize_difficulty(row.get(key))
            if label is not None:
                return label
    return None


def split_context_completion(text: str) -> Tuple[str, str]:
    text = " ".join(str(text).strip().split())
    split_at = text.rfind(". ")
    if split_at < 0:
        return "", text
    context = text[: split_at + 1].strip()
    completion = text[split_at + 2 :].strip()
    return context, completion


def row_context_completion(row: Dict[str, object], text: str) -> Tuple[str, str]:
    context = str(row.get("context", "") or "").strip()
    completion = str(row.get("completion", "") or "").strip()
    if completion:
        return context, completion
    return split_context_completion(text)


def load_text_rows(path: Path, *, text_column: str) -> List[TextRow]:
    suffix = path.suffix.lower()
    rows: List[TextRow] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if text_column not in (reader.fieldnames or []):
                raise ValueError(f"{path} does not contain text column {text_column!r}")
            for row in reader:
                text = str(row.get(text_column, "")).strip()
                if text:
                    context, completion = row_context_completion(row, text)
                    rows.append(
                        TextRow(
                            text=text,
                            difficulty=infer_row_difficulty(row),
                            source=dict(row),
                            context=context,
                            completion=completion,
                        )
                    )
        return rows

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    continue
                text = str(payload.get(text_column, "")).strip()
                if text:
                    context, completion = row_context_completion(payload, text)
                    rows.append(
                        TextRow(
                            text=text,
                            difficulty=infer_row_difficulty(payload),
                            source=payload,
                            context=context,
                            completion=completion,
                        )
                    )
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            text = line.strip()
            if text:
                context, completion = split_context_completion(text)
                rows.append(
                    TextRow(
                        text=text,
                        difficulty=None,
                        source={"line_index": str(idx)},
                        context=context,
                        completion=completion,
                    )
                )
    return rows


def select_rows(
    rows: Sequence[TextRow],
    *,
    difficulty: str,
    balance_mixed: bool,
    max_examples: int | None,
    seed: int,
) -> List[TextRow]:
    if difficulty not in {"mixed", "all", *DIFFICULTY_LABELS}:
        raise ValueError(f"Unsupported difficulty={difficulty!r}")

    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)

    if difficulty in DIFFICULTY_LABELS:
        selected = [row for row in selected if row.difficulty == difficulty]
    elif difficulty == "mixed" and balance_mixed:
        groups: Dict[str, List[TextRow]] = {label: [] for label in DIFFICULTY_LABELS}
        for row in selected:
            if row.difficulty in groups:
                groups[row.difficulty].append(row)
        if all(groups[label] for label in DIFFICULTY_LABELS):
            if max_examples is None:
                per_label = min(len(groups[label]) for label in DIFFICULTY_LABELS)
                quotas = {label: per_label for label in DIFFICULTY_LABELS}
            else:
                base = max_examples // len(DIFFICULTY_LABELS)
                remainder = max_examples % len(DIFFICULTY_LABELS)
                quotas = {
                    label: base + (1 if idx < remainder else 0)
                    for idx, label in enumerate(DIFFICULTY_LABELS)
                }
            balanced: List[TextRow] = []
            for label in DIFFICULTY_LABELS:
                rng.shuffle(groups[label])
                balanced.extend(groups[label][: quotas[label]])
            rng.shuffle(balanced)
            selected = balanced

    if max_examples is not None and difficulty != "mixed":
        selected = selected[: max(0, int(max_examples))]
    if not selected:
        raise ValueError(f"No rows selected for difficulty={difficulty!r}")
    return selected


def split_train_val(rows: Sequence[TextRow], *, val_frac: float, seed: int) -> Tuple[List[TextRow], List[TextRow]]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_frac))
    if val_count == 0 and val_frac > 0.0 and len(shuffled) > 1:
        val_count = 1
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        train_rows, val_rows = shuffled, []
    return train_rows, val_rows


def text_and_loss_parts(row: TextRow | str) -> Tuple[str, str, str]:
    if isinstance(row, TextRow):
        context = row.context.strip()
        completion = row.completion.strip()
        if completion:
            return row.text, context, completion
        context, completion = split_context_completion(row.text)
        return row.text, context, completion
    text = str(row).strip()
    context, completion = split_context_completion(text)
    return text, context, completion


def mixed_example_loss_weights(
    *,
    context_token_count: int,
    completion_token_count_with_eos: int,
    completion_loss_ratio: float,
) -> List[float]:
    weights = [0.0] * (context_token_count + completion_token_count_with_eos)
    context_supervised_count = max(0, context_token_count - 1)
    completion_supervised_count = max(0, completion_token_count_with_eos)
    if context_supervised_count == 0 and completion_supervised_count == 0:
        return weights

    completion_mass = completion_loss_ratio if completion_supervised_count else 0.0
    context_mass = 1.0 - completion_loss_ratio if context_supervised_count else 0.0
    if context_supervised_count == 0:
        completion_mass = 1.0
    if completion_supervised_count == 0:
        context_mass = 1.0

    if context_supervised_count:
        context_weight = context_mass / float(context_supervised_count)
        for idx in range(1, context_token_count):
            weights[idx] = context_weight
    if completion_supervised_count:
        completion_weight = completion_mass / float(completion_supervised_count)
        for idx in range(context_token_count, context_token_count + completion_token_count_with_eos):
            weights[idx] = completion_weight
    return weights


def pack_texts(
    rows: Sequence[TextRow | str],
    tokenizer,
    *,
    block_size: int,
    drop_remainder: bool = False,
    loss_mode: str = "full",
    completion_loss_ratio: float = 0.7,
    mixed_full_loss_ratio: float = 0.7,
    seed: int = 0,
) -> Tuple[PackedCausalDataset, PackStats]:
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    if loss_mode not in {"full", "completion", "mixed", "weighted"}:
        raise ValueError(f"Unsupported loss_mode={loss_mode!r}")
    if not 0.0 <= completion_loss_ratio <= 1.0:
        raise ValueError(f"completion_loss_ratio must be in [0, 1], got {completion_loss_ratio}")
    if not 0.0 <= mixed_full_loss_ratio <= 1.0:
        raise ValueError(f"mixed_full_loss_ratio must be in [0, 1], got {mixed_full_loss_ratio}")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("GPT-2 tokenizer must expose eos_token_id for packing.")

    token_ids: List[int] = []
    label_ids: List[int] = []
    loss_weight_ids: List[float] = []
    mixed_completion_indices: set[int] = set()
    if loss_mode == "mixed":
        completion_count = int(round(len(rows) * (1.0 - mixed_full_loss_ratio)))
        completion_count = min(len(rows), max(0, completion_count))
        mixed_completion_indices = set(random.Random(seed).sample(range(len(rows)), completion_count))

    full_loss_examples = 0
    completion_loss_examples = 0
    weighted_loss_examples = 0

    for row_idx, row in enumerate(rows):
        text, context, completion = text_and_loss_parts(row)
        example_loss_mode = loss_mode
        if loss_mode == "mixed":
            example_loss_mode = "completion" if row_idx in mixed_completion_indices else "full"

        if example_loss_mode == "full":
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                token_ids.extend(ids)
                token_ids.append(int(eos_id))
                label_ids.extend(ids)
                label_ids.append(int(eos_id))
                loss_weight_ids.extend([1.0] * (len(ids) + 1))
                full_loss_examples += 1
            continue

        if not completion:
            continue
        completion_text = completion
        if context and not completion_text.startswith((" ", "\n", "\t")):
            completion_text = " " + completion_text
        context_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if not completion_ids:
            continue
        ids = context_ids + completion_ids + [int(eos_id)]
        if example_loss_mode == "completion":
            # Completion-only loss can overweight short relation words such as
            # above/below/left/right. The scenarios vary, but relation-token
            # weighting may still be worth testing explicitly later.
            labels = [-100] * len(context_ids) + completion_ids + [int(eos_id)]
            weights = [0.0] * len(context_ids) + [1.0] * (len(completion_ids) + 1)
            completion_loss_examples += 1
        else:
            labels = ids
            weights = mixed_example_loss_weights(
                context_token_count=len(context_ids),
                completion_token_count_with_eos=len(completion_ids) + 1,
                completion_loss_ratio=completion_loss_ratio,
            )
            weighted_loss_examples += 1
        token_ids.extend(ids)
        label_ids.extend(labels)
        loss_weight_ids.extend(weights)

    if not token_ids:
        raise ValueError("No tokens produced from selected training text.")
    if len(token_ids) != len(label_ids) or len(token_ids) != len(loss_weight_ids):
        raise RuntimeError("Internal packing error: token/label/weight streams have different lengths.")

    input_blocks: List[torch.Tensor] = []
    mask_blocks: List[torch.Tensor] = []
    label_blocks: List[torch.Tensor] = []
    weight_blocks: List[torch.Tensor] = []
    padded_tokens = 0
    loss_tokens = 0
    loss_weight_sum = 0.0

    for start in range(0, len(token_ids), block_size):
        chunk = token_ids[start : start + block_size]
        label_chunk = label_ids[start : start + block_size]
        weight_chunk = loss_weight_ids[start : start + block_size]
        if len(chunk) < 2 and input_blocks:
            break
        if len(chunk) < block_size and drop_remainder:
            continue
        pad_count = block_size - len(chunk)
        padded_tokens += max(0, pad_count)
        shifted_pairs = list(zip(label_chunk[1:], weight_chunk[1:]))
        loss_tokens += sum(1 for label, weight in shifted_pairs if label != -100 and weight > 0.0)
        loss_weight_sum += sum(float(weight) for label, weight in shifted_pairs if label != -100)
        input_chunk = chunk + [int(eos_id)] * max(0, pad_count)
        labels = label_chunk + [-100] * max(0, pad_count)
        loss_weights = weight_chunk + [0.0] * max(0, pad_count)
        attention_mask = [1] * len(chunk) + [0] * max(0, pad_count)
        input_blocks.append(torch.tensor(input_chunk, dtype=torch.long))
        label_blocks.append(torch.tensor(labels, dtype=torch.long))
        weight_blocks.append(torch.tensor(loss_weights, dtype=torch.float32))
        mask_blocks.append(torch.tensor(attention_mask, dtype=torch.long))

    if not input_blocks:
        raise ValueError("Packing produced zero blocks; reduce block_size or disable drop_remainder.")

    stats = PackStats(
        examples=len(rows),
        tokens_with_eos=len(token_ids),
        blocks=len(input_blocks),
        padded_tokens=padded_tokens,
        loss_tokens=loss_tokens,
        loss_token_fraction=float(loss_tokens) / float(max(1, len(token_ids))),
        loss_weight_sum=loss_weight_sum,
        full_loss_examples=full_loss_examples,
        completion_loss_examples=completion_loss_examples,
        weighted_loss_examples=weighted_loss_examples,
    )
    return PackedCausalDataset(input_blocks, mask_blocks, label_blocks, weight_blocks), stats


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2_tokenizer(name_or_path: str, *, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint: str, *, local_files_only: bool, torch_dtype: str):
    kwargs = {"local_files_only": local_files_only}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    model.config.use_cache = False
    return model


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def lr_for_step(
    *,
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
    min_lr_ratio: float,
) -> float:
    if schedule == "constant":
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step + 1) / float(warmup_steps)
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule={schedule!r}")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr * min_lr_ratio
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(total_steps - warmup_steps)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + coeff * (base_lr - min_lr)


def batch_causal_lm_loss(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor | None, int, float]:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    shift_weights = batch.get("loss_weights")
    if shift_weights is not None:
        shift_weights = shift_weights[..., 1:].contiguous().to(dtype=shift_logits.dtype)
    else:
        shift_weights = torch.ones_like(shift_labels, dtype=shift_logits.dtype)
    valid_mask = shift_labels != -100
    shift_weights = shift_weights * valid_mask.to(dtype=shift_weights.dtype)
    loss_tokens = int((shift_labels != -100).sum().item())
    loss_weight_sum = float(shift_weights.sum().item())
    if loss_tokens <= 0 or loss_weight_sum <= 0.0:
        return None, 0, 0.0
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    weighted_loss = (token_losses * shift_weights.view(-1)).sum()
    return weighted_loss / shift_weights.sum(), loss_tokens, loss_weight_sum


@torch.no_grad()
def evaluate_val_loss(model, loader: DataLoader, device: torch.device) -> float | None:
    if loader is None:
        return None
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _loss_tokens, loss_weight_sum = batch_causal_lm_loss(model, batch)
        if loss is None:
            continue
        loss_sum += float(loss.item()) * loss_weight_sum
        weight_sum += loss_weight_sum
    return loss_sum / weight_sum if weight_sum else None


def import_ewok_eval(ewok_variant: str):
    os.environ["EWOK_VARIANT"] = ewok_variant
    from evaluation import ewok as ewok_eval

    return ewok_eval


def run_ewok_eval_record(
    *,
    model,
    tokenizer,
    ewok_eval,
    ewok_batch_size: int,
    step: int,
    epoch: float,
    run_label: str,
    checkpoint: str,
    learning_rate: float,
    train_loss_last: float | None,
    val_loss: float | None,
    tokens_seen: int,
    loss_tokens_seen: int,
    loss_weight_seen: float,
    ewok_items_path: Path,
    show_progress: bool,
) -> Dict:
    model.eval()
    with torch.no_grad():
        metrics_by_method, per_item = ewok_eval.evaluate(
            model,
            tokenizer,
            batch_size=ewok_batch_size,
            return_per_item=True,
            score_reduction="mean",
            return_all_methods=True,
            show_progress=show_progress,
        )

    timestamp = datetime.now().isoformat()
    babylm = metrics_by_method[BABYLM_COMPLETION_CHOICE]
    context = metrics_by_method.get(EWOK_CONTEXT_SENSITIVITY)

    for rec in per_item:
        item_record = dict(rec)
        item_record.update(
            {
                "type": "ewok_item_mean",
                "step": int(step),
                "epoch": float(epoch),
                "timestamp": timestamp,
                "run_label": run_label,
            }
        )
        append_jsonl(ewok_items_path, item_record)

    record = {
        "step": int(step),
        "epoch": float(epoch),
        "timestamp": timestamp,
        "run_label": run_label,
        "checkpoint": checkpoint,
        "learning_rate": float(learning_rate),
        "train_loss_last": None if train_loss_last is None else float(train_loss_last),
        "val_loss": None if val_loss is None else float(val_loss),
        "tokens_seen_global_approx": int(tokens_seen),
        "input_tokens_seen": int(tokens_seen),
        "loss_tokens_seen": int(loss_tokens_seen),
        "loss_token_fraction_seen": float(loss_tokens_seen) / float(max(1, tokens_seen)),
        "loss_weight_seen": float(loss_weight_seen),
        "ewok_reductions": ["mean"],
        "ewok_primary_reduction": "mean",
        "eval_official": to_jsonable(babylm["domain_scores_official"]),
        "eval_full": to_jsonable(babylm["domain_scores_full"]),
        "eval_margin_stats": to_jsonable(babylm.get("domain_margin_stats")),
        "eval_official_mean": to_jsonable(babylm["domain_scores_official"]),
        "eval_full_mean": to_jsonable(babylm["domain_scores_full"]),
        "eval_margin_stats_mean": to_jsonable(babylm.get("domain_margin_stats")),
        "eval_babylm_completion_choice_official": to_jsonable(babylm["domain_scores_official"]),
        "eval_babylm_completion_choice_full": to_jsonable(babylm["domain_scores_full"]),
        "eval_babylm_completion_choice_margin_stats": to_jsonable(babylm.get("domain_margin_stats")),
        "eval_babylm_completion_choice_official_mean": to_jsonable(babylm["domain_scores_official"]),
        "eval_babylm_completion_choice_full_mean": to_jsonable(babylm["domain_scores_full"]),
        "eval_babylm_completion_choice_margin_stats_mean": to_jsonable(babylm.get("domain_margin_stats")),
    }
    if context is not None:
        record.update(
            {
                "eval_context_sensitivity_official_mean": to_jsonable(context["domain_scores_official"]),
                "eval_context_sensitivity_full_mean": to_jsonable(context["domain_scores_full"]),
                "eval_context_sensitivity_margin_stats_mean": to_jsonable(context.get("domain_margin_stats")),
                "eval_ewok_paper_context_sensitivity_official_mean": to_jsonable(context["domain_scores_official"]),
                "eval_ewok_paper_context_sensitivity_full_mean": to_jsonable(context["domain_scores_full"]),
                "eval_ewok_paper_context_sensitivity_margin_stats_mean": to_jsonable(
                    context.get("domain_margin_stats")
                ),
            }
        )

    full = babylm["domain_scores_full"]
    margins = babylm.get("domain_margin_stats") or {}
    avg = pair_to_scalar(full.get("average"))
    spatial_acc = pair_to_scalar(full.get(SPATIAL_DOMAIN))
    spatial_margin = None
    if isinstance(margins.get(SPATIAL_DOMAIN), dict):
        spatial_margin = margins[SPATIAL_DOMAIN].get("mean_signed_m")
    print(
        f"EWoK mean @ epoch {epoch:.3f}, step {step}: "
        f"avg={avg if avg is not None else float('nan'):.4f}, "
        f"spatial={spatial_acc if spatial_acc is not None else float('nan'):.4f}, "
        f"spatial_margin={spatial_margin if spatial_margin is not None else float('nan'):.4f}"
    )
    return record


def run_synthetic_spatial_eval_record(
    *,
    model,
    tokenizer,
    synthetic_items,
    batch_size: int,
    step: int,
    epoch: float,
    run_label: str,
    items_path: Path,
) -> Dict:
    if not synthetic_items:
        return {}

    model.eval()
    summary, per_item = evaluate_synthetic_spatial(
        model=model,
        tokenizer=tokenizer,
        items=synthetic_items,
        batch_size=batch_size,
        score_reduction="mean",
    )
    timestamp = datetime.now().isoformat()
    for rec in per_item:
        item_record = dict(rec)
        item_record.update(
            {
                "type": "synthetic_spatial_item_mean",
                "step": int(step),
                "epoch": float(epoch),
                "timestamp": timestamp,
                "run_label": run_label,
            }
        )
        append_jsonl(items_path, item_record)

    overall = summary.get("overall", {})
    by_tier = summary.get("by_tier", {})
    in_format = by_tier.get("in_format", {})
    paraphrase = by_tier.get("paraphrase", {})
    composition = by_tier.get("composition", {})
    print(
        f"Synthetic spatial @ epoch {epoch:.3f}, step {step}: "
        f"overall={float(overall.get('acc_combined', float('nan'))):.4f}, "
        f"in_format={float(in_format.get('acc_combined', float('nan'))):.4f}, "
        f"paraphrase={float(paraphrase.get('acc_combined', float('nan'))):.4f}, "
        f"composition={float(composition.get('acc_combined', float('nan'))):.4f}"
    )
    return {
        "eval_synthetic_spatial_overall": summary.get("overall"),
        "eval_synthetic_spatial_by_tier": summary.get("by_tier"),
        "eval_synthetic_spatial_by_concept": summary.get("by_concept"),
        "eval_synthetic_spatial_by_template_family": summary.get("by_template_family"),
        "eval_synthetic_spatial_by_contrast_type": summary.get("by_contrast_type"),
    }


def maybe_add_synthetic_spatial_eval(
    *,
    record: Dict,
    model,
    tokenizer,
    synthetic_items,
    batch_size: int,
    step: int,
    epoch: float,
    run_label: str,
    items_path: Path,
) -> Dict:
    if not synthetic_items:
        return record
    synthetic_record = run_synthetic_spatial_eval_record(
        model=model,
        tokenizer=tokenizer,
        synthetic_items=synthetic_items,
        batch_size=batch_size,
        step=step,
        epoch=epoch,
        run_label=run_label,
        items_path=items_path,
    )
    record.update(synthetic_record)
    return record


def pair_to_scalar(value) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return 0.5 * (float(value[0]) + float(value[1]))
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [record for record in payload if isinstance(record, dict)]


def ordered_domains_from_runs(run_records: Sequence[Tuple[str, List[Dict]]]) -> List[str]:
    seen = set()
    for _, records in run_records:
        for record in records:
            full = record.get("eval_babylm_completion_choice_full_mean", record.get("eval_full_mean"))
            if isinstance(full, dict):
                seen.update(str(domain) for domain in full if str(domain) != "average")
            margins = record.get("eval_babylm_completion_choice_margin_stats_mean", record.get("eval_margin_stats_mean"))
            if isinstance(margins, dict):
                seen.update(str(domain) for domain in margins if str(domain) != "average")
    ordered = [domain for domain in DOMAIN_ORDER if domain in seen]
    return ordered + sorted(seen.difference(ordered))


def extract_accuracy_series(records: Sequence[Dict], domain: str) -> List[Tuple[float, float]]:
    out = []
    for record in records:
        full = record.get("eval_babylm_completion_choice_full_mean", record.get("eval_full_mean"))
        if not isinstance(full, dict):
            continue
        y = pair_to_scalar(full.get(domain))
        x = record.get("epoch", record.get("step"))
        if y is not None and isinstance(x, (int, float)):
            out.append((float(x), float(y)))
    return sorted(out, key=lambda item: item[0])


def extract_margin_series(records: Sequence[Dict], domain: str) -> List[Tuple[float, float]]:
    out = []
    for record in records:
        margins = record.get("eval_babylm_completion_choice_margin_stats_mean", record.get("eval_margin_stats_mean"))
        if not isinstance(margins, dict):
            continue
        stats = margins.get(domain)
        if not isinstance(stats, dict):
            continue
        y = stats.get("mean_signed_m")
        x = record.get("epoch", record.get("step"))
        if isinstance(y, (int, float)) and isinstance(x, (int, float)):
            out.append((float(x), float(y)))
    return sorted(out, key=lambda item: item[0])


def extract_synthetic_spatial_series(
    records: Sequence[Dict],
    *,
    group_key: str,
    group_name: str,
    metric: str,
) -> List[Tuple[float, float]]:
    out = []
    for record in records:
        groups = record.get(group_key)
        if not isinstance(groups, dict):
            continue
        stats = groups.get(group_name)
        if not isinstance(stats, dict):
            continue
        y = stats.get("acc_combined" if metric == "accuracy" else "mean_signed_m")
        x = record.get("epoch", record.get("step"))
        if isinstance(y, (int, float)) and isinstance(x, (int, float)):
            out.append((float(x), float(y)))
    return sorted(out, key=lambda item: item[0])


def _plot_series_by_groups(
    run_records: Sequence[Tuple[str, List[Dict]]],
    out_path: Path,
    *,
    group_key: str,
    groups: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    dpi: int,
) -> Path | None:
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    line_styles = ["-", "--", "-.", ":"]
    plotted = 0
    for run_idx, (label, records) in enumerate(run_records):
        color = colors[run_idx % len(colors)] if colors else None
        for group_idx, group in enumerate(groups):
            series = extract_synthetic_spatial_series(
                records,
                group_key=group_key,
                group_name=group,
                metric=metric,
            )
            if not series:
                continue
            ax.plot(
                [x for x, _ in series],
                [y for _, y in series],
                marker="o",
                markersize=3,
                linewidth=1.5,
                linestyle=line_styles[group_idx % len(line_styles)],
                color=color,
                label=f"{label} / {group}",
            )
            plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    if metric == "accuracy":
        ax.axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
        ax.set_ylim(0.0, 1.0)
    else:
        ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_synthetic_spatial_concept_grid(
    run_records: Sequence[Tuple[str, List[Dict]]],
    out_path: Path,
    *,
    metric: str,
    dpi: int,
) -> Path | None:
    if plt is None:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    flat_axes = axes.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    plotted_any = False
    for idx, concept in enumerate(SYNTHETIC_SPATIAL_CONCEPTS):
        ax = flat_axes[idx]
        for run_idx, (label, records) in enumerate(run_records):
            series = extract_synthetic_spatial_series(
                records,
                group_key="eval_synthetic_spatial_by_concept",
                group_name=concept,
                metric=metric,
            )
            if not series:
                continue
            plotted_any = True
            ax.plot(
                [x for x, _ in series],
                [y for _, y in series],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[run_idx % len(colors)] if colors else None,
                label=label,
            )
        if metric == "accuracy":
            ax.axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Acc")
        else:
            ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
            ax.set_ylabel("Margin")
        ax.set_title(concept)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7)
    if not plotted_any:
        plt.close(fig)
        return None
    title = "Synthetic Spatial Concept Accuracy" if metric == "accuracy" else "Synthetic Spatial Concept Margin"
    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_synthetic_spatial_transfer_gap(
    run_records: Sequence[Tuple[str, List[Dict]]],
    out_path: Path,
    *,
    dpi: int,
) -> Path | None:
    if plt is None:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    plotted = 0
    for run_idx, (label, records) in enumerate(run_records):
        color = colors[run_idx % len(colors)] if colors else None
        ewok = dict(extract_accuracy_series(records, SPATIAL_DOMAIN))
        in_format = dict(
            extract_synthetic_spatial_series(
                records,
                group_key="eval_synthetic_spatial_by_tier",
                group_name="in_format",
                metric="accuracy",
            )
        )
        paraphrase = dict(
            extract_synthetic_spatial_series(
                records,
                group_key="eval_synthetic_spatial_by_tier",
                group_name="paraphrase",
                metric="accuracy",
            )
        )
        xs = sorted(set(ewok).intersection(in_format).intersection(paraphrase))
        if not xs:
            continue
        axes[0].plot(xs, [ewok[x] for x in xs], color=color, linewidth=1.6, linestyle="-", label=f"{label} / EWoK")
        axes[0].plot(
            xs,
            [in_format[x] for x in xs],
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"{label} / in_format",
        )
        axes[0].plot(
            xs,
            [paraphrase[x] for x in xs],
            color=color,
            linewidth=1.4,
            linestyle=":",
            label=f"{label} / paraphrase",
        )
        axes[1].plot(
            xs,
            [in_format[x] - ewok[x] for x in xs],
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"{label} / in_format-EWoK",
        )
        axes[1].plot(
            xs,
            [paraphrase[x] - ewok[x] for x in xs],
            color=color,
            linewidth=1.4,
            linestyle=":",
            label=f"{label} / paraphrase-EWoK",
        )
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    axes[0].axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Synthetic vs EWoK Spatial Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.0, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Synthetic - EWoK")
    axes[1].set_title("Transfer Gap")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_domain_grid(
    run_records: Sequence[Tuple[str, List[Dict]]],
    out_path: Path,
    *,
    metric: str,
    dpi: int,
) -> Path | None:
    if plt is None:
        print("[warn] matplotlib is unavailable; skipping plots.")
        return None

    domains = ordered_domains_from_runs(run_records)
    if not domains:
        return None

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
    flat_axes = axes.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    extract = extract_accuracy_series if metric == "accuracy" else extract_margin_series

    for idx, domain in enumerate(domains[: len(flat_axes)]):
        ax = flat_axes[idx]
        plotted = 0
        for run_idx, (label, records) in enumerate(run_records):
            series = extract(records, domain)
            if not series:
                continue
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                color=colors[run_idx % len(colors)] if colors else None,
                label=label,
            )
            plotted += 1
        if metric == "accuracy":
            ax.axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Full mean acc", fontsize=9)
        else:
            ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
            ax.set_ylabel("Mean signed margin", fontsize=9)
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, alpha=0.25)
        if plotted and idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    title = "EWoK BabyLM Completion Full Mean"
    title += " Accuracy by Domain" if metric == "accuracy" else " Margin by Domain"
    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_spatial_main(run_records: Sequence[Tuple[str, List[Dict]]], out_path: Path, *, dpi: int) -> Path | None:
    if plt is None:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for run_idx, (label, records) in enumerate(run_records):
        color = colors[run_idx % len(colors)] if colors else None
        acc = extract_accuracy_series(records, SPATIAL_DOMAIN)
        margin = extract_margin_series(records, SPATIAL_DOMAIN)
        if acc:
            axes[0].plot(
                [x for x, _ in acc],
                [y for _, y in acc],
                marker="o",
                linewidth=1.8,
                color=color,
                label=label,
            )
        if margin:
            axes[1].plot(
                [x for x, _ in margin],
                [y for _, y in margin],
                marker="s",
                linewidth=1.8,
                color=color,
                label=label,
            )

    axes[0].axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Full mean acc")
    axes[0].set_title("Spatial Relations EWoK Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color="#999999", linewidth=0.9, linestyle=(0, (4, 2)))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean signed margin")
    axes[1].set_title("Spatial Relations EWoK Margin")
    axes[1].grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_plots(run_infos: Sequence[Tuple[str, Path]], output_dir: Path, *, dpi: int) -> List[Path]:
    run_records = [(label, metric_records(path)) for label, path in run_infos]
    run_records = [(label, records) for label, records in run_records if records]
    if not run_records:
        return []

    created = []
    for metric, filename in (
        ("accuracy", "ewok_completion_full_mean_accuracy_domains_4x3_by_epoch.png"),
        ("margin", "ewok_completion_full_mean_margin_domains_4x3_by_epoch.png"),
    ):
        path = plot_domain_grid(run_records, output_dir / filename, metric=metric, dpi=dpi)
        if path is not None:
            created.append(path)
    path = plot_spatial_main(run_records, output_dir / "spatial_relations_main_by_epoch.png", dpi=dpi)
    if path is not None:
        created.append(path)
    for metric, filename, ylabel in (
        ("accuracy", "synthetic_spatial_accuracy_by_tier.png", "Synthetic acc"),
        ("margin", "synthetic_spatial_margin_by_tier.png", "Synthetic margin"),
    ):
        path = _plot_series_by_groups(
            run_records,
            output_dir / filename,
            group_key="eval_synthetic_spatial_by_tier",
            groups=SYNTHETIC_SPATIAL_TIERS,
            metric=metric,
            title=f"Synthetic Spatial {metric.title()} by Tier",
            ylabel=ylabel,
            dpi=dpi,
        )
        if path is not None:
            created.append(path)
    for metric, filename in (
        ("accuracy", "synthetic_spatial_accuracy_by_concept.png"),
        ("margin", "synthetic_spatial_margin_by_concept.png"),
    ):
        path = plot_synthetic_spatial_concept_grid(run_records, output_dir / filename, metric=metric, dpi=dpi)
        if path is not None:
            created.append(path)
    path = plot_synthetic_spatial_transfer_gap(
        run_records,
        output_dir / "synthetic_spatial_transfer_gap.png",
        dpi=dpi,
    )
    if path is not None:
        created.append(path)
    return created


def save_selected_rows(rows: Sequence[TextRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "context", "completion", "difficulty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "text": row.text,
                    "context": row.context,
                    "completion": row.completion,
                    "difficulty": row.difficulty or "",
                }
            )


def infer_template_preset_from_data_path(path: Path) -> str:
    name = str(path)
    for preset in ("v21", "v20", "v19", "v15", "v14", "v13", "v12", "v11", "v10", "v9", "v8", "v7", "v6", "v5", "v4", "v3"):
        if preset in name:
            return preset
    return "v4"


def train_one_run(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    learning_rate: float,
    tokenizer,
    train_rows: Sequence[TextRow],
    val_rows: Sequence[TextRow],
    run_dir: Path,
) -> Tuple[str, Path]:
    set_all_seeds(args.seed)
    device = resolve_device(args.device)
    loss_label = args.loss_mode
    if args.loss_mode == "mixed":
        loss_label = f"mixedfull{args.mixed_full_loss_ratio:g}"
    elif args.loss_mode == "weighted":
        loss_label = f"weighted{args.completion_loss_ratio:g}"
    label = f"{safe_name(checkpoint)} lr={learning_rate:g} loss={loss_label}"

    train_dataset, train_pack_stats = pack_texts(
        train_rows,
        tokenizer,
        block_size=args.block_size,
        drop_remainder=args.drop_remainder,
        loss_mode=args.loss_mode,
        completion_loss_ratio=args.completion_loss_ratio,
        mixed_full_loss_ratio=args.mixed_full_loss_ratio,
        seed=args.seed,
    )
    val_dataset = None
    val_pack_stats = None
    if val_rows:
        val_dataset, val_pack_stats = pack_texts(
            val_rows,
            tokenizer,
            block_size=args.block_size,
            drop_remainder=False,
            loss_mode=args.loss_mode,
            completion_loss_ratio=args.completion_loss_ratio,
            mixed_full_loss_ratio=args.mixed_full_loss_ratio,
            seed=args.seed + 1,
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    val_loader = (
        None
        if val_dataset is None
        else DataLoader(
            val_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=max(0, min(args.num_workers, 2)),
        )
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    save_selected_rows(train_rows, run_dir / "selected_train_rows.csv")
    if val_rows:
        save_selected_rows(val_rows, run_dir / "selected_val_rows.csv")

    metrics_path = run_dir / "step_metrics.json"
    scalars_path = run_dir / "scalars.jsonl"
    ewok_items_path = run_dir / "ewok_items.jsonl"
    synthetic_spatial_items_path = run_dir / "synthetic_spatial_items.jsonl"
    step_metrics: List[Dict] = []

    synthetic_spatial_items = []
    synthetic_spatial_batch_size = args.synthetic_spatial_eval_batch_size or args.ewok_batch_size
    if args.synthetic_spatial_eval != "off":
        synthetic_seed = (
            args.seed + 100000
            if args.synthetic_spatial_eval_seed is None
            else int(args.synthetic_spatial_eval_seed)
        )
        synthetic_template_preset = (
            args.synthetic_spatial_eval_template_preset
            or infer_template_preset_from_data_path(args.data)
        )
        synthetic_spatial_items = generate_synthetic_spatial_eval_items(
            n_per_tier=args.synthetic_spatial_eval_n_per_tier,
            seed=synthetic_seed,
            template_preset=synthetic_template_preset,
        )
        write_synthetic_spatial_eval_dataset(
            synthetic_spatial_items,
            run_dir / "synthetic_spatial_eval_dataset.jsonl",
        )

    atomic_write_json(
        run_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "script": str(Path(__file__).resolve()),
            "checkpoint": checkpoint,
            "learning_rate": learning_rate,
            "args": vars(args),
            "train_pack_stats": train_pack_stats.__dict__,
            "val_pack_stats": None if val_pack_stats is None else val_pack_stats.__dict__,
            "synthetic_spatial_eval_items": len(synthetic_spatial_items),
            "synthetic_spatial_eval_batch_size": synthetic_spatial_batch_size,
        },
    )

    print(f"Loading model: {checkpoint}")
    model = load_model(checkpoint, local_files_only=args.local_files_only, torch_dtype=args.torch_dtype)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_steps = max(1, int(args.epochs * updates_per_epoch))
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    eval_interval_steps = max(1, int(round(args.epoch_eval * updates_per_epoch)))
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    set_optimizer_lr(
        optimizer,
        lr_for_step(
            step=0,
            base_lr=learning_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule=args.lr_schedule,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )

    ewok_eval = import_ewok_eval(args.ewok_variant)
    tokens_seen = 0
    loss_tokens_seen = 0
    loss_weight_seen = 0.0
    update_step = 0
    last_train_loss = None
    last_eval_step = None

    print(
        f"Run {label}: train_blocks={len(train_dataset)}, val_blocks={len(val_dataset) if val_dataset else 0}, "
        f"updates_per_epoch={updates_per_epoch}, total_steps={total_steps}, eval_every={eval_interval_steps} steps, "
        f"train_loss_token_fraction={train_pack_stats.loss_token_fraction:.3f}, "
        f"loss_examples(full={train_pack_stats.full_loss_examples}, "
        f"completion={train_pack_stats.completion_loss_examples}, weighted={train_pack_stats.weighted_loss_examples})"
    )

    if args.eval_at_start:
        val_loss = evaluate_val_loss(model, val_loader, device)
        record = run_ewok_eval_record(
            model=model,
            tokenizer=tokenizer,
            ewok_eval=ewok_eval,
            ewok_batch_size=args.ewok_batch_size,
            step=0,
            epoch=0.0,
            run_label=label,
            checkpoint=checkpoint,
            learning_rate=learning_rate,
            train_loss_last=None,
            val_loss=val_loss,
            tokens_seen=tokens_seen,
            loss_tokens_seen=loss_tokens_seen,
            loss_weight_seen=loss_weight_seen,
            ewok_items_path=ewok_items_path,
            show_progress=args.ewok_progress,
        )
        record = maybe_add_synthetic_spatial_eval(
            record=record,
            model=model,
            tokenizer=tokenizer,
            synthetic_items=synthetic_spatial_items,
            batch_size=synthetic_spatial_batch_size,
            step=0,
            epoch=0.0,
            run_label=label,
            items_path=synthetic_spatial_items_path,
        )
        step_metrics.append(record)
        atomic_write_json(metrics_path, step_metrics)
        append_jsonl(scalars_path, {"type": "ewok", **record})
        last_eval_step = 0

    model.train()
    stop_training = False
    for epoch_idx in range(int(math.ceil(args.epochs))):
        if stop_training:
            break
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())
            loss_mean, batch_loss_tokens, batch_loss_weight = batch_causal_lm_loss(model, batch)
            if loss_mean is None:
                continue
            loss = loss_mean / max(1, args.grad_accum_steps)
            loss.backward()
            accum_count += 1
            loss_tokens_seen += batch_loss_tokens
            loss_weight_seen += batch_loss_weight
            last_train_loss = float(loss_mean.item())

            should_step = accum_count >= args.grad_accum_steps or batch_idx == len(train_loader) - 1
            if not should_step:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            update_step += 1

            next_lr = lr_for_step(
                step=update_step,
                base_lr=learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                schedule=args.lr_schedule,
                min_lr_ratio=args.min_lr_ratio,
            )
            set_optimizer_lr(optimizer, next_lr)

            epoch_float = update_step / float(updates_per_epoch)
            if update_step % args.log_every == 0:
                print(
                    f"{label} step {update_step}/{total_steps} "
                    f"epoch={epoch_float:.3f} loss={last_train_loss:.4f} lr={next_lr:.3e} "
                    f"loss_tokens={loss_tokens_seen} loss_frac={loss_tokens_seen / max(1, tokens_seen):.3f}"
                )

            due_for_eval = update_step % eval_interval_steps == 0
            if due_for_eval:
                val_loss = evaluate_val_loss(model, val_loader, device)
                record = run_ewok_eval_record(
                    model=model,
                    tokenizer=tokenizer,
                    ewok_eval=ewok_eval,
                    ewok_batch_size=args.ewok_batch_size,
                    step=update_step,
                    epoch=epoch_float,
                    run_label=label,
                    checkpoint=checkpoint,
                    learning_rate=learning_rate,
                    train_loss_last=last_train_loss,
                    val_loss=val_loss,
                    tokens_seen=tokens_seen,
                    loss_tokens_seen=loss_tokens_seen,
                    loss_weight_seen=loss_weight_seen,
                    ewok_items_path=ewok_items_path,
                    show_progress=args.ewok_progress,
                )
                record = maybe_add_synthetic_spatial_eval(
                    record=record,
                    model=model,
                    tokenizer=tokenizer,
                    synthetic_items=synthetic_spatial_items,
                    batch_size=synthetic_spatial_batch_size,
                    step=update_step,
                    epoch=epoch_float,
                    run_label=label,
                    items_path=synthetic_spatial_items_path,
                )
                step_metrics.append(record)
                atomic_write_json(metrics_path, step_metrics)
                append_jsonl(scalars_path, {"type": "ewok", **record})
                last_eval_step = update_step
                model.train()
                if args.save_every_eval:
                    ckpt_dir = run_dir / "checkpoints" / f"step_{update_step:06d}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

            if update_step >= total_steps:
                stop_training = True
                break

    if last_eval_step != update_step:
        epoch_float = update_step / float(updates_per_epoch)
        val_loss = evaluate_val_loss(model, val_loader, device)
        record = run_ewok_eval_record(
            model=model,
            tokenizer=tokenizer,
            ewok_eval=ewok_eval,
            ewok_batch_size=args.ewok_batch_size,
            step=update_step,
            epoch=epoch_float,
            run_label=label,
            checkpoint=checkpoint,
            learning_rate=learning_rate,
            train_loss_last=last_train_loss,
            val_loss=val_loss,
            tokens_seen=tokens_seen,
            loss_tokens_seen=loss_tokens_seen,
            loss_weight_seen=loss_weight_seen,
            ewok_items_path=ewok_items_path,
            show_progress=args.ewok_progress,
        )
        record = maybe_add_synthetic_spatial_eval(
            record=record,
            model=model,
            tokenizer=tokenizer,
            synthetic_items=synthetic_spatial_items,
            batch_size=synthetic_spatial_batch_size,
            step=update_step,
            epoch=epoch_float,
            run_label=label,
            items_path=synthetic_spatial_items_path,
        )
        step_metrics.append(record)
        atomic_write_json(metrics_path, step_metrics)
        append_jsonl(scalars_path, {"type": "ewok", **record})

    if args.save_final:
        final_dir = run_dir / "final_model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model to {final_dir}")

    write_plots([(label, metrics_path)], run_dir / "plots", dpi=args.plot_dpi)
    return label, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Generated CSV/JSONL/TXT training data.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/research/bos_aligned_proto/spatial_synth_training"),
    )
    parser.add_argument("--checkpoints", nargs="+", default=["gpt2-medium"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[5e-5])
    parser.add_argument("--tokenizer-name", default="gpt2", help="Default GPT-2 tokenizer.")
    parser.add_argument("--difficulty", choices=("mixed", "all", *DIFFICULTY_LABELS), default="mixed")
    parser.add_argument("--no-balance-mixed", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument(
        "--loss-mode",
        choices=("full", "completion", "mixed", "weighted"),
        default="full",
        help=(
            "full trains on every token; completion masks setup/context tokens; "
            "mixed uses full loss for some examples and completion-only loss for the rest; "
            "weighted gives completion and context weighted loss shares within every example."
        ),
    )
    parser.add_argument(
        "--mixed-full-loss-ratio",
        type=float,
        default=0.7,
        help="For --loss-mode mixed, fraction of examples trained with full loss; the rest use completion-only loss.",
    )
    parser.add_argument(
        "--completion-loss-ratio",
        type=float,
        default=0.7,
        help="For --loss-mode weighted, fraction of each example's loss mass assigned to completion tokens.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--epoch-eval",
        type=float,
        default=0.5,
        help="Evaluate every this many epochs, e.g. 0.5 for twice per epoch.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ewok-batch-size", type=int, default=8)
    parser.add_argument(
        "--ewok-variant",
        choices=("fast", "full"),
        default="full",
        help="Dataset variant; scoring is BabyLM completion full mean.",
    )
    parser.add_argument("--ewok-progress", action="store_true")
    parser.add_argument(
        "--synthetic-spatial-eval",
        choices=("off", "three_tier"),
        default="off",
        help="Optional held-out synthetic EWoK-style spatial eval run at each EWoK eval step.",
    )
    parser.add_argument(
        "--synthetic-spatial-eval-n-per-tier",
        type=int,
        default=300,
        help="Number of held-out synthetic spatial eval items per tier.",
    )
    parser.add_argument(
        "--synthetic-spatial-eval-seed",
        type=int,
        default=None,
        help="Held-out synthetic spatial eval seed; defaults to --seed + 100000.",
    )
    parser.add_argument(
        "--synthetic-spatial-eval-template-preset",
        default=None,
        help="Preset recorded for eval provenance; defaults to inferring from the data path.",
    )
    parser.add_argument(
        "--synthetic-spatial-eval-batch-size",
        type=int,
        default=None,
        help="Batch size for synthetic spatial eval; defaults to --ewok-batch-size.",
    )
    parser.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-dpi", type=int, default=140)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epoch_eval <= 0:
        raise ValueError("--epoch-eval must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0.0 <= args.completion_loss_ratio <= 1.0:
        raise ValueError("--completion-loss-ratio must be in [0, 1]")
    if not 0.0 <= args.mixed_full_loss_ratio <= 1.0:
        raise ValueError("--mixed-full-loss-ratio must be in [0, 1]")
    if args.synthetic_spatial_eval_n_per_tier <= 0:
        raise ValueError("--synthetic-spatial-eval-n-per-tier must be positive")
    if args.synthetic_spatial_eval_batch_size is not None and args.synthetic_spatial_eval_batch_size <= 0:
        raise ValueError("--synthetic-spatial-eval-batch-size must be positive")

    set_all_seeds(args.seed)
    rows = load_text_rows(args.data, text_column=args.text_column)
    selected = select_rows(
        rows,
        difficulty=args.difficulty,
        balance_mixed=not args.no_balance_mixed,
        max_examples=args.max_train_examples,
        seed=args.seed,
    )
    train_rows, val_rows = split_train_val(selected, val_frac=args.val_frac, seed=args.seed + 17)
    counts = Counter(row.difficulty or "<unknown>" for row in selected)
    print(
        f"Loaded {len(rows)} rows from {args.data}; selected {len(selected)} "
        f"({dict(counts)}), train={len(train_rows)}, val={len(val_rows)}"
    )

    tokenizer = load_gpt2_tokenizer(args.tokenizer_name, local_files_only=args.local_files_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_infos: List[Tuple[str, Path]] = []
    for checkpoint in args.checkpoints:
        for learning_rate in args.learning_rates:
            loss_name = args.loss_mode
            if args.loss_mode == "mixed":
                loss_name = f"mixedfull{args.mixed_full_loss_ratio:g}"
            elif args.loss_mode == "weighted":
                loss_name = f"weighted{args.completion_loss_ratio:g}"
            run_name = f"{safe_name(checkpoint)}_lr{learning_rate:g}_loss{loss_name}_seed{args.seed}"
            run_dir = args.output_dir / run_name
            run_infos.append(
                train_one_run(
                    args=args,
                    checkpoint=checkpoint,
                    learning_rate=learning_rate,
                    tokenizer=tokenizer,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    run_dir=run_dir,
                )
            )

    if len(run_infos) > 1:
        created = write_plots(run_infos, args.output_dir / "summary_plots", dpi=args.plot_dpi)
        if created:
            print("Wrote summary plots:")
            for path in created:
                print(f"  {path}")


if __name__ == "__main__":
    main()
