#!/usr/bin/env python3
"""Evaluate EWoK with prompted direct-choice templates on local HF models.

This helper is intentionally separate from `run_queue.py` so we can experiment
with prompting strategies without perturbing the raw continuation-likelihood
pipeline. It reuses the existing local download/model-load path, then swaps in
instruction-style EWoK prompts that ask the model to answer with `1` or `2`.

Two inference modes are supported:

- `choice_answer_logprob`: score the answer strings `1` and `2` under the prompt
  and compare their conditional log-likelihoods.
- `choice_generate`: greedily generate a short response, parse the first valid
  `1` or `2`, and turn that into discrete EWoK-style margins.
- `context_choice_generate`: show one target statement and two contexts,
  greedily generate a short response, parse the selected context, and score
  EWoK context sensitivity directly.
- `statement_true_logprob`: score each candidate statement independently and
  choose by `log P(True)`.
- `statement_true_false_margin`: score each candidate statement independently
  and choose by `log P(True) - log P(False)`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
from importlib import import_module

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

rq = import_module("moonshotGPT.hf_ewok_eval.run_queue")

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "moonshotGPT/runs/hf_ewok_eval/prompted"
DEFAULT_PROMPT_TEMPLATE = "direct_choice"
DEFAULT_INFERENCE_MODE = "choice_answer_logprob"
DEFAULT_EWOK_VARIANT = "fast"
DEFAULT_SCORE_REDUCTION = "mean"
DEFAULT_MARGIN_EPS = 1e-6
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_ANSWER_SEPARATOR = " "
DEFAULT_TRUE_LABEL = "True"
DEFAULT_FALSE_LABEL = "False"
DEFAULT_TARGET_PERMUTATION_MODE = "original"
DEFAULT_TARGET_PERMUTATION_SEED = 0
DEFAULT_QUEUE_SUMMARY_NAME = "queue_summary.json"
DEFAULT_SHOW_PROGRESS = True

INFERENCE_MODES = (
    "choice_answer_logprob",
    "choice_generate",
    "context_choice_generate",
    "statement_true_logprob",
    "statement_true_false_margin",
)
TARGET_PERMUTATION_MODES = ("original", "alternate", "random")
BUILTIN_PROMPT_TEMPLATES = {
    "direct_choice": """# INSTRUCTIONS
You will be shown one context and two possible continuations.
Your job is to decide which continuation better fits the context.
Answer with exactly one token: 1 or 2.

# TASK
## Context
"{{Ci}}"

## Candidate continuations
1. "{{T1}}"
2. "{{T2}}"

## Task
Which continuation makes more sense given the context?
Answer only 1 or 2.

## Response
""",
    "careful_choice": """# INSTRUCTIONS
You will be shown one context and two possible continuations.
Decide which continuation better fits the context.

Before answering, think carefully about:
- who is doing what to whom
- whether any entities have been swapped
- whether attributes or state changes make one continuation more plausible

Do not reveal your reasoning.
Answer with exactly one token: 1 or 2.

# TASK
## Context
"{{Ci}}"

## Candidate continuations
1. "{{T1}}"
2. "{{T2}}"

## Task
Which continuation makes more sense given the context?
Answer only 1 or 2.

## Response
""",
}


@dataclass(frozen=True)
class PromptChoiceInference:
    """Parsed prompt-level inference details for one context."""

    raw_response_text: str | None
    predicted_choice: int | None
    response_valid: bool
    choice_1_score: float
    choice_2_score: float
    scoring_signal: str | None = None
    choice_1_logp_true: float | None = None
    choice_1_logp_false: float | None = None
    choice_1_true_false_margin: float | None = None
    choice_2_logp_true: float | None = None
    choice_2_logp_false: float | None = None
    choice_2_true_false_margin: float | None = None


def _utc_now_iso() -> str:
    return rq._utc_now_iso()


def _sanitize_name(value: str) -> str:
    return rq._sanitize_name(value)


def _optional_str(value: Any) -> str | None:
    return rq._optional_str(value)


def _optional_positive_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    resolved = float(value)
    if resolved <= 0:
        raise ValueError(f"Expected a positive float, got: {value}")
    return resolved


def _maybe_tqdm(iterable: Iterable[Any], *, enabled: bool, **kwargs: Any) -> Iterable[Any]:
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _load_shared_ewok_module(variant: str) -> Any:
    os.environ["EWOK_VARIANT"] = str(variant).strip().lower()
    module_name = "moonshotGPT.evaluation.ewok"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _filter_ewok_df(*, ewok_df: Any, domains: Sequence[str], limit: int | None) -> Any:
    filtered = ewok_df
    if domains:
        wanted = {str(domain).strip() for domain in domains if str(domain).strip()}
        filtered = filtered[filtered["Domain"].isin(sorted(wanted))]
    if limit is not None:
        filtered = filtered.head(int(limit))
    return filtered.copy()


@contextmanager
def _temporary_module_attr(module: Any, attr_name: str, value: Any):
    original = getattr(module, attr_name)
    setattr(module, attr_name, value)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


@contextmanager
def _temporary_tokenizer_padding_side(tokenizer: Any, padding_side: str):
    if not hasattr(tokenizer, "padding_side"):
        yield
        return
    original = tokenizer.padding_side
    tokenizer.padding_side = str(padding_side)
    try:
        yield
    finally:
        tokenizer.padding_side = original


def _resolve_prompt_template(
    *,
    prompt_template_name: str,
    prompt_template_file: str | None,
    prompt_label: str | None,
) -> tuple[str, str, str]:
    if prompt_template_file:
        path = Path(prompt_template_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {path}")
        label = _optional_str(prompt_label) or path.stem
        return _sanitize_name(label), str(path), path.read_text(encoding="utf-8")

    name = str(prompt_template_name).strip().lower()
    if name not in BUILTIN_PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt template: {prompt_template_name}. "
            f"Expected one of: {sorted(BUILTIN_PROMPT_TEMPLATES)}"
        )
    return name, f"builtin:{name}", BUILTIN_PROMPT_TEMPLATES[name]


def _render_choice_prompt(
    *,
    template_text: str,
    context_text: str,
    target_1_text: str,
    target_2_text: str,
) -> str:
    rendered = str(template_text)
    replacements = {
        "{{Ci}}": str(context_text),
        "{{context}}": str(context_text),
        "{{T1}}": str(target_1_text),
        "{{candidate1}}": str(target_1_text),
        "{{T2}}": str(target_2_text),
        "{{candidate2}}": str(target_2_text),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _render_statement_prompt(
    *,
    template_text: str,
    context_text: str,
    statement_text: str,
) -> str:
    rendered = str(template_text)
    replacements = {
        "{{Ci}}": str(context_text),
        "{{context}}": str(context_text),
        "{{Tj}}": str(statement_text),
        "{{statement}}": str(statement_text),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _render_context_choice_prompt(
    *,
    template_text: str,
    statement_text: str,
    context_1_text: str,
    context_2_text: str,
) -> str:
    rendered = str(template_text)
    replacements = {
        "{{Tj}}": str(statement_text),
        "{{statement}}": str(statement_text),
        "{{C1}}": str(context_1_text),
        "{{context1}}": str(context_1_text),
        "{{C2}}": str(context_2_text),
        "{{context2}}": str(context_2_text),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _should_swap_targets_for_row(
    *,
    row_index: int,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> bool:
    mode = str(target_permutation_mode).strip().lower()
    if mode == "original":
        return False
    if mode == "alternate":
        return bool(int(row_index) % 2)
    if mode == "random":
        seed = int(target_permutation_seed)
        mixed = (int(row_index) * 1103515245 + 12345 + seed) & 0x7FFFFFFF
        return bool(mixed % 2)
    raise ValueError(
        f"Unsupported target_permutation_mode: {target_permutation_mode}. "
        f"Expected one of: {TARGET_PERMUTATION_MODES}"
    )


def _canonical_choice_from_display_choice(
    displayed_choice: int | None,
    *,
    swapped: bool,
) -> int | None:
    if displayed_choice is None:
        return None
    if displayed_choice not in {1, 2}:
        return None
    if not swapped:
        return int(displayed_choice)
    return 2 if int(displayed_choice) == 1 else 1


def _canonical_scores_from_display_scores(
    *,
    displayed_choice_1_score: float,
    displayed_choice_2_score: float,
    swapped: bool,
) -> tuple[float, float]:
    if not swapped:
        return float(displayed_choice_1_score), float(displayed_choice_2_score)
    return float(displayed_choice_2_score), float(displayed_choice_1_score)


def _display_gold_choice_for_target(*, canonical_target: int, swapped: bool) -> int:
    if canonical_target not in {1, 2}:
        raise ValueError(f"canonical_target must be 1 or 2, got: {canonical_target}")
    if not swapped:
        return int(canonical_target)
    return 2 if int(canonical_target) == 1 else 1


def _parse_choice_response_text(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None

    answer_matches = re.findall(
        r"(?i)answer\s*:\s*([12])\b",
        stripped,
    )
    if answer_matches:
        return int(answer_matches[-1])

    first_char = stripped[0]
    if first_char == "1":
        return 1
    if first_char == "2":
        return 2

    match = re.search(r"(?<!\d)([12])(?!\d)", stripped)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_device(model: Any, device_override: str | torch.device | None = None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _resolve_bos_token_id(tokenizer: Any) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise RuntimeError(
        "Tokenizer must define bos_token_id, eos_token_id, or pad_token_id for prompted EWoK evaluation."
    )


def _per_token_log_likelihood(
    model: Any,
    tokenizer: Any,
    input_texts: Sequence[str],
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _resolve_device(model, device_override=device)
    inputs = tokenizer(
        list(input_texts),
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    bos_token_id = _resolve_bos_token_id(tokenizer)
    bos_tensor = torch.full(
        (int(input_ids.shape[0]), 1),
        bos_token_id,
        device=device,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    bos_attention = torch.ones(
        (int(attention_mask.shape[0]), 1),
        device=device,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([bos_attention, attention_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

    logits = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    shifted_input_ids = input_ids[:, 1:]
    token_logprobs = log_probs.gather(
        dim=-1,
        index=shifted_input_ids.unsqueeze(-1),
    ).squeeze(-1)
    return token_logprobs, inputs["attention_mask"]


def _conditional_target_token_logps(
    model: Any,
    tokenizer: Any,
    *,
    prefixes: Sequence[str],
    targets: Sequence[str],
    answer_separator: str,
    batch_size: int,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    device = _resolve_device(model, device_override=device)
    if len(prefixes) != len(targets):
        raise ValueError("prefixes and targets must have the same length.")

    all_results: list[torch.Tensor] = []
    for start in range(0, len(prefixes), int(batch_size)):
        batch_prefixes = list(prefixes[start : start + int(batch_size)])
        batch_targets = list(targets[start : start + int(batch_size)])

        prefix_id_rows = [
            tokenizer.encode(str(prefix), add_special_tokens=False)
            for prefix in batch_prefixes
        ]
        target_id_rows = [
            tokenizer.encode(f"{answer_separator}{target}", add_special_tokens=False)
            for target in batch_targets
        ]
        if any(len(ids) == 0 for ids in target_id_rows):
            raise ValueError("At least one scored target encoded to zero tokens.")

        bos_token_id = _resolve_bos_token_id(tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = bos_token_id

        sequence_rows = [
            [bos_token_id] + list(prefix_ids) + list(target_ids)
            for prefix_ids, target_ids in zip(prefix_id_rows, target_id_rows)
        ]
        target_spans = [
            (len(prefix_ids), len(prefix_ids) + len(target_ids))
            for prefix_ids, target_ids in zip(prefix_id_rows, target_id_rows)
        ]
        max_length = max(len(ids) for ids in sequence_rows)
        padded_rows = [
            ids + [int(pad_token_id)] * (max_length - len(ids))
            for ids in sequence_rows
        ]
        attention_rows = [
            [1] * len(ids) + [0] * (max_length - len(ids))
            for ids in sequence_rows
        ]
        input_ids = torch.tensor(padded_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        logits = logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        shifted_input_ids = input_ids[:, 1:]
        token_logprobs = log_probs.gather(
            dim=-1,
            index=shifted_input_ids.unsqueeze(-1),
        ).squeeze(-1)

        for row_idx, (target_start, target_end) in enumerate(target_spans):
            all_results.append(
                token_logprobs[row_idx, target_start:target_end].detach().cpu()
            )
    return all_results


def _reduce_token_logps(token_logps: torch.Tensor, score_reduction: str) -> float:
    resolved = str(score_reduction).strip().lower()
    if resolved not in {"sum", "mean"}:
        raise ValueError(f"score_reduction must be 'sum' or 'mean', got: {score_reduction}")
    if token_logps.numel() == 0:
        return 0.0
    if resolved == "sum":
        return float(token_logps.sum().item())
    return float(token_logps.mean().item())


def _scores_from_choice_prediction(predicted_choice: int | None) -> tuple[float, float]:
    if predicted_choice == 1:
        return 1.0, 0.0
    if predicted_choice == 2:
        return 0.0, 1.0
    return 0.0, 0.0


def _prompt_level_inference_from_scores(
    *,
    choice_1_score: float,
    choice_2_score: float,
) -> PromptChoiceInference:
    if choice_1_score > choice_2_score:
        predicted_choice = 1
    elif choice_2_score > choice_1_score:
        predicted_choice = 2
    else:
        predicted_choice = None
    return PromptChoiceInference(
        raw_response_text=None,
        predicted_choice=predicted_choice,
        response_valid=True,
        choice_1_score=float(choice_1_score),
        choice_2_score=float(choice_2_score),
    )


def _generate_choice_inference(
    model: Any,
    tokenizer: Any,
    *,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    show_progress: bool,
) -> list[PromptChoiceInference]:
    device = _resolve_device(model)
    results: list[PromptChoiceInference] = []
    prompt_iter = _maybe_tqdm(
        range(0, len(prompts), int(batch_size)),
        enabled=show_progress,
        desc="Generating prompted choices",
        total=(len(prompts) + int(batch_size) - 1) // int(batch_size),
        leave=False,
    )
    with _temporary_tokenizer_padding_side(tokenizer, "left"):
        for start in prompt_iter:
            chunk = list(prompts[start : start + int(batch_size)])
            inputs = tokenizer(
                chunk,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )

            new_token_ids = generated[:, int(input_ids.shape[1]) :]
            decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
            for raw_text in decoded:
                parsed_choice = _parse_choice_response_text(raw_text)
                choice_1_score, choice_2_score = _scores_from_choice_prediction(parsed_choice)
                results.append(
                    PromptChoiceInference(
                        raw_response_text=str(raw_text),
                        predicted_choice=parsed_choice,
                        response_valid=parsed_choice in {1, 2},
                        choice_1_score=float(choice_1_score),
                        choice_2_score=float(choice_2_score),
                    )
                )
    return results


def _build_choice_record(
    *,
    domain: str,
    row_index: int,
    score_reduction: str,
    inference_mode: str,
    prompt_template_name: str,
    prompt_template_source: str,
    prompt_c1: Any,
    prompt_c2: Any,
    inference_c1: PromptChoiceInference,
    inference_c2: PromptChoiceInference,
    swap_targets_c1: bool,
    swap_targets_c2: bool,
    margin_eps: float,
    store_prompts: bool,
) -> dict[str, Any]:
    displayed_predicted_choice_c1 = inference_c1.predicted_choice
    displayed_predicted_choice_c2 = inference_c2.predicted_choice
    predicted_choice_c1 = _canonical_choice_from_display_choice(
        displayed_predicted_choice_c1,
        swapped=bool(swap_targets_c1),
    )
    predicted_choice_c2 = _canonical_choice_from_display_choice(
        displayed_predicted_choice_c2,
        swapped=bool(swap_targets_c2),
    )

    s11, s12 = _canonical_scores_from_display_scores(
        displayed_choice_1_score=inference_c1.choice_1_score,
        displayed_choice_2_score=inference_c1.choice_2_score,
        swapped=bool(swap_targets_c1),
    )
    s21, s22 = _canonical_scores_from_display_scores(
        displayed_choice_1_score=inference_c2.choice_1_score,
        displayed_choice_2_score=inference_c2.choice_2_score,
        swapped=bool(swap_targets_c2),
    )

    m1 = float(s11 - s12)
    m2 = float(s22 - s21)
    m = float(0.5 * (m1 + m2))

    k1 = float(s11 - s21)
    k2 = float(s22 - s12)
    k = float(0.5 * (k1 + k2))

    record = {
        "domain": str(domain),
        "row_index": int(row_index),
        "score_reduction": str(score_reduction),
        "prompt_inference_mode": str(inference_mode),
        "prompt_template_name": str(prompt_template_name),
        "prompt_template_source": str(prompt_template_source),
        "target_permutation_swapped_c1": bool(swap_targets_c1),
        "target_permutation_swapped_c2": bool(swap_targets_c2),
        "display_target_1_id_c1": 2 if bool(swap_targets_c1) else 1,
        "display_target_2_id_c1": 1 if bool(swap_targets_c1) else 2,
        "display_target_1_id_c2": 2 if bool(swap_targets_c2) else 1,
        "display_target_2_id_c2": 1 if bool(swap_targets_c2) else 2,
        "gold_choice_c1": 1,
        "gold_choice_c2": 2,
        "display_gold_choice_c1": _display_gold_choice_for_target(
            canonical_target=1,
            swapped=bool(swap_targets_c1),
        ),
        "display_gold_choice_c2": _display_gold_choice_for_target(
            canonical_target=2,
            swapped=bool(swap_targets_c2),
        ),
        "predicted_choice_c1": predicted_choice_c1,
        "predicted_choice_c2": predicted_choice_c2,
        "displayed_predicted_choice_c1": displayed_predicted_choice_c1,
        "displayed_predicted_choice_c2": displayed_predicted_choice_c2,
        "response_text_c1": inference_c1.raw_response_text,
        "response_text_c2": inference_c2.raw_response_text,
        "response_valid_c1": bool(inference_c1.response_valid),
        "response_valid_c2": bool(inference_c2.response_valid),
        "choice_1_score_given_c1_prompt": s11,
        "choice_2_score_given_c1_prompt": s12,
        "choice_2_score_given_c2_prompt": s22,
        "choice_1_score_given_c2_prompt": s21,
        "margin_official_m1": m1,
        "margin_symmetric_m2": m2,
        "margin_combined": m,
        "correct_official": bool(m1 > 0.0),
        "correct_symmetric": bool(m2 > 0.0),
        "correct_combined": bool(m > 0.0),
        "near_tie_official": bool(abs(m1) < float(margin_eps)),
        "near_tie_symmetric": bool(abs(m2) < float(margin_eps)),
        "near_tie_combined": bool(abs(m) < float(margin_eps)),
        "babylm_completion_choice_margin_official_m1": m1,
        "babylm_completion_choice_margin_symmetric_m2": m2,
        "babylm_completion_choice_margin_combined": m,
        "babylm_completion_choice_correct_official": bool(m1 > 0.0),
        "babylm_completion_choice_correct_symmetric": bool(m2 > 0.0),
        "babylm_completion_choice_correct_combined": bool(m > 0.0),
        "babylm_completion_choice_near_tie_official": bool(abs(m1) < float(margin_eps)),
        "babylm_completion_choice_near_tie_symmetric": bool(abs(m2) < float(margin_eps)),
        "babylm_completion_choice_near_tie_combined": bool(abs(m) < float(margin_eps)),
        "ewok_context_sensitivity_margin_official_k1": k1,
        "ewok_context_sensitivity_margin_symmetric_k2": k2,
        "ewok_context_sensitivity_margin_combined": k,
        "ewok_context_sensitivity_correct_official": bool(k1 > 0.0),
        "ewok_context_sensitivity_correct_symmetric": bool(k2 > 0.0),
        "ewok_context_sensitivity_correct_combined": bool(k > 0.0),
        "ewok_context_sensitivity_near_tie_official": bool(abs(k1) < float(margin_eps)),
        "ewok_context_sensitivity_near_tie_symmetric": bool(abs(k2) < float(margin_eps)),
        "ewok_context_sensitivity_near_tie_combined": bool(abs(k) < float(margin_eps)),
        "ewok_paper_context_sensitivity_margin_official_k1": k1,
        "ewok_paper_context_sensitivity_margin_symmetric_k2": k2,
        "ewok_paper_context_sensitivity_margin_combined": k,
        "ewok_paper_context_sensitivity_correct_official": bool(k1 > 0.0),
        "ewok_paper_context_sensitivity_correct_symmetric": bool(k2 > 0.0),
        "ewok_paper_context_sensitivity_correct_combined": bool(k > 0.0),
        "ewok_paper_context_sensitivity_near_tie_official": bool(abs(k1) < float(margin_eps)),
        "ewok_paper_context_sensitivity_near_tie_symmetric": bool(abs(k2) < float(margin_eps)),
        "ewok_paper_context_sensitivity_near_tie_combined": bool(abs(k) < float(margin_eps)),
    }
    if inference_c1.scoring_signal is not None or inference_c2.scoring_signal is not None:
        record.update(
            {
                "statement_scoring_signal": inference_c1.scoring_signal or inference_c2.scoring_signal,
                "statement_logp_true_T1_given_C1": inference_c1.choice_1_logp_true,
                "statement_logp_false_T1_given_C1": inference_c1.choice_1_logp_false,
                "statement_true_false_margin_T1_given_C1": inference_c1.choice_1_true_false_margin,
                "statement_logp_true_T2_given_C1": inference_c1.choice_2_logp_true,
                "statement_logp_false_T2_given_C1": inference_c1.choice_2_logp_false,
                "statement_true_false_margin_T2_given_C1": inference_c1.choice_2_true_false_margin,
                "statement_logp_true_T2_given_C2": inference_c2.choice_2_logp_true,
                "statement_logp_false_T2_given_C2": inference_c2.choice_2_logp_false,
                "statement_true_false_margin_T2_given_C2": inference_c2.choice_2_true_false_margin,
                "statement_logp_true_T1_given_C2": inference_c2.choice_1_logp_true,
                "statement_logp_false_T1_given_C2": inference_c2.choice_1_logp_false,
                "statement_true_false_margin_T1_given_C2": inference_c2.choice_1_true_false_margin,
            }
        )
    if store_prompts:
        record["prompt_c1"] = prompt_c1
        record["prompt_c2"] = prompt_c2
    return record


def _build_context_choice_record(
    *,
    domain: str,
    row_index: int,
    score_reduction: str,
    inference_mode: str,
    prompt_template_name: str,
    prompt_template_source: str,
    prompt_t1: Any,
    prompt_t2: Any,
    inference_t1: PromptChoiceInference,
    inference_t2: PromptChoiceInference,
    swap_contexts_t1: bool,
    swap_contexts_t2: bool,
    margin_eps: float,
    store_prompts: bool,
) -> dict[str, Any]:
    displayed_predicted_context_t1 = inference_t1.predicted_choice
    displayed_predicted_context_t2 = inference_t2.predicted_choice
    predicted_context_t1 = _canonical_choice_from_display_choice(
        displayed_predicted_context_t1,
        swapped=bool(swap_contexts_t1),
    )
    predicted_context_t2 = _canonical_choice_from_display_choice(
        displayed_predicted_context_t2,
        swapped=bool(swap_contexts_t2),
    )

    s11, s21 = _canonical_scores_from_display_scores(
        displayed_choice_1_score=inference_t1.choice_1_score,
        displayed_choice_2_score=inference_t1.choice_2_score,
        swapped=bool(swap_contexts_t1),
    )
    s12, s22 = _canonical_scores_from_display_scores(
        displayed_choice_1_score=inference_t2.choice_1_score,
        displayed_choice_2_score=inference_t2.choice_2_score,
        swapped=bool(swap_contexts_t2),
    )

    m1 = float(s11 - s12)
    m2 = float(s22 - s21)
    m = float(0.5 * (m1 + m2))

    k1 = float(s11 - s21)
    k2 = float(s22 - s12)
    k = float(0.5 * (k1 + k2))

    record = {
        "domain": str(domain),
        "row_index": int(row_index),
        "score_reduction": str(score_reduction),
        "prompt_inference_mode": str(inference_mode),
        "prompt_template_name": str(prompt_template_name),
        "prompt_template_source": str(prompt_template_source),
        "context_choice_prompt_layout": "statement_with_two_contexts",
        "target_permutation_swapped_c1": bool(swap_contexts_t1),
        "target_permutation_swapped_c2": bool(swap_contexts_t2),
        "context_permutation_swapped_t1": bool(swap_contexts_t1),
        "context_permutation_swapped_t2": bool(swap_contexts_t2),
        "display_context_1_id_t1": 2 if bool(swap_contexts_t1) else 1,
        "display_context_2_id_t1": 1 if bool(swap_contexts_t1) else 2,
        "display_context_1_id_t2": 2 if bool(swap_contexts_t2) else 1,
        "display_context_2_id_t2": 1 if bool(swap_contexts_t2) else 2,
        "gold_choice_c1": 1,
        "gold_choice_c2": 2,
        "gold_context_t1": 1,
        "gold_context_t2": 2,
        "display_gold_choice_c1": _display_gold_choice_for_target(
            canonical_target=1,
            swapped=bool(swap_contexts_t1),
        ),
        "display_gold_choice_c2": _display_gold_choice_for_target(
            canonical_target=2,
            swapped=bool(swap_contexts_t2),
        ),
        "display_gold_context_t1": _display_gold_choice_for_target(
            canonical_target=1,
            swapped=bool(swap_contexts_t1),
        ),
        "display_gold_context_t2": _display_gold_choice_for_target(
            canonical_target=2,
            swapped=bool(swap_contexts_t2),
        ),
        "predicted_choice_c1": predicted_context_t1,
        "predicted_choice_c2": predicted_context_t2,
        "predicted_context_t1": predicted_context_t1,
        "predicted_context_t2": predicted_context_t2,
        "displayed_predicted_choice_c1": displayed_predicted_context_t1,
        "displayed_predicted_choice_c2": displayed_predicted_context_t2,
        "displayed_predicted_context_t1": displayed_predicted_context_t1,
        "displayed_predicted_context_t2": displayed_predicted_context_t2,
        "response_text_c1": inference_t1.raw_response_text,
        "response_text_c2": inference_t2.raw_response_text,
        "response_text_t1_context_choice": inference_t1.raw_response_text,
        "response_text_t2_context_choice": inference_t2.raw_response_text,
        "response_valid_c1": bool(inference_t1.response_valid),
        "response_valid_c2": bool(inference_t2.response_valid),
        "response_valid_t1_context_choice": bool(inference_t1.response_valid),
        "response_valid_t2_context_choice": bool(inference_t2.response_valid),
        "choice_1_score_given_c1_prompt": s11,
        "choice_2_score_given_c1_prompt": s12,
        "choice_2_score_given_c2_prompt": s22,
        "choice_1_score_given_c2_prompt": s21,
        "context_1_score_given_t1_prompt": s11,
        "context_2_score_given_t1_prompt": s21,
        "context_2_score_given_t2_prompt": s22,
        "context_1_score_given_t2_prompt": s12,
        "margin_official_m1": m1,
        "margin_symmetric_m2": m2,
        "margin_combined": m,
        "correct_official": bool(m1 > 0.0),
        "correct_symmetric": bool(m2 > 0.0),
        "correct_combined": bool(m > 0.0),
        "near_tie_official": bool(abs(m1) < float(margin_eps)),
        "near_tie_symmetric": bool(abs(m2) < float(margin_eps)),
        "near_tie_combined": bool(abs(m) < float(margin_eps)),
        "babylm_completion_choice_margin_official_m1": m1,
        "babylm_completion_choice_margin_symmetric_m2": m2,
        "babylm_completion_choice_margin_combined": m,
        "babylm_completion_choice_correct_official": bool(m1 > 0.0),
        "babylm_completion_choice_correct_symmetric": bool(m2 > 0.0),
        "babylm_completion_choice_correct_combined": bool(m > 0.0),
        "babylm_completion_choice_near_tie_official": bool(abs(m1) < float(margin_eps)),
        "babylm_completion_choice_near_tie_symmetric": bool(abs(m2) < float(margin_eps)),
        "babylm_completion_choice_near_tie_combined": bool(abs(m) < float(margin_eps)),
        "ewok_context_sensitivity_margin_official_k1": k1,
        "ewok_context_sensitivity_margin_symmetric_k2": k2,
        "ewok_context_sensitivity_margin_combined": k,
        "ewok_context_sensitivity_correct_official": bool(k1 > 0.0),
        "ewok_context_sensitivity_correct_symmetric": bool(k2 > 0.0),
        "ewok_context_sensitivity_correct_combined": bool(k > 0.0),
        "ewok_context_sensitivity_near_tie_official": bool(abs(k1) < float(margin_eps)),
        "ewok_context_sensitivity_near_tie_symmetric": bool(abs(k2) < float(margin_eps)),
        "ewok_context_sensitivity_near_tie_combined": bool(abs(k) < float(margin_eps)),
        "ewok_paper_context_sensitivity_margin_official_k1": k1,
        "ewok_paper_context_sensitivity_margin_symmetric_k2": k2,
        "ewok_paper_context_sensitivity_margin_combined": k,
        "ewok_paper_context_sensitivity_correct_official": bool(k1 > 0.0),
        "ewok_paper_context_sensitivity_correct_symmetric": bool(k2 > 0.0),
        "ewok_paper_context_sensitivity_correct_combined": bool(k > 0.0),
        "ewok_paper_context_sensitivity_near_tie_official": bool(abs(k1) < float(margin_eps)),
        "ewok_paper_context_sensitivity_near_tie_symmetric": bool(abs(k2) < float(margin_eps)),
        "ewok_paper_context_sensitivity_near_tie_combined": bool(abs(k) < float(margin_eps)),
    }
    if store_prompts:
        record["prompt_c1"] = prompt_t1
        record["prompt_c2"] = prompt_t2
        record["prompt_t1_context_choice"] = prompt_t1
        record["prompt_t2_context_choice"] = prompt_t2
    return record


def _build_domain_prompts(
    *,
    df: Any,
    template_text: str,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[list[str], list[str], list[bool], list[bool]]:
    prompts_c1: list[str] = []
    prompts_c2: list[str] = []
    swaps_c1: list[bool] = []
    swaps_c2: list[bool] = []
    for _, row in df.iterrows():
        row_index = int(row["index"]) if "index" in row else int(len(prompts_c1))
        swap_targets = _should_swap_targets_for_row(
            row_index=row_index,
            target_permutation_mode=target_permutation_mode,
            target_permutation_seed=target_permutation_seed,
        )
        target_1_text = row["Target2"] if swap_targets else row["Target1"]
        target_2_text = row["Target1"] if swap_targets else row["Target2"]
        prompts_c1.append(
            _render_choice_prompt(
                template_text=template_text,
                context_text=row["Context1"],
                target_1_text=target_1_text,
                target_2_text=target_2_text,
            )
        )
        prompts_c2.append(
            _render_choice_prompt(
                template_text=template_text,
                context_text=row["Context2"],
                target_1_text=target_1_text,
                target_2_text=target_2_text,
            )
        )
        swaps_c1.append(bool(swap_targets))
        swaps_c2.append(bool(swap_targets))
    return prompts_c1, prompts_c2, swaps_c1, swaps_c2


def _score_domain_rows_choice_answer_logprob(
    model: Any,
    tokenizer: Any,
    *,
    df: Any,
    template_text: str,
    batch_size: int,
    score_reduction: str,
    answer_separator: str,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[list[PromptChoiceInference], list[PromptChoiceInference], list[str], list[str], list[bool], list[bool]]:
    prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _build_domain_prompts(
        df=df,
        template_text=template_text,
        target_permutation_mode=target_permutation_mode,
        target_permutation_seed=target_permutation_seed,
    )

    answer_one = ["1"] * len(prompts_c1)
    answer_two = ["2"] * len(prompts_c1)
    r11 = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts_c1,
        targets=answer_one,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )
    r12 = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts_c1,
        targets=answer_two,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )
    r22 = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts_c2,
        targets=answer_two,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )
    r21 = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts_c2,
        targets=answer_one,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )

    inference_c1 = [
        _prompt_level_inference_from_scores(
            choice_1_score=_reduce_token_logps(logps_1, score_reduction),
            choice_2_score=_reduce_token_logps(logps_2, score_reduction),
        )
        for logps_1, logps_2 in zip(r11, r12)
    ]
    inference_c2 = [
        PromptChoiceInference(
            raw_response_text=None,
            predicted_choice=(
                2
                if _reduce_token_logps(logps_2, score_reduction) > _reduce_token_logps(logps_1, score_reduction)
                else 1
                if _reduce_token_logps(logps_1, score_reduction) > _reduce_token_logps(logps_2, score_reduction)
                else None
            ),
            response_valid=True,
            choice_1_score=_reduce_token_logps(logps_1, score_reduction),
            choice_2_score=_reduce_token_logps(logps_2, score_reduction),
        )
        for logps_1, logps_2 in zip(r21, r22)
    ]
    return inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2


def _build_domain_statement_prompts(
    *,
    df: Any,
    template_text: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompts_c1: list[dict[str, str]] = []
    prompts_c2: list[dict[str, str]] = []
    for _, row in df.iterrows():
        prompts_c1.append(
            {
                "choice_1_prompt": _render_statement_prompt(
                    template_text=template_text,
                    context_text=row["Context1"],
                    statement_text=row["Target1"],
                ),
                "choice_2_prompt": _render_statement_prompt(
                    template_text=template_text,
                    context_text=row["Context1"],
                    statement_text=row["Target2"],
                ),
            }
        )
        prompts_c2.append(
            {
                "choice_1_prompt": _render_statement_prompt(
                    template_text=template_text,
                    context_text=row["Context2"],
                    statement_text=row["Target1"],
                ),
                "choice_2_prompt": _render_statement_prompt(
                    template_text=template_text,
                    context_text=row["Context2"],
                    statement_text=row["Target2"],
                ),
            }
        )
    return prompts_c1, prompts_c2


def _statement_logprob_values(
    model: Any,
    tokenizer: Any,
    *,
    prompts: Sequence[str],
    batch_size: int,
    score_reduction: str,
    answer_separator: str,
) -> list[dict[str, float]]:
    true_targets = [DEFAULT_TRUE_LABEL] * len(prompts)
    false_targets = [DEFAULT_FALSE_LABEL] * len(prompts)
    true_logps = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts,
        targets=true_targets,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )
    false_logps = _conditional_target_token_logps(
        model,
        tokenizer,
        prefixes=prompts,
        targets=false_targets,
        answer_separator=answer_separator,
        batch_size=batch_size,
    )
    values: list[dict[str, float]] = []
    for true_tokens, false_tokens in zip(true_logps, false_logps):
        logp_true = _reduce_token_logps(true_tokens, score_reduction)
        logp_false = _reduce_token_logps(false_tokens, score_reduction)
        values.append(
            {
                "logp_true": float(logp_true),
                "logp_false": float(logp_false),
                "true_false_margin": float(logp_true - logp_false),
            }
        )
    return values


def _statement_score_from_values(*, values: dict[str, float], inference_mode: str) -> float:
    if inference_mode == "statement_true_logprob":
        return float(values["logp_true"])
    if inference_mode == "statement_true_false_margin":
        return float(values["true_false_margin"])
    raise ValueError(f"Unsupported statement inference mode: {inference_mode}")


def _statement_scoring_signal(inference_mode: str) -> str:
    if inference_mode == "statement_true_logprob":
        return "logp_true"
    if inference_mode == "statement_true_false_margin":
        return "true_false_margin"
    raise ValueError(f"Unsupported statement inference mode: {inference_mode}")


def _prompt_level_statement_inference(
    *,
    choice_1_values: dict[str, float],
    choice_2_values: dict[str, float],
    inference_mode: str,
) -> PromptChoiceInference:
    choice_1_score = _statement_score_from_values(
        values=choice_1_values,
        inference_mode=inference_mode,
    )
    choice_2_score = _statement_score_from_values(
        values=choice_2_values,
        inference_mode=inference_mode,
    )
    if choice_1_score > choice_2_score:
        predicted_choice = 1
    elif choice_2_score > choice_1_score:
        predicted_choice = 2
    else:
        predicted_choice = None
    return PromptChoiceInference(
        raw_response_text=None,
        predicted_choice=predicted_choice,
        response_valid=True,
        choice_1_score=float(choice_1_score),
        choice_2_score=float(choice_2_score),
        scoring_signal=_statement_scoring_signal(inference_mode),
        choice_1_logp_true=float(choice_1_values["logp_true"]),
        choice_1_logp_false=float(choice_1_values["logp_false"]),
        choice_1_true_false_margin=float(choice_1_values["true_false_margin"]),
        choice_2_logp_true=float(choice_2_values["logp_true"]),
        choice_2_logp_false=float(choice_2_values["logp_false"]),
        choice_2_true_false_margin=float(choice_2_values["true_false_margin"]),
    )


def _score_domain_rows_statement_logprob(
    model: Any,
    tokenizer: Any,
    *,
    df: Any,
    template_text: str,
    inference_mode: str,
    batch_size: int,
    score_reduction: str,
    answer_separator: str,
) -> tuple[list[PromptChoiceInference], list[PromptChoiceInference], list[dict[str, str]], list[dict[str, str]], list[bool], list[bool]]:
    prompts_c1, prompts_c2 = _build_domain_statement_prompts(
        df=df,
        template_text=template_text,
    )
    num_rows = len(prompts_c1)
    flat_prompts = (
        [item["choice_1_prompt"] for item in prompts_c1]
        + [item["choice_2_prompt"] for item in prompts_c1]
        + [item["choice_1_prompt"] for item in prompts_c2]
        + [item["choice_2_prompt"] for item in prompts_c2]
    )
    flat_values = _statement_logprob_values(
        model,
        tokenizer,
        prompts=flat_prompts,
        batch_size=batch_size,
        score_reduction=score_reduction,
        answer_separator=answer_separator,
    )
    c1_choice_1 = flat_values[0:num_rows]
    c1_choice_2 = flat_values[num_rows : 2 * num_rows]
    c2_choice_1 = flat_values[2 * num_rows : 3 * num_rows]
    c2_choice_2 = flat_values[3 * num_rows : 4 * num_rows]

    inference_c1 = [
        _prompt_level_statement_inference(
            choice_1_values=choice_1_values,
            choice_2_values=choice_2_values,
            inference_mode=inference_mode,
        )
        for choice_1_values, choice_2_values in zip(c1_choice_1, c1_choice_2)
    ]
    inference_c2 = [
        _prompt_level_statement_inference(
            choice_1_values=choice_1_values,
            choice_2_values=choice_2_values,
            inference_mode=inference_mode,
        )
        for choice_1_values, choice_2_values in zip(c2_choice_1, c2_choice_2)
    ]
    swaps_c1 = [False] * len(prompts_c1)
    swaps_c2 = [False] * len(prompts_c2)
    return inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2


def _score_domain_rows_choice_generate(
    model: Any,
    tokenizer: Any,
    *,
    df: Any,
    template_text: str,
    batch_size: int,
    max_new_tokens: int,
    show_progress: bool,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[list[PromptChoiceInference], list[PromptChoiceInference], list[str], list[str], list[bool], list[bool]]:
    prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _build_domain_prompts(
        df=df,
        template_text=template_text,
        target_permutation_mode=target_permutation_mode,
        target_permutation_seed=target_permutation_seed,
    )
    inference_c1 = _generate_choice_inference(
        model,
        tokenizer,
        prompts=prompts_c1,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
    )
    inference_c2 = _generate_choice_inference(
        model,
        tokenizer,
        prompts=prompts_c2,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
    )
    return inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2


def _build_domain_context_choice_prompts(
    *,
    df: Any,
    template_text: str,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[list[str], list[str], list[bool], list[bool]]:
    prompts_t1: list[str] = []
    prompts_t2: list[str] = []
    swaps_t1: list[bool] = []
    swaps_t2: list[bool] = []
    for _, row in df.iterrows():
        row_index = int(row["index"]) if "index" in row else int(len(prompts_t1))
        swap_contexts = _should_swap_targets_for_row(
            row_index=row_index,
            target_permutation_mode=target_permutation_mode,
            target_permutation_seed=target_permutation_seed,
        )
        context_1_text = row["Context2"] if swap_contexts else row["Context1"]
        context_2_text = row["Context1"] if swap_contexts else row["Context2"]
        prompts_t1.append(
            _render_context_choice_prompt(
                template_text=template_text,
                statement_text=row["Target1"],
                context_1_text=context_1_text,
                context_2_text=context_2_text,
            )
        )
        prompts_t2.append(
            _render_context_choice_prompt(
                template_text=template_text,
                statement_text=row["Target2"],
                context_1_text=context_1_text,
                context_2_text=context_2_text,
            )
        )
        swaps_t1.append(bool(swap_contexts))
        swaps_t2.append(bool(swap_contexts))
    return prompts_t1, prompts_t2, swaps_t1, swaps_t2


def _score_domain_rows_context_choice_generate(
    model: Any,
    tokenizer: Any,
    *,
    df: Any,
    template_text: str,
    batch_size: int,
    max_new_tokens: int,
    show_progress: bool,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[list[PromptChoiceInference], list[PromptChoiceInference], list[str], list[str], list[bool], list[bool]]:
    prompts_t1, prompts_t2, swaps_t1, swaps_t2 = _build_domain_context_choice_prompts(
        df=df,
        template_text=template_text,
        target_permutation_mode=target_permutation_mode,
        target_permutation_seed=target_permutation_seed,
    )
    inference_t1 = _generate_choice_inference(
        model,
        tokenizer,
        prompts=prompts_t1,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
    )
    inference_t2 = _generate_choice_inference(
        model,
        tokenizer,
        prompts=prompts_t2,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
    )
    return inference_t1, inference_t2, prompts_t1, prompts_t2, swaps_t1, swaps_t2


def prompted_ewok_score_records_all_methods(
    *,
    model: Any,
    tokenizer: Any,
    ewok_df: Any,
    template_text: str,
    prompt_template_name: str,
    prompt_template_source: str,
    inference_mode: str,
    batch_size: int,
    score_reduction: str,
    margin_eps: float,
    answer_separator: str,
    max_new_tokens: int,
    target_permutation_mode: str,
    target_permutation_seed: int,
    store_prompts: bool,
    show_progress: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    domains = list(ewok_df["Domain"].unique())
    domain_iter = _maybe_tqdm(
        domains,
        enabled=show_progress,
        desc=f"Prompted EWoK ({inference_mode})",
        total=len(domains),
        leave=False,
    )
    for domain in domain_iter:
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()
        if inference_mode == "choice_answer_logprob":
            inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _score_domain_rows_choice_answer_logprob(
                model,
                tokenizer,
                df=df,
                template_text=template_text,
                batch_size=batch_size,
                score_reduction=score_reduction,
                answer_separator=answer_separator,
                target_permutation_mode=target_permutation_mode,
                target_permutation_seed=target_permutation_seed,
            )
        elif inference_mode == "choice_generate":
            inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _score_domain_rows_choice_generate(
                model,
                tokenizer,
                df=df,
                template_text=template_text,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                show_progress=show_progress,
                target_permutation_mode=target_permutation_mode,
                target_permutation_seed=target_permutation_seed,
            )
        elif inference_mode == "context_choice_generate":
            inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _score_domain_rows_context_choice_generate(
                model,
                tokenizer,
                df=df,
                template_text=template_text,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                show_progress=show_progress,
                target_permutation_mode=target_permutation_mode,
                target_permutation_seed=target_permutation_seed,
            )
        elif inference_mode in {"statement_true_logprob", "statement_true_false_margin"}:
            inference_c1, inference_c2, prompts_c1, prompts_c2, swaps_c1, swaps_c2 = _score_domain_rows_statement_logprob(
                model,
                tokenizer,
                df=df,
                template_text=template_text,
                inference_mode=inference_mode,
                batch_size=batch_size,
                score_reduction=score_reduction,
                answer_separator=answer_separator,
            )
        else:
            raise ValueError(f"Unsupported inference mode: {inference_mode}")

        for idx in range(len(df)):
            if inference_mode == "context_choice_generate":
                records.append(
                    _build_context_choice_record(
                        domain=str(domain),
                        row_index=int(df.loc[idx, "index"]),
                        score_reduction=score_reduction,
                        inference_mode=inference_mode,
                        prompt_template_name=prompt_template_name,
                        prompt_template_source=prompt_template_source,
                        prompt_t1=prompts_c1[idx],
                        prompt_t2=prompts_c2[idx],
                        inference_t1=inference_c1[idx],
                        inference_t2=inference_c2[idx],
                        swap_contexts_t1=swaps_c1[idx],
                        swap_contexts_t2=swaps_c2[idx],
                        margin_eps=margin_eps,
                        store_prompts=store_prompts,
                    )
                )
            else:
                records.append(
                    _build_choice_record(
                        domain=str(domain),
                        row_index=int(df.loc[idx, "index"]),
                        score_reduction=score_reduction,
                        inference_mode=inference_mode,
                        prompt_template_name=prompt_template_name,
                        prompt_template_source=prompt_template_source,
                        prompt_c1=prompts_c1[idx],
                        prompt_c2=prompts_c2[idx],
                        inference_c1=inference_c1[idx],
                        inference_c2=inference_c2[idx],
                        swap_targets_c1=swaps_c1[idx],
                        swap_targets_c2=swaps_c2[idx],
                        margin_eps=margin_eps,
                        store_prompts=store_prompts,
                    )
                )
    return records


def _summarize_records_with_shared_logic(
    *,
    shared_ewok_module: Any,
    ewok_df: Any,
    records: list[dict[str, Any]],
    margin_eps: float,
) -> dict[str, Any]:
    summarize_fn = getattr(shared_ewok_module, "_summarize_records_all_methods", None)
    if summarize_fn is None:
        raise RuntimeError(
            "moonshotGPT.evaluation.ewok is missing _summarize_records_all_methods; "
            "cannot reuse the shared summary logic."
        )
    with _temporary_module_attr(shared_ewok_module, "ewok_df", ewok_df):
        return summarize_fn(records, float(margin_eps))


def _ewok_official_average(metric_block: Optional[dict[str, Any]]) -> float | None:
    if not isinstance(metric_block, dict):
        return None
    domain_scores_full = metric_block.get("domain_scores_full")
    if not isinstance(domain_scores_full, dict):
        return None
    average = domain_scores_full.get("average")
    if isinstance(average, (list, tuple)) and average:
        return float(average[0])
    return None


def _build_ewok_payload(
    *,
    shared_ewok_module: Any,
    ewok_df: Any,
    metrics_by_method_mean: dict[str, Any],
    batch_size: int,
    elapsed_seconds: float,
    prompt_template_name: str,
    prompt_template_source: str,
    inference_mode: str,
    score_reduction: str,
    answer_separator: str,
    max_new_tokens: int,
    target_permutation_mode: str,
    target_permutation_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    babylm_mean = metrics_by_method_mean.get(shared_ewok_module.BABYLM_COMPLETION_CHOICE)
    context_mean = metrics_by_method_mean.get(shared_ewok_module.EWOK_CONTEXT_SENSITIVITY)
    babylm_full_mean = babylm_mean.get("domain_scores_full") if isinstance(babylm_mean, dict) else None
    context_full_mean = context_mean.get("domain_scores_full") if isinstance(context_mean, dict) else None

    summary = {
        "babylm_completion_choice_full_mean": babylm_full_mean,
        "ewok_context_sensitivity_full_mean": context_full_mean,
        "babylm_completion_choice_full_mean_clean_average": rq._ewok_full_mean_clean_average(
            babylm_full_mean
        ),
        "ewok_context_sensitivity_full_mean_clean_average": rq._ewok_full_mean_clean_average(
            context_full_mean
        ),
        "babylm_completion_choice_official_mean_average": _ewok_official_average(babylm_mean),
        "ewok_context_sensitivity_official_mean_average": _ewok_official_average(context_mean),
        "num_items": int(len(ewok_df)),
        "elapsed_seconds": float(elapsed_seconds),
    }

    payload = {
        "ewok_source": str(shared_ewok_module.SRC),
        "batch_size": int(batch_size),
        "elapsed_seconds": float(elapsed_seconds),
        "prompting": {
            "prompt_template_name": str(prompt_template_name),
            "prompt_template_source": str(prompt_template_source),
            "inference_mode": str(inference_mode),
            "score_reduction": str(score_reduction),
            "answer_separator": str(answer_separator),
            "true_label": DEFAULT_TRUE_LABEL,
            "false_label": DEFAULT_FALSE_LABEL,
            "max_new_tokens": int(max_new_tokens),
            "target_permutation_mode": str(target_permutation_mode),
            "target_permutation_seed": int(target_permutation_seed),
        },
        "mean": {
            "metrics_by_method": metrics_by_method_mean,
            "num_items": int(len(ewok_df)),
        },
        "summary": summary,
    }
    return payload, summary


def _evaluate_prompted_with_oom_retries(
    *,
    eval_fn: Callable[[int], list[dict[str, Any]]],
    start_batch_size: int,
    status_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    batch_size = max(1, int(start_batch_size))
    attempts: list[dict[str, Any]] = []
    while True:
        started = time.time()
        try:
            if status_callback is not None:
                status_callback(f"batch_size={int(batch_size)}")
            records = eval_fn(batch_size)
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "status": "completed",
                    "elapsed_seconds": float(time.time() - started),
                }
            )
            return records, batch_size, attempts
        except RuntimeError as exc:
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "status": "failed",
                    "elapsed_seconds": float(time.time() - started),
                    "error": str(exc),
                }
            )
            if batch_size <= 1 or not rq._is_oom_error(exc):
                raise
            rq._release_memory()
            batch_size = max(1, batch_size // 2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False))
            handle.write("\n")


def _output_paths_for_model(
    *,
    output_root: Path,
    spec: Any,
    prompt_template_name: str,
    inference_mode: str,
    target_permutation_mode: str,
) -> dict[str, Path]:
    run_slug_parts = [str(spec.model_slug), str(prompt_template_name), str(inference_mode)]
    if str(target_permutation_mode).strip().lower() != DEFAULT_TARGET_PERMUTATION_MODE:
        run_slug_parts.append(f"perm_{str(target_permutation_mode).strip().lower()}")
    run_slug = _sanitize_name("__".join(run_slug_parts))
    output_dir = output_root / run_slug
    return {
        "run_slug": Path(run_slug),
        "output_dir": output_dir,
        "manifest_path": output_dir / "manifest.json",
        "metrics_path": output_dir / "ewok_metrics.json",
        "items_path": output_dir / "ewok_items.jsonl",
        "summary_path": output_dir / "summary.json",
        "run_log_path": output_dir / rq.DEFAULT_MODEL_LOG_NAME,
    }


def _run_single_model(
    *,
    spec: Any,
    output_root: Path,
    downloads_root: Path,
    dtype_name: str,
    disable_xet: bool,
    hf_token: str | None,
    download_retries: int,
    shared_ewok_module: Any,
    ewok_df: Any,
    variant: str,
    domains: Sequence[str],
    limit: int | None,
    prompt_template_name: str,
    prompt_template_source: str,
    template_text: str,
    inference_mode: str,
    score_reduction: str,
    margin_eps: float,
    answer_separator: str,
    max_new_tokens: int,
    target_permutation_mode: str,
    target_permutation_seed: int,
    overwrite: bool,
    show_progress: bool,
    store_prompts: bool,
    emit_summary: bool = True,
) -> dict[str, Any]:
    paths = _output_paths_for_model(
        output_root=output_root,
        spec=spec,
        prompt_template_name=prompt_template_name,
        inference_mode=inference_mode,
        target_permutation_mode=target_permutation_mode,
    )
    output_dir = paths["output_dir"]
    manifest_path = paths["manifest_path"]
    metrics_path = paths["metrics_path"]
    items_path = paths["items_path"]
    summary_path = paths["summary_path"]
    run_log_path = paths["run_log_path"]

    started_at = _utc_now_iso()
    started = time.time()
    manifest = None
    progress_callback = rq._compose_progress_callbacks(
        rq._make_phase_logger(run_log_path, model_slug=str(paths["run_slug"])),
    )

    hardware_info = rq.detect_hardware()
    dtype = rq._auto_dtype(dtype_name, hardware_info)
    bnb_available = rq._bitsandbytes_available()
    load_attempts = rq.build_load_attempts(
        spec,
        hardware_info,
        dtype,
        bitsandbytes_available=bnb_available,
    )

    cleanup_path: Path | None = downloads_root / spec.model_slug
    model = None
    tokenizer = None
    status = "failed"
    top_error: str | None = None
    top_traceback: str | None = None
    ewok_summary: dict[str, Any] | None = None
    load_strategy: str | None = None
    quantization: str | None = None
    num_gpus_used: int | None = None
    batch_size_final: int | None = None

    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise RuntimeError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model": {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "param_count_b": float(spec.param_count_b),
                "trust_remote_code": bool(spec.trust_remote_code),
                "tokenizer_id": spec.tokenizer_id,
                "model_slug": spec.model_slug,
            },
            "started_at": started_at,
            "status": "running",
            "paths": {
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "ewok_metrics_path": str(metrics_path),
                "ewok_items_path": str(items_path),
                "run_log_path": str(run_log_path),
                "downloads_root": str(downloads_root),
            },
            "hardware": hardware_info,
            "dtype": str(dtype),
            "bitsandbytes_available": bool(bnb_available),
            "settings": {
                "disable_xet": bool(disable_xet),
                "download_retries": int(download_retries),
                "variant": variant,
                "domains": list(domains),
                "limit": limit,
                "prompt_template_name": str(prompt_template_name),
                "prompt_template_source": str(prompt_template_source),
                "inference_mode": str(inference_mode),
                "score_reduction": str(score_reduction),
                "margin_eps": float(margin_eps),
                "answer_separator": str(answer_separator),
                "max_new_tokens": int(max_new_tokens),
                "target_permutation_mode": str(target_permutation_mode),
                "target_permutation_seed": int(target_permutation_seed),
                "store_prompts": bool(store_prompts),
            },
            "download": {},
            "load_attempts": [],
            "evaluation": {},
            "cleanup": {
                "attempted": False,
                "removed_download_dir": False,
                "error": None,
            },
        }
        _write_json(manifest_path, manifest)

        rq._notify_progress(progress_callback, "start", spec.model_id)
        rq._notify_progress(progress_callback, "download", spec.model_id)
        download_started = time.time()
        asset_paths = rq._download_assets(
            spec,
            download_root=downloads_root,
            hf_token=hf_token,
            disable_xet=disable_xet,
            max_retries=download_retries,
        )
        cleanup_path = Path(asset_paths["download_dir"])
        manifest["download"] = {
            "status": "completed",
            "elapsed_seconds": float(time.time() - download_started),
            "disable_xet": bool(disable_xet),
            "download_dir": str(asset_paths["download_dir"]),
            "model_dir": str(asset_paths["model_dir"]),
            "tokenizer_dir": str(asset_paths["tokenizer_dir"]),
            "tokenizer_source": asset_paths["tokenizer_source"],
            "tokenizer_mode": asset_paths["tokenizer_mode"],
            "weight_family": asset_paths.get("weight_family"),
            "ignore_patterns": asset_paths.get("ignore_patterns"),
        }
        _write_json(manifest_path, manifest)
        rq._notify_progress(
            progress_callback,
            "downloaded",
            f"{float(time.time() - download_started):.1f}s",
        )

        if not load_attempts:
            raise RuntimeError("No viable load strategy was available for the detected hardware.")

        for attempt_idx, attempt in enumerate(load_attempts, start=1):
            attempt_record: dict[str, Any] = {
                "load_strategy": attempt["load_strategy"],
                "quantization": attempt.get("quantization"),
                "device_mode": attempt["device_mode"],
                "start_batch_size": int(attempt["start_batch_size"]),
                "started_at": _utc_now_iso(),
            }
            attempt_started = time.time()
            try:
                rq._notify_progress(
                    progress_callback,
                    "load",
                    f"attempt {attempt_idx}/{len(load_attempts)} {attempt['load_strategy']}",
                )
                model, tokenizer = rq._load_model_and_tokenizer_for_attempt(
                    spec,
                    asset_paths=asset_paths,
                    attempt=attempt,
                    dtype=dtype,
                    hardware=hardware_info,
                )
                attempt_record["status"] = "model_loaded"
                attempt_record["load_elapsed_seconds"] = float(time.time() - attempt_started)
                attempt_record["model_loader"] = getattr(model, "_hf_ewok_loader_name", None)
                attempt_record["attn_implementation"] = getattr(
                    model,
                    "_hf_ewok_attn_implementation",
                    None,
                )
                load_strategy = str(attempt["load_strategy"])
                quantization = attempt.get("quantization")
                num_gpus_used = rq._count_model_gpus(model)

                eval_started = time.time()
                rq._notify_progress(
                    progress_callback,
                    "evaluate",
                    f"{load_strategy} batch_size={int(attempt['start_batch_size'])}",
                )

                records, batch_size_final, eval_attempts = _evaluate_prompted_with_oom_retries(
                    eval_fn=lambda batch_size: prompted_ewok_score_records_all_methods(
                        model=model,
                        tokenizer=tokenizer,
                        ewok_df=ewok_df,
                        template_text=template_text,
                        prompt_template_name=prompt_template_name,
                        prompt_template_source=prompt_template_source,
                        inference_mode=inference_mode,
                        batch_size=batch_size,
                        score_reduction=score_reduction,
                        margin_eps=margin_eps,
                        answer_separator=answer_separator,
                        max_new_tokens=max_new_tokens,
                        target_permutation_mode=target_permutation_mode,
                        target_permutation_seed=target_permutation_seed,
                        store_prompts=store_prompts,
                        show_progress=show_progress,
                    ),
                    start_batch_size=int(attempt["start_batch_size"]),
                    status_callback=lambda detail: rq._notify_progress(
                        progress_callback,
                        "evaluate",
                        f"{load_strategy} {detail}",
                    ),
                )
                metrics_by_method = _summarize_records_with_shared_logic(
                    shared_ewok_module=shared_ewok_module,
                    ewok_df=ewok_df,
                    records=records,
                    margin_eps=margin_eps,
                )
                ewok_payload, ewok_summary = _build_ewok_payload(
                    shared_ewok_module=shared_ewok_module,
                    ewok_df=ewok_df,
                    metrics_by_method_mean=metrics_by_method,
                    batch_size=batch_size_final,
                    elapsed_seconds=float(time.time() - eval_started),
                    prompt_template_name=prompt_template_name,
                    prompt_template_source=prompt_template_source,
                    inference_mode=inference_mode,
                    score_reduction=score_reduction,
                    answer_separator=answer_separator,
                    max_new_tokens=max_new_tokens,
                    target_permutation_mode=target_permutation_mode,
                    target_permutation_seed=target_permutation_seed,
                )
                _write_json(metrics_path, ewok_payload)
                _write_jsonl(items_path, records)

                attempt_record["status"] = "completed"
                attempt_record["num_gpus_used"] = num_gpus_used
                attempt_record["evaluation_attempts"] = eval_attempts
                attempt_record["evaluation_elapsed_seconds"] = float(time.time() - eval_started)
                manifest["load_attempts"].append(attempt_record)
                status = "completed"
                top_error = None
                top_traceback = None
                rq._notify_progress(
                    progress_callback,
                    "completed",
                    f"{load_strategy} batch_size={int(batch_size_final)}",
                )
                break
            except Exception as exc:
                attempt_record["status"] = "failed"
                attempt_record["error"] = str(exc)
                attempt_record["traceback"] = traceback.format_exc()
                attempt_record["elapsed_seconds"] = float(time.time() - attempt_started)
                manifest["load_attempts"].append(attempt_record)
                top_error = str(exc)
                top_traceback = attempt_record["traceback"]
                rq._release_memory()
                model = None
                tokenizer = None
                rq._notify_progress(
                    progress_callback,
                    "retry" if rq._is_capacity_error(exc) else "failed_load",
                    f"{attempt['load_strategy']}: {str(exc)}",
                )
                if not rq._is_capacity_error(exc):
                    status = "failed_load"
                    break
        else:
            status = "failed_capacity"

        if status != "completed" and top_error is None:
            top_error = "No load attempt completed successfully."

    except Exception as exc:
        status = "failed"
        top_error = str(exc)
        top_traceback = traceback.format_exc()
    finally:
        rq._notify_progress(progress_callback, "cleanup", None)
        model = None
        tokenizer = None
        rq._release_memory()

        cleanup_error = None
        removed = False
        if cleanup_path is not None:
            try:
                rq._remove_tree(cleanup_path)
                removed = not cleanup_path.exists()
            except Exception as exc:
                cleanup_error = str(exc)

        elapsed_seconds = float(time.time() - started)
        summary = {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "param_count_b": float(spec.param_count_b),
            "status": status,
            "load_strategy": load_strategy,
            "num_gpus_used": num_gpus_used,
            "quantization": quantization,
            "batch_size_final": batch_size_final,
            "elapsed_seconds": elapsed_seconds,
            "error": top_error,
            "provider": "local_hf",
            "eval_method": "prompted_choice",
            "prompt_template_name": prompt_template_name,
            "prompt_template_source": prompt_template_source,
            "inference_mode": inference_mode,
            "target_permutation_mode": str(target_permutation_mode),
            "target_permutation_seed": int(target_permutation_seed),
            "ewok_variant": variant,
            "ewok_source": str(shared_ewok_module.SRC),
            "babylm_completion_choice_full_mean": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("babylm_completion_choice_full_mean")
            ),
            "ewok_context_sensitivity_full_mean": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("ewok_context_sensitivity_full_mean")
            ),
            "ewok_babylm_completion_full_mean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else rq._ewok_full_mean_average(ewok_summary.get("babylm_completion_choice_full_mean"))
            ),
            "ewok_context_sensitivity_full_mean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else rq._ewok_full_mean_average(ewok_summary.get("ewok_context_sensitivity_full_mean"))
            ),
            "ewok_babylm_completion_full_mean_clean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("babylm_completion_choice_full_mean_clean_average")
            ),
            "ewok_context_sensitivity_full_mean_clean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("ewok_context_sensitivity_full_mean_clean_average")
            ),
            "ewok_babylm_completion_official_mean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("babylm_completion_choice_official_mean_average")
            ),
            "ewok_context_sensitivity_official_mean_avg": (
                None
                if not isinstance(ewok_summary, dict)
                else ewok_summary.get("ewok_context_sensitivity_official_mean_average")
            ),
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
        }
        _write_json(summary_path, summary)

        if manifest is not None:
            manifest["status"] = status
            manifest["finished_at"] = summary["finished_at"]
            manifest["error"] = top_error
            if top_traceback is not None:
                manifest["traceback"] = top_traceback
            manifest["evaluation"] = {
                "batch_size_final": batch_size_final,
                "load_strategy": load_strategy,
                "num_gpus_used": num_gpus_used,
                "quantization": quantization,
                "summary": ewok_summary,
            }
            manifest["cleanup"] = {
                "attempted": cleanup_path is not None,
                "removed_download_dir": bool(removed),
                "error": cleanup_error,
            }
            _write_json(manifest_path, manifest)

        if emit_summary:
            stream = sys.stdout if str(status) == "completed" else sys.stderr
            print(json.dumps(summary, indent=2, sort_keys=False), file=stream)

    return summary


def _queue_status_from_summaries(summaries: Sequence[dict[str, Any]]) -> str:
    if not summaries:
        return "completed"
    if all(str(item.get("status")) == "completed" for item in summaries):
        return "completed"
    if all(str(item.get("status")) != "completed" for item in summaries):
        return "failed"
    return "partial_failed"


def _run_model_queue(
    *,
    specs: Sequence[Any],
    output_root: Path,
    downloads_root: Path,
    dtype_name: str,
    disable_xet: bool,
    hf_token: str | None,
    download_retries: int,
    shared_ewok_module: Any,
    ewok_df: Any,
    variant: str,
    domains: Sequence[str],
    limit: int | None,
    prompt_template_name: str,
    prompt_template_source: str,
    template_text: str,
    inference_mode: str,
    score_reduction: str,
    margin_eps: float,
    answer_separator: str,
    max_new_tokens: int,
    target_permutation_mode: str,
    target_permutation_seed: int,
    overwrite: bool,
    show_progress: bool,
    store_prompts: bool,
    config_path: str | None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()
    summaries: list[dict[str, Any]] = []
    model_iter = _maybe_tqdm(
        specs,
        enabled=show_progress and len(specs) > 1,
        desc="Prompted EWoK model queue",
        total=len(specs),
        leave=False,
    )
    for spec in model_iter:
        summaries.append(
            _run_single_model(
                spec=spec,
                output_root=output_root,
                downloads_root=downloads_root,
                dtype_name=dtype_name,
                disable_xet=disable_xet,
                hf_token=hf_token,
                download_retries=download_retries,
                shared_ewok_module=shared_ewok_module,
                ewok_df=ewok_df,
                variant=variant,
                domains=domains,
                limit=limit,
                prompt_template_name=prompt_template_name,
                prompt_template_source=prompt_template_source,
                template_text=template_text,
                inference_mode=inference_mode,
                score_reduction=score_reduction,
                margin_eps=margin_eps,
                answer_separator=answer_separator,
                max_new_tokens=max_new_tokens,
                target_permutation_mode=target_permutation_mode,
                target_permutation_seed=target_permutation_seed,
                overwrite=overwrite,
                show_progress=show_progress,
                store_prompts=store_prompts,
                emit_summary=True,
            )
        )

    completed = [item["model_id"] for item in summaries if str(item.get("status")) == "completed"]
    failed = [item["model_id"] for item in summaries if str(item.get("status")) != "completed"]
    queue_summary = {
        "status": _queue_status_from_summaries(summaries),
        "provider": "local_hf",
        "eval_method": "prompted_choice",
        "config": str(config_path) if config_path else None,
        "output_root": str(output_root),
        "prompt_template_name": prompt_template_name,
        "prompt_template_source": prompt_template_source,
        "inference_mode": inference_mode,
        "target_permutation_mode": str(target_permutation_mode),
        "target_permutation_seed": int(target_permutation_seed),
        "num_models": int(len(specs)),
        "num_completed": int(len(completed)),
        "num_failed": int(len(failed)),
        "completed_model_ids": completed,
        "failed_model_ids": failed,
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "results": list(summaries),
    }
    _write_json(output_root / DEFAULT_QUEUE_SUMMARY_NAME, queue_summary)
    return queue_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate local HF models on EWoK using prompted direct-choice templates."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--model", help="Single Hugging Face model ID to evaluate.")
    target_group.add_argument(
        "--config",
        default=None,
        help="YAML queue config using the same model format as run_queue.py.",
    )
    parser.add_argument(
        "--param-count-b",
        type=float,
        default=None,
        help="Required in --model mode if you want size-aware plotting/metadata later.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional model revision label for --model mode.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for --model mode.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="",
        help="Optional tokenizer override for --model mode.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Where to write prompted eval outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--downloads-root",
        default=str(rq.DEFAULT_DOWNLOADS_ROOT),
        help=f"Where to stage temporary model downloads. Default: {rq.DEFAULT_DOWNLOADS_ROOT}",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Requested model dtype. Default: auto",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=rq.DEFAULT_DOWNLOAD_RETRIES,
        help=f"Download retry count. Default: {rq.DEFAULT_DOWNLOAD_RETRIES}",
    )
    xet_group = parser.add_mutually_exclusive_group()
    xet_group.add_argument(
        "--disable-xet",
        dest="disable_xet",
        action="store_true",
        help="Disable the hf_xet/Xet download backend.",
    )
    xet_group.add_argument(
        "--enable-xet",
        dest="disable_xet",
        action="store_false",
        help="Allow Hugging Face to use the hf_xet/Xet backend.",
    )
    parser.set_defaults(disable_xet=rq.DEFAULT_DISABLE_XET)

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        choices=sorted(BUILTIN_PROMPT_TEMPLATES.keys()),
        help=f"Built-in prompt template to use. Default: {DEFAULT_PROMPT_TEMPLATE}",
    )
    prompt_group.add_argument(
        "--prompt-template-file",
        default=None,
        help="Path to a custom prompt template file using placeholders like {{Ci}}, {{T1}}, {{T2}}.",
    )
    parser.add_argument(
        "--prompt-label",
        default="",
        help="Optional label for a custom prompt file; used in output folder names.",
    )
    parser.add_argument(
        "--inference-mode",
        default=DEFAULT_INFERENCE_MODE,
        choices=INFERENCE_MODES,
        help=f"Prompted inference mode. Default: {DEFAULT_INFERENCE_MODE}",
    )
    parser.add_argument(
        "--score-reduction",
        default=DEFAULT_SCORE_REDUCTION,
        choices=["sum", "mean"],
        help=f"How to aggregate target token logprobs for answer-logprob mode. Default: {DEFAULT_SCORE_REDUCTION}",
    )
    parser.add_argument(
        "--answer-separator",
        default=DEFAULT_ANSWER_SEPARATOR,
        help="String inserted between the prompt prefix and candidate answer in answer-logprob mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation budget for choice_generate mode. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--margin-eps",
        type=float,
        default=DEFAULT_MARGIN_EPS,
        help=f"Near-tie threshold. Default: {DEFAULT_MARGIN_EPS}",
    )
    parser.add_argument(
        "--target-permutation-mode",
        default=DEFAULT_TARGET_PERMUTATION_MODE,
        choices=TARGET_PERMUTATION_MODES,
        help=(
            "How to order displayed candidate targets in prompted choice mode. "
            f"Default: {DEFAULT_TARGET_PERMUTATION_MODE}"
        ),
    )
    parser.add_argument(
        "--target-permutation-seed",
        type=int,
        default=DEFAULT_TARGET_PERMUTATION_SEED,
        help=(
            "Seed used when --target-permutation-mode=random. "
            f"Default: {DEFAULT_TARGET_PERMUTATION_SEED}"
        ),
    )
    parser.add_argument(
        "--ewok-variant",
        default=DEFAULT_EWOK_VARIANT,
        help=f"EWoK variant to load. Default: {DEFAULT_EWOK_VARIANT}",
    )
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        default=[],
        help="Optional domain filter. Pass multiple times to keep multiple domains.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit after domain filtering.",
    )
    parser.add_argument(
        "--store-prompts",
        action="store_true",
        help="Include rendered prompts in ewok_items.jsonl for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prompted-eval output directory.",
    )
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        help="Force progress bars on.",
    )
    parser.add_argument(
        "--hide-progress",
        dest="show_progress",
        action="store_false",
        help="Force progress bars off.",
    )
    parser.set_defaults(show_progress=DEFAULT_SHOW_PROGRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    margin_eps = float(args.margin_eps)
    if margin_eps < 0:
        raise SystemExit(f"margin_eps must be >= 0, got: {margin_eps}")
    if int(args.max_new_tokens) <= 0:
        raise SystemExit(f"max_new_tokens must be > 0, got: {args.max_new_tokens}")
    if args.limit is not None and int(args.limit) <= 0:
        raise SystemExit(f"limit must be > 0, got: {args.limit}")

    prompt_template_name, prompt_template_source, template_text = _resolve_prompt_template(
        prompt_template_name=args.prompt_template,
        prompt_template_file=args.prompt_template_file,
        prompt_label=args.prompt_label,
    )
    variant = str(args.ewok_variant).strip().lower()
    shared_ewok_module = _load_shared_ewok_module(variant)
    ewok_df = _filter_ewok_df(
        ewok_df=shared_ewok_module.ewok_df,
        domains=args.domains,
        limit=args.limit,
    )
    if len(ewok_df) == 0:
        raise SystemExit("No EWoK rows remained after filtering.")

    output_root = Path(args.output_root).expanduser().resolve()
    downloads_root = Path(args.downloads_root).expanduser().resolve()

    if args.config:
        specs = rq.load_queue_config(args.config)
        if not specs:
            raise SystemExit(f"No models were found in queue config: {args.config}")
        queue_summary = _run_model_queue(
            specs=specs,
            output_root=output_root,
            downloads_root=downloads_root,
            dtype_name=args.dtype,
            disable_xet=bool(args.disable_xet),
            hf_token=_optional_str(args.hf_token),
            download_retries=int(args.download_retries),
            shared_ewok_module=shared_ewok_module,
            ewok_df=ewok_df,
            variant=variant,
            domains=args.domains,
            limit=args.limit,
            prompt_template_name=prompt_template_name,
            prompt_template_source=prompt_template_source,
            template_text=template_text,
            inference_mode=args.inference_mode,
            score_reduction=args.score_reduction,
            margin_eps=margin_eps,
            answer_separator=args.answer_separator,
            max_new_tokens=int(args.max_new_tokens),
            target_permutation_mode=args.target_permutation_mode,
            target_permutation_seed=int(args.target_permutation_seed),
            overwrite=bool(args.overwrite),
            show_progress=bool(args.show_progress),
            store_prompts=bool(args.store_prompts),
            config_path=args.config,
        )
        print(json.dumps(queue_summary, indent=2, sort_keys=False))
        return 0 if queue_summary["status"] == "completed" else 1

    if args.param_count_b is None:
        raise SystemExit("--param-count-b is required in --model mode.")

    spec = rq.ModelSpec(
        model_id=str(args.model).strip(),
        param_count_b=float(args.param_count_b),
        revision=_optional_str(args.revision),
        trust_remote_code=bool(args.trust_remote_code),
        tokenizer_id=_optional_str(args.tokenizer_id),
    )
    summary = _run_single_model(
        spec=spec,
        output_root=output_root,
        downloads_root=downloads_root,
        dtype_name=args.dtype,
        disable_xet=bool(args.disable_xet),
        hf_token=_optional_str(args.hf_token),
        download_retries=int(args.download_retries),
        shared_ewok_module=shared_ewok_module,
        ewok_df=ewok_df,
        variant=variant,
        domains=args.domains,
        limit=args.limit,
        prompt_template_name=prompt_template_name,
        prompt_template_source=prompt_template_source,
        template_text=template_text,
        inference_mode=args.inference_mode,
        score_reduction=args.score_reduction,
        margin_eps=margin_eps,
        answer_separator=args.answer_separator,
        max_new_tokens=int(args.max_new_tokens),
        target_permutation_mode=args.target_permutation_mode,
        target_permutation_seed=int(args.target_permutation_seed),
        overwrite=bool(args.overwrite),
        show_progress=bool(args.show_progress),
        store_prompts=bool(args.store_prompts),
        emit_summary=True,
    )
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
