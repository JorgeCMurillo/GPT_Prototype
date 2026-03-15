#!/usr/bin/env python3
"""Evaluate causal language models on EWoK with multiple scoring conventions.

The shared language-model evaluator supports two complementary metrics:

- BabyLM completion choice:
  Hold the context fixed and compare the correct target against the distractor.
- EWoK paper context sensitivity:
  Hold the target fixed and compare the correct context against the distractor.

Backward compatibility is preserved: `evaluate(...)` still returns the BabyLM
completion-choice outputs by default. Call `evaluate(..., return_all_methods=True)`
to retrieve both methods in one pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    from .ewok_data import load_ewok_df
except ImportError:
    from ewok_data import load_ewok_df


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
BABYLM_COMPLETION_CHOICE_SCORING = BABYLM_COMPLETION_CHOICE
EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
EWOK_PAPER_CONTEXT_SENSITIVITY = EWOK_CONTEXT_SENSITIVITY


def _resolve_device(model, device_override=None):
    if device_override is not None:
        return torch.device(device_override)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return device


def per_token_log_likelihood(model, tokenizer, input_texts, device=None):
    device = _resolve_device(model, device)
    inputs = tokenizer(input_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    batch_size = input_ids.shape[0]
    bos_token_id = tokenizer.bos_token_id
    bos_tensor = torch.full((batch_size, 1), bos_token_id, device=device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    ones_tensor = torch.ones((batch_size, 1), device=device)
    attn_mask = torch.cat([ones_tensor, attn_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = getattr(outputs, "logits", outputs["logits"])

    logits = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = input_ids[:, 1:]
    token_logprobs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs, inputs.attention_mask


def per_token_conditional_log_likelihood(model, tokenizer, contexts, targets, device=None, batch_size=8):
    device = _resolve_device(model, device)
    all_results = []

    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        batch_texts = [c + " " + t for c, t in zip(batch_contexts, batch_targets)]
        batch_context_lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in batch_contexts]
        batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, batch_texts, device)

        for j in range(len(batch_contexts)):
            start_idx = batch_context_lengths[j]
            valid_length = batch_attn_masks[j].sum().item()
            row_result = batch_log_probs[j, start_idx:valid_length]
            all_results.append(row_result)

    return all_results


ewok_df, SRC = load_ewok_df()
ewok_df = ewok_df.convert_dtypes()


def _validate_score_reduction(score_reduction: str) -> str:
    score_reduction = str(score_reduction).strip().lower()
    if score_reduction not in {"sum", "mean"}:
        raise ValueError(f"score_reduction must be 'sum' or 'mean', got: {score_reduction}")
    return score_reduction


def _reduce_token_logps(token_logps, score_reduction: str) -> float:
    if token_logps.numel() == 0:
        return 0.0
    if score_reduction == "sum":
        return float(token_logps.sum().item())
    return float(token_logps.mean().item())


def _score_domain_rows(model, tokenizer, df, *, batch_size: int, score_reduction: str):
    ctx1 = df["Context1"].tolist()
    tgt1 = df["Target1"].tolist()
    ctx2 = df["Context2"].tolist()
    tgt2 = df["Target2"].tolist()

    r11 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt1, batch_size=batch_size)
    r12 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt2, batch_size=batch_size)
    r22 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt2, batch_size=batch_size)
    r21 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt1, batch_size=batch_size)

    s11 = np.array([_reduce_token_logps(x, score_reduction) for x in r11], dtype=np.float64)
    s12 = np.array([_reduce_token_logps(x, score_reduction) for x in r12], dtype=np.float64)
    s22 = np.array([_reduce_token_logps(x, score_reduction) for x in r22], dtype=np.float64)
    s21 = np.array([_reduce_token_logps(x, score_reduction) for x in r21], dtype=np.float64)
    return s11, s12, s22, s21


def ewok_score_records_all_methods(
    model,
    tokenizer,
    batch_size=8,
    score_reduction="sum",
    margin_eps: float = 1e-6,
):
    """Build per-item EWoK records for BabyLM and paper-style scoring methods."""
    score_reduction = _validate_score_reduction(score_reduction)
    margin_eps = float(margin_eps)
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    records = []
    domains = ewok_df["Domain"].unique()

    for domain in domains:
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()
        s11, s12, s22, s21 = _score_domain_rows(
            model,
            tokenizer,
            df,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )

        m1 = s11 - s12
        m2 = s22 - s21
        m = 0.5 * (m1 + m2)

        k1 = s11 - s21
        k2 = s22 - s12
        k = 0.5 * (k1 + k2)

        for i in range(len(df)):
            rec = {
                "domain": domain,
                "row_index": int(df.loc[i, "index"]),
                "score_reduction": score_reduction,
                "S11_logp_T1_given_C1": float(s11[i]),
                "S12_logp_T2_given_C1": float(s12[i]),
                "S22_logp_T2_given_C2": float(s22[i]),
                "S21_logp_T1_given_C2": float(s21[i]),
                "margin_official_m1": float(m1[i]),
                "margin_symmetric_m2": float(m2[i]),
                "margin_combined": float(m[i]),
                "correct_official": bool(m1[i] > 0.0),
                "correct_symmetric": bool(m2[i] > 0.0),
                "correct_combined": bool(m[i] > 0.0),
                "near_tie_official": bool(abs(m1[i]) < margin_eps),
                "near_tie_symmetric": bool(abs(m2[i]) < margin_eps),
                "near_tie_combined": bool(abs(m[i]) < margin_eps),
                "babylm_completion_choice_margin_official_m1": float(m1[i]),
                "babylm_completion_choice_margin_symmetric_m2": float(m2[i]),
                "babylm_completion_choice_margin_combined": float(m[i]),
                "babylm_completion_choice_correct_official": bool(m1[i] > 0.0),
                "babylm_completion_choice_correct_symmetric": bool(m2[i] > 0.0),
                "babylm_completion_choice_correct_combined": bool(m[i] > 0.0),
                "babylm_completion_choice_near_tie_official": bool(abs(m1[i]) < margin_eps),
                "babylm_completion_choice_near_tie_symmetric": bool(abs(m2[i]) < margin_eps),
                "babylm_completion_choice_near_tie_combined": bool(abs(m[i]) < margin_eps),
                "ewok_context_sensitivity_margin_official_k1": float(k1[i]),
                "ewok_context_sensitivity_margin_symmetric_k2": float(k2[i]),
                "ewok_context_sensitivity_margin_combined": float(k[i]),
                "ewok_context_sensitivity_correct_official": bool(k1[i] > 0.0),
                "ewok_context_sensitivity_correct_symmetric": bool(k2[i] > 0.0),
                "ewok_context_sensitivity_correct_combined": bool(k[i] > 0.0),
                "ewok_context_sensitivity_near_tie_official": bool(abs(k1[i]) < margin_eps),
                "ewok_context_sensitivity_near_tie_symmetric": bool(abs(k2[i]) < margin_eps),
                "ewok_context_sensitivity_near_tie_combined": bool(abs(k[i]) < margin_eps),
                "ewok_paper_context_sensitivity_margin_official_k1": float(k1[i]),
                "ewok_paper_context_sensitivity_margin_symmetric_k2": float(k2[i]),
                "ewok_paper_context_sensitivity_margin_combined": float(k[i]),
                "ewok_paper_context_sensitivity_correct_official": bool(k1[i] > 0.0),
                "ewok_paper_context_sensitivity_correct_symmetric": bool(k2[i] > 0.0),
                "ewok_paper_context_sensitivity_correct_combined": bool(k[i] > 0.0),
                "ewok_paper_context_sensitivity_near_tie_official": bool(abs(k1[i]) < margin_eps),
                "ewok_paper_context_sensitivity_near_tie_symmetric": bool(abs(k2[i]) < margin_eps),
                "ewok_paper_context_sensitivity_near_tie_combined": bool(abs(k[i]) < margin_eps),
            }
            records.append(rec)

    return records


ewok_per_item_records = ewok_score_records_all_methods


def _summarize_babylm_completion_choice_metrics(records, margin_eps: float):
    domain_scores_official = {}
    domain_scores_full = {}
    domain_margin_stats = {}
    macro_metric_keys = ["acc_combined", "mean_signed_m", "mean_abs_m", "tie_rate_m"]
    macro_metric_values = {k: [] for k in macro_metric_keys}
    total_items = 0
    average_score = 0.0

    for domain in ewok_df["Domain"].unique():
        domain_records = [r for r in records if r["domain"] == domain]
        m1 = np.array([r["margin_official_m1"] for r in domain_records], dtype=np.float64)
        m2 = np.array([r["margin_symmetric_m2"] for r in domain_records], dtype=np.float64)
        m = np.array([r["margin_combined"] for r in domain_records], dtype=np.float64)

        acc1 = float((m1 > 0.0).mean())
        acc2 = float((m2 > 0.0).mean())
        acc_combined = float((m > 0.0).mean())

        domain_scores_official[domain] = acc1
        domain_scores_full[domain] = (acc1, acc2)
        average_score += acc1

        stats = {
            "n": int(len(domain_records)),
            "acc_combined": acc_combined,
            "mean_signed_m": float(m.mean()),
            "mean_abs_m": float(np.abs(m).mean()),
            "tie_rate_m": float((np.abs(m) < margin_eps).mean()),
        }
        domain_margin_stats[domain] = stats
        total_items += int(len(domain_records))
        for key in macro_metric_keys:
            macro_metric_values[key].append(stats[key])

    avg = average_score / len(ewok_df["Domain"].unique())
    domain_scores_full["average"] = (avg, avg)
    domain_margin_stats["average"] = {
        "n": int(total_items),
        "acc_combined": float(np.mean(macro_metric_values["acc_combined"])),
        "mean_signed_m": float(np.mean(macro_metric_values["mean_signed_m"])),
        "mean_abs_m": float(np.mean(macro_metric_values["mean_abs_m"])),
        "tie_rate_m": float(np.mean(macro_metric_values["tie_rate_m"])),
    }

    return {
        "label": "BabyLM Completion Choice Scoring",
        "domain_scores_official": domain_scores_official,
        "domain_scores_full": domain_scores_full,
        "domain_margin_stats": domain_margin_stats,
    }


def _summarize_ewok_paper_context_sensitivity_metrics(records, margin_eps: float):
    domain_scores_official = {}
    domain_scores_full = {}
    domain_margin_stats = {}
    macro_metric_keys = ["acc_combined", "mean_signed_k", "mean_abs_k", "tie_rate_k"]
    macro_metric_values = {k: [] for k in macro_metric_keys}
    total_items = 0

    for domain in ewok_df["Domain"].unique():
        domain_records = [r for r in records if r["domain"] == domain]
        k1 = np.array(
            [r["ewok_context_sensitivity_margin_official_k1"] for r in domain_records],
            dtype=np.float64,
        )
        k2 = np.array(
            [r["ewok_context_sensitivity_margin_symmetric_k2"] for r in domain_records],
            dtype=np.float64,
        )
        k = np.array(
            [r["ewok_context_sensitivity_margin_combined"] for r in domain_records],
            dtype=np.float64,
        )

        acc1 = float((k1 > 0.0).mean())
        acc2 = float((k2 > 0.0).mean())
        acc_combined = float((k > 0.0).mean())

        domain_scores_official[domain] = acc1
        domain_scores_full[domain] = (acc1, acc2)
        stats = {
            "n": int(len(domain_records)),
            "acc_combined": acc_combined,
            "mean_signed_k": float(k.mean()),
            "mean_abs_k": float(np.abs(k).mean()),
            "tie_rate_k": float((np.abs(k) < margin_eps).mean()),
        }
        domain_margin_stats[domain] = stats
        total_items += int(len(domain_records))
        for key in macro_metric_keys:
            macro_metric_values[key].append(stats[key])

    avg_official = float(np.mean(list(domain_scores_official.values())))
    avg_symmetric = float(np.mean([values[1] for values in domain_scores_full.values()]))
    domain_scores_full["average"] = (avg_official, avg_symmetric)
    domain_margin_stats["average"] = {
        "n": int(total_items),
        "acc_combined": float(np.mean(macro_metric_values["acc_combined"])),
        "mean_signed_k": float(np.mean(macro_metric_values["mean_signed_k"])),
        "mean_abs_k": float(np.mean(macro_metric_values["mean_abs_k"])),
        "tie_rate_k": float(np.mean(macro_metric_values["tie_rate_k"])),
    }

    return {
        "label": "EWoK Paper Context Sensitivity",
        "domain_scores_official": domain_scores_official,
        "domain_scores_full": domain_scores_full,
        "domain_margin_stats": domain_margin_stats,
    }


def _summarize_records_all_methods(records, margin_eps: float):
    return {
        BABYLM_COMPLETION_CHOICE: _summarize_babylm_completion_choice_metrics(records, margin_eps),
        EWOK_CONTEXT_SENSITIVITY: _summarize_ewok_paper_context_sensitivity_metrics(records, margin_eps),
    }


def _legacy_results_from_metrics(metrics_by_method, *, per_item, return_per_item: bool):
    babylm_metrics = metrics_by_method[BABYLM_COMPLETION_CHOICE]
    if return_per_item:
        return (
            babylm_metrics["domain_scores_official"],
            babylm_metrics["domain_scores_full"],
            per_item,
            babylm_metrics["domain_margin_stats"],
        )
    return (
        babylm_metrics["domain_scores_official"],
        babylm_metrics["domain_scores_full"],
        babylm_metrics["domain_margin_stats"],
    )


def evaluate(
    model,
    tokenizer,
    batch_size=8,
    return_per_item=False,
    score_reduction="sum",
    margin_eps: float = 1e-6,
    return_all_methods: bool = False,
):
    """Evaluate EWoK under BabyLM and paper-style scoring conventions."""
    score_reduction = _validate_score_reduction(score_reduction)
    margin_eps = float(margin_eps)
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    print(f"evaluating model on EWOK dataset... (score_reduction={score_reduction})")
    records = ewok_score_records_all_methods(
        model,
        tokenizer,
        batch_size=batch_size,
        score_reduction=score_reduction,
        margin_eps=margin_eps,
    )
    metrics_by_method = _summarize_records_all_methods(records, margin_eps)
    per_item = records if return_per_item else None

    if return_all_methods:
        if return_per_item:
            return metrics_by_method, per_item
        return metrics_by_method

    return _legacy_results_from_metrics(
        metrics_by_method,
        per_item=per_item,
        return_per_item=return_per_item,
    )


def evaluate_babylm_completion_choice(
    model,
    tokenizer,
    batch_size=8,
    return_per_item=False,
    score_reduction="sum",
    margin_eps: float = 1e-6,
):
    """Evaluate only the BabyLM completion-choice scoring view."""
    return evaluate(
        model,
        tokenizer,
        batch_size=batch_size,
        return_per_item=return_per_item,
        score_reduction=score_reduction,
        margin_eps=margin_eps,
        return_all_methods=False,
    )


def evaluate_all_ewok_scoring_methods(
    model,
    tokenizer,
    batch_size=8,
    return_per_item=False,
    score_reduction="sum",
    margin_eps: float = 1e-6,
):
    """Evaluate both BabyLM and EWoK paper-style scoring views."""
    return evaluate(
        model,
        tokenizer,
        batch_size=batch_size,
        return_per_item=return_per_item,
        score_reduction=score_reduction,
        margin_eps=margin_eps,
        return_all_methods=True,
    )


__all__ = [
    "BABYLM_COMPLETION_CHOICE",
    "BABYLM_COMPLETION_CHOICE_SCORING",
    "EWOK_CONTEXT_SENSITIVITY",
    "EWOK_PAPER_CONTEXT_SENSITIVITY",
    "SRC",
    "evaluate",
    "evaluate_babylm_completion_choice",
    "evaluate_all_ewok_scoring_methods",
    "ewok_df",
    "ewok_score_records_all_methods",
    "ewok_per_item_records",
    "per_token_conditional_log_likelihood",
    "per_token_log_likelihood",
]
