#!/usr/bin/env python3
"""Evaluate causal language models on EWoK with multiple scoring conventions.

The shared language-model evaluator supports three complementary metrics:

- BabyLM completion choice:
  Hold the context fixed and compare the correct target against the distractor.
- EWoK paper context sensitivity:
  Hold the target fixed and compare the correct context against the distractor.
- PMI completion choice:
  Hold the context fixed and subtract each target's BOS-only likelihood before
  comparing the correct target against the distractor.

Backward compatibility is preserved: `evaluate(...)` still returns the BabyLM
completion-choice outputs by default. Call `evaluate(..., return_all_methods=True)`
to retrieve all three methods in one pass. Combined accuracy averages the two
decision accuracies; it does not threshold the average margin.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from .ewok_data import load_ewok_df
except ImportError:
    from ewok_data import load_ewok_df


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
BABYLM_COMPLETION_CHOICE_SCORING = BABYLM_COMPLETION_CHOICE
EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
EWOK_PAPER_CONTEXT_SENSITIVITY = EWOK_CONTEXT_SENSITIVITY
PMI_COMPLETION_CHOICE = "pmi_completion_choice"


def _maybe_tqdm(iterable, *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _resolve_device(model, device_override=None):
    if device_override is not None:
        return torch.device(device_override)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return device


def resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise RuntimeError("Tokenizer must define bos_token_id, eos_token_id, or pad_token_id for EWoK evaluation.")


def per_token_log_likelihood(model, tokenizer, input_texts, device=None):
    device = _resolve_device(model, device)
    inputs = tokenizer(input_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    batch_size = input_ids.shape[0]
    bos_token_id = resolve_bos_token_id(tokenizer)
    bos_tensor = torch.full((batch_size, 1), bos_token_id, device=device, dtype=input_ids.dtype)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    ones_tensor = torch.ones((batch_size, 1), device=device, dtype=attn_mask.dtype)
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


def per_token_unconditional_log_likelihood(model, tokenizer, texts, device=None, batch_size=8):
    """Score text from the evaluator BOS token with no EWoK context."""
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, batch_texts, device)

        for j in range(len(batch_texts)):
            valid_length = batch_attn_masks[j].sum().item()
            all_results.append(batch_log_probs[j, :valid_length])

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
    rb1 = per_token_unconditional_log_likelihood(model, tokenizer, tgt1, batch_size=batch_size)
    rb2 = per_token_unconditional_log_likelihood(model, tokenizer, tgt2, batch_size=batch_size)

    s11 = np.array([_reduce_token_logps(x, score_reduction) for x in r11], dtype=np.float64)
    s12 = np.array([_reduce_token_logps(x, score_reduction) for x in r12], dtype=np.float64)
    s22 = np.array([_reduce_token_logps(x, score_reduction) for x in r22], dtype=np.float64)
    s21 = np.array([_reduce_token_logps(x, score_reduction) for x in r21], dtype=np.float64)
    b1 = np.array([_reduce_token_logps(x, score_reduction) for x in rb1], dtype=np.float64)
    b2 = np.array([_reduce_token_logps(x, score_reduction) for x in rb2], dtype=np.float64)
    return s11, s12, s22, s21, b1, b2


def ewok_score_records_all_methods(
    model,
    tokenizer,
    batch_size=8,
    score_reduction="sum",
    margin_eps: float = 1e-6,
    show_progress: bool = False,
):
    """Build per-item EWoK records for BabyLM, paper-style, and PMI scoring."""
    score_reduction = _validate_score_reduction(score_reduction)
    margin_eps = float(margin_eps)
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    records = []
    domains = ewok_df["Domain"].unique()

    domain_iter = _maybe_tqdm(
        domains,
        enabled=show_progress,
        desc=f"EWoK domains ({score_reduction})",
        total=len(domains),
        leave=False,
    )
    for domain in domain_iter:
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()
        s11, s12, s22, s21, b1, b2 = _score_domain_rows(
            model,
            tokenizer,
            df,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )

        p11 = s11 - b1
        p12 = s12 - b2
        p22 = s22 - b2
        p21 = s21 - b1

        m1 = s11 - s12
        m2 = s22 - s21
        m = 0.5 * (m1 + m2)

        k1 = s11 - s21
        k2 = s22 - s12
        k = 0.5 * (k1 + k2)

        pm1 = p11 - p12
        pm2 = p22 - p21
        pm = 0.5 * (pm1 + pm2)

        for i in range(len(df)):
            m1_correct = bool(m1[i] > 0.0)
            m2_correct = bool(m2[i] > 0.0)
            m_correct_combined = 0.5 * (float(m1_correct) + float(m2_correct))
            k1_correct = bool(k1[i] > 0.0)
            k2_correct = bool(k2[i] > 0.0)
            k_correct_combined = 0.5 * (float(k1_correct) + float(k2_correct))
            pm1_correct = bool(pm1[i] > 0.0)
            pm2_correct = bool(pm2[i] > 0.0)
            pm_correct_combined = 0.5 * (float(pm1_correct) + float(pm2_correct))
            rec = {
                "domain": domain,
                "row_index": int(df.loc[i, "index"]),
                "score_reduction": score_reduction,
                "S11_logp_T1_given_C1": float(s11[i]),
                "S12_logp_T2_given_C1": float(s12[i]),
                "S22_logp_T2_given_C2": float(s22[i]),
                "S21_logp_T1_given_C2": float(s21[i]),
                "B1_logp_T1_given_BOS": float(b1[i]),
                "B2_logp_T2_given_BOS": float(b2[i]),
                "margin_official_m1": float(m1[i]),
                "margin_symmetric_m2": float(m2[i]),
                "margin_combined": float(m[i]),
                "correct_official": m1_correct,
                "correct_symmetric": m2_correct,
                "correct_combined": m_correct_combined,
                "near_tie_official": bool(abs(m1[i]) < margin_eps),
                "near_tie_symmetric": bool(abs(m2[i]) < margin_eps),
                "near_tie_combined": bool(abs(m[i]) < margin_eps),
                "babylm_completion_choice_margin_official_m1": float(m1[i]),
                "babylm_completion_choice_margin_symmetric_m2": float(m2[i]),
                "babylm_completion_choice_margin_combined": float(m[i]),
                "babylm_completion_choice_correct_official": m1_correct,
                "babylm_completion_choice_correct_symmetric": m2_correct,
                "babylm_completion_choice_correct_combined": m_correct_combined,
                "babylm_completion_choice_near_tie_official": bool(abs(m1[i]) < margin_eps),
                "babylm_completion_choice_near_tie_symmetric": bool(abs(m2[i]) < margin_eps),
                "babylm_completion_choice_near_tie_combined": bool(abs(m[i]) < margin_eps),
                "ewok_context_sensitivity_margin_official_k1": float(k1[i]),
                "ewok_context_sensitivity_margin_symmetric_k2": float(k2[i]),
                "ewok_context_sensitivity_margin_combined": float(k[i]),
                "ewok_context_sensitivity_correct_official": k1_correct,
                "ewok_context_sensitivity_correct_symmetric": k2_correct,
                "ewok_context_sensitivity_correct_combined": k_correct_combined,
                "ewok_context_sensitivity_near_tie_official": bool(abs(k1[i]) < margin_eps),
                "ewok_context_sensitivity_near_tie_symmetric": bool(abs(k2[i]) < margin_eps),
                "ewok_context_sensitivity_near_tie_combined": bool(abs(k[i]) < margin_eps),
                "ewok_paper_context_sensitivity_margin_official_k1": float(k1[i]),
                "ewok_paper_context_sensitivity_margin_symmetric_k2": float(k2[i]),
                "ewok_paper_context_sensitivity_margin_combined": float(k[i]),
                "ewok_paper_context_sensitivity_correct_official": k1_correct,
                "ewok_paper_context_sensitivity_correct_symmetric": k2_correct,
                "ewok_paper_context_sensitivity_correct_combined": k_correct_combined,
                "ewok_paper_context_sensitivity_near_tie_official": bool(abs(k1[i]) < margin_eps),
                "ewok_paper_context_sensitivity_near_tie_symmetric": bool(abs(k2[i]) < margin_eps),
                "ewok_paper_context_sensitivity_near_tie_combined": bool(abs(k[i]) < margin_eps),
                "pmi_completion_choice_S11_T1_given_C1": float(p11[i]),
                "pmi_completion_choice_S12_T2_given_C1": float(p12[i]),
                "pmi_completion_choice_S22_T2_given_C2": float(p22[i]),
                "pmi_completion_choice_S21_T1_given_C2": float(p21[i]),
                "pmi_completion_choice_margin_official_m1": float(pm1[i]),
                "pmi_completion_choice_margin_symmetric_m2": float(pm2[i]),
                "pmi_completion_choice_margin_combined": float(pm[i]),
                "pmi_completion_choice_correct_official": pm1_correct,
                "pmi_completion_choice_correct_symmetric": pm2_correct,
                "pmi_completion_choice_correct_combined": pm_correct_combined,
                "pmi_completion_choice_near_tie_official": bool(abs(pm1[i]) < margin_eps),
                "pmi_completion_choice_near_tie_symmetric": bool(abs(pm2[i]) < margin_eps),
                "pmi_completion_choice_near_tie_combined": bool(abs(pm[i]) < margin_eps),
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
        acc_combined = 0.5 * (acc1 + acc2)

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
        acc_combined = 0.5 * (acc1 + acc2)

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


def _records_have_pmi_completion_choice(records) -> bool:
    required = (
        "pmi_completion_choice_margin_official_m1",
        "pmi_completion_choice_margin_symmetric_m2",
        "pmi_completion_choice_margin_combined",
    )
    return bool(records) and all(all(key in record for key in required) for record in records)


def _summarize_pmi_completion_choice_metrics(records, margin_eps: float):
    domain_scores_official = {}
    domain_scores_full = {}
    domain_margin_stats = {}
    macro_metric_keys = ["acc_combined", "mean_signed_pmi_m", "mean_abs_pmi_m", "tie_rate_pmi_m"]
    macro_metric_values = {k: [] for k in macro_metric_keys}
    total_items = 0

    for domain in ewok_df["Domain"].unique():
        domain_records = [r for r in records if r["domain"] == domain]
        m1 = np.array(
            [r["pmi_completion_choice_margin_official_m1"] for r in domain_records],
            dtype=np.float64,
        )
        m2 = np.array(
            [r["pmi_completion_choice_margin_symmetric_m2"] for r in domain_records],
            dtype=np.float64,
        )
        m = np.array(
            [r["pmi_completion_choice_margin_combined"] for r in domain_records],
            dtype=np.float64,
        )

        acc1 = float((m1 > 0.0).mean())
        acc2 = float((m2 > 0.0).mean())
        acc_combined = 0.5 * (acc1 + acc2)

        domain_scores_official[domain] = acc1
        domain_scores_full[domain] = (acc1, acc2)
        stats = {
            "n": int(len(domain_records)),
            "acc_combined": acc_combined,
            "mean_signed_pmi_m": float(m.mean()),
            "mean_abs_pmi_m": float(np.abs(m).mean()),
            "tie_rate_pmi_m": float((np.abs(m) < margin_eps).mean()),
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
        "mean_signed_pmi_m": float(np.mean(macro_metric_values["mean_signed_pmi_m"])),
        "mean_abs_pmi_m": float(np.mean(macro_metric_values["mean_abs_pmi_m"])),
        "tie_rate_pmi_m": float(np.mean(macro_metric_values["tie_rate_pmi_m"])),
    }

    return {
        "label": "PMI Completion Choice Scoring",
        "domain_scores_official": domain_scores_official,
        "domain_scores_full": domain_scores_full,
        "domain_margin_stats": domain_margin_stats,
    }


def _summarize_records_all_methods(records, margin_eps: float):
    metrics_by_method = {
        BABYLM_COMPLETION_CHOICE: _summarize_babylm_completion_choice_metrics(records, margin_eps),
        EWOK_CONTEXT_SENSITIVITY: _summarize_ewok_paper_context_sensitivity_metrics(records, margin_eps),
    }
    if _records_have_pmi_completion_choice(records):
        metrics_by_method[PMI_COMPLETION_CHOICE] = _summarize_pmi_completion_choice_metrics(records, margin_eps)
    return metrics_by_method


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
    show_progress: bool = False,
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
        show_progress=show_progress,
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
    show_progress: bool = False,
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
        show_progress=show_progress,
    )


def evaluate_all_ewok_scoring_methods(
    model,
    tokenizer,
    batch_size=8,
    return_per_item=False,
    score_reduction="sum",
    margin_eps: float = 1e-6,
    show_progress: bool = False,
):
    """Evaluate BabyLM completion choice, context sensitivity, and PMI scoring."""
    return evaluate(
        model,
        tokenizer,
        batch_size=batch_size,
        return_per_item=return_per_item,
        score_reduction=score_reduction,
        margin_eps=margin_eps,
        return_all_methods=True,
        show_progress=show_progress,
    )


__all__ = [
    "BABYLM_COMPLETION_CHOICE",
    "BABYLM_COMPLETION_CHOICE_SCORING",
    "EWOK_CONTEXT_SENSITIVITY",
    "EWOK_PAPER_CONTEXT_SENSITIVITY",
    "PMI_COMPLETION_CHOICE",
    "SRC",
    "evaluate",
    "evaluate_babylm_completion_choice",
    "evaluate_all_ewok_scoring_methods",
    "ewok_df",
    "ewok_score_records_all_methods",
    "ewok_per_item_records",
    "per_token_conditional_log_likelihood",
    "per_token_log_likelihood",
    "per_token_unconditional_log_likelihood",
    "resolve_bos_token_id",
]
