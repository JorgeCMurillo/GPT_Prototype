#!/usr/bin/env python3
import torch
from typing import List, Optional
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch

def per_token_log_likelihood(model, tokenizer, input_texts, device="cuda"):
    # Tokenize the batch with padding
    # return_tensors="pt" creates a rectangular tensor (Batch, Max_Seq_Len)
    inputs = tokenizer(input_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)
    
    # Get batch size
    batch_size = input_ids.shape[0]

    # Prepend BOS token (Batch-wise)
    bos_token_id = tokenizer.bos_token_id
    bos_tensor = torch.full((batch_size, 1), bos_token_id, device=device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    # Prepend Attention Mask (Batch-wise)
    ones_tensor = torch.ones((batch_size, 1), device=device)
    attn_mask = torch.cat([ones_tensor, attn_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs['logits']

    # Remove last logit (shift left)
    logits = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    # Remove first input_id (BOS) for aligning labels
    input_ids = input_ids[:, 1:]

    # Gather log probs at the specific input indices
    token_logprobs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Note: token_logprobs now contains values for PADDING tokens at the end too.
    # We return the whole thing and let the caller slice it.
    return token_logprobs, inputs.attention_mask

def per_token_conditional_log_likelihood(model, tokenizer, contexts, targets, device="cuda"):
    # 1. Prepare batch of texts exactly as you did
    texts = [c + " " + t for c, t in zip(contexts, targets)]
    
    # 2. Calculate context lengths for each item in the batch
    # We do this in a loop because they can be different lengths
    context_token_lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in contexts]

    # 3. Run the model on the batch (High Performance)
    batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, texts, device)

    # 4. Slice the results row-by-row to match your original logic
    results = []
    for i in range(len(contexts)):
        # Get the start index for this specific row
        start_idx = context_token_lengths[i]
        
        # Get the total length of valid tokens (excluding padding)
        # sum() gives the count of real tokens (non-padded)
        valid_length = batch_attn_masks[i].sum().item()
        
        # Slice: from context_end up to the actual end of the sentence (ignoring padding)
        # We use valid_length because batching adds padding zeros at the end which we don't want
        row_result = batch_log_probs[i, start_idx:valid_length]
        
        results.append(row_result)

    return results
def per_token_conditional_log_likelihood(model, tokenizer, contexts, targets, device="cuda", batch_size=8):
    all_results = []
    
    # Process data in chunks of batch_size
    for i in range(0, len(contexts), batch_size):
        # 1. Slice the current batch
        batch_contexts = contexts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        
        # 2. Prepare batch of texts (Context + " " + Target)
        batch_texts = [c + " " + t for c, t in zip(batch_contexts, batch_targets)]
        
        # 3. Calculate context lengths for this specific batch
        #    (We need this to know where to start slicing the probabilities)
        batch_context_lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in batch_contexts]

        # 4. Run the model on the current batch
        #    per_token_log_likelihood handles padding and tokenization internally
        batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, batch_texts, device)

        # 5. Process results for this batch
        for j in range(len(batch_contexts)):
            # Get the start index for this specific row
            start_idx = batch_context_lengths[j]
            
            # Get the total length of valid tokens (excluding padding)
            valid_length = batch_attn_masks[j].sum().item()
            
            # Slice: from context_end up to the actual end of the sentence
            # We assume batch_log_probs[j] corresponds to batch_texts[j]
            row_result = batch_log_probs[j, start_idx:valid_length]
            
            all_results.append(row_result)

    return all_results
import io
import os
import zipfile
import pandas as pd
from pathlib import Path

_LEGACY_SRC = Path(
    "/home/jorge/tokenPred/babylm_10m/test_eval/evaluation-pipeline-2025/evaluation_data/fast_eval/ewok_fast"
)
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_ZIP = _THIS_DIR / "ewok_fast_jsonl.zip"
_DEFAULT_DIR = _THIS_DIR / "ewok_fast"
_DEFAULT_ZIP_PASSWORD = os.environ.get("EWOK_ZIP_PASSWORD", "ew2026")


def _load_ewok_from_dir(src_dir: Path):
    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in directory: {src_dir}")
    return pd.concat(
        [pd.read_json(fp, lines=True, encoding="utf-8") for fp in jsonl_files],
        ignore_index=True,
        sort=False,
    )


def _load_ewok_from_zip(zip_path: Path, password: Optional[str]):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".jsonl"))
        if not members:
            raise FileNotFoundError(f"No .jsonl files found in zip: {zip_path}")

        frames = []
        for name in members:
            info = zf.getinfo(name)
            member_is_encrypted = bool(info.flag_bits & 0x1)
            pwd = None
            if member_is_encrypted:
                if not password:
                    raise RuntimeError(
                        f"Zip member '{name}' is encrypted. Set EWOK_ZIP_PASSWORD."
                    )
                pwd = password.encode("utf-8")
            try:
                with zf.open(name, "r", pwd=pwd) as f:
                    frames.append(pd.read_json(io.TextIOWrapper(f, encoding="utf-8"), lines=True))
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "password" in msg or "encrypted" in msg:
                    raise RuntimeError(
                        f"Failed to decrypt '{name}' in {zip_path}. "
                        "Set EWOK_ZIP_PASSWORD to the zip password."
                    ) from exc
                raise

    return pd.concat(frames, ignore_index=True, sort=False)


def _ewok_source_candidates():
    candidates = []
    env_src = os.environ.get("EWOK_SRC")
    if env_src:
        candidates.append(Path(env_src))
    candidates.extend([_DEFAULT_ZIP, _DEFAULT_DIR, _LEGACY_SRC])

    # Keep order but remove duplicates.
    deduped = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _load_ewok_df():
    errors = []
    candidates = _ewok_source_candidates()
    for src in candidates:
        try:
            if src.is_file() and src.suffix.lower() == ".zip":
                return _load_ewok_from_zip(src, _DEFAULT_ZIP_PASSWORD), src
            if src.is_dir():
                return _load_ewok_from_dir(src), src
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    searched = ", ".join(str(c) for c in candidates)
    details = "; ".join(errors) if errors else "No candidate source exists."
    raise FileNotFoundError(
        f"Could not load EWoK data. Searched: {searched}. Details: {details}"
    )


ewok_df, SRC = _load_ewok_df()
ewok_df = ewok_df.convert_dtypes()

def _validate_score_reduction(score_reduction: str) -> str:
    score_reduction = str(score_reduction).strip().lower()
    if score_reduction not in {"sum", "mean"}:
        raise ValueError(f"score_reduction must be 'sum' or 'mean', got: {score_reduction}")
    return score_reduction


def _reduce_token_logps(token_logps, score_reduction: str) -> float:
    # token_logps are per-token values for log P(T_i | C_j).
    # We either aggregate with:
    # - sum: log P(T_i | C_j) over all tokens in T_i
    # - mean: average per-token log P(T_i | C_j)
    if token_logps.numel() == 0:
        return 0.0
    if score_reduction == "sum":
        return float(token_logps.sum().item())
    return float(token_logps.mean().item())


def ewok_per_item_records(model, tokenizer, batch_size=8, score_reduction="sum"):
    """
    Build per-item EWOK records with pairwise comparisons.

    Notation:
    - T1, T2 are the two targets.
    - C1, C2 are the two contexts.
    - We score P(T_i | C_j) using either sum or mean token log-probabilities.

    For each item, we compute:
    - m1 = score(P(T1 | C1)) - score(P(T2 | C1))   (official)
    - m2 = score(P(T2 | C2)) - score(P(T1 | C2))   (symmetric)
    """
    score_reduction = _validate_score_reduction(score_reduction)
    records = []
    domains = ewok_df["Domain"].unique()

    for domain in domains:
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()  # keep original index
        ctx1 = df["Context1"].tolist()
        tgt1 = df["Target1"].tolist()
        ctx2 = df["Context2"].tolist()
        tgt2 = df["Target2"].tolist()

        r11 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt1, batch_size=batch_size)
        r12 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt2, batch_size=batch_size)
        r22 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt2, batch_size=batch_size)
        r21 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt1, batch_size=batch_size)

        for i in range(len(df)):
            # S11 means score(P(T1 | C1)); similarly for S12/S22/S21.
            S11 = _reduce_token_logps(r11[i], score_reduction)
            S12 = _reduce_token_logps(r12[i], score_reduction)
            S22 = _reduce_token_logps(r22[i], score_reduction)
            S21 = _reduce_token_logps(r21[i], score_reduction)

            # m1/m2 are preference margins under C1 and C2 respectively.
            m1 = S11 - S12
            m2 = S22 - S21

            rec = {
                "domain": domain,
                "row_index": int(df.loc[i, "index"]),   # original df index
                "score_reduction": score_reduction,
                "S11_logp_T1_given_C1": S11,
                "S12_logp_T2_given_C1": S12,
                "S22_logp_T2_given_C2": S22,
                "S21_logp_T1_given_C2": S21,
                "margin_official_m1": m1,
                "margin_symmetric_m2": m2,
                "correct_official": (m1 > 0.0),
                "correct_symmetric": (m2 > 0.0),
                "tie_official": (m1 == 0.0),
                "tie_symmetric": (m2 == 0.0),
            }
            records.append(rec)

    return records

def evaluate(model, tokenizer, batch_size=8, return_per_item=False, score_reduction="sum", margin_eps: float = 1e-6):
    """
    Evaluate EWOK domain accuracies.

    For each row we compare the following scores:
    - official: score(P(T1 | C1)) > score(P(T2 | C1))
    - symmetric: score(P(T2 | C2)) > score(P(T1 | C2))

    score(...) can be:
    - 'sum': sequence log-probability (sum of token log-probs)
    - 'mean': average token log-probability

    margin_eps defines near-ties as abs(margin) < margin_eps.
    """
    score_reduction = _validate_score_reduction(score_reduction)
    margin_eps = float(margin_eps)
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    domains = ewok_df["Domain"].unique()
    domain_scores_official = {}
    domain_scores_full = {}
    domain_margin_stats = {}
    average_score = 0.0
    print(f'evaluating model on EWOK dataset... (score_reduction={score_reduction})')
    per_item = [] if return_per_item else None
    macro_metric_keys = ["acc_combined", "mean_signed_m", "mean_abs_m", "tie_rate_m"]
    macro_metric_values = {k: [] for k in macro_metric_keys}
    total_items = 0

    for domain in domains:
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()
        ctx1 = df["Context1"].tolist()
        tgt1 = df["Target1"].tolist()
        ctx2 = df["Context2"].tolist()
        tgt2 = df["Target2"].tolist()

        r11 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt1, batch_size=batch_size)
        r12 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt2, batch_size=batch_size)
        r22 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt2, batch_size=batch_size)
        r21 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt1, batch_size=batch_size)

        m1 = np.array(
            [_reduce_token_logps(a, score_reduction) - _reduce_token_logps(b, score_reduction) for a, b in zip(r11, r12)]
        )
        m2 = np.array(
            [_reduce_token_logps(a, score_reduction) - _reduce_token_logps(b, score_reduction) for a, b in zip(r22, r21)]
        )
        m = 0.5 * (m1 + m2)

        acc1 = float((m1 > 0).mean())
        acc2 = float((m2 > 0).mean())
        acc_combined = float((m > 0).mean())

        domain_scores_full[domain] = (acc1, acc2)
        domain_scores_official[domain] = acc1
        average_score += acc1

        stats = {
            "n": int(len(df)),
            "acc_combined": float(acc_combined),
            "mean_signed_m": float(m.mean()),
            "mean_abs_m": float(np.abs(m).mean()),
            "tie_rate_m": float((np.abs(m) < margin_eps).mean()),
        }
        domain_margin_stats[domain] = stats
        total_items += int(len(df))
        for k in macro_metric_keys:
            macro_metric_values[k].append(stats[k])

        if return_per_item:
            for i in range(len(df)):
                per_item.append({
                    "domain": domain,
                    "row_index": int(df.loc[i, "index"]),
                    "score_reduction": score_reduction,
                    "margin_official_m1": float(m1[i]),
                    "margin_symmetric_m2": float(m2[i]),
                    "margin_combined": float(m[i]),
                    "correct_official": bool(m1[i] > 0),
                    "correct_symmetric": bool(m2[i] > 0),
                    "correct_combined": bool(m[i] > 0),
                    "near_tie_official": bool(abs(m1[i]) < margin_eps),
                    "near_tie_symmetric": bool(abs(m2[i]) < margin_eps),
                    "near_tie_combined": bool(abs(m[i]) < margin_eps),
                })

    avg = average_score / len(domains)
    domain_scores_full["average"] = (avg, avg)
    domain_margin_stats["average"] = {
        "n": int(total_items),
        "acc_combined": float(np.mean(macro_metric_values["acc_combined"])),
        "mean_signed_m": float(np.mean(macro_metric_values["mean_signed_m"])),
        "mean_abs_m": float(np.mean(macro_metric_values["mean_abs_m"])),
        "tie_rate_m": float(np.mean(macro_metric_values["tie_rate_m"])),
    }

    if return_per_item:
        return domain_scores_official, domain_scores_full, per_item, domain_margin_stats
    return domain_scores_official, domain_scores_full, domain_margin_stats
