#!/usr/bin/env python3
"""Evaluate causal language models on BLiMP-fast with BOS-prepended scoring.

This module loads local BLiMP JSONL subsets, compares good/bad sentence pairs,
and reports aggregate, per-subset, and per-item grammar metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
_DEFAULT_DIR = _PROJECT_DIR / "blimp_fast"
_LEGACY_SRC = Path(
    "/home/jorge/tokenPred/babylm_10m/evaluation-pipeline-2025/evaluation_data/fast_eval/blimp_fast"
)


def _normalize_optional_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _blimp_source_candidates(data_dir: Optional[str] = None) -> List[Path]:
    candidates: List[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))

    env_src = os.environ.get("BLIMP_SRC")
    if env_src:
        candidates.append(Path(env_src))

    candidates.extend([_DEFAULT_DIR, _THIS_DIR / "blimp_fast", _LEGACY_SRC])

    deduped: List[Path] = []
    seen = set()
    for c in candidates:
        key = str(c.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _resolve_blimp_dir(data_dir: Optional[str] = None) -> Path:
    errors = []
    for src in _blimp_source_candidates(data_dir):
        try:
            src = src.expanduser().resolve()
            if not src.is_dir():
                continue
            jsonl_files = sorted(src.glob("*.jsonl"))
            if not jsonl_files:
                errors.append(f"{src}: no .jsonl files found")
                continue
            return src
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    searched = ", ".join(str(c) for c in _blimp_source_candidates(data_dir))
    details = "; ".join(errors) if errors else "No candidate source exists."
    raise FileNotFoundError(
        f"Could not resolve BLiMP data directory. Searched: {searched}. Details: {details}"
    )


def load_blimp_records(
    data_dir: Optional[str] = None,
    max_examples_per_subset: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Load BLiMP-fast JSONL records from a local directory."""
    src_dir = _resolve_blimp_dir(data_dir)
    max_examples_per_subset = _normalize_optional_limit(max_examples_per_subset)
    records: List[Dict[str, Any]] = []

    for fp in sorted(src_dir.glob("*.jsonl")):
        kept = 0
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = json.loads(line)
                subset_uid = str(raw.get("UID") or fp.stem)
                field = str(raw.get("field") or "supplement").strip() or "supplement"
                if field == "syntax_semantics":
                    field = "syntax/semantics"
                linguistics_term = str(raw.get("linguistics_term") or "supplement").strip() or "supplement"
                records.append(
                    {
                        "row_index": int(len(records)),
                        "subset_row_index": int(kept),
                        "subset_uid": subset_uid,
                        "field": field,
                        "linguistics_term": linguistics_term,
                        "pair_id": raw.get("pair_id", raw.get("row")),
                        "source_file": fp.name,
                        "sentence_good": str(raw["sentence_good"]),
                        "sentence_bad": str(raw["sentence_bad"]),
                    }
                )
                kept += 1
                if max_examples_per_subset is not None and kept >= max_examples_per_subset:
                    break

    if not records:
        raise FileNotFoundError(f"No BLiMP records loaded from {src_dir}")
    return records, src_dir


def _resolve_device(model, device=None):
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_bos_token_id(tokenizer) -> int:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return int(bos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise RuntimeError(
        "Tokenizer must define bos_token_id or eos_token_id for BOS-style full-sentence scoring."
    )


def per_token_sentence_log_likelihood(
    model,
    tokenizer,
    input_texts: Sequence[str],
    *,
    batch_size: int = 8,
    device=None,
) -> List[torch.Tensor]:
    """Return per-token log-probs for complete sentences with a prepended BOS-style token."""
    device = _resolve_device(model, device=device)
    bos_token_id = _resolve_bos_token_id(tokenizer)
    all_results: List[torch.Tensor] = []

    for start in range(0, len(input_texts), int(batch_size)):
        batch_texts = list(input_texts[start : start + int(batch_size)])
        inputs = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs.input_ids.to(device)
        attn_mask = inputs.attention_mask.to(device)
        batch_rows = int(input_ids.shape[0])

        bos_tensor = torch.full((batch_rows, 1), bos_token_id, dtype=input_ids.dtype, device=device)
        bos_attn = torch.ones((batch_rows, 1), dtype=attn_mask.dtype, device=device)
        model_input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        model_attn_mask = torch.cat([bos_attn, attn_mask], dim=1)

        with torch.no_grad():
            outputs = model(input_ids=model_input_ids, attention_mask=model_attn_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs["logits"]

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = model_input_ids[:, 1:]
        token_logp = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1).detach().cpu()
        orig_attn_mask = inputs.attention_mask.detach().cpu()

        for row_idx in range(batch_rows):
            valid_length = int(orig_attn_mask[row_idx].sum().item())
            all_results.append(token_logp[row_idx, :valid_length])

    return all_results


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


def _aggregate_accuracy_by_key(per_item_records, key: str) -> Dict[str, float]:
    buckets: Dict[str, Dict[str, int]] = {}
    for rec in per_item_records:
        group = str(rec.get(key, "<NA>"))
        bucket = buckets.setdefault(group, {"correct": 0, "n": 0})
        bucket["correct"] += 1 if bool(rec.get("correct", False)) else 0
        bucket["n"] += 1

    out: Dict[str, float] = {}
    accs: List[float] = []
    for group in sorted(buckets.keys()):
        bucket = buckets[group]
        n = int(bucket["n"])
        if n <= 0:
            continue
        acc = float(bucket["correct"] / n)
        out[group] = acc
        accs.append(acc)
    if accs:
        out["average"] = float(np.mean(accs))
    return out


def _aggregate_margin_stats_by_key(per_item_records, key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for rec in per_item_records:
        group = str(rec.get(key, "<NA>"))
        bucket = buckets.setdefault(group, {"margin": [], "correct": [], "tie": []})
        bucket["margin"].append(float(rec["margin"]))
        bucket["correct"].append(1.0 if bool(rec["correct"]) else 0.0)
        bucket["tie"].append(1.0 if bool(rec["near_tie"]) else 0.0)

    out: Dict[str, Dict[str, float]] = {}
    accs: List[float] = []
    abs_ms: List[float] = []
    tie_rates: List[float] = []
    for group in sorted(buckets.keys()):
        margins = np.asarray(buckets[group]["margin"], dtype=np.float64)
        correct = np.asarray(buckets[group]["correct"], dtype=np.float64)
        ties = np.asarray(buckets[group]["tie"], dtype=np.float64)
        if margins.size == 0:
            continue
        stats = {
            "n": int(margins.size),
            "accuracy": float(correct.mean()),
            "mean_signed_m": float(margins.mean()),
            "mean_abs_m": float(np.abs(margins).mean()),
            "tie_rate": float(ties.mean()),
        }
        out[group] = stats
        accs.append(stats["accuracy"])
        abs_ms.append(stats["mean_abs_m"])
        tie_rates.append(stats["tie_rate"])

    if out:
        total_n = sum(int(v["n"]) for v in out.values())
        all_margins = np.asarray([float(rec["margin"]) for rec in per_item_records], dtype=np.float64)
        all_correct = np.asarray([1.0 if bool(rec["correct"]) else 0.0 for rec in per_item_records], dtype=np.float64)
        all_ties = np.asarray([1.0 if bool(rec["near_tie"]) else 0.0 for rec in per_item_records], dtype=np.float64)
        out["average"] = {
            "n": int(total_n),
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "mean_signed_m": float(all_margins.mean()) if all_margins.size else 0.0,
            "mean_abs_m": float(np.mean(abs_ms)) if abs_ms else 0.0,
            "tie_rate": float(np.mean(tie_rates)) if tie_rates else 0.0,
            "accuracy_item_weighted": float(all_correct.mean()) if all_correct.size else 0.0,
            "tie_rate_item_weighted": float(all_ties.mean()) if all_ties.size else 0.0,
        }
    return out


def blimp_per_item_records(
    model,
    tokenizer,
    *,
    records: Sequence[Dict[str, Any]],
    batch_size: int = 8,
    score_reduction: str = "sum",
    margin_eps: float = 1e-6,
    device=None,
) -> List[Dict[str, Any]]:
    score_reduction = _validate_score_reduction(score_reduction)
    margin_eps = float(margin_eps)
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    good_scores = per_token_sentence_log_likelihood(
        model,
        tokenizer,
        [rec["sentence_good"] for rec in records],
        batch_size=batch_size,
        device=device,
    )
    bad_scores = per_token_sentence_log_likelihood(
        model,
        tokenizer,
        [rec["sentence_bad"] for rec in records],
        batch_size=batch_size,
        device=device,
    )

    per_item: List[Dict[str, Any]] = []
    for rec, good_logps, bad_logps in zip(records, good_scores, bad_scores):
        good_score = _reduce_token_logps(good_logps, score_reduction)
        bad_score = _reduce_token_logps(bad_logps, score_reduction)
        margin = float(good_score - bad_score)
        per_item.append(
            {
                "row_index": int(rec["row_index"]),
                "subset_row_index": int(rec["subset_row_index"]),
                "subset_uid": str(rec["subset_uid"]),
                "field": str(rec["field"]),
                "linguistics_term": str(rec["linguistics_term"]),
                "pair_id": rec.get("pair_id"),
                "source_file": str(rec["source_file"]),
                "score_reduction": score_reduction,
                "score_good": float(good_score),
                "score_bad": float(bad_score),
                "margin": margin,
                "correct": bool(margin > 0.0),
                "tie": bool(margin == 0.0),
                "near_tie": bool(abs(margin) < margin_eps),
            }
        )
    return per_item


def evaluate(
    model,
    tokenizer,
    *,
    batch_size: int = 8,
    return_per_item: bool = False,
    score_reduction: str = "sum",
    margin_eps: float = 1e-6,
    data_dir: Optional[str] = None,
    max_examples_per_subset: Optional[int] = None,
    records: Optional[Sequence[Dict[str, Any]]] = None,
    source_path: Optional[Path] = None,
    device=None,
) -> Dict[str, Any]:
    score_reduction = _validate_score_reduction(score_reduction)
    if records is None:
        loaded_records, loaded_source = load_blimp_records(
            data_dir=data_dir,
            max_examples_per_subset=max_examples_per_subset,
        )
        records = loaded_records
        source_path = loaded_source
    elif source_path is None:
        source_path = Path("<preloaded>")

    print(
        "evaluating model on BLiMP dataset... "
        f"(score_reduction={score_reduction}, num_examples={len(records)}, source={source_path})"
    )
    per_item = blimp_per_item_records(
        model,
        tokenizer,
        records=records,
        batch_size=batch_size,
        score_reduction=score_reduction,
        margin_eps=margin_eps,
        device=device,
    )

    correct = np.asarray([1.0 if rec["correct"] else 0.0 for rec in per_item], dtype=np.float64)
    margins = np.asarray([float(rec["margin"]) for rec in per_item], dtype=np.float64)
    near_ties = np.asarray([1.0 if rec["near_tie"] else 0.0 for rec in per_item], dtype=np.float64)

    by_uid = _aggregate_accuracy_by_key(per_item, "subset_uid")
    by_field = _aggregate_accuracy_by_key(per_item, "field")
    by_linguistics_term = _aggregate_accuracy_by_key(per_item, "linguistics_term")
    margin_stats_by_uid = _aggregate_margin_stats_by_key(per_item, "subset_uid")

    result = {
        "score_reduction": score_reduction,
        "source": str(source_path),
        "num_examples": int(len(per_item)),
        "num_subsets": int(len({rec["subset_uid"] for rec in per_item})),
        "accuracy": float(correct.mean()) if correct.size else 0.0,
        "accuracy_macro_uid": float(by_uid.get("average", 0.0)),
        "mean_signed_m": float(margins.mean()) if margins.size else 0.0,
        "mean_abs_m": float(np.abs(margins).mean()) if margins.size else 0.0,
        "tie_rate": float(near_ties.mean()) if near_ties.size else 0.0,
        "by_uid": by_uid,
        "by_field": by_field,
        "by_linguistics_term": by_linguistics_term,
        "margin_stats_by_uid": margin_stats_by_uid,
    }
    if return_per_item:
        result["per_item"] = per_item
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on BLiMP-fast")
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--data-dir", default="", help="Directory containing BLiMP-fast JSONL files")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for sentence scoring")
    parser.add_argument(
        "--max-examples-per-subset",
        type=int,
        default=0,
        help="Optional cap per BLiMP subset for faster smoke tests (0 uses all)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device (default: auto)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local files/cache only",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the combined BLiMP metrics as JSON",
    )
    return parser.parse_args()


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = _parse_args()
    device = _get_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    start_time = time.time()
    metrics_sum = evaluate(
        model,
        tokenizer,
        batch_size=args.batch_size,
        score_reduction="sum",
        data_dir=(args.data_dir or None),
        max_examples_per_subset=args.max_examples_per_subset,
        device=device,
    )
    metrics_mean = evaluate(
        model,
        tokenizer,
        batch_size=args.batch_size,
        score_reduction="mean",
        data_dir=(args.data_dir or None),
        max_examples_per_subset=args.max_examples_per_subset,
        device=device,
    )
    elapsed = time.time() - start_time

    print("BLiMP-fast results")
    print(f"  Model: {args.model}")
    print(f"  Source: {metrics_sum['source']}")
    print(f"  Num examples: {metrics_sum['num_examples']}")
    print(f"  Num subsets: {metrics_sum['num_subsets']}")
    print(
        f"  Sum accuracy: {metrics_sum['accuracy']:.4f} "
        f"(UID macro {metrics_sum['accuracy_macro_uid']:.4f})"
    )
    print(
        f"  Mean accuracy: {metrics_mean['accuracy']:.4f} "
        f"(UID macro {metrics_mean['accuracy_macro_uid']:.4f})"
    )
    print(f"  Elapsed: {elapsed:.2f}s")

    if args.output_json:
        payload = {
            "model": args.model,
            "device": str(device),
            "elapsed_seconds": float(elapsed),
            "sum": metrics_sum,
            "mean": metrics_mean,
        }
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
