#!/usr/bin/env python3
"""Evaluate the direct EWoK context-choice prompt on a local HF checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from moonshotGPT.evaluation import ewok  # noqa: E402
from moonshotGPT.evaluation.ewok_data import load_ewok_df  # noqa: E402
from moonshotGPT.hf_ewok_eval import prompted_ewok_eval as prompted  # noqa: E402


PROMPT_FILE = PROJECT_DIR / "hf_ewok_eval/prompts/qwen3_context_sensitivity/context_choice_answer_only.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("fast", "full"), default="fast")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--context-order", choices=("original", "alternate", "random"), default="alternate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comparison-summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max new tokens must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df, source = load_ewok_df(args.variant)
    df = df.reset_index(drop=True).convert_dtypes()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        local_files_only=True,
    ).to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    started = time.monotonic()
    records = prompted.prompted_ewok_score_records_all_methods(
        model=model,
        tokenizer=tokenizer,
        ewok_df=df,
        template_text=PROMPT_FILE.read_text(encoding="utf-8"),
        prompt_template_name="context_choice_answer_only_32tok",
        prompt_template_source=str(PROMPT_FILE),
        inference_mode="context_choice_generate",
        batch_size=args.batch_size,
        score_reduction="mean",
        margin_eps=1e-6,
        answer_separator=" ",
        max_new_tokens=args.max_new_tokens,
        target_permutation_mode=args.context_order,
        target_permutation_seed=args.seed,
        store_prompts=False,
        show_progress=True,
    )
    elapsed = time.monotonic() - started
    metrics = prompted._summarize_records_with_shared_logic(
        shared_ewok_module=ewok,
        ewok_df=df,
        records=records,
        margin_eps=1e-6,
    )
    context = metrics[ewok.EWOK_CONTEXT_SENSITIVITY]
    completion = metrics[ewok.BABYLM_COMPLETION_CHOICE]
    valid_t1 = sum(bool(row["response_valid_t1_context_choice"]) for row in records)
    valid_t2 = sum(bool(row["response_valid_t2_context_choice"]) for row in records)
    both_valid = sum(
        bool(row["response_valid_t1_context_choice"] and row["response_valid_t2_context_choice"])
        for row in records
    )
    choices_t1 = Counter(str(row["displayed_predicted_context_t1"]) for row in records)
    choices_t2 = Counter(str(row["displayed_predicted_context_t2"]) for row in records)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "ewok_variant": args.variant,
        "ewok_source": str(source),
        "num_items": len(records),
        "num_domains": len(df["Domain"].unique()),
        "prompt_file": str(PROMPT_FILE),
        "method": "context_choice_generate",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "context_order": args.context_order,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "response_validity": {
            "t1_valid_count": valid_t1,
            "t2_valid_count": valid_t2,
            "both_valid_count": both_valid,
            "t1_displayed_choices": dict(choices_t1),
            "t2_displayed_choices": dict(choices_t2),
        },
        "context_sensitivity": context,
        "completion_choice": completion,
    }
    if args.comparison_summary:
        comparison = json.loads(Path(args.comparison_summary).read_text(encoding="utf-8"))
        if len(records) != int(comparison["num_items"]) or args.variant != comparison["ewok_variant"]:
            raise ValueError("Comparison summary uses different EWoK rows")
        summary["comparison_summary"] = str(Path(args.comparison_summary).resolve())
        summary["comparison"] = {
            "raw_context_t1": comparison["regular"]["macro_by_domain"]["context_t1"],
            "true_false_context_t1": comparison["true_false"]["macro_by_domain"]["context_t1"],
            "prompted_context_t1": context["domain_scores_full"]["average"][0],
            "raw_context_both_sides_mean": comparison["regular"]["macro_by_domain"]["context_both_sides_mean"],
            "true_false_context_both_sides_mean": comparison["true_false"]["macro_by_domain"]["context_both_sides_mean"],
            "prompted_context_both_sides_mean": context["domain_margin_stats"]["average"]["acc_combined"],
        }
    with (output_dir / "ewok_items.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "num_items": len(records),
        "context_t1": context["domain_scores_full"]["average"][0],
        "context_t2": context["domain_scores_full"]["average"][1],
        "context_both_sides_mean": context["domain_margin_stats"]["average"]["acc_combined"],
        "response_validity": summary["response_validity"],
        "comparison": summary.get("comparison"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
