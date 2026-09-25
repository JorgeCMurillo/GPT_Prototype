#!/usr/bin/env python3
"""Score 1 versus 2 after the EWoK direct context-choice prompt."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from moonshotGPT.evaluation.ewok_data import load_ewok_df  # noqa: E402
from moonshotGPT.hf_ewok_eval import prompted_ewok_eval as prompted  # noqa: E402


PROMPT_FILE = PROJECT_DIR / "hf_ewok_eval/prompts/qwen3_context_sensitivity/context_choice_answer_only.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("fast", "full"), default="fast")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df, source = load_ewok_df(args.variant)
    df = df.reset_index(drop=True)
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    prompts_t1, prompts_t2, swaps_t1, swaps_t2 = prompted._build_domain_context_choice_prompts(
        df=df.reset_index(),
        template_text=prompt_text,
        target_permutation_mode="alternate",
        target_permutation_seed=0,
    )
    prompts = prompts_t1 + prompts_t2
    print(f"Scoring {len(prompts)} prompts with both answer labels", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        local_files_only=True,
    ).to(torch.device(args.device)).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    score_1 = prompted._conditional_target_token_logps(
        model, tokenizer,
        prefixes=prompts,
        targets=["1"] * len(prompts),
        answer_separator="",
        batch_size=args.batch_size,
    )
    score_2 = prompted._conditional_target_token_logps(
        model, tokenizer,
        prefixes=prompts,
        targets=["2"] * len(prompts),
        answer_separator="",
        batch_size=args.batch_size,
    )
    scores = [(float(a.sum().item()), float(b.sum().item())) for a, b in zip(score_1, score_2, strict=True)]
    n = len(df)
    records = []
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for index, row in df.iterrows():
        item = {"row_index": int(index), "domain": str(row["Domain"])}
        for side, offset, swaps in (("t1", 0, swaps_t1), ("t2", n, swaps_t2)):
            a, b = scores[index + offset]
            predicted = 1 if a > b else 2 if b > a else None
            gold = (2 if swaps[index] else 1) if side == "t1" else (1 if swaps[index] else 2)
            item[side] = {
                "logp_1": a,
                "logp_2": b,
                "displayed_gold": gold,
                "displayed_prediction": predicted,
                "context_order_swapped": bool(swaps[index]),
                "correct": predicted == gold,
            }
        records.append(item)
        by_domain[item["domain"]].append(item)

    domain_scores = {
        domain: {
            side: sum(item[side]["correct"] for item in items) / len(items)
            for side in ("t1", "t2")
        }
        for domain, items in sorted(by_domain.items())
    }
    macro = {
        side: sum(scores_by_side[side] for scores_by_side in domain_scores.values()) / len(domain_scores)
        for side in ("t1", "t2")
    }
    macro["both_sides_mean"] = (macro["t1"] + macro["t2"]) / 2
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "ewok_variant": args.variant,
        "ewok_source": str(source),
        "num_items": n,
        "prompt_file": str(PROMPT_FILE),
        "answer_labels": ["1", "2"],
        "answer_separator": "",
        "context_order": "alternate",
        "dtype": args.dtype,
        "device": args.device,
        "macro_by_domain": macro,
        "domains": domain_scores,
        "displayed_prediction_counts": {
            side: {str(choice): sum(item[side]["displayed_prediction"] == choice for item in records) for choice in (1, 2, None)}
            for side in ("t1", "t2")
        },
    }
    with (output_dir / "ewok_items.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_items": n, "macro_by_domain": macro,
                      "displayed_prediction_counts": summary["displayed_prediction_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
