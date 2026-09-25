#!/usr/bin/env python3
"""Compare raw EWoK continuation scores with prompted True/False scores."""

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

from moonshotGPT.evaluation import ewok  # noqa: E402
from moonshotGPT.evaluation.ewok_data import load_ewok_df  # noqa: E402
from moonshotGPT.hf_ewok_eval import prompted_ewok_eval as prompted  # noqa: E402


PROMPT_FILE = PROJECT_DIR / "hf_ewok_eval/prompts/qwen3_statement_logprob/statement_support_true_false.txt"
PAIRS = (("C1T1", "Context1", "Target1"), ("C1T2", "Context1", "Target2"),
         ("C2T2", "Context2", "Target2"), ("C2T1", "Context2", "Target1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("fast", "full"), default="fast")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    return parser.parse_args()


def summarize(records: list[dict], method: str) -> dict:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_domain[record["domain"]].append(record)

    keys = ("completion_c1", "completion_c2", "context_t1", "context_t2")

    def rates(rows: list[dict]) -> dict[str, float]:
        result = {key: sum(row[method][key] for row in rows) / len(rows) for key in keys}
        result["completion_both_sides_mean"] = (result["completion_c1"] + result["completion_c2"]) / 2
        result["context_both_sides_mean"] = (result["context_t1"] + result["context_t2"]) / 2
        return result

    domain_rates = {domain: rates(rows) for domain, rows in sorted(by_domain.items())}
    macro = {
        key: sum(domain_rates[domain][key] for domain in domain_rates) / len(domain_rates)
        for key in (*keys, "completion_both_sides_mean", "context_both_sides_mean")
    }
    return {"macro_by_domain": macro, "micro_by_item": rates(records), "domains": domain_rates}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df, source = load_ewok_df(args.variant)
    df = df.reset_index(drop=True)
    template = PROMPT_FILE.read_text(encoding="utf-8")

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

    records = [
        {"row_index": int(index), "domain": str(row.Domain), "regular": {}, "true_false": {}}
        for index, row in df.iterrows()
    ]
    for pair_name, context_key, target_key in PAIRS:
        print(f"Scoring regular {pair_name}", flush=True)
        token_logps = ewok.per_token_conditional_log_likelihood(
            model,
            tokenizer,
            df[context_key].astype(str).tolist(),
            df[target_key].astype(str).tolist(),
            device=device,
            batch_size=args.batch_size,
        )
        for record, token_scores in zip(records, token_logps, strict=True):
            record["regular"][pair_name] = float(token_scores.mean().item())

        print(f"Scoring True/False {pair_name}", flush=True)
        prompts = [
            prompted._render_statement_prompt(
                template_text=template,
                context_text=context,
                statement_text=target,
            )
            for context, target in zip(df[context_key], df[target_key], strict=True)
        ]
        values = prompted._statement_logprob_values(
            model,
            tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            score_reduction="mean",
            answer_separator=" ",
        )
        for record, value in zip(records, values, strict=True):
            record["true_false"][pair_name] = value

    for record in records:
        for method in ("regular", "true_false"):
            scores = record[method]
            if method == "true_false":
                scores = {key: value["true_false_margin"] for key, value in scores.items()}
                record[method + "_margins"] = scores
            record[method + "_correct"] = {
                "completion_c1": scores["C1T1"] > scores["C1T2"],
                "completion_c2": scores["C2T2"] > scores["C2T1"],
                "context_t1": scores["C1T1"] > scores["C2T1"],
                "context_t2": scores["C2T2"] > scores["C1T2"],
            }

    metric_rows = [
        {
            "domain": record["domain"],
            "regular": record["regular_correct"],
            "true_false": record["true_false_correct"],
        }
        for record in records
    ]
    regular_summary = summarize(metric_rows, "regular")
    true_false_summary = summarize(metric_rows, "true_false")
    paired = {}
    tie_counts = {"regular": {}, "true_false": {}}
    for key in ("completion_c1", "completion_c2", "context_t1", "context_t2"):
        left, right = {
            "completion_c1": ("C1T1", "C1T2"),
            "completion_c2": ("C2T2", "C2T1"),
            "context_t1": ("C1T1", "C2T1"),
            "context_t2": ("C2T2", "C1T2"),
        }[key]
        tie_counts["regular"][key] = sum(row["regular"][left] == row["regular"][right] for row in records)
        tie_counts["true_false"][key] = sum(
            row["true_false_margins"][left] == row["true_false_margins"][right] for row in records
        )
        paired[key] = {
            "both_correct": sum(row["regular"][key] and row["true_false"][key] for row in metric_rows),
            "regular_only": sum(row["regular"][key] and not row["true_false"][key] for row in metric_rows),
            "true_false_only": sum(not row["regular"][key] and row["true_false"][key] for row in metric_rows),
            "both_wrong": sum(not row["regular"][key] and not row["true_false"][key] for row in metric_rows),
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "ewok_variant": args.variant,
        "ewok_source": str(source),
        "num_items": len(records),
        "num_domains": len(regular_summary["domains"]),
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": args.dtype,
        "regular_scoring": "mean log P(target tokens | context); context + space + target",
        "true_false_scoring": "mean log P(' True' | prompt) - mean log P(' False' | prompt)",
        "true_false_prompt_file": str(PROMPT_FILE),
        "regular": regular_summary,
        "true_false": true_false_summary,
        "paired_counts": paired,
        "tie_counts": tie_counts,
    }
    with (output_dir / "ewok_items.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "regular_macro": regular_summary["macro_by_domain"],
        "true_false_macro": true_false_summary["macro_by_domain"],
        "paired_counts": paired,
        "tie_counts": tie_counts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
