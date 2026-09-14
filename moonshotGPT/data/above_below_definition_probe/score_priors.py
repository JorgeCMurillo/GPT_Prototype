#!/usr/bin/env python3
"""Measure above/below word and target-only priors with the probe checkpoint."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from evaluation.ewok import (per_token_conditional_log_likelihood,
                             per_token_unconditional_log_likelihood,
                             resolve_bos_token_id)


def describe(scores, tokenizer, text):
    values = scores.detach().float().cpu().tolist()
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == len(values)
    return {"text": text, "token_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "target_token_count": len(values),
            "token_log_probs": values,
            "mean_log_likelihood": sum(values) / len(values),
            "sum_log_likelihood": sum(values)}


def comparison(upper, lower):
    return {"upper": upper, "lower": lower,
            "mean_gap_upper_minus_lower": upper["mean_log_likelihood"] - lower["mean_log_likelihood"],
            "sum_gap_upper_minus_lower": upper["sum_log_likelihood"] - lower["sum_log_likelihood"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required for this checkpoint evaluation"
    out = args.results_dir
    with (ROOT / "generated/probes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    target_forms = Counter((row["Target1"], row["Target2"]) for row in rows
                           if row["direction"] in ("definition_to_word", "synonym_to_word"))
    assert len(target_forms) == 4
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    bare = ("above", "below")
    sentence_texts = sorted({text for pair in target_forms for text in pair})
    texts = [*bare, *sentence_texts]
    with torch.inference_mode():
        raw = per_token_unconditional_log_likelihood(model, tokenizer, texts,
                                                      device="cuda", batch_size=8)
    by_text = {text: describe(score, tokenizer, text) for text, score in zip(texts, raw)}
    bare_comparison = comparison(by_text["above"], by_text["below"])
    sentences = [{"pair_count_in_probe": count,
                  **comparison(by_text[upper], by_text[lower])}
                 for (upper, lower), count in target_forms.items()]

    prefixes = ("One object is", "The first object is", "The second object is",
                "The first item is", "The second item is")
    for prefix in prefixes:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        for word in bare:
            joined_ids = tokenizer.encode(prefix + " " + word, add_special_tokens=False)
            assert joined_ids[:len(prefix_ids)] == prefix_ids
    contexts = [prefix for prefix in prefixes for _ in bare]
    targets = [word for _ in prefixes for word in bare]
    with torch.inference_mode():
        conditional = per_token_conditional_log_likelihood(model, tokenizer, contexts,
            targets, device="cuda", batch_size=8)
    prefix_comparisons = []
    for index, prefix in enumerate(prefixes):
        upper_scores, lower_scores = conditional[2 * index:2 * index + 2]
        # The evaluator scores the token span after the context, including the
        # joining space. It can tokenize differently from a BOS-only bare word.
        upper = {"text": "above", "target_token_count": len(upper_scores),
                 "token_log_probs": upper_scores.detach().float().cpu().tolist(),
                 "mean_log_likelihood": float(upper_scores.float().mean()),
                 "sum_log_likelihood": float(upper_scores.float().sum())}
        lower = {"text": "below", "target_token_count": len(lower_scores),
                 "token_log_probs": lower_scores.detach().float().cpu().tolist(),
                 "mean_log_likelihood": float(lower_scores.float().mean()),
                 "sum_log_likelihood": float(lower_scores.float().sum())}
        prefix_comparisons.append({"prefix": prefix, **comparison(upper, lower)})
    result = {"model": str(Path(args.model).resolve()),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "probe_sha256": hashlib.sha256((ROOT / "generated/probes.jsonl").read_bytes()).hexdigest(),
              "device_name": torch.cuda.get_device_name(0),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "bos_token_id": resolve_bos_token_id(tokenizer),
              "convention": "BOS-only targets as written, no added leading space or EOS; mean log likelihood primary for full targets",
              "bare_word": bare_comparison,
              "prefix_conditioned_words": prefix_comparisons,
              "full_target_sentences": sentences}
    (out / "word_priors.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Above versus below: target-only likelihoods", "",
             "Same Qwen3 checkpoint and float32/eager scoring as the definition probe. Higher log likelihood means more preferred. BOS-only targets are scored exactly as written, without an added leading space or EOS. Full-sentence means match the probe's primary per-target-token reduction; full-sentence sums are also saved in `word_priors.json`.", "",
             "| Scoring condition | Above log likelihood | Below log likelihood | Above minus below | Target tokens (above/below) |",
             "|---|---:|---:|---:|---:|"]
    def table_row(label, item):
        a, b = item["upper"], item["lower"]
        lines.append(f"| {label} | {a['mean_log_likelihood']:.4f} | {b['mean_log_likelihood']:.4f} | {item['mean_gap_upper_minus_lower']:+.4f} | {a['target_token_count']}/{b['target_token_count']} |")
    table_row("Bare word from BOS", bare_comparison)
    for item in prefix_comparisons:
        table_row(f"After ‘{item['prefix']}’", item)
    lines += ["", "The bare-word row is a no-context prior. Prefix rows score the word after a neutral sentence fragment and are not BOS-only priors. A preference in either setting can help explain response bias, but neither by itself measures whether the model uses the above/below context correctly.", "",
              "## Full sentence targets from the probe", "",
              "| Above target | Below target | Pairs using form | Above mean | Below mean | Above minus below |",
              "|---|---|---:|---:|---:|---:|"]
    for item in sentences:
        a, b = item["upper"], item["lower"]
        lines.append(f"| {a['text']} | {b['text']} | {item['pair_count_in_probe']} | {a['mean_log_likelihood']:.4f} | {b['mean_log_likelihood']:.4f} | {item['mean_gap_upper_minus_lower']:+.4f} |")
    lines += ["", "Sentence-level prior gaps include all words and punctuation. They should be compared to the conditional full-target scores in `item_scores.csv`, rather than treated as the probability of the isolated relation word.", ""]
    (out / "word_priors.md").write_text("\n".join(lines))
    report_path = out / "report.md"
    link = "See [above/below word priors](word_priors.md) for BOS-only and prefix-conditioned likelihoods."
    if report_path.exists():
        current = report_path.read_text()
        if link not in current:
            report_path.write_text(current.rstrip() + "\n\n" + link + "\n")
    print(out / "word_priors.md", flush=True)


if __name__ == "__main__":
    main()
