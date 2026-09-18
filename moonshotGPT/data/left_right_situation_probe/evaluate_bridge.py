#!/usr/bin/env python3
"""Matched explicit-summary-bridge diagnostic on saved direct-label controls."""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import score_probes, summarize, write_csv

BRIDGE = "To summarize the positions:"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    base = args.baseline_dir
    metadata = json.loads((base / "summary.json").read_text())
    baseline = [json.loads(line) for line in (base / "item_scores.jsonl").read_text().splitlines()]
    baseline = [r for r in baseline if r["event_family"] == "direct_label_control"]
    inputs = {r["probe_id"]: r for r in map(json.loads, (base / "input_probes.jsonl").read_text().splitlines())}
    probes = []
    for original in baseline:
        row = dict(inputs[original["probe_id"]])
        row["base_probe_id"] = row["probe_id"]
        row["probe_id"] += "__summary_bridge"
        row["bridge"] = BRIDGE
        for i in (1, 2):
            row[f"Context{i}"] += " " + BRIDGE
            # Word counts follow the parent generator's regex convention.
            row[f"Context{i}_word_count"] = len(re.findall(r"\b\w+\b", row[f"Context{i}"]))
            assert row[f"Target{i}"] == original[f"Target{i}"]
        probes.append(row)
    assert len(probes) == 96 and torch.cuda.is_available()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "input_probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probes))
    (out / "baseline_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in baseline))
    tokenizer = AutoTokenizer.from_pretrained(metadata["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(metadata["model"], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    items, unique = score_probes(probes, model, tokenizer, args.batch_size, out / "token_scores.jsonl")
    (out / "item_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    write_csv(out / "item_scores.csv", items)
    lookup = {r["probe_id"]: r for r in baseline}
    matches = []
    for row in items:
        before = lookup[row["base_probe_id"]]
        matches.append({"base_probe_id": before["probe_id"], "variant_probe_id": row["probe_id"],
            "entity_order_match": row["context_entity_order"] == row["target_entity_order"],
            "baseline_accuracy": before["binary_accuracy"], "bridge_accuracy": row["binary_accuracy"],
            "baseline_both_correct": before["both_correct"], "bridge_both_correct": row["both_correct"],
            "left_context_choice_changed": before["prediction_in_left_context"] != row["prediction_in_left_context"],
            "right_context_choice_changed": before["prediction_in_right_context"] != row["prediction_in_right_context"]})
    write_csv(out / "matched_changes.csv", matches)
    groups = {}
    for name, same in (("same_entity_order", True), ("reversed_entity_order", False), ("all", None)):
        select = lambda r: same is None or (r["context_entity_order"] == r["target_entity_order"]) == same
        groups[name] = {"baseline": summarize([r for r in baseline if select(r)]),
                        "bridge": summarize([r for r in items if select(r)])}
    # Reconstruct scores and decisions from saved token-level values.
    token_lookup = {}
    for line in (out / "token_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        token_lookup[row["context"], row["target"]] = float(torch.tensor(row["token_log_probs"], dtype=torch.float32).mean())
    for row in items:
        for c in (1, 2):
            for t in (1, 2):
                assert row[f"S{c}{t}"] == token_lookup[row[f"Context{c}"], row[f"Target{t}"]]
        assert row["left_correct"] == (row["S11"] > row["S12"])
        assert row["right_correct"] == (row["S22"] > row["S21"])
    summary = {"model": metadata["model"], "baseline_dir": str(base.resolve()), "bridge": BRIDGE,
        "created_utc": datetime.now(timezone.utc).isoformat(), "n_pairs": len(items),
        "n_unique_sequences": unique, "score_reduction": "mean", "pmi": False,
        "dtype": "float32", "attention": "eager", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "baseline_scores_sha256": hashlib.sha256((base / "item_scores.jsonl").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((out / "input_probes.jsonl").read_bytes()).hexdigest(),
        "groups": groups, "validation": "saved token means and choices reconstructed successfully"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Explicit summary bridge: direct-label controls", "",
        f'Appended **{BRIDGE}** to each context, keeping both candidate answers unchanged.', "",
        "Same Qwen3 359M step-19,500 checkpoint and raw mean full-target likelihood scoring as the baseline; no PMI. The bridge is conditioning text and is not included in the scored target.", "",
        "| Entity order | Pairs | Baseline accuracy | Bridge accuracy | Baseline both correct | Bridge both correct |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, data in groups.items():
        a, b = data["baseline"], data["bridge"]
        report.append(f"| {name} | {a['n_pairs']} | {a['binary_accuracy']:.2%} | {b['binary_accuracy']:.2%} | {a['both_correct_fraction']:.2%} | {b['both_correct_fraction']:.2%} |")
    report.extend(["", "This is a matched prompt-format diagnostic on 96 direct-label pairs, not a rerun of the physical-situation families. A bridge effect supports sensitivity to how restatement is prompted; it does not uniquely establish repetition as the mechanism. The bridge also changes length and discourse cues.", "", summary["validation"], ""])
    (out / "report.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
