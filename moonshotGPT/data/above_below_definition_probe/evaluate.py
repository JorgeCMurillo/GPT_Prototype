#!/usr/bin/env python3
"""Score above/below definition and synonym pairs with mean target-token likelihood."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from evaluation.ewok import per_token_conditional_log_likelihood
from data.spatial_bias_report.generate import generate_one as generate_bias_table
from data.above_below_definition_probe.analyze_entity_order import analyze as analyze_entity_order
from data.above_below_definition_probe.render_bias import render as render_bias


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    assert n
    return {
        "n_pairs": n,
        "n_judgments": 2 * n,
        "choice_accuracy": sum(r["above_correct"] + r["below_correct"] for r in rows) / (2 * n),
        "above_gold_accuracy": sum(r["above_correct"] for r in rows) / n,
        "below_gold_accuracy": sum(r["below_correct"] for r in rows) / n,
        "both_correct_fraction": sum(r["paired_success"] for r in rows) / n,
        "above_chosen_fraction": sum((r["above_margin"] > 0) + (r["below_margin"] < 0)
                                     for r in rows) / (2 * n),
        "context_sensitivity_accuracy": sum(r["context_sensitivity_accuracy"]
                                            for r in rows) / n,
        "exact_choice_ties": sum((r["above_margin"] == 0) + (r["below_margin"] == 0)
                                 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required for this checkpoint evaluation."
    probe_path = ROOT / "generated/probes.jsonl"
    probes = [json.loads(line) for line in probe_path.read_text().splitlines()]
    manifest = json.loads((ROOT / "generated/manifest.json").read_text())
    assert len(probes) == manifest["paired_probes"]
    assert len({p["probe_id"] for p in probes}) == len(probes)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for path in (probe_path, ROOT / "components.json", ROOT / "generated/manifest.json",
                 ROOT / "generated/entity_order_matches.csv"):
        (args.out_dir / ("input_" + path.name)).write_bytes(path.read_bytes())

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters on "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    contexts, targets, keys = [], [], []
    for probe in probes:
        for c, t in ((1, 1), (1, 2), (2, 2), (2, 1)):
            context, target = probe[f"Context{c}"], probe[f"Target{t}"]
            ctx_ids = tokenizer.encode(context, add_special_tokens=False)
            joined_ids = tokenizer.encode(context + " " + target, add_special_tokens=False)
            assert joined_ids[:len(ctx_ids)] == ctx_ids, "Context tokenization boundary changed"
            assert len(joined_ids) > len(ctx_ids)
            contexts.append(context)
            targets.append(target)
            keys.append((probe["probe_id"], f"S{c}{t}"))
    with torch.inference_mode():
        token_scores = per_token_conditional_log_likelihood(
            model, tokenizer, contexts, targets, device="cuda", batch_size=args.batch_size)
    assert len(token_scores) == 4 * len(probes)
    lookup = {}
    with (args.out_dir / "token_scores.jsonl").open("w") as stream:
        for (probe_id, combination), scores in zip(keys, token_scores):
            values = scores.detach().float().cpu()
            assert len(values) and torch.isfinite(values).all()
            lookup[probe_id, combination] = values
            stream.write(json.dumps({"probe_id": probe_id, "combination": combination,
                                     "target_token_log_probs": values.tolist()}) + "\n")
    results = []
    for probe in probes:
        scores = {k: float(lookup[probe["probe_id"], k].mean())
                  for k in ("S11", "S12", "S22", "S21")}
        above_margin = scores["S11"] - scores["S12"]
        below_margin = scores["S22"] - scores["S21"]
        context_above = scores["S11"] - scores["S21"]
        context_below = scores["S22"] - scores["S12"]
        results.append({
            **probe, "score_reduction": "mean", **scores,
            **{f"{k}_target_token_count": len(lookup[probe["probe_id"], k]) for k in scores},
            "above_margin": above_margin, "below_margin": below_margin,
            "above_correct": above_margin > 0, "below_correct": below_margin > 0,
            "completion_choice_accuracy": ((above_margin > 0) + (below_margin > 0)) / 2,
            "paired_success": above_margin > 0 and below_margin > 0,
            "context_sensitivity_above_margin": context_above,
            "context_sensitivity_below_margin": context_below,
            "context_sensitivity_accuracy": ((context_above > 0) + (context_below > 0)) / 2,
        })
    write_csv(args.out_dir / "item_scores.csv", results)
    (args.out_dir / "item_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))

    groupings = (("probe_family", "direction"),
                 ("probe_family", "direction", "design_block"),
                 ("probe_family", "direction", "phrase_pair_id"),
                 ("probe_family", "direction", "definition_structure_id"),
                 ("probe_family", "direction", "context_stem_id"),
                 ("probe_family", "direction", "design_block", "context_entity_order"),
                 ("probe_family", "direction", "design_block", "entity_noun_id"))
    grouped = []
    for factors in groupings:
        buckets = defaultdict(list)
        for row in results:
            buckets[tuple(row[f] for f in factors)].append(row)
        for key, subset in buckets.items():
            grouped.append({"grouping": "+".join(factors), "group": "|".join(key),
                            **summarize(subset)})
    write_csv(args.out_dir / "grouped_scores.csv", grouped)
    families = {}
    for family in ("literal_definition", "lexical_synonym"):
        selected = [r for r in results if r["probe_family"] == family]
        by_direction = {direction: summarize([r for r in selected if r["direction"] == direction])
                        for direction in sorted({r["direction"] for r in selected})}
        families[family] = {"by_direction": by_direction,
                            "direction_balanced_accuracy": sum(
                                s["choice_accuracy"] for s in by_direction.values()) / len(by_direction)}
    summary = {
        "model": str(Path(args.model).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "components_sha256": hashlib.sha256((ROOT / "components.json").read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0), "dtype": "float32",
        "attention_implementation": "eager", "batch_size": args.batch_size,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "primary_reduction": "mean", "n_pairs": len(results), "families": families,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Above/below definition and synonym probe", "",
              "Qwen3 step-19,500 checkpoint; binary choice between two full target sentences, scored by mean target-token log likelihood. Each pair includes an above-compatible and a below-compatible context. Exact ties count as incorrect.", "",
              "| Probe family | Direction | Pairs | Choice accuracy | Above gold | Below gold | Both correct | Above choice share | Context sensitivity |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in grouped:
        if row["grouping"] != "probe_family+direction":
            continue
        family, direction = row["group"].split("|")
        report.append(f"| {family} | {direction} | {row['n_pairs']} | " + " | ".join(
            f"{row[k]:.2%}" for k in ("choice_accuracy", "above_gold_accuracy",
                "below_gold_accuracy", "both_correct_fraction", "above_chosen_fraction",
                "context_sensitivity_accuracy")) + " |")
    report += ["", "Direction-balanced choice accuracy averages the two mapping directions within each family. The over/under lexical-synonym rows are separate because those words can suggest vertical alignment. The entity-order extension adds cases where the second entity is the subject; these invert the relational phrase while keeping the first entity as the target subject. Repeated stems, nouns, and structures are related stimuli, not independent semantic cases.",
               "", "See `bias_table.md` for constant-answer and paired-context behavior; `grouped_scores.csv` and `bias_table.csv` include phrase-pair, noun, structure, and entity-order splits.", ""]
    (args.out_dir / "report.md").write_text("\n".join(report))
    generate_bias_table(args.out_dir)
    analyze_entity_order(args.out_dir)
    render_bias(args.out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
