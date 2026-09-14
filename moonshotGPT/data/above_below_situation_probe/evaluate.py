#!/usr/bin/env python3
"""Score matched above/below situations with binary mean-token likelihood."""

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


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    assert n
    return {"n_pairs": n, "n_judgments": 2*n,
            "binary_accuracy": sum(r["binary_accuracy"] for r in rows)/n,
            "above_gold_accuracy": sum(r["above_correct"] for r in rows)/n,
            "below_gold_accuracy": sum(r["below_correct"] for r in rows)/n,
            "both_correct_fraction": sum(r["both_correct"] for r in rows)/n,
            "above_word_chosen_fraction": sum(r["above_word_chosen_count"] for r in rows)/(2*n),
            "context_sensitivity_accuracy": sum(r["context_sensitivity_accuracy"] for r in rows)/n,
            "exact_choice_ties": sum(r["exact_choice_ties"] for r in rows)}


def score_probes(probes, model, tokenizer, batch_size):
    keys = []
    for row in probes:
        for c, t in ((1, 1), (1, 2), (2, 2), (2, 1)):
            context, target = row[f"Context{c}"], row[f"Target{t}"]
            prefix = tokenizer.encode(context, add_special_tokens=False)
            joined = tokenizer.encode(context + " " + target, add_special_tokens=False)
            assert joined[:len(prefix)] == prefix and len(joined) > len(prefix)
            keys.append((context, target))
    unique = list(dict.fromkeys(keys))
    contexts = [c for c, _ in unique]
    targets = [t for _, t in unique]
    with torch.inference_mode():
        token_scores = per_token_conditional_log_likelihood(
            model, tokenizer, contexts, targets, device="cuda", batch_size=batch_size)
    scores = {key: value.detach().float().cpu()
              for key, value in zip(unique, token_scores)}
    assert len(scores) == len(unique)
    results = []
    for row in probes:
        values = {label: scores[row[f"Context{c}"], row[f"Target{t}"]]
                  for label, c, t in (("S11", 1, 1), ("S12", 1, 2),
                                      ("S22", 2, 2), ("S21", 2, 1))}
        means = {label: float(tokens.mean()) for label, tokens in values.items()}
        above_margin = means["S11"] - means["S12"]
        below_margin = means["S22"] - means["S21"]
        context_above = means["S11"] - means["S21"]
        context_below = means["S22"] - means["S12"]
        above_correct, below_correct = above_margin > 0, below_margin > 0
        first_word = row["target1_relation_word"]
        above_chosen = ((above_margin > 0) if first_word == "above" else (above_margin < 0))
        above_chosen += ((below_margin < 0) if first_word == "above" else (below_margin > 0))
        results.append({**row, **means,
            **{label + "_target_token_count": len(tokens) for label, tokens in values.items()},
            "score_reduction": "mean", "above_margin": above_margin,
            "below_margin": below_margin,
            "above_correct": above_correct, "below_correct": below_correct,
            "binary_accuracy": (above_correct + below_correct)/2,
            "both_correct": above_correct and below_correct,
            "above_word_chosen_count": above_chosen,
            "context_sensitivity_above_margin": context_above,
            "context_sensitivity_below_margin": context_below,
            "context_sensitivity_accuracy": ((context_above > 0)+(context_below > 0))/2,
            "exact_choice_ties": (above_margin == 0)+(below_margin == 0),
            "prediction_in_above_context": "above" if above_margin > 0 else "below" if above_margin < 0 else "tie",
            "prediction_in_below_context": "below" if below_margin > 0 else "above" if below_margin < 0 else "tie",
        })
    return results, len(unique)


def variant_consistency(items, links):
    by_id = {row["probe_id"]: row for row in items}
    assert len(by_id) == len(items)
    rows = []
    for link in links:
        a, b = by_id[link["base_probe_id"]], by_id[link["variant_probe_id"]]
        rows.append({**link,
            "base_accuracy": a["binary_accuracy"],
            "variant_accuracy": b["binary_accuracy"],
            "above_prediction_flip": a["prediction_in_above_context"] != b["prediction_in_above_context"],
            "below_prediction_flip": a["prediction_in_below_context"] != b["prediction_in_below_context"],
            "both_predictions_same": (a["prediction_in_above_context"] == b["prediction_in_above_context"]
                                      and a["prediction_in_below_context"] == b["prediction_in_below_context"]),
            "both_variants_fully_correct": a["both_correct"] and b["both_correct"],
        })
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["control"]].append(row)
    summary = {control: {"n_matches": len(subset),
        "above_prediction_flip_fraction": sum(r["above_prediction_flip"] for r in subset)/len(subset),
        "below_prediction_flip_fraction": sum(r["below_prediction_flip"] for r in subset)/len(subset),
        "both_predictions_same_fraction": sum(r["both_predictions_same"] for r in subset)/len(subset),
        "both_variants_fully_correct_fraction": sum(r["both_variants_fully_correct"] for r in subset)/len(subset)}
        for control, subset in sorted(grouped.items())}
    return rows, summary


def make_report(summary):
    out = ["# Above/below situation probe", "",
        "Binary choice between two complete target sentences, scored by mean target-token log likelihood. Each pair has one above-compatible and one below-compatible context; exact ties count as wrong.",
        "", "| Family | Evidence | Pairs | Accuracy | Above gold | Below gold | Both correct | Above word chosen |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for family, data in summary["families"].items():
        for evidence, result in data["by_evidence"].items():
            out.append(f"| {family} | {evidence} | {result['n_pairs']} | " +
                       " | ".join(f"{result[key]:.2%}" for key in (
                           "binary_accuracy", "above_gold_accuracy", "below_gold_accuracy",
                           "both_correct_fraction", "above_word_chosen_fraction")) + " |")
        result = data["all_evidence"]
        aggregate_label = "both label forms" if family == "direct_label_control" else "both evidence forms"
        out.append(f"| **{family}** | **{aggregate_label}** | {result['n_pairs']} | " +
                   " | ".join(f"**{result[key]:.2%}**" for key in (
                       "binary_accuracy", "above_gold_accuracy", "below_gold_accuracy",
                       "both_correct_fraction", "above_word_chosen_fraction")) + " |")
    overall = summary["overall_applied"]
    direct = summary["direct_label_control"]
    out.extend(["", f"**Equal-family applied mean:** {overall['equal_family_binary_accuracy']:.2%} across five physical families. The direct-label control is separate ({direct['binary_accuracy']:.2%}).",
        "", "The family mean first balances the two evidence forms and all wording, object, and entity-order variants within each family, then gives the five physical families equal weight. The repeated variants are matched measurements of a small set of physical cases, not independent scene samples.",
        "", "`target_moves_without_crossing` holds upward or downward motion constant across the above and below contexts. Its score tests whether the model uses the final relative position rather than motion direction alone. `both_move` includes preserved and reversed vertical order.",
        "", "`variant_consistency_summary.json` records matched prediction flips for context order, target order, evidence type, and wording length. Target order also inverts the relation word in the answer, so its flip rate is not a pure syntax effect. Length-band prefixes vary in wording as well as length. `bias_table.md` separates lexical answer preference by target order.", ""])
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required for this checkpoint evaluation"
    probe_path = ROOT / "generated/probes.jsonl"
    probes = [json.loads(line) for line in probe_path.read_text().splitlines()]
    manifest = json.loads((ROOT / "generated/manifest.json").read_text())
    assert len(probes) == manifest["paired_rows"]
    links = read_csv(ROOT / "generated/variant_matches.csv")
    assert len(links) == manifest["variant_matches"]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for path in (probe_path, ROOT / "components.json", ROOT / "generated/manifest.json",
                 ROOT / "generated/variant_matches.csv"):
        (out / ("input_" + path.name)).write_bytes(path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters; scoring {len(probes)} pairs", flush=True)
    items, unique = score_probes(probes, model, tokenizer, args.batch_size)
    write_csv(out / "item_scores.csv", items)
    (out / "item_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items))
    grouped = []
    groupings = (("event_family",), ("event_family", "evidence_type"),
                 ("event_family", "event_subtype"),
                 ("event_family", "length_band"),
                 ("event_family", "context_entity_order"),
                 ("event_family", "target_entity_order"),
                 ("event_family", "object_pair_id"))
    for factors in groupings:
        buckets = defaultdict(list)
        for row in items:
            buckets[tuple(row[f] for f in factors)].append(row)
        for values, subset in buckets.items():
            grouped.append({"grouping": "+".join(factors), "group": "|".join(values),
                            **summarize(subset)})
    write_csv(out / "grouped_scores.csv", grouped)
    family_names = ("static_placement", "target_crosses", "reference_crosses",
                    "target_moves_without_crossing", "both_move")
    families = {}
    for family in (*family_names, "direct_label_control"):
        subset = [r for r in items if r["event_family"] == family]
        evidence_names = sorted({r["evidence_type"] for r in subset})
        families[family] = {"all_evidence": summarize(subset),
            "by_evidence": {evidence: summarize([r for r in subset if r["evidence_type"] == evidence])
                            for evidence in evidence_names}}
    consistency_rows, consistency_summary = variant_consistency(items, links)
    write_csv(out / "variant_consistency.csv", consistency_rows)
    (out / "variant_consistency_summary.json").write_text(
        json.dumps(consistency_summary, indent=2) + "\n")
    summary = {"model": str(Path(args.model).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "components_sha256": hashlib.sha256((ROOT / "components.json").read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0), "dtype": "float32",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "score_reduction": "mean", "batch_size": args.batch_size,
        "n_pairs": len(probes), "n_unique_context_target_sequences": unique,
        "families": families,
        "overall_applied": {"equal_family_binary_accuracy": sum(
            families[f]["all_evidence"]["binary_accuracy"] for f in family_names)/len(family_names),
            "physical_family_count": len(family_names),
            "physical_pair_rows": sum(families[f]["all_evidence"]["n_pairs"] for f in family_names)},
        "direct_label_control": families["direct_label_control"]["all_evidence"],
        "variant_consistency": consistency_summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text(make_report(summary))
    generate_bias_table(out)
    print(json.dumps({"report": str(out / "report.md"),
                      "equal_family_binary_accuracy": summary["overall_applied"]["equal_family_binary_accuracy"],
                      "direct_label_binary_accuracy": summary["direct_label_control"]["binary_accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
