#!/usr/bin/env python3
"""Score full targets for the closer/farther extension and binary analysis."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv
from evaluation.ewok import (per_token_conditional_log_likelihood,
    per_token_unconditional_log_likelihood, resolve_bos_token_id)

TARGETS = ["Target1", "Target2", "Target3"]
LABELS = dict(zip(TARGETS, ["closer", "farther", "unchanged"]))
PARENT_RESULT = (ROOT.parents[2] / "runs/research/bos_aligned_proto/closer_farther_probe"
    / "qwen3_359m_step19500_v1_2")


def decision(values, correct):
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    margin = values[correct] - max(v for k, v in values.items() if k != correct)
    prediction = ordered[0][0] if ordered[0][1] > ordered[1][1] else "tie"
    return prediction, margin, margin > 0


def summarize(rows, method):
    predictions = Counter(r[f"{method}_prediction"] for r in rows)
    return {"n_contexts": len(rows),
        "n_correct": sum(r[f"{method}_correct"] for r in rows),
        "accuracy": sum(r[f"{method}_correct"] for r in rows) / len(rows),
        "mean_gold_margin": sum(r[f"{method}_gold_margin"] for r in rows) / len(rows),
        **{f"predicted_{label}": predictions[target] for target, label in LABELS.items()},
        "ties": predictions["tie"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required."
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in [ROOT / "components.json", *sorted((ROOT / "generated").glob("*"))]:
        (out / ("input_" + path.name)).write_bytes(path.read_bytes())
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (out / "input_probes.jsonl").read_text().splitlines()]
    manifest = json.loads((out / "input_manifest.json").read_text())
    assert len(rows) == manifest["context_rows"] == len({r["probe_id"] for r in rows})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    sequences = list(dict.fromkeys((r["Context"], r[k]) for r in rows for k in TARGETS))
    token_ids = {}
    for context, target in sequences:
        prefix = tokenizer.encode(context, add_special_tokens=False)
        joined = tokenizer.encode(context + " " + target, add_special_tokens=False)
        assert joined[:len(prefix)] == prefix and len(joined) > len(prefix)
        token_ids[context, target] = joined[len(prefix):]
    scores = {}
    print(f"Scoring {len(sequences)} sequences on {torch.cuda.get_device_name(0)}.", flush=True)
    with torch.inference_mode(), (out / "token_scores.jsonl").open("w") as stream:
        for start in range(0, len(sequences), 384):
            chunk = sequences[start:start+384]
            values = per_token_conditional_log_likelihood(model, tokenizer,
                [c for c, _ in chunk], [t for _, t in chunk], device="cuda",
                batch_size=args.batch_size)
            for pair, tensor in zip(chunk, values):
                tensor = tensor.detach().float().cpu()
                assert len(tensor) == len(token_ids[pair]) and torch.isfinite(tensor).all()
                scores[pair] = float(tensor.mean())
                stream.write(json.dumps({"context": pair[0], "target": pair[1],
                    "target_token_ids": token_ids[pair],
                    "target_token_log_probs": tensor.tolist()}) + "\n")
            stream.flush()
            print(f"Scored {min(start+384, len(sequences))}/{len(sequences)}.", flush=True)
    targets = list(dict.fromkeys(r[k] for r in rows for k in TARGETS))
    priors = {}
    with torch.inference_mode(), (out / "target_priors.jsonl").open("w") as stream:
        values = per_token_unconditional_log_likelihood(model, tokenizer, targets,
            device="cuda", batch_size=args.batch_size)
        for target, tensor in zip(targets, values):
            tensor = tensor.detach().float().cpu()
            assert len(tensor) == len(tokenizer.encode(target, add_special_tokens=False))
            assert torch.isfinite(tensor).all()
            priors[target] = float(tensor.mean())
            stream.write(json.dumps({"target": target, "mean_log_likelihood": priors[target],
                "target_token_ids": tokenizer.encode(target, add_special_tokens=False),
                "target_token_log_probs": tensor.tolist()}) + "\n")
    results = []
    for row in rows:
        scored = {**row, "score_reduction": "mean"}
        for method in ["raw", "pmi"]:
            values = {k: scores[row["Context"], row[k]] -
                (priors[row[k]] if method == "pmi" else 0) for k in TARGETS}
            pred, margin, correct = decision(values, row["correct_target"])
            scored.update({method + "_" + k + "_mean": v for k, v in values.items()})
            scored.update({method + "_prediction": pred,
                method + "_gold_margin": margin, method + "_correct": correct})
        results.append(scored)
    write_csv(out / "item_scores.csv", results)
    (out / "item_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    groupings = [(), ("condition_id",), ("numeric_information",), ("length_band",),
        ("outcome",), ("condition_id", "numeric_information"),
        ("condition_id", "numeric_information", "length_band"),
        ("condition_id", "numeric_information", "length_band", "outcome"),
        ("condition_id", "length_band", "outcome"),
        ("condition_id", "numeric_information", "context_entity_order", "outcome"),
        ("context_entity_order", "outcome"), ("name_id",), ("object_id",),
        ("numeric_case_id",), ("unit_id",)]
    grouped, confusion = [], []
    for factors in groupings:
        groups = defaultdict(list)
        for row in results:
            groups[tuple(row[k] for k in factors)].append(row)
        for key, subset in groups.items():
            for method in ["raw", "pmi"]:
                descriptor = {"method": method, "grouping": "+".join(factors) or "overall",
                    "group": "|".join(str(v) for v in key) or "all"}
                grouped.append({**descriptor, **summarize(subset, method)})
                if factors in [(), ("condition_id",), ("condition_id", "numeric_information")]:
                    counts = Counter((r["outcome"], r[method + "_prediction"]) for r in subset)
                    for gold in LABELS.values():
                        for prediction in [*TARGETS, "tie"]:
                            confusion.append({**descriptor, "gold": gold,
                                "predicted": LABELS.get(prediction, prediction),
                                "count": counts[gold, prediction]})
    write_csv(out / "grouped_scores.csv", grouped)
    write_csv(out / "confusion_matrices.csv", confusion)
    by_id = {r["probe_id"]: r for r in results}
    parent_summary = json.loads((PARENT_RESULT / "summary.json").read_text())
    assert Path(parent_summary["model"]).resolve() == Path(args.model).resolve()
    assert parent_summary["score_reduction"] == "mean"
    assert parent_summary["probe_sha256"] == manifest["source_sha256"]["parent_probes"]
    assert parent_summary["dtype"] == "float32" and parent_summary["attention_implementation"] == "eager"
    parent_rows = {r["probe_id"]: r for r in map(json.loads,
        (PARENT_RESULT / "item_scores.jsonl").open())}
    match_scores = []
    for match in csv.DictReader((out / "input_matches.csv").open()):
        a = by_id.get(match["reference_probe_id"], parent_rows.get(match["reference_probe_id"]))
        b = by_id[match["variant_probe_id"]]
        assert a is not None and a["outcome"] == b["outcome"]
        assert all(a[k] == b[k] for k in TARGETS)
        for method in ["raw", "pmi"]:
            match_scores.append({**match, "method": method,
                "reference_correct": a[method + "_correct"],
                "variant_correct": b[method + "_correct"],
                "same_prediction": a[method + "_prediction"] == b[method + "_prediction"],
                "wrong_to_right": not a[method + "_correct"] and b[method + "_correct"],
                "right_to_wrong": a[method + "_correct"] and not b[method + "_correct"]})
    write_csv(out / "matched_scores.csv", match_scores)
    match_groups = defaultdict(list)
    for row in match_scores:
        match_groups[row["control"], row["method"]].append(row)
    match_summary = [{"control": control, "method": method, "n_matches": len(subset),
        "reference_accuracy": sum(r["reference_correct"] for r in subset)/len(subset),
        "variant_accuracy": sum(r["variant_correct"] for r in subset)/len(subset),
        "same_prediction_fraction": sum(r["same_prediction"] for r in subset)/len(subset),
        "wrong_to_right": sum(r["wrong_to_right"] for r in subset),
        "right_to_wrong": sum(r["right_to_wrong"] for r in subset)}
        for (control, method), subset in match_groups.items()]
    write_csv(out / "match_summary.csv", match_summary)
    consistency = []
    length_groups = defaultdict(list)
    for row in results:
        if row["condition_id"] != "direct_label":
            length_groups[row["length_match_id"]].append(row)
    assert len(length_groups) == manifest["length_groups"]
    for key, subset in length_groups.items():
        assert len(subset) == 3 and {r["length_band"] for r in subset} == {
            "compact", "standard", "expanded"}
        assert len({r["context_entity_order"] for r in subset}) == 1
        for method in ["raw", "pmi"]:
            consistency.append({"length_match_id": key, "method": method,
                "condition_id": subset[0]["condition_id"],
                "numeric_information": subset[0]["numeric_information"],
                "context_entity_order": subset[0]["context_entity_order"],
                "outcome": subset[0]["outcome"],
                "n_correct": sum(r[method + "_correct"] for r in subset),
                "all_correct": all(r[method + "_correct"] for r in subset),
                "same_prediction": len({r[method + "_prediction"] for r in subset}) == 1})
    write_csv(out / "length_consistency.csv", consistency)
    consistency_summary = []
    for method in ["raw", "pmi"]:
        for condition, mode in [("reference_object_moves", "specific"),
            ("reference_object_moves", "absent"),
            ("both_move_same_separation", "specific"),
            ("both_move_same_separation", "absent")]:
            subset = [r for r in consistency if r["method"] == method and
                r["condition_id"] == condition and r["numeric_information"] == mode]
            consistency_summary.append({"method": method, "condition_id": condition,
                "numeric_information": mode, "n_groups": len(subset),
                "all_correct_fraction": sum(r["all_correct"] for r in subset)/len(subset),
                "same_prediction_fraction": sum(r["same_prediction"] for r in subset)/len(subset)})
    write_csv(out / "length_consistency_summary.csv", consistency_summary)
    summary = {"model": str(Path(args.model).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["version"], "source_sha256": sources,
        "parent_result": str(PARENT_RESULT),
        "parent_probe_sha256_verified": True, "score_reduction": "mean",
        "dtype": "float32", "attention_implementation": "eager",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bos_token_id": resolve_bos_token_id(tokenizer),
        "unique_conditional_sequences": len(sequences), "unique_target_priors": len(priors),
        "three_way_diagnostic_chance": 1/3,
        "primary_binary_summary_file": "binary_summary.json",
        "primary_binary_report_file": "report.md",
        "overall_applied_summary_file": "overall_summary.json",
        "overall_applied_report_file": "overall_report.md",
        "pmi_convention": "Secondary: mean logp(T|C) minus mean logp(T|BOS); target-only text as written, no added leading space or EOS.",
        "ties": "Exact maximum ties count as incorrect.",
        "overall": {method: summarize(results, method) for method in ["raw", "pmi"]},
        "groups": grouped, "match_summary": match_summary,
        "length_consistency": consistency_summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Qwen3 closer/farther: new event families", "",
        "Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation. Three-way uniform-choice chance is 33.33%.",
        "Scores for numeric and nonnumeric versions are reported separately. Direct labels are lexical controls and are not pooled with movement inference.", "",
        "| Condition | Numeric information | Length | N | Raw choice | PMI choice |",
        "|---|---|---|---:|---:|---:|"]
    for row in grouped:
        if row["method"] == "raw" and row["grouping"] == "condition_id+numeric_information+length_band":
            other = next(x for x in grouped if x["method"] == "pmi" and
                x["grouping"] == row["grouping"] and x["group"] == row["group"])
            condition, mode, band = row["group"].split("|")
            report.append(f"| {condition} | {mode} | {band} | {row['n_contexts']} | {row['accuracy']:.2%} | {other['accuracy']:.2%} |")
    report += ["", "## Outcome accuracy", "",
        "| Condition | Numeric information | Length | Gold outcome | N | Raw choice | PMI choice |",
        "|---|---|---|---|---:|---:|---:|"]
    for row in grouped:
        if row["method"] == "raw" and row["grouping"] == "condition_id+numeric_information+length_band+outcome":
            other = next(x for x in grouped if x["method"] == "pmi" and
                x["grouping"] == row["grouping"] and x["group"] == row["group"])
            condition, mode, band, outcome = row["group"].split("|")
            report.append(f"| {condition} | {mode} | {band} | {outcome} | {row['n_contexts']} | {row['accuracy']:.2%} | {other['accuracy']:.2%} |")
    report += ["", "## Wording consistency", "",
        "| Condition | Numeric information | Groups | All three correct (raw) | Same prediction (raw) |",
        "|---|---|---:|---:|---:|"]
    for row in consistency_summary:
        if row["method"] == "raw":
            report.append(f"| {row['condition_id']} | {row['numeric_information']} | {row['n_groups']} | {row['all_correct_fraction']:.2%} | {row['same_prediction_fraction']:.2%} |")
    report += ["", "## Matched comparisons", "",
        "| Control | N | Reference accuracy (raw) | New accuracy (raw) | Same prediction | Fixed | Broken |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in match_summary:
        if row["method"] == "raw":
            report.append(f"| {row['control']} | {row['n_matches']} | {row['reference_accuracy']:.2%} | {row['variant_accuracy']:.2%} | {row['same_prediction_fraction']:.2%} | {row['wrong_to_right']} | {row['right_to_wrong']} |")
    report += ["", "Parent standard comparisons are paired on entity, unit, numeric case, outcome, targets, and initial/final separation; they describe different physical events. Parent item scores were reused only after verifying the checkpoint, mean scoring convention, and source probe hash.",
        "The reference-object closer/farther minimal-pair analysis is saved separately in contrast_report.md. Its binary choice and fixed-target context-sensitivity scores should be considered alongside the three-way confusion results.",
        "Length groups share event coordinates and target sentences. Their wording also changes syntax, so length effects are not isolated. Numeric cases and entity variants repeat the same templates and are not independent scenarios.",
        "Direct-label rows contain the comparative word and serve as an easy ceiling control. Their score should not be interpreted as movement inference. Exact ties count as incorrect. Saved inputs, token scores, target priors, confusion matrices, match outcomes, and item scores support further diagnosis."]
    (out / "three_way_diagnostic.md").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)
    subprocess.run([sys.executable, str(ROOT / "analyze_contrasts.py"),
        "--results-dir", str(out)], check=True)
    subprocess.run([sys.executable, str(ROOT / "analyze_binary.py"),
        "--results-dir", str(out)], check=True)
    if manifest["version"] == "1.2":
        subprocess.run([sys.executable, str(ROOT / "analyze_overall.py"),
            "--results-dir", str(out)], check=True)


if __name__ == "__main__":
    main()
