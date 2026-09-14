#!/usr/bin/env python3
"""Analyze paired closer/farther contexts from saved extension item scores."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.results_dir
    items = [json.loads(line) for line in (out / "item_scores.jsonl").open()]
    groups = defaultdict(dict)
    reference_rows = 0
    for row in items:
        if row["condition_id"] != "reference_object_moves":
            continue
        reference_rows += 1
        key = (row["numeric_information"], row["name_id"], row["object_id"],
               row["numeric_case_id"], row["unit_id"], row["length_band"],
               row["context_entity_order"])
        assert row["outcome"] not in groups[key]
        groups[key][row["outcome"]] = row
    assert len(groups) == reference_rows//2
    pairs = []
    for key, group in groups.items():
        assert set(group) == {"closer", "farther"}
        c, f = group["closer"], group["farther"]
        assert all(c[k] == f[k] for k in ["Target1", "Target2", "Target3",
            "name_id", "object_id", "numeric_information", "length_band"])
        result = {"closer_probe_id": c["probe_id"], "farther_probe_id": f["probe_id"],
            "numeric_information": key[0], "name_id": key[1], "object_id": key[2],
            "numeric_case_id": key[3], "unit_id": key[4], "length_band": key[5],
            "context_entity_order": key[6]}
        for method in ["raw", "pmi"]:
            a = c[f"{method}_Target1_mean"] - c[f"{method}_Target2_mean"]
            b = f[f"{method}_Target2_mean"] - f[f"{method}_Target1_mean"]
            result.update({f"{method}_closer_margin": a,
                f"{method}_farther_margin": b,
                f"{method}_closer_correct": a > 0,
                f"{method}_farther_correct": b > 0,
                f"{method}_both_correct": a > 0 and b > 0,
                f"{method}_close_choice_fraction": ((a > 0) + (b < 0)) / 2})
        close_context = c["raw_Target1_mean"] - f["raw_Target1_mean"]
        far_context = f["raw_Target2_mean"] - c["raw_Target2_mean"]
        assert abs(close_context -
            (c["pmi_Target1_mean"] - f["pmi_Target1_mean"])) < 1e-9
        assert abs(far_context -
            (f["pmi_Target2_mean"] - c["pmi_Target2_mean"])) < 1e-9
        result.update({"context_closer_margin": close_context,
            "context_farther_margin": far_context,
            "context_closer_correct": close_context > 0,
            "context_farther_correct": far_context > 0,
            "context_both_correct": close_context > 0 and far_context > 0})
        pairs.append(result)
    write_csv(out / "contrast_pair_scores.csv", pairs)
    aggregate = []
    for factors in [(), ("numeric_information",),
                    ("numeric_information", "length_band"),
                    ("numeric_information", "context_entity_order"),
                    ("numeric_information", "length_band", "context_entity_order")]:
        by_factor = defaultdict(list)
        for row in pairs:
            by_factor[tuple(row[k] for k in factors)].append(row)
        for key, subset in by_factor.items():
            n = len(subset)
            result = {"grouping": "+".join(factors) or "overall",
                "group": "|".join(key) or "all", "n_pairs": n,
                "n_judgments": 2*n}
            for method in ["raw", "pmi", "context"]:
                result.update({f"{method}_accuracy": sum(
                    r[f"{method}_closer_correct"] + r[f"{method}_farther_correct"]
                    for r in subset)/(2*n),
                    f"{method}_closer_accuracy": sum(
                        r[f"{method}_closer_correct"] for r in subset)/n,
                    f"{method}_farther_accuracy": sum(
                        r[f"{method}_farther_correct"] for r in subset)/n,
                    f"{method}_both_correct": sum(
                        r[f"{method}_both_correct"] for r in subset)/n})
            result["raw_close_choice_fraction"] = sum(
                r["raw_close_choice_fraction"] for r in subset)/n
            aggregate.append(result)
    write_csv(out / "contrast_summary.csv", aggregate)
    (out / "contrast_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    report = ["# Reference-object movement: paired contrast", "",
        "The primary contrast for this event family compares only closer versus farther targets under matched toward/away contexts. The earlier three-way score also offered a same-distance target, which the model selected on every raw movement example.",
        "Each pair shares names, objects, targets, numeric case, unit, and length. Context sensitivity compares the same target across the two contexts; PMI target-only subtraction cancels from it.", "",
        "| Numeric evidence | Wording | Pairs | Raw binary choice | PMI binary choice | Context sensitivity | Raw close-choice rate | Both raw contexts correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        if row["grouping"] == "numeric_information+length_band":
            mode, band = row["group"].split("|")
            report.append(f"| {mode} | {band} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['context_accuracy']:.2%} | {row['raw_close_choice_fraction']:.2%} | {row['raw_both_correct']:.2%} |")
    report += ["", "| Numeric evidence | Pairs | Raw binary choice | PMI binary choice | Context sensitivity | Raw close-choice rate |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        if row["grouping"] == "numeric_information":
            report.append(f"| {row['group']} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['context_accuracy']:.2%} | {row['raw_close_choice_fraction']:.2%} |")
    report += ["", "| Numeric evidence | Context order | Pairs | Raw binary choice | PMI binary choice | Context sensitivity | Raw close-choice rate |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        if row["grouping"] == "numeric_information+context_entity_order":
            mode, order = row["group"].split("|")
            report.append(f"| {mode} | {order} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['context_accuracy']:.2%} | {row['raw_close_choice_fraction']:.2%} |")
    report += ["", "Uniform binary-choice chance is 50%. The toward/away contexts differ in direction wording (one word versus two), not in names, objects, targets, or event constraints. Binary and context-sensitivity scores near chance indicate that removing the same-distance distractor does not reveal robust discrimination here."]
    (out / "contrast_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
