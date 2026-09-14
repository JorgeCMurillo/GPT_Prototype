#!/usr/bin/env python3
"""Make three balanced two-answer contrasts from saved event-extension scores."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import write_csv
from data.spatial_bias_report.generate import generate_one as generate_bias_table

TARGET = {"closer": "Target1", "farther": "Target2", "unchanged": "Target3"}
CONTRASTS = [("closer_vs_farther", "closer", "farther"),
             ("closer_vs_unchanged", "closer", "unchanged"),
             ("farther_vs_unchanged", "farther", "unchanged")]


def summarize(rows):
    n = len(rows)
    assert n > 0
    out = {"n_pairs": n, "n_judgments": 2*n}
    for method in ["raw", "pmi"]:
        out.update({method + "_accuracy": sum(
            r[method + "_first_correct"] + r[method + "_second_correct"]
            for r in rows)/(2*n),
            method + "_first_accuracy": sum(r[method + "_first_correct"] for r in rows)/n,
            method + "_second_accuracy": sum(r[method + "_second_correct"] for r in rows)/n,
            method + "_both_correct": sum(r[method + "_both_correct"] for r in rows)/n,
            method + "_first_choice_fraction": sum(
                r[method + "_first_chosen_count"] for r in rows)/(2*n),
            method + "_ties": sum(r[method + "_ties"] for r in rows)})
    if rows[0]["contrast"] == "closer_vs_farther":
        out.update({"context_accuracy": sum(
            r["context_first_correct"] + r["context_second_correct"]
            for r in rows)/(2*n),
            "context_both_correct": sum(r["context_both_correct"] for r in rows)/n})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.results_dir
    inputs = [json.loads(line) for line in (out / "item_scores.jsonl").open()]
    manifest = json.loads((out / "input_manifest.json").read_text())
    assert len(inputs) == manifest["context_rows"]
    groups = defaultdict(dict)
    for r in inputs:
        if r["condition_id"] == "direct_label":
            key = ("direct_label", r["name_id"], r["object_id"], r["template_id"])
        else:
            key = (r["numeric_information"], r["name_id"], r["object_id"],
                   r["numeric_case_id"], r["unit_id"], r["length_band"],
                   r["context_entity_order"])
        assert r["outcome"] not in groups[key]
        groups[key][r["outcome"]] = r
    assert len(groups) == len(inputs)//3
    assert all(set(g) == set(TARGET) for g in groups.values())
    pairs, judgments = [], []
    for group in groups.values():
        for contrast, first, second in CONTRASTS:
            a, b = group[first], group[second]
            assert all(a[k] == b[k] for k in ["name_id", "object_id", *TARGET.values()])
            mode = "direct_label" if a["condition_id"] == "direct_label" else a["numeric_information"]
            band = a["length_band"]
            pair_id = a["probe_id"] + "__vs__" + b["probe_id"]
            row = {"pair_id": pair_id, "contrast": contrast,
                "first_outcome": first, "second_outcome": second,
                "first_probe_id": a["probe_id"], "second_probe_id": b["probe_id"],
                "first_condition_id": a["condition_id"],
                "second_condition_id": b["condition_id"],
                "first_event_family": a["event_family"],
                "second_event_family": b["event_family"],
                "first_entity_order_match_id": a.get("entity_order_match_id", ""),
                "second_entity_order_match_id": b.get("entity_order_match_id", ""),
                "mode": mode, "length_band": band,
                "context_entity_order": a["context_entity_order"],
                "direct_context_entity_order": a["context_entity_order"] if mode == "direct_label" else "",
                "name_id": a["name_id"], "object_id": a["object_id"],
                "numeric_case_id": a["numeric_case_id"], "unit_id": a["unit_id"],
                "same_event_family": a["condition_id"] == b["condition_id"]}
            for method in ["raw", "pmi"]:
                first_target, second_target = TARGET[first], TARGET[second]
                first_margin = a[f"{method}_{first_target}_mean"] - a[f"{method}_{second_target}_mean"]
                second_margin = b[f"{method}_{second_target}_mean"] - b[f"{method}_{first_target}_mean"]
                row.update({method + "_first_margin": first_margin,
                    method + "_second_margin": second_margin,
                    method + "_first_correct": first_margin > 0,
                    method + "_second_correct": second_margin > 0,
                    method + "_both_correct": first_margin > 0 and second_margin > 0,
                    method + "_first_chosen_count": (first_margin > 0) + (second_margin < 0),
                    method + "_ties": (first_margin == 0) + (second_margin == 0)})
                for source, score_row, gold, margin in [
                    ("first", a, first, first_margin), ("second", b, second, second_margin)]:
                    judgments.append({"pair_id": pair_id, "contrast": contrast,
                        "mode": mode, "length_band": band, "context_role": source,
                        "context_entity_order": score_row["context_entity_order"],
                        "event_family": score_row["event_family"],
                        "name_id": score_row["name_id"], "object_id": score_row["object_id"],
                        "numeric_case_id": score_row["numeric_case_id"],
                        "unit_id": score_row["unit_id"],
                        "entity_order_match_id": score_row.get("entity_order_match_id", ""),
                        "condition_id": score_row["condition_id"],
                        "probe_id": score_row["probe_id"], "gold": gold,
                        "method": method, "margin": margin, "correct": margin > 0,
                        "prediction": gold if margin > 0 else
                        (second if gold == first else first) if margin < 0 else "tie"})
            if contrast == "closer_vs_farther":
                assert row["same_event_family"]
                first_ctx = a["raw_Target1_mean"] - b["raw_Target1_mean"]
                second_ctx = b["raw_Target2_mean"] - a["raw_Target2_mean"]
                assert abs(first_ctx -
                    (a["pmi_Target1_mean"] - b["pmi_Target1_mean"])) < 1e-9
                assert abs(second_ctx -
                    (b["pmi_Target2_mean"] - a["pmi_Target2_mean"])) < 1e-9
                row.update({"context_first_margin": first_ctx,
                    "context_second_margin": second_ctx,
                    "context_first_correct": first_ctx > 0,
                    "context_second_correct": second_ctx > 0,
                    "context_both_correct": first_ctx > 0 and second_ctx > 0})
            pairs.append(row)
    assert len(pairs) == len(inputs) and len(judgments) == 2*2*len(inputs)
    assert Counter(r["contrast"] for r in pairs) == {
        c: len(groups) for c, _, _ in CONTRASTS}
    write_csv(out / "binary_pair_scores.csv", pairs)
    write_csv(out / "binary_judgments.csv", judgments)
    grouped = []
    for factors in [("contrast",), ("contrast", "mode"),
                    ("contrast", "mode", "length_band"),
                    ("contrast", "mode", "context_entity_order"),
                    ("contrast", "mode", "length_band", "context_entity_order")]:
        buckets = defaultdict(list)
        for row in pairs:
            buckets[tuple(row[k] for k in factors)].append(row)
        for key, subset in buckets.items():
            grouped.append({"grouping": "+".join(factors),
                "group": "|".join(str(k) for k in key), **summarize(subset)})
    write_csv(out / "binary_grouped_scores.csv", grouped)
    order_groups = defaultdict(dict)
    for row in pairs:
        key = tuple(row[k] for k in ["contrast", "mode", "length_band", "name_id",
            "object_id", "numeric_case_id", "unit_id"])
        order = row["context_entity_order"]
        assert order not in order_groups[key]
        order_groups[key][order] = row
    order_pairs = []
    for key, group in order_groups.items():
        if set(group) != {"person_first", "object_first"}:
            assert manifest["version"] == "1.1"
            continue
        person, obj = group["person_first"], group["object_first"]
        order_pair = {"contrast": key[0], "mode": key[1], "length_band": key[2],
            "name_id": key[3], "object_id": key[4], "numeric_case_id": key[5],
            "unit_id": key[6], "person_first_pair_id": person["pair_id"],
            "object_first_pair_id": obj["pair_id"]}
        for method in ["raw", "pmi"]:
            order_pair.update({
                method + "_person_first_accuracy": (
                    person[method + "_first_correct"] + person[method + "_second_correct"])/2,
                method + "_object_first_accuracy": (
                    obj[method + "_first_correct"] + obj[method + "_second_correct"])/2,
                method + "_first_prediction_same": person[method + "_first_correct"] == obj[method + "_first_correct"],
                method + "_second_prediction_same": person[method + "_second_correct"] == obj[method + "_second_correct"],
                method + "_any_prediction_flip": (
                    person[method + "_first_correct"] != obj[method + "_first_correct"] or
                    person[method + "_second_correct"] != obj[method + "_second_correct"]),
                method + "_all_four_correct": person[method + "_both_correct"] and obj[method + "_both_correct"]})
        order_pairs.append(order_pair)
    if manifest["version"] == "1.2":
        assert len(order_pairs) == len(groups)//2 * len(CONTRASTS)
    write_csv(out / "entity_order_pair_scores.csv", order_pairs)
    order_summary = []
    for factors in [("contrast", "mode"), ("contrast", "mode", "length_band")]:
        buckets = defaultdict(list)
        for row in order_pairs:
            buckets[tuple(row[k] for k in factors)].append(row)
        for key, subset in buckets.items():
            n = len(subset)
            summary_row = {"grouping": "+".join(factors), "group": "|".join(key),
                "n_order_pairs": n}
            for method in ["raw", "pmi"]:
                for metric in ["person_first_accuracy", "object_first_accuracy",
                               "first_prediction_same", "second_prediction_same",
                               "any_prediction_flip", "all_four_correct"]:
                    summary_row[method + "_" + metric] = sum(
                        r[method + "_" + metric] for r in subset)/n
            order_summary.append(summary_row)
    write_csv(out / "entity_order_summary.csv", order_summary)
    summary = {"primary_metric": "binary mean-target-token choice",
        "uniform_choice_chance": 0.5,
        "n_unique_contexts": len(inputs), "n_binary_pairs": len(pairs),
        "n_binary_judgments": len(pairs)*2,
        "context_sensitivity_scope": "same-family closer/farther pairs only",
        "groups": grouped, "entity_order_summary": order_summary}
    (out / "binary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Closer/farther event extension: binary evaluation", "",
        "Qwen3 359M at 19.5k steps. Each decision compares exactly two full target sentences using mean target-token log likelihood. Raw binary choice is primary; target-only PMI is secondary. Uniform-choice chance is 50%.",
        f"Every contrast has {len(groups):,} balanced pairs ({2*len(groups):,} judgments). Direct labels are lexical ceiling controls. Numeric cases, wording lengths, and entity orders repeat underlying event patterns.", "",
        "| Contrast | Evidence | Pairs | Raw choice | PMI choice | First gold | Second gold | Both correct | First chosen |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in grouped:
        if row["grouping"] == "contrast+mode":
            contrast, mode = row["group"].split("|")
            report.append(f"| {contrast} | {mode} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['raw_first_accuracy']:.2%} | {row['raw_second_accuracy']:.2%} | {row['raw_both_correct']:.2%} | {row['raw_first_choice_fraction']:.2%} |")
    report += ["", "'First chosen' is the raw fraction selecting the first named answer in the contrast; a value near 100% can produce 50% accuracy without tracking the context. The two context types in closer/farther are within the same family. Closer/unchanged and farther/unchanged pair an object-movement context with a co-motion context, matched on entities, starting separation, numeric case, unit, wording band, and context entity order; event type and other syntax also change.",
        "", "## Wording bands", "",
        "| Contrast | Evidence | Band | Pairs | Raw choice | PMI choice | Both correct |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in grouped:
        if row["grouping"] == "contrast+mode+length_band":
            contrast, mode, band = row["group"].split("|")
            report.append(f"| {contrast} | {mode} | {band} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['raw_both_correct']:.2%} |")
    report += ["", "## Context entity order", "",
        "| Contrast | Evidence | Context order | Pairs | Raw choice | PMI choice | Both correct |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in grouped:
        if row["grouping"] == "contrast+mode+context_entity_order":
            contrast, mode, order = row["group"].split("|")
            report.append(f"| {contrast} | {mode} | {order} | {row['n_pairs']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['raw_both_correct']:.2%} |")
    if order_summary:
        report += ["", "## Matched entity-order consistency", "",
            "Each row pairs person-first and object-first contexts with the same event, outcome, entities, distances, targets, and wording band. Order also changes local phrasing or clause position; the flip rate is diagnostic rather than a pure order effect.", "",
            "| Contrast | Evidence | Order pairs | Person-first raw | Object-first raw | Any raw prediction flip | All four correct |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for row in order_summary:
            if row["grouping"] == "contrast+mode":
                contrast, mode = row["group"].split("|")
                report.append(f"| {contrast} | {mode} | {row['n_order_pairs']} | {row['raw_person_first_accuracy']:.2%} | {row['raw_object_first_accuracy']:.2%} | {row['raw_any_prediction_flip']:.2%} | {row['raw_all_four_correct']:.2%} |")
    report += ["", "## Fixed-target context sensitivity", "",
        "For closer versus farther, the same target is compared across matched toward/away contexts; target-only PMI subtraction cancels. This measure is not applied to the other contrasts because their paired contexts come from different physical event families.", "",
        "| Evidence | Pairs | Context sensitivity | Both targets correct |",
        "|---|---:|---:|---:|"]
    for row in grouped:
        if row["grouping"] == "contrast+mode" and row["group"].startswith("closer_vs_farther|"):
            mode = row["group"].split("|")[1]
            report.append(f"| {mode} | {row['n_pairs']} | {row['context_accuracy']:.2%} | {row['context_both_correct']:.2%} |")
    report += ["", "The archived three-way report offered closer, farther, and same-distance simultaneously and is retained as a secondary response-preference diagnostic. It is not the primary accuracy measure. All binary judgments were calculated from the saved conditional target scores; no model rerun or three-answer ranking was used for the results above."]
    (out / "report.md").write_text("\n".join(report) + "\n")
    generate_bias_table(out)
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
