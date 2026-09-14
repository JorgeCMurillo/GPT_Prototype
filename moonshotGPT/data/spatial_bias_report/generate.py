#!/usr/bin/env python3
"""Make comparable answer-preference tables from saved spatial-probe decisions."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPATIAL_RESULTS = REPO / "runs/research/bos_aligned_proto"
MARKER = "See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior."


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value):
    if value in (True, "True"):
        return True
    if value in (False, "False"):
        return False
    raise ValueError(f"Expected a Boolean, received {value!r}")


def predictions(first_margin, second_margin):
    """Margins favor the gold first answer in C1 and gold second answer in C2."""
    first_margin, second_margin = float(first_margin), float(second_margin)
    return (("first" if first_margin > 0 else "second" if first_margin < 0 else "tie"),
            ("second" if second_margin > 0 else "first" if second_margin < 0 else "tie"))


def pair(record, labels, attrs, margins, context=None):
    assert len(labels) == 2 and labels[0] != labels[1]
    return {"pair_id": record.get("pair_id", record.get("probe_id")), "first_label": labels[0],
        "second_label": labels[1], "attrs": attrs,
        "predictions": {m: predictions(*values) for m, values in margins.items()},
        "context_correct": context or {}}


def close_far_labels(row):
    condition = row.get("target_negation_pattern_id") or row.get("condition_id")
    return {
        "negate_close": ("close", "not close"),
        "negate_far": ("not far", "far"),
        "both_negated": ("not far", "not close"),
    }.get(condition, ("close", "far"))


def from_paired_items(path):
    rows = read_csv(path)
    fields = set(rows[0])
    paired = []
    if {"raw_close_margin", "raw_far_margin"} <= fields:
        kind = "close_far_evidence"
        for row in rows:
            assert row.get("score_reduction", "mean") == "mean"
            attrs = {k: row.get(k, "") for k in ["condition_id", "evidence_type",
                "mode", "wording_family_id", "scene_role", "scene_id", "context_structure",
                "context_entity_order", "target_order", "include_in_primary_scene_summary"]}
            margins = {m: (row[m + "_close_margin"], row[m + "_far_margin"])
                       for m in ["raw", "pmi"]}
            context = (parse_bool(row["context_close_correct"]),
                       parse_bool(row["context_far_correct"]))
            paired.append(pair(row, ("close", "far"), attrs, margins,
                               {"context": context}))
    elif {"above_margin", "below_margin"} <= fields:
        kind = "above_below_definition"
        labels_by_direction = {
            "word_to_definition": ("above definition", "below definition"),
            "definition_to_word": ("above", "below"),
            "word_to_synonym": ("over", "under"),
            "synonym_to_word": ("above", "below"),
        }
        for row in rows:
            if row["score_reduction"] != "mean":
                continue
            attrs = {k: row.get(k, "") for k in ["probe_family", "direction",
                "design_block", "phrase_pair_id", "phrase_register",
                "definition_structure_id", "context_stem_id", "entity_noun_id",
                "context_entity_order"]}
            context = (float(row["context_sensitivity_above_margin"]) > 0,
                       float(row["context_sensitivity_below_margin"]) > 0)
            paired.append(pair(row, labels_by_direction[row["direction"]], attrs,
                {"raw": (row["above_margin"], row["below_margin"])},
                {"context": context}))
    elif {"close_margin", "far_margin"} <= fields:
        kind = "close_far_definition" if "direction" in fields else "close_far_situation"
        for row in rows:
            if row["score_reduction"] != "mean":
                continue
            attrs = {k: row.get(k, "") for k in ["direction", "condition_id",
                "target_negation_pattern_id", "scenario_family", "context_stem_id",
                "adjective_pair_id", "modifier_id", "definition_structure_id",
                "context_structure_label", "context_form_id", "reference_form_id",
                "reference_order_id", "sentence_structure_id"]}
            labels = ("close definition", "far definition") if row.get(
                "direction") == "word_to_definition" else close_far_labels(row)
            context = (float(row["context_sensitivity_close_margin"]) > 0,
                float(row["context_sensitivity_far_margin"]) > 0)
            paired.append(pair(row, labels, attrs,
                {"raw": (row["close_margin"], row["far_margin"])},
                {"context": context}))
    else:
        raise ValueError(f"Unrecognized paired item schema: {path}")
    assert len({p["pair_id"] for p in paired}) == len(paired)
    return kind, paired


def from_event_pairs(path):
    rows = read_csv(path)
    paired = []
    for row in rows:
        attrs = {k: row.get(k, "") for k in ["contrast", "mode", "length_band",
            "context_entity_order", "direct_context_entity_order", "first_event_family",
            "second_event_family", "same_event_family"]}
        margins = {m: (row[m + "_first_margin"], row[m + "_second_margin"])
                   for m in ["raw", "pmi"]}
        context = {}
        if row["contrast"] == "closer_vs_farther":
            context = {"context": (parse_bool(row["context_first_correct"]),
                                   parse_bool(row["context_second_correct"]))}
        paired.append(pair(row, (row["first_outcome"], row["second_outcome"]),
                           attrs, margins, context))
    return "closer_farther_event", paired


def from_non_numeric_pairs(path):
    rows = read_csv(path)
    paired = []
    for row in rows:
        attrs = {k: row.get(k, "") for k in ["dataset", "length_band",
            "template_family", "template_id", "object_id"]}
        margins = {m: (row[m + "_a_margin"], row[m + "_b_margin"])
                   for m in ["raw", "pmi"]}
        context = {"context": (parse_bool(row["context_a_correct"]),
                               parse_bool(row["context_b_correct"]))}
        paired.append(pair(row, ("closer", "farther"), attrs, margins, context))
    return "closer_farther_person_movement", paired


def from_parent_binary(path, matches_path):
    items = read_csv(path)
    by_id = {(r["method"], r["probe_id"], r["contrast"]): r for r in items}
    assert len(by_id) == len(items)
    paired = []
    for link in read_csv(matches_path):
        method = link["method"]
        a = by_id[method, link["probe_a"], link["contrast"]]
        b = by_id[method, link["probe_b"], link["contrast"]]
        labels = tuple(link["contrast"].split("_vs_"))
        assert a["outcome"] == labels[0] and b["outcome"] == labels[1]
        pair_id = link["contrast"] + "__" + link["matched_group_id"] + "__" + link["description_family"]
        attrs = {k: link.get(k, "") for k in ["contrast", "description_family", "length_band"]}
        attrs["condition_id"] = a["condition_id"] + " vs " + b["condition_id"]
        record = {"pair_id": pair_id}
        first = "first" if a["predicted_outcome"] == labels[0] else "second" if a["predicted_outcome"] == labels[1] else "tie"
        second = "first" if b["predicted_outcome"] == labels[0] else "second" if b["predicted_outcome"] == labels[1] else "tie"
        paired.append({"pair_id": pair_id, "first_label": labels[0],
            "second_label": labels[1], "attrs": attrs,
            "predictions": {method: (first, second)},
            "context_correct": {"context": (parse_bool(link["context_a_correct"]),
                                            parse_bool(link["context_b_correct"]))}
        })
    # The input has one row per method; combine methods into one matched pair.
    combined = {}
    for row in paired:
        key = row["pair_id"]
        if key not in combined:
            combined[key] = row
        else:
            assert all(row[k] == combined[key][k] for k in ["first_label", "second_label", "attrs"])
            combined[key]["predictions"].update(row["predictions"])
    assert all(set(r["predictions"]) == {"raw", "pmi"} for r in combined.values())
    return "closer_farther_parent_binary", list(combined.values())


def detect(out):
    if (out / "binary_pair_scores.csv").exists():
        return from_event_pairs(out / "binary_pair_scores.csv")
    if (out / "pair_scores.csv").exists():
        fields = set(read_csv(out / "pair_scores.csv")[0])
        if {"raw_a_margin", "raw_b_margin"} <= fields:
            return from_non_numeric_pairs(out / "pair_scores.csv")
    if (out / "matched_pair_scores.csv").exists() and (out / "item_scores.csv").exists():
        return from_parent_binary(out / "item_scores.csv", out / "matched_pair_scores.csv")
    if (out / "item_scores.csv").exists():
        fields = set(read_csv(out / "item_scores.csv")[0])
        if ({"raw_close_margin", "raw_far_margin"} <= fields or
            {"close_margin", "far_margin"} <= fields or
            {"above_margin", "below_margin"} <= fields):
            return from_paired_items(out / "item_scores.csv")
    raise ValueError("No supported matched binary pair scores in this report directory")


def groupings(kind, pairs):
    if kind == "above_below_definition":
        return [("probe_family", "direction"),
                ("probe_family", "direction", "design_block"),
                ("probe_family", "direction", "phrase_pair_id"),
                ("probe_family", "direction", "definition_structure_id"),
                ("probe_family", "direction", "context_stem_id"),
                ("probe_family", "direction", "design_block", "context_entity_order"),
                ("probe_family", "direction", "design_block", "entity_noun_id")]
    if kind == "close_far_definition":
        return [("direction",), ("direction", "adjective_pair_id"),
                ("direction", "modifier_id"),
                ("direction", "definition_structure_id"),
                ("direction", "context_stem_id")]
    if kind == "close_far_situation":
        extra = [("condition_id", "scenario_family"),
                 ("condition_id", "reference_order_id"),
                 ("condition_id", "sentence_structure_id")]
        if any(p["attrs"].get("context_form_id") for p in pairs):
            extra.append(("context_form_id", "reference_form_id"))
        return [("condition_id",), *extra]
    if kind == "close_far_evidence":
        attrs = pairs[0]["attrs"]
        if any(p["attrs"].get("mode") for p in pairs):
            return [("mode", "wording_family_id"),
                    ("mode", "wording_family_id", "target_order"),
                    ("mode", "wording_family_id", "context_structure"),
                    ("mode", "wording_family_id", "context_entity_order")]
        if any(p["attrs"].get("scene_role") for p in pairs):
            return [("scene_role",), ("scene_role", "scene_id"),
                    ("scene_role", "context_structure", "target_order")]
        return [("condition_id",), ("condition_id", "context_entity_order"),
                ("condition_id", "target_order")]
    if kind == "closer_farther_event":
        return [("contrast", "mode"), ("contrast", "mode", "length_band"),
                ("contrast", "mode", "context_entity_order"),
                ("contrast", "mode", "first_event_family", "second_event_family")]
    if kind == "closer_farther_person_movement":
        return [("dataset",), ("dataset", "length_band"),
                ("dataset", "template_family")]
    if kind == "closer_farther_parent_binary":
        return [("contrast", "description_family"),
                ("contrast", "description_family", "length_band")]
    raise AssertionError(kind)


def summarize(subset, method, grouping, group, kind):
    n = len(subset)
    assert n
    labels = {(p["first_label"], p["second_label"]) for p in subset}
    assert len(labels) == 1, (grouping, group, labels)
    first_label, second_label = labels.pop()
    patterns = Counter()
    chosen = Counter()
    context = []
    for p in subset:
        a, b = p["predictions"][method]
        chosen.update([a, b])
        pattern = {( "first", "second"): "both_correct",
                   ("first", "first"): "always_first",
                   ("second", "second"): "always_second",
                   ("second", "first"): "both_wrong"}.get((a, b), "tie_in_pair")
        patterns[pattern] += 1
        if "context" in p["context_correct"]:
            context.extend(p["context_correct"]["context"])
    assert sum(patterns.values()) == n and sum(chosen.values()) == 2*n
    first_gold = (patterns["both_correct"] + patterns["always_first"])/n
    second_gold = (patterns["both_correct"] + patterns["always_second"])/n
    accuracy = (first_gold + second_gold)/2
    first_choice = chosen["first"]/(2*n)
    second_choice = chosen["second"]/(2*n)
    assert abs(first_choice + second_choice + chosen["tie"]/(2*n)-1) < 1e-12
    return {"source_kind": kind, "grouping": grouping, "group": group,
        "method": method, "first_answer": first_label, "second_answer": second_label,
        "n_pairs": n, "n_judgments": 2*n,
        "accuracy": accuracy, "first_gold_accuracy": first_gold,
        "second_gold_accuracy": second_gold,
        "first_chosen_fraction": first_choice,
        "second_chosen_fraction": second_choice,
        "choice_gap_percentage_points": 100*(first_choice-second_choice),
        "both_correct_fraction": patterns["both_correct"]/n,
        "always_first_fraction": patterns["always_first"]/n,
        "always_second_fraction": patterns["always_second"]/n,
        "both_wrong_fraction": patterns["both_wrong"]/n,
        "pair_with_tie_fraction": patterns["tie_in_pair"]/n,
        "tie_judgments": chosen["tie"],
        "context_sensitivity_accuracy": sum(context)/len(context) if context else ""}


def generate_one(out, link_report=True):
    kind, pairs = detect(out)
    assert pairs
    specs = groupings(kind, pairs)
    rows = []
    for factors in specs:
        buckets = defaultdict(list)
        for p in pairs:
            key = tuple(p["attrs"].get(k, "") for k in factors)
            buckets[key].append(p)
        for key, subset in buckets.items():
            label = "|".join(key)
            for method in ["raw", "pmi"]:
                if all(method in p["predictions"] for p in subset):
                    rows.append(summarize(subset, method, "+".join(factors), label, kind))
    assert rows
    write_csv(out / "bias_table.csv", rows)
    primary = "+".join(specs[0])
    selected = [r for r in rows if r["grouping"] == primary and r["method"] == "raw"]
    report = ["# Answer-preference diagnostics", "",
        "Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.",
        "`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.", "",
        "| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        report.append(f"| {r['group'].replace('|', ' / ')} | {r['first_answer']} / {r['second_answer']} | {r['n_pairs']} | {r['accuracy']:.2%} | {r['first_gold_accuracy']:.2%} | {r['second_gold_accuracy']:.2%} | {r['first_chosen_fraction']:.2%} / {r['second_chosen_fraction']:.2%} | {r['both_correct_fraction']:.2%} | {r['always_first_fraction']:.2%} | {r['always_second_fraction']:.2%} | {r['both_wrong_fraction']:.2%} |")
    report += ["", "`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.", ""]
    (out / "bias_table.md").write_text("\n".join(report))
    metadata = {"source_kind": kind, "n_unique_pairs": len(pairs),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [out / "binary_pair_scores.csv", out / "pair_scores.csv",
                      out / "matched_pair_scores.csv", out / "item_scores.csv"] if p.exists()},
        "primary_grouping": primary, "groupings": ["+".join(s) for s in specs],
        "n_table_rows": len(rows), "raw_table_file": "bias_table.csv",
        "display_file": "bias_table.md"}
    (out / "bias_table_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report_path = out / "report.md"
    if link_report and report_path.exists():
        original = report_path.read_text()
        if MARKER not in original:
            report_path.write_text(original.rstrip() + "\n\n" + MARKER + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", type=Path)
    group.add_argument("--all-spatial", action="store_true")
    args = parser.parse_args()
    if args.results_dir:
        out = args.results_dir.resolve()
        result = generate_one(out)
        print(json.dumps({"directory": str(out), **result}, indent=2))
        return
    completed, skipped = [], []
    for report in SPATIAL_RESULTS.rglob("report.md"):
        out = report.parent
        if not any(term in str(out) for term in ["close_far", "closer_farther", "above_below"]):
            continue
        try:
            completed.append((str(out), generate_one(out)))
        except ValueError as exc:
            skipped.append((str(out), str(exc)))
    print(json.dumps({"completed": len(completed), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
