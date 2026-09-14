#!/usr/bin/env python3
"""Equal-weight applied movement macro from binary pair scores and prior person movement."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from data.closer_farther_probe.generate import write_csv

PERSON_RESULT = (REPO / "runs/research/bos_aligned_proto/closer_farther_probe"
    / "qwen3_359m_step19500_non_numeric_v1_2")
DEFINITION_RESULT = (REPO / "runs/research/bos_aligned_proto/close_far_definition_probe"
    / "qwen3_359m_step19500_v1")
FAMILIES = ["person_moves_reference_stationary", "reference_object_moves",
            "object_moves_vs_both_move_cross_family"]
CONTRASTS = ["closer_vs_farther", "closer_vs_unchanged", "farther_vs_unchanged"]
MODES = ["absent", "specific"]
BANDS = ["compact", "standard", "expanded"]
ORDERS = ["person_first", "object_first"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    assert value in ("True", "False")
    return value == "True"


def cell_mean(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.results_dir
    manifest = json.loads((out / "input_manifest.json").read_text())
    assert manifest["version"] == "1.2", "The macro is defined for the order-balanced v1.2 probe."
    evaluation = json.loads((out / "summary.json").read_text())
    person_evaluation = json.loads((PERSON_RESULT / "summary.json").read_text())
    definition_evaluation = json.loads((DEFINITION_RESULT / "summary.json").read_text())
    for key in ["model", "score_reduction", "dtype", "attention_implementation"]:
        assert evaluation[key] == person_evaluation[key], key
    for key in ["model", "dtype", "attention_implementation"]:
        assert evaluation[key] == definition_evaluation[key], key
    assert definition_evaluation["primary_reduction"] == "mean"
    assert evaluation["score_reduction"] == "mean"

    new_pairs = read_csv(out / "binary_pair_scores.csv")
    assert len(new_pairs) == manifest["context_rows"]
    original_objects = {r["object_id"] for r in read_csv(out / "input_direct_examples.csv")}
    assert len(original_objects) == 3
    rows = []
    for pair in new_pairs:
        if pair["mode"] == "direct_label":
            continue
        contrast = pair["contrast"]
        family = ("reference_object_moves" if contrast == "closer_vs_farther"
                  else "object_moves_vs_both_move_cross_family")
        assert pair["context_entity_order"] in ORDERS
        if family == "reference_object_moves":
            assert truth(pair["same_event_family"])
        else:
            assert not truth(pair["same_event_family"])
        row = {"source": "event_extension_v1_2", "family": family,
            "contrast": contrast, "mode": pair["mode"],
            "length_band": pair["length_band"],
            "context_entity_order": pair["context_entity_order"],
            "name_id": pair["name_id"], "object_id": pair["object_id"],
            "numeric_case_id": pair["numeric_case_id"], "unit_id": pair["unit_id"],
            "pair_id": pair["pair_id"]}
        for method in ["raw", "pmi"]:
            row[method + "_first_correct"] = truth(pair[method + "_first_correct"])
            row[method + "_second_correct"] = truth(pair[method + "_second_correct"])
            row[method + "_accuracy"] = (
                row[method + "_first_correct"] + row[method + "_second_correct"]) / 2
            row[method + "_both_correct"] = truth(pair[method + "_both_correct"])
            row[method + "_first_choice_fraction"] = int(pair[method + "_first_chosen_count"]) / 2
        rows.append(row)

    parent_pairs = read_csv(PERSON_RESULT / "pair_scores.csv")
    for pair in parent_pairs:
        if pair["object_id"] not in original_objects or pair["length_band"] not in BANDS:
            continue
        assert pair["dataset"] in ("non_numeric", "numeric_aligned")
        mode = "absent" if pair["dataset"] == "non_numeric" else "specific"
        row = {"source": "person_movement_v1_2", "family": FAMILIES[0],
            "contrast": "closer_vs_farther", "mode": mode,
            "length_band": pair["length_band"],
            "context_entity_order": "legacy_wording",
            "name_id": pair["name_id"], "object_id": pair["object_id"],
            "numeric_case_id": pair["numeric_case_id"], "unit_id": pair["unit_id"],
            "pair_id": pair["pair_id"]}
        for method in ["raw", "pmi"]:
            first = truth(pair[method + "_a_correct"])
            second = truth(pair[method + "_b_correct"])
            row[method + "_first_correct"] = first
            row[method + "_second_correct"] = second
            row[method + "_accuracy"] = (first + second) / 2
            row[method + "_both_correct"] = first and second
            first_margin = float(pair[method + "_a_margin"])
            second_margin = float(pair[method + "_b_margin"])
            row[method + "_first_choice_fraction"] = (
                (first_margin > 0) + (second_margin < 0)) / 2
        rows.append(row)

    assert Counter((r["family"], r["mode"]) for r in rows) == {
        (FAMILIES[0], "absent"): 36, (FAMILIES[0], "specific"): 1296,
        (FAMILIES[1], "absent"): 72, (FAMILIES[1], "specific"): 2592,
        (FAMILIES[2], "absent"): 144, (FAMILIES[2], "specific"): 5184}
    assert len(rows) == 9324
    write_csv(out / "overall_applied_pairs.csv", rows)

    # The hierarchy avoids weighting numeric cases 36 times more heavily than
    # nonnumeric cases or weighting a family by its number of wording variants.
    cells = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ["family", "contrast", "mode",
                                    "length_band", "context_entity_order"])
        cells[key].append(row)
    cell_scores = []
    for key, subset in cells.items():
        cell_scores.append({"family": key[0], "contrast": key[1], "mode": key[2],
            "length_band": key[3], "context_entity_order": key[4],
            "n_pairs": len(subset),
            **{method + "_" + metric: cell_mean(subset, method + "_" + metric)
               for method in ["raw", "pmi"]
               for metric in ["accuracy", "first_correct", "second_correct",
                              "both_correct", "first_choice_fraction"]}})
    write_csv(out / "overall_applied_cells.csv", cell_scores)
    contrast_scores = []
    for family in FAMILIES:
        expected_contrasts = (["closer_vs_farther"] if family != FAMILIES[2]
                              else CONTRASTS[1:])
        expected_orders = ["legacy_wording"] if family == FAMILIES[0] else ORDERS
        for contrast in expected_contrasts:
            modes = []
            for mode in MODES:
                subset = [r for r in cell_scores if r["family"] == family and
                          r["contrast"] == contrast and r["mode"] == mode]
                assert {(r["length_band"], r["context_entity_order"]) for r in subset} == {
                    (band, order) for band in BANDS for order in expected_orders}
                modes.append({"family": family, "contrast": contrast, "mode": mode,
                    "n_pairs": sum(r["n_pairs"] for r in subset),
                    **{method + "_" + metric: cell_mean(subset, method + "_" + metric)
                       for method in ["raw", "pmi"]
                       for metric in ["accuracy", "first_correct", "second_correct",
                                      "both_correct", "first_choice_fraction"]}})
            assert len(modes) == 2
            for mode_row in modes:
                contrast_scores.append(mode_row)
            contrast_scores.append({"family": family, "contrast": contrast,
                "mode": "equal_numeric_nonnumeric_mean",
                "n_pairs": sum(r["n_pairs"] for r in modes),
                **{method + "_" + metric: cell_mean(modes, method + "_" + metric)
                   for method in ["raw", "pmi"]
                   for metric in ["accuracy", "first_correct", "second_correct",
                                  "both_correct", "first_choice_fraction"]}})
    write_csv(out / "overall_contrast_scores.csv", contrast_scores)
    macros = [r for r in contrast_scores if r["mode"] == "equal_numeric_nonnumeric_mean"]
    family_scores = []
    for family in FAMILIES:
        subset = [r for r in macros if r["family"] == family]
        assert len(subset) == (2 if family == FAMILIES[2] else 1)
        family_scores.append({"family": family,
            "contrasts": [r["contrast"] for r in subset],
            "n_pairs": sum(r["n_pairs"] for r in subset),
            **{method + "_" + metric: cell_mean(subset, method + "_" + metric)
               for method in ["raw", "pmi"]
               for metric in ["accuracy", "both_correct", "first_choice_fraction"]}})
    overall = {method + "_" + metric: cell_mean(family_scores, method + "_" + metric)
        for method in ["raw", "pmi"]
        for metric in ["accuracy", "both_correct", "first_choice_fraction"]}
    direct = [r for r in new_pairs if r["mode"] == "direct_label"]
    assert len(direct) == 72 and Counter(r["contrast"] for r in direct) == {
        contrast: 24 for contrast in CONTRASTS}
    direct_controls = {method + "_accuracy": sum(
        truth(r[method + "_first_correct"]) + truth(r[method + "_second_correct"])
        for r in direct)/(2*len(direct)) for method in ["raw", "pmi"]}
    definition_controls = definition_evaluation["scores"]["mean"]
    summary = {"score_definition": "Equal mean of three applied comparison families; within each: equal contrast mean, then equal numeric/nonnumeric mean, then equal wording-band/entity-order mean, then paired-case mean.",
        "primary_metric": "raw binary mean-full-target-token accuracy",
        "uniform_choice_chance": 0.5,
        "model": evaluation["model"], "score_reduction": "mean",
        "included_sources": [str(out), str(PERSON_RESULT)],
        "person_movement_source": "Aligned nonnumeric and numeric person-movement pairs, restricted to the three original objects and compact/standard/expanded forms.",
        "family_scores": family_scores, "contrast_scores": contrast_scores,
        "overall_applied_mean": overall, "direct_label_control": direct_controls,
        "definition_control": {"source": str(DEFINITION_RESULT),
            "direction_balanced_accuracy": definition_controls["direction_balanced_accuracy"],
            "word_to_definition_accuracy": definition_controls["by_direction"]["word_to_definition"]["completion_choice_accuracy"],
            "definition_to_word_accuracy": definition_controls["by_direction"]["definition_to_word"]["completion_choice_accuracy"]},
        "total_applied_pairs": len(rows)}
    (out / "overall_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Closer/farther applied movement score", "",
        "Each accuracy is a binary choice between two full targets, scored by mean target-token log likelihood. The raw score is primary; PMI is secondary. Uniform-choice chance is 50%.",
        "The overall mean gives equal weight to three comparison families. Within a family, it averages matched case judgments, then wording lengths and available entity orders, numeric and nonnumeric evidence, and finally the family's contrasts. Direct labels and literal definitions are separate controls.", "",
        "| Comparison family | Contrasts | Pairs | Raw accuracy | PMI accuracy | Both contexts correct |",
        "|---|---|---:|---:|---:|---:|"]
    labels = {FAMILIES[0]: "Person moves; reference stationary",
        FAMILIES[1]: "Reference object moves",
        FAMILIES[2]: "Reference object moves vs both move (cross-family)"}
    for row in family_scores:
        report.append(f"| {labels[row['family']]} | {', '.join(row['contrasts'])} | {row['n_pairs']:,} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['raw_both_correct']:.2%} |")
    report += [f"| **Overall applied mean** | Equal family weight | {len(rows):,} | **{overall['raw_accuracy']:.2%}** | {overall['pmi_accuracy']:.2%} | {overall['raw_both_correct']:.2%} |",
        "", "The cross-family row averages closer-versus-unchanged and farther-versus-unchanged equally. Its two contexts differ in physical event and syntax; it does not isolate a within-family unchanged judgment. The person-moving row uses the previously evaluated matched numeric/nonnumeric probe and has no matched entity-order variants. Numeric and nonnumeric evidence receive equal weight despite very different row counts.",
        "", "## Contrast details", "",
        "| Family | Contrast | Evidence | Pairs | Raw accuracy | Closer/farther-side accuracy | Other-side accuracy | First answer chosen |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in contrast_scores:
        if row["mode"] not in MODES:
            continue
        report.append(f"| {labels[row['family']]} | {row['contrast']} | {row['mode']} | {row['n_pairs']:,} | {row['raw_accuracy']:.2%} | {row['raw_first_correct']:.2%} | {row['raw_second_correct']:.2%} | {row['raw_first_choice_fraction']:.2%} |")
    report += ["", "## Controls and scope", "",
        f"Direct comparative labels: {direct_controls['raw_accuracy']:.2%} raw, {direct_controls['pmi_accuracy']:.2%} PMI across the three binary contrasts. They are excluded from the applied mean.",
        f"Literal close/far definitions (separate categorical probe): {definition_controls['by_direction']['word_to_definition']['completion_choice_accuracy']:.2%} word-to-definition and {definition_controls['by_direction']['definition_to_word']['completion_choice_accuracy']:.2%} definition-to-word; equal direction mean {definition_controls['direction_balanced_accuracy']:.2%}. This is excluded from the closer/farther movement mean.",
        "The three family rows are diagnostic categories, not independent random samples: names, objects, numbers, units, lengths, and entity orders reuse underlying cases. Inspect `overall_applied_cells.csv`, `overall_applied_pairs.csv`, and `entity_order_summary.csv` for failures hidden by the mean.", ""]
    (out / "overall_report.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
