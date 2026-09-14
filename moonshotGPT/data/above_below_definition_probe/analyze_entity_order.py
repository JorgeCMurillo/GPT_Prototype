#!/usr/bin/env python3
"""Compare logically matched first- and second-subject definition probes."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


LINK = "See [matched entity-order diagnostics](entity_order_report.md) for the first- versus second-subject comparison."


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predicted(row, gold):
    margin = float(row[f"{gold}_margin"])
    if margin == 0:
        return "tie"
    return gold if margin > 0 else ("below" if gold == "above" else "above")


def analyze(out):
    items = {row["probe_id"]: row for row in read_csv(out / "item_scores.csv")}
    assert len(items) == len(read_csv(out / "item_scores.csv"))
    matches = read_csv(out / "input_entity_order_matches.csv")
    rows = []
    for match in matches:
        first = items[match["first_subject_probe_id"]]
        second = items[match["second_subject_probe_id"]]
        assert all(first[k] == second[k] == match[k] for k in (
            "probe_family", "phrase_pair_id", "entity_noun_id", "definition_structure_id"))
        assert first["Target1"] == second["Target1"]
        assert first["Target2"] == second["Target2"]
        first_preds = tuple(predicted(first, gold) for gold in ("above", "below"))
        second_preds = tuple(predicted(second, gold) for gold in ("above", "below"))
        first_correct = sum(pred == gold for pred, gold in zip(first_preds, ("above", "below")))
        second_correct = sum(pred == gold for pred, gold in zip(second_preds, ("above", "below")))
        rows.append({
            **match,
            "first_subject_accuracy": first_correct / 2,
            "second_subject_accuracy": second_correct / 2,
            "first_subject_both_correct": first_correct == 2,
            "second_subject_both_correct": second_correct == 2,
            "all_four_correct": first_correct == second_correct == 2,
            "above_prediction_flip": first_preds[0] != second_preds[0],
            "below_prediction_flip": first_preds[1] != second_preds[1],
            "any_prediction_flip": first_preds != second_preds,
            "first_subject_above_prediction": first_preds[0],
            "first_subject_below_prediction": first_preds[1],
            "second_subject_above_prediction": second_preds[0],
            "second_subject_below_prediction": second_preds[1],
        })
    assert len(rows) == len(matches)
    write_csv(out / "entity_order_pair_scores.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["probe_family"]].append(row)
    summary = {}
    for family, group in groups.items():
        n = len(group)
        summary[family] = {"n_matches": n,
            **{field: sum(row[field] for row in group) / n for field in (
                "first_subject_accuracy", "second_subject_accuracy",
                "first_subject_both_correct", "second_subject_both_correct",
                "all_four_correct", "above_prediction_flip",
                "below_prediction_flip", "any_prediction_flip")}}
    (out / "entity_order_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Matched entity-order diagnostics", "",
        "Each match holds the phrase pair, noun, target sentences, and clause structure fixed. The second-subject context inverts the relational wording—for example, ‘second item is lower than first item’ corresponds to ‘first item is above second item.’ This is an inverse-relation test, not an order-only syntax manipulation.", "",
        "| Probe family | Matched pairs | First-subject accuracy | Second-subject accuracy | Both contexts correct in both orders | Any prediction flip |",
        "|---|---:|---:|---:|---:|---:|"]
    for family, values in summary.items():
        report.append(f"| {family} | {values['n_matches']} | " + " | ".join(
            f"{values[field]:.2%}" for field in (
                "first_subject_accuracy", "second_subject_accuracy",
                "all_four_correct", "any_prediction_flip")) + " |")
    report += ["", "`entity_order_pair_scores.csv` contains each matched contrast and separate above- and below-context prediction flips. The matched variants reuse phrase pairs and nouns; percentages describe this probe set rather than independent semantic cases.", ""]
    (out / "entity_order_report.md").write_text("\n".join(report))
    main_report = out / "report.md"
    if main_report.exists():
        current = main_report.read_text()
        if LINK not in current:
            main_report.write_text(current.rstrip() + "\n\n" + LINK + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results_dir), indent=2))


if __name__ == "__main__":
    main()
