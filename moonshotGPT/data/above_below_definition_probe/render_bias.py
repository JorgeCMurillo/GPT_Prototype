#!/usr/bin/env python3
"""Render an explicit above/below preference view from the shared bias table."""

import argparse
import csv
import json
from pathlib import Path


LINK = "See [above/below preference table](above_below_bias.md) for explicit upper- and lower-answer counts."
FRACTIONS = ("accuracy", "first_gold_accuracy", "second_gold_accuracy",
             "first_chosen_fraction", "second_chosen_fraction",
             "both_correct_fraction", "always_first_fraction",
             "always_second_fraction", "both_wrong_fraction")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def combine(rows):
    n = sum(int(row["n_pairs"]) for row in rows)
    assert n > 0
    return {"n_pairs": n, "tie_judgments": sum(int(row["tie_judgments"]) for row in rows),
            **{field: sum(float(row[field]) * int(row["n_pairs"])
        for row in rows) / n for field in FRACTIONS}}


def percentage(value):
    return f"{100 * value:.2f}%"


def add_table(lines, selected):
    lines += ["| Set | Pairs | Accuracy | Upper chosen | Lower chosen | Ties | Upper gold correct | Lower gold correct | Both contexts correct | Same upper answer twice |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in selected:
        n = row["n_pairs"]
        choices = 2 * n
        upper = round(row["first_chosen_fraction"] * choices)
        lower = round(row["second_chosen_fraction"] * choices)
        ties = row["tie_judgments"]
        assert upper + lower + ties == choices
        lines.append(f"| {label} | {n} | {percentage(row['accuracy'])} | "
            f"{upper}/{choices} ({percentage(row['first_chosen_fraction'])}) | "
            f"{lower}/{choices} ({percentage(row['second_chosen_fraction'])}) | {ties} | "
            f"{percentage(row['first_gold_accuracy'])} | "
            f"{percentage(row['second_gold_accuracy'])} | "
            f"{percentage(row['both_correct_fraction'])} | "
            f"{percentage(row['always_first_fraction'])} |")
    lines.append("")


def render(out):
    rows = [row for row in read_csv(out / "bias_table.csv") if row["method"] == "raw"]
    primary = {row["group"]: row for row in rows
               if row["grouping"] == "probe_family+direction"}
    literal = [primary["literal_definition|word_to_definition"],
               primary["literal_definition|definition_to_word"]]
    synonym = [primary["lexical_synonym|word_to_synonym"],
               primary["lexical_synonym|synonym_to_word"]]
    summary = json.loads((out / "summary.json").read_text())
    lines = ["# Above/below answer preference", "",
        "Each probe has one upper-compatible and one lower-compatible context, and both targets are scored after each context. Gold answers are balanced 50/50. ‘Upper chosen’ measures the model's response preference; accuracy measures whether the chosen answer matches the context. ‘Same upper answer twice’ counts pairs in which the same answer wins in both contexts. In word-to-definition rows the targets are height descriptions, not the literal words *above* and *below*.", "",
        "## Literal height definitions", ""]
    add_table(lines, [("All literal pairs (row-weighted)", combine(literal)),
                      ("Above/below → height definition", combine(literal[:1])),
                      ("Height definition → above/below", combine(literal[1:]))])
    lines += [f"Direction-balanced accuracy: {percentage(summary['families']['literal_definition']['direction_balanced_accuracy'])}. The row-weighted accuracy above differs because there are 24 forward and 40 reverse pairs. Both measures exclude over/under.", "",
              "## Over/under synonyms", "",
              "The upper target is *over* in word-to-synonym rows and *above* in synonym-to-word rows. These are lexical mappings and remain separate from height definitions.", ""]
    add_table(lines, [("All synonym pairs (row-weighted)", combine(synonym)),
                      ("Above/below → over/under", combine(synonym[:1])),
                      ("Over/under → above/below", combine(synonym[1:]))])
    lines += [f"Direction-balanced synonym accuracy: {percentage(summary['families']['lexical_synonym']['direction_balanced_accuracy'])}.", "",
              "## Height-phrase sensitivity", ""]
    phrase_labels = {"vertical_higher_lower": "vertically higher / lower",
                     "greater_lesser_height": "greater / lesser height",
                     "higher_up_lower_down": "higher up / lower down",
                     "higher_lower": "higher / lower"}
    phrase_rows = []
    for phrase, label in phrase_labels.items():
        matching = [row for row in rows if row["grouping"] ==
                    "probe_family+direction+phrase_pair_id" and
                    row["group"].startswith("literal_definition|") and
                    row["group"].endswith("|" + phrase)]
        assert len(matching) == 2
        phrase_rows.append((label, combine(matching)))
    add_table(lines, phrase_rows)
    lines += ["## Reverse-context entity order", "",
              "These rows compare first-subject and second-subject contexts within the added inverse-relation block. The noun, phrase pair, clause structure, and target sentences are matched; the relational wording also inverts.", ""]
    order_rows = []
    for order in ("first_subject", "second_subject"):
        match = [row for row in rows if row["grouping"] ==
                 "probe_family+direction+design_block+context_entity_order" and
                 row["group"] == "literal_definition|definition_to_word|entity_order_extension|" + order]
        assert len(match) == 1
        order_rows.append((order.replace("_", " "), combine(match)))
    add_table(lines, order_rows)
    lines += ["The phrase and order cells reuse names, templates, and targets, so their counts are repeated diagnostic variants rather than independent semantic cases. See `bias_table.csv` for every split and `entity_order_report.md` for direct matched-order flips.", ""]
    (out / "above_below_bias.md").write_text("\n".join(lines))
    report_path = out / "report.md"
    if report_path.exists():
        current = report_path.read_text()
        if LINK not in current:
            report_path.write_text(current.rstrip() + "\n\n" + LINK + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    render(args.results_dir)
    print(args.results_dir / "above_below_bias.md")


if __name__ == "__main__":
    main()
