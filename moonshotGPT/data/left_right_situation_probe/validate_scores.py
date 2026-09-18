#!/usr/bin/env python3
"""Independently reconstruct saved likelihood choices and family summaries."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results
    tokens = {}
    for line in (out / "token_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        values = row["token_log_probs"]
        assert values and all(math.isfinite(v) for v in values)
        key = row["context"], row["target"]
        assert key not in tokens
        tokens[key] = mean(values)
    items = [json.loads(line) for line in (out / "item_scores.jsonl").read_text().splitlines()]
    inputs = [json.loads(line) for line in (out / "input_probes.jsonl").read_text().splitlines()]
    assert len(items) == len(inputs) == 2400
    for row, original in zip(items, inputs):
        assert all(row[k] == v for k, v in original.items())
        for c in (1, 2):
            for t in (1, 2):
                assert math.isclose(row[f"S{c}{t}"], tokens[row[f"Context{c}"], row[f"Target{t}"]], abs_tol=2e-6)
        predictions = []
        correct = []
        for c in (1, 2):
            scores = row[f"S{c}1"], row[f"S{c}2"]
            selected = 0 if scores[0] == scores[1] else 1 if scores[0] > scores[1] else 2
            correct.append(selected == c)
            predictions.append("tie" if not selected else row[f"target{selected}_relation_word"])
        assert row["left_correct"] == correct[0] and row["right_correct"] == correct[1]
        assert row["binary_accuracy"] == sum(correct) / 2
        assert row["both_correct"] == all(correct)
        assert row["left_word_chosen_count"] == predictions.count("left")
        assert row["exact_choice_ties"] == predictions.count("tie")
    summary = json.loads((out / "summary.json").read_text())
    family_scores = []
    for family, result in summary["families"].items():
        subset = [r for r in items if r["event_family"] == family]
        accuracy = mean(r["binary_accuracy"] for r in subset)
        assert math.isclose(result["all_evidence"]["binary_accuracy"], accuracy)
        assert math.isclose(result["all_evidence"]["both_correct_fraction"], mean(r["both_correct"] for r in subset))
        if family != "direct_label_control":
            family_scores.append(accuracy)
    assert math.isclose(summary["overall_applied"]["equal_family_binary_accuracy"], mean(family_scores))
    by_id = {r["probe_id"]: r for r in items}
    with (out / "variant_consistency.csv").open() as stream:
        links = list(csv.DictReader(stream))
    for link in links:
        a, b = by_id[link["base_probe_id"]], by_id[link["variant_probe_id"]]
        for truth in ("left", "right"):
            flipped = a[f"prediction_in_{truth}_context"] != b[f"prediction_in_{truth}_context"]
            assert link[f"{truth}_prediction_flip"] == str(flipped)
    result = {"validation": "passed", "pairs": len(items), "unique_sequences": len(tokens),
              "matched_links": len(links), "checks": ["input preservation", "finite token scores",
              "token means agree within 2e-6 (float32 reduction)", "all choices and lexical preferences",
              "pair correctness and ties", "family averages", "matched prediction flips"]}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
