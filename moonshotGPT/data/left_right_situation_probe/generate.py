#!/usr/bin/env python3
"""Generate fixed-screen left/right scenarios with observable scene metadata."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMED = ("far-left", "inner-left", "center", "inner-right", "far-right")
ORDERS = {
    "named_positions": "far-left, inner-left, center, inner-right, far-right",
    "numbered_left_to_right": "1, 2, 3, 4, 5",
    "numbered_right_to_left": "5, 4, 3, 2, 1",
    "lettered_positions": "A, B, C, D, E",
}


def position_label(position, evidence):
    if evidence == "named_positions":
        return NAMED[position - 1]
    if evidence == "lettered_positions":
        return "ABCDE"[position - 1]
    if evidence == "numbered_right_to_left":
        return str(6 - position)
    assert evidence == "numbered_left_to_right"
    return str(position)


def relation(target, reference):
    assert target != reference
    return "left" if target < reference else "right"


def motion(points):
    return "still" if points[0] == points[1] else "right" if points[1] > points[0] else "left"


def frame(length):
    return {
        "compact": "You view a fixed screen straight on. ",
        "standard": "You view a fixed screen straight on, with left and right taken from your viewpoint. ",
        "expanded": "You view a fixed screen straight on, with left and right taken from your viewpoint. The screen and your viewpoint stay unchanged throughout the scene. ",
    }[length]


def physical_context(case, truth, pair, evidence, length, order):
    prefix = f"Fixed screen slots, left to right: {ORDERS[evidence]}. "
    prefix += {"compact": "", "standard": "The screen stays fixed. ",
               "expanded": "The screen and viewpoint stay fixed. "}[length]
    clauses = {}
    for entity in ("target", "reference"):
        start, end = case[truth][entity]
        first, last = position_label(start, evidence), position_label(end, evidence)
        subject = f"The {pair[entity]} icon"
        if length != "expanded":
            if start != end:
                clauses[entity] = f"{subject} moves from {first} to {last}."
            else:
                verb = "is" if case["event_family"] == "static_placement" else "stays"
                clauses[entity] = f"{subject} {verb} at {last}."
            continue
        if start == end:
            clauses[entity] = f"{subject} starts at {first} and stays there."
        else:
            clauses[entity] = f"{subject} starts at {first}, then moves to {last}."
    sequence = ("target", "reference") if order == "target_first" else ("reference", "target")
    return prefix + " ".join(clauses[e] for e in sequence)


def sentence(pair, truth, order, positioned=False):
    subject, reference = pair["target"], pair["reference"]
    if order == "reference_first":
        subject, reference = reference, subject
        truth = "right" if truth == "left" else "left"
    verb = "is positioned" if positioned else "is"
    return f"The {subject} icon {verb} to the {truth} of the {reference} icon.", truth


def validate_case(case):
    family = case["event_family"]
    for truth in ("left", "right"):
        a, b = case[truth]["target"], case[truth]["reference"]
        assert all(type(n) is int and 1 <= n <= 5 for n in a + b)
        assert relation(a[1], b[1]) == truth
        crossed = relation(a[0], b[0]) != truth
        if family == "static_placement":
            assert a[0] == a[1] and b[0] == b[1]
        elif family == "target_crosses":
            assert a[0] != a[1] and b[0] == b[1] and crossed
        elif family == "reference_crosses":
            assert a[0] == a[1] and b[0] != b[1] and crossed
        elif family == "target_moves_without_crossing":
            assert a[0] != a[1] and b[0] == b[1] and not crossed
        else:
            assert family == "both_move" and a[0] != a[1] and b[0] != b[1]
            assert crossed == (case["event_subtype"] == "order_reversed")
    if family == "target_moves_without_crossing":
        assert motion(case["left"]["target"]) == motion(case["right"]["target"])


def make_row(case, pair, evidence, length, context_order, target_order):
    direct = case is None
    case_id = evidence if direct else case["id"]
    row = {
        "probe_id": "__".join((case_id, pair["id"], evidence, length, context_order, target_order)),
        "probe_version": "1.2", "probe_family": "control" if direct else "applied",
        "event_family": "direct_label_control" if direct else case["event_family"],
        "event_subtype": evidence.removeprefix("direct_label_") if direct else case["event_subtype"],
        "case_id": case_id, "object_pair_id": pair["id"],
        "target_object": pair["target"], "reference_object": pair["reference"],
        "reference_frame": "reader_view_fixed_screen", "evidence_type": evidence,
        "displayed_slot_labels_left_to_right": "" if direct else ORDERS[evidence],
        "length_band": length, "context_entity_order": context_order,
        "last_mentioned_entity": "reference" if context_order == "target_first" else "target",
        "target_entity_order": target_order,
        "correct_target_for_context1": "Target1", "correct_target_for_context2": "Target2",
    }
    for i, truth in enumerate(("left", "right"), 1):
        row[f"Target{i}"], row[f"target{i}_relation_word"] = sentence(pair, truth, target_order)
        row[f"context{i}_target_relation"] = truth
        row[f"Context{i}"] = (frame(length) + sentence(pair, truth, context_order, evidence == "direct_label_positioned")[0]
                              if direct else physical_context(case, truth, pair, evidence, length, context_order))
        state = None if direct else case[truth]
        for entity in ("target", "reference"):
            row[f"{entity}_motion_{truth}"] = "" if direct else motion(state[entity])
            for point, index in (("start", 0), ("end", 1)):
                row[f"{entity}_{point}_{truth}"] = "" if direct else state[entity][index]
                row[f"{entity}_{point}_label_{truth}"] = "" if direct else position_label(state[entity][index], evidence)
        row[f"initial_relation_{truth}"] = "" if direct else relation(state["target"][0], state["reference"][0])
        row[f"final_relation_{truth}"] = truth
        row[f"order_reversed_{truth}"] = "" if direct else row[f"initial_relation_{truth}"] != truth
    for field in ("Context1", "Context2", "Target1", "Target2"):
        row[field + "_word_count"] = len(re.findall(r"\b\w+\b", row[field]))
    assert row["Context1"] != row["Context2"] and row["Target1"] != row["Target2"]
    return row


def make_matches(rows):
    factors = ("case_id", "object_pair_id", "evidence_type", "length_band", "context_entity_order", "target_entity_order")
    lookup = {tuple(r[k] for k in factors): r for r in rows}
    assert len(lookup) == len(rows)
    matches = []
    for row in rows:
        variants = []
        for field in ("context_entity_order", "target_entity_order"):
            if row[field] == "target_first":
                variants.append((field, {field: "reference_first"}))
        if row["length_band"] == "compact":
            variants.extend(("length_band", {"length_band": v}) for v in ("standard", "expanded"))
        if row["evidence_type"] == "named_positions":
            variants.extend(("evidence_type", {"evidence_type": v}) for v in ORDERS if v != "named_positions")
        if row["evidence_type"] == "numbered_left_to_right":
            variants.append(("numbering_direction", {"evidence_type": "numbered_right_to_left"}))
        if row["evidence_type"] == "direct_label_plain":
            variants.append(("direct_label_style", {"case_id": "direct_label_positioned", "evidence_type": "direct_label_positioned"}))
        for control, changes in variants:
            other = lookup[tuple(changes.get(k, row[k]) for k in factors)]
            fixed = ("Context1", "Context2") if control == "target_entity_order" else ("Target1", "Target2")
            assert all(row[k] == other[k] for k in fixed)
            for truth in ("left", "right"):
                for entity in ("target", "reference"):
                    for point in ("start", "end"):
                        key = f"{entity}_{point}_{truth}"
                        assert row[key] == other[key]
            matches.append({"control": control, "base_probe_id": row["probe_id"], "variant_probe_id": other["probe_id"], "case_id": row["case_id"], "event_family": row["event_family"]})
    return matches


def build(config):
    for case in config["cases"]:
        validate_case(case)
    rows = [make_row(*args) for args in product(config["cases"], config["object_pairs"], config["evidence_types"], config["length_bands"], config["context_entity_orders"], config["target_entity_orders"])]
    rows.extend(make_row(None, pair, "direct_label_" + style, length, co, to)
                for pair, style, length, co, to in product(config["object_pairs"], config["direct_label_styles"], config["length_bands"], config["context_entity_orders"], config["target_entity_orders"]))
    assert len(rows) == 2400 and len({r["probe_id"] for r in rows}) == len(rows)
    assert Counter(r["event_family"] for r in rows) == {"static_placement": 576, "target_crosses": 384, "reference_crosses": 384, "target_moves_without_crossing": 384, "both_move": 576, "direct_label_control": 96}
    return rows, make_matches(rows)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    source = ROOT / "components.json"
    config = json.loads(source.read_text())
    rows, matches = build(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "probes.csv", rows)
    (args.out_dir / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_csv(args.out_dir / "variant_matches.csv", matches)
    review = [r for r in rows if r["object_pair_id"] == "ball_cone" and r["length_band"] == "compact" and r["context_entity_order"] == r["target_entity_order"] == "target_first"]
    write_csv(args.out_dir / "review_examples.csv", review)
    catalog = [{"case_id": c["id"], "event_family": c["event_family"], "event_subtype": c["event_subtype"], **{f"{entity}_{point}_{truth}": c[truth][entity][i] for truth in ("left", "right") for entity in ("target", "reference") for point, i in (("start", 0), ("end", 1))}} for c in config["cases"]]
    write_csv(args.out_dir / "case_catalog.csv", catalog)
    report = ["# Fixed-frame left/right review examples", "", "All examples use ball/cone, compact wording, and target-first context and answer orders.", ""]
    for r in review:
        report.extend([f"## {r['case_id']} / {r['evidence_type']}", "", f"C1: {r['Context1']}", "", f"T1: {r['Target1']}", "", f"C2: {r['Context2']}", "", f"T2: {r['Target2']}", ""])
    (args.out_dir / "review_examples.md").write_text("\n".join(report))
    manifest = {"probe_name": config["probe_name"], "version": config["version"], "paired_rows": len(rows), "binary_judgments": 2 * len(rows), "conditional_likelihoods": 4 * len(rows), "applied_rows": 2304, "direct_label_control_rows": 96, "physical_cases": len(config["cases"]), "object_pairs": len(config["object_pairs"]), "by_event_family": dict(Counter(r["event_family"] for r in rows)), "by_evidence_type": dict(Counter(r["evidence_type"] for r in rows)), "variant_matches": len(matches), "components_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "validation": "passed"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
