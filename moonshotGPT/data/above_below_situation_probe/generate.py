#!/usr/bin/env python3
"""Generate matched above/below scene pairs and diagnostic variants."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHELF_LABELS = {1: "bottom", 2: "lower", 3: "middle", 4: "upper", 5: "top"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def feet(n):
    return f"{n} foot" if n == 1 else f"{n} feet"


def intro(evidence, length):
    if evidence == "named_shelves":
        return {
            "compact": "The rack's shelves run from bottom through lower, middle, and upper to top. ",
            "standard": "In the room, a five-tier rack has bottom, lower, middle, upper, and top shelves, in that vertical order. ",
            "expanded": "In the room, a fixed five-tier rack has bottom, lower, middle, upper, and top shelves, in that vertical order. Each shelf stays at its level throughout the scene. ",
        }[length]
    if evidence == "numbered_steps":
        return {
            "compact": "Steps 1 to 5 go up. ",
            "standard": "On a staircase, steps 1 to 5 go up. ",
            "expanded": "On a staircase, steps 1 to 5 go up. The steps stay fixed as the objects move. ",
        }[length]
    if evidence == "numbered_floors":
        return {
            "compact": "Floors are numbered upward, 1 to 5. ",
            "standard": "In a building, floors are numbered upward, 1 to 5. ",
            "expanded": "In a building, floors are numbered upward, 1 to 5. The floors stay fixed as the objects move. ",
        }[length]
    assert evidence == "measured_height", evidence
    return {
        "compact": "Heights are measured from the floor. ",
        "standard": "In the room, the heights of both objects are measured from the same floor. ",
        "expanded": "In the room, the heights of both objects are measured from the same level floor. The floor stays fixed while the objects move or remain still. ",
    }[length]


def entity_clause(noun, positions, evidence, is_static):
    start, end = positions
    if evidence == "named_shelves":
        first, last = SHELF_LABELS[start], SHELF_LABELS[end]
        if is_static:
            return f"the {noun} is on the {last} shelf"
        if start == end:
            return f"the {noun} remains on the {last} shelf"
        return f"the {noun} moves from the {first} shelf to the {last} shelf"
    if evidence in ("numbered_steps", "numbered_floors"):
        place = "step" if evidence == "numbered_steps" else "floor"
        if is_static:
            return f"the {noun} is on {place} {end}"
        if start == end:
            return f"the {noun} stays on {place} {end}"
        return f"the {noun} moves from {place} {start} to {place} {end}"
    assert evidence == "measured_height", evidence
    if is_static:
        return f"the {noun} is at a height of {feet(end)} from the floor"
    if start == end:
        return f"the {noun} remains at a height of {feet(end)} from the floor"
    return (f"the {noun} moves from a height of {feet(start)} to a height of "
            f"{feet(end)}, measured from the floor")


def physical_context(case, state, pair, evidence, length, order):
    static = case["event_family"] == "static_placement"
    clauses = {
        "target": entity_clause(pair["target"], state["target"], evidence, static),
        "reference": entity_clause(pair["reference"], state["reference"], evidence, static),
    }
    sequence = ("target", "reference") if order == "target_first" else ("reference", "target")
    return (intro(evidence, length) +
            ". ".join(clauses[entity].capitalize() for entity in sequence) + ".")


def direct_context(pair, truth, length, order, style):
    target, reference = pair["target"], pair["reference"]
    verb = "is" if style == "plain" else "is positioned"
    if order == "target_first":
        sentence = f"The {target} {verb} {truth} the {reference}."
    else:
        inverse = "below" if truth == "above" else "above"
        sentence = f"The {reference} {verb} {inverse} the {target}."
    prefix = {
        "compact": "",
        "standard": "In the room, ",
        "expanded": "In the room, the two objects have fixed positions. ",
    }[length]
    return prefix + (sentence[0].lower() + sentence[1:] if length == "standard" else sentence)


def targets(pair, order):
    target, reference = pair["target"], pair["reference"]
    if order == "target_first":
        return (f"The {target} is above the {reference}.",
                f"The {target} is below the {reference}.", "above", "below")
    return (f"The {reference} is below the {target}.",
            f"The {reference} is above the {target}.", "below", "above")


def sign(value):
    assert value != 0
    return 1 if value > 0 else -1


def motion(positions):
    start, end = positions
    return "up" if end > start else "down" if end < start else "still"


def validate_case(case):
    assert case["event_family"] in {
        "static_placement", "target_crosses", "reference_crosses",
        "target_moves_without_crossing", "both_move"}
    for truth in ("above", "below"):
        state = case[truth]
        a0, a1 = state["target"]
        b0, b1 = state["reference"]
        assert all(1 <= n <= 5 for n in (a0, a1, b0, b1))
        assert sign(a1 - b1) == (1 if truth == "above" else -1)
        start_sign, end_sign = sign(a0 - b0), sign(a1 - b1)
        family = case["event_family"]
        if family == "static_placement":
            assert a0 == a1 and b0 == b1
        elif family == "target_crosses":
            assert a0 != a1 and b0 == b1 and start_sign != end_sign
        elif family == "reference_crosses":
            assert a0 == a1 and b0 != b1 and start_sign != end_sign
        elif family == "target_moves_without_crossing":
            assert a0 != a1 and b0 == b1 and start_sign == end_sign
        else:
            assert a0 != a1 and b0 != b1
            if case["event_subtype"] == "order_reversed":
                assert start_sign != end_sign
            else:
                assert start_sign == end_sign
    if case["event_family"] == "target_moves_without_crossing":
        assert motion(case["above"]["target"]) == motion(case["below"]["target"])


def make_row(case, pair, evidence, length, context_order, target_order):
    is_direct = case is None
    case_id = evidence if is_direct else case["id"]
    family = "direct_label_control" if is_direct else case["event_family"]
    probe_id = "__".join((case_id, pair["id"], evidence, length, context_order, target_order))
    target1, target2, word1, word2 = targets(pair, target_order)
    if is_direct:
        style = evidence.removeprefix("direct_label_")
        assert style in ("plain", "positioned")
        context1 = direct_context(pair, "above", length, context_order, style)
        context2 = direct_context(pair, "below", length, context_order, style)
        positions = {k: "" for k in (
            "target_start_above", "target_end_above", "reference_start_above",
            "reference_end_above", "target_start_below", "target_end_below",
            "reference_start_below", "reference_end_below")}
    else:
        context1 = physical_context(case, case["above"], pair, evidence, length, context_order)
        context2 = physical_context(case, case["below"], pair, evidence, length, context_order)
        positions = {f"{entity}_{point}_{truth}": case[truth][entity][index]
                     for truth in ("above", "below")
                     for entity in ("target", "reference")
                     for point, index in (("start", 0), ("end", 1))}
    row = {
        "probe_id": probe_id,
        "probe_version": "1.3",
        "probe_family": "control" if is_direct else "applied",
        "event_family": family,
        "event_subtype": style if is_direct else case["event_subtype"],
        "case_id": case_id,
        "object_pair_id": pair["id"],
        "target_object": pair["target"],
        "reference_object": pair["reference"],
        "evidence_type": evidence,
        "length_band": length,
        "context_entity_order": context_order,
        "target_entity_order": target_order,
        "target1_relation_word": word1,
        "target2_relation_word": word2,
        "target_motion_above": "" if is_direct else motion(case["above"]["target"]),
        "target_motion_below": "" if is_direct else motion(case["below"]["target"]),
        "reference_motion_above": "" if is_direct else motion(case["above"]["reference"]),
        "reference_motion_below": "" if is_direct else motion(case["below"]["reference"]),
        "Context1": context1,
        "Context2": context2,
        "Target1": target1,
        "Target2": target2,
        "correct_target_for_context1": "Target1",
        "correct_target_for_context2": "Target2",
        **positions,
    }
    for field in ("Context1", "Context2", "Target1", "Target2"):
        row[field + "_word_count"] = len(re.findall(r"\b\w+\b", row[field]))
    assert context1 != context2 and target1 != target2
    if not is_direct:
        assert not re.search(r"\b(?:above|below)\b", context1 + " " + context2, re.I)
    return row


def make_matches(rows):
    by_key = {(
        row["case_id"], row["object_pair_id"], row["evidence_type"],
        row["length_band"], row["context_entity_order"], row["target_entity_order"]
    ): row for row in rows}
    assert len(by_key) == len(rows)
    matches = []
    for row in rows:
        case, obj, evidence, length, context_order, target_order = (
            row["case_id"], row["object_pair_id"], row["evidence_type"],
            row["length_band"], row["context_entity_order"], row["target_entity_order"])
        key = (case, obj, evidence, length, context_order, target_order)
        comparisons = []
        if context_order == "target_first":
            comparisons.append(("context_entity_order", (
                case, obj, evidence, length, "reference_first", target_order)))
        if target_order == "target_first":
            comparisons.append(("target_entity_order", (
                case, obj, evidence, length, context_order, "reference_first")))
        if length == "compact":
            for variant in ("standard", "expanded"):
                comparisons.append(("length_band", (
                    case, obj, evidence, variant, context_order, target_order)))
        if evidence == "named_shelves":
            for variant in ("measured_height", "numbered_steps", "numbered_floors"):
                comparisons.append(("evidence_type", (
                    case, obj, variant, length, context_order, target_order)))
        if evidence == "direct_label_plain":
            comparisons.append(("direct_label_style", (
                "direct_label_positioned", obj, "direct_label_positioned",
                length, context_order, target_order)))
        for control, other_key in comparisons:
            other = by_key[other_key]
            if control in ("context_entity_order", "length_band", "evidence_type", "direct_label_style"):
                assert row["Target1"] == other["Target1"] and row["Target2"] == other["Target2"]
            if control == "target_entity_order":
                assert row["Context1"] == other["Context1"] and row["Context2"] == other["Context2"]
            matches.append({"control": control, "base_probe_id": row["probe_id"],
                            "variant_probe_id": other["probe_id"],
                            "event_family": row["event_family"], "case_id": case,
                            "object_pair_id": obj, "evidence_type": evidence,
                            "length_band": length})
    assert len({(m["control"], m["base_probe_id"], m["variant_probe_id"])
                for m in matches}) == len(matches)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    config_path = ROOT / "components.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(config["object_pairs"]) == 4 and len(config["cases"]) == 12
    for case in config["cases"]:
        validate_case(case)
    rows = [make_row(case, pair, evidence, length, context_order, target_order)
            for case, pair, evidence, length, context_order, target_order in product(
                config["cases"], config["object_pairs"], config["evidence_types"],
                config["length_bands"], config["context_entity_orders"],
                config["target_entity_orders"])]
    rows += [make_row(None, pair, "direct_label_" + style, length, context_order, target_order)
             for pair, style, length, context_order, target_order in product(
                config["object_pairs"], config["direct_label_styles"], config["length_bands"],
                config["context_entity_orders"], config["target_entity_orders"])]
    assert len(rows) == 2400 and len({r["probe_id"] for r in rows}) == len(rows)
    assert Counter(r["event_family"] for r in rows) == {
        "static_placement": 576, "target_crosses": 384,
        "reference_crosses": 384, "target_moves_without_crossing": 384,
        "both_move": 576, "direct_label_control": 96}
    assert Counter(r["target_entity_order"] for r in rows) == {
        "target_first": 1200, "reference_first": 1200}
    matches = make_matches(rows)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "probes.csv", rows)
    (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_csv(out / "variant_matches.csv", matches)
    catalog = [{"case_id": case["id"], "event_family": case["event_family"],
                "event_subtype": case["event_subtype"],
                **{f"{truth}_{entity}_{point}": case[truth][entity][index]
                   for truth in ("above", "below")
                   for entity in ("target", "reference")
                   for point, index in (("start", 0), ("end", 1))}}
               for case in config["cases"]]
    write_csv(out / "case_catalog.csv", catalog)
    review = [r for r in rows if r["object_pair_id"] == "ball_cone"
              and r["evidence_type"] in ("named_shelves", "numbered_steps", "numbered_floors")
              and r["length_band"] == "compact"
              and r["context_entity_order"] == "target_first"
              and r["target_entity_order"] == "target_first"]
    write_csv(out / "review_examples.csv", review)
    manifest = {"probe_name": config["probe_name"], "version": config["version"],
                "paired_rows": len(rows), "binary_judgments": len(rows) * 2,
                "conditional_likelihoods": len(rows) * 4,
                "applied_rows": 2304, "direct_label_control_rows": 96,
                "by_event_family": dict(Counter(r["event_family"] for r in rows)),
                "by_evidence_type": dict(Counter(r["evidence_type"] for r in rows)),
                "object_pairs": len(config["object_pairs"]),
                "physical_cases": len(config["cases"]),
                "variant_matches": len(matches),
                "components_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "validation": "passed"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
