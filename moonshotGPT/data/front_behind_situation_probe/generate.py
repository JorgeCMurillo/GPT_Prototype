#!/usr/bin/env python3
"""Render front/behind pairs from positions and an explicit facing vector."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def relation(target, reference, facing):
    if facing not in (-1, 1) or target == reference:
        raise ValueError("Require distinct positions and facing +1 or -1")
    return "front" if (target - reference) * facing > 0 else "behind"


def reflect(scene, size=5):
    return {entity: [size + 1 - p for p in scene[entity]]
            for entity in ("target", "reference")}


def signature(scene):
    return tuple(scene[entity][i] for entity in ("target", "reference") for i in (0, 1))


def canonical_cases(config):
    """Merge seeds already supplied as each other's reflected counterpart."""
    cases, seen = [], {}
    size = config["position_count"]
    ids = set()
    for seed in config["cases"]:
        if seed["id"] in ids:
            raise ValueError("Duplicate seed ID")
        ids.add(seed["id"])
        a, b = seed["target"], seed["reference"]
        if len(a) != 2 or len(b) != 2 or any(type(p) is not int or not 1 <= p <= size for p in a + b):
            raise ValueError("Invalid scene endpoints")
        initial, final = relation(a[0], b[0], 1), relation(a[1], b[1], 1)
        am, bm, cross = a[0] != a[1], b[0] != b[1], initial != final
        conditions = {"static_placement": not am and not bm,
                      "target_crosses": am and not bm and cross,
                      "reference_crosses": not am and bm and cross,
                      "target_moves_without_crossing": am and not bm and not cross,
                      "both_move": am and bm}
        if not conditions.get(seed["family"], False):
            raise ValueError(f"Wrong event family: {seed['id']}")
        mirror = reflect(seed, size)
        key = (seed["family"], tuple(sorted((signature(seed), signature(mirror)))))
        if key in seen:
            seen[key]["source_seed_ids"].append(seed["id"])
        else:
            case = {**seed, "source_seed_ids": [seed["id"]], "mirror": mirror}
            cases.append(case)
            seen[key] = case
    return cases


def answer(target, reference, word, order):
    phrase = "in front of" if word == "front" else "behind"
    if order == "target_first":
        return f"{target} is {phrase} {reference}."
    # Keep the SAME reference's facing frame; do not assume A's orientation.
    return f"{phrase.capitalize()} {reference} is {target}."


def motion(start, end, facing):
    return "still" if start == end else "forward" if (end - start) * facing > 0 else "backward"


def base_row(config, case, entity, evidence, order, facing, context_order="template_specific", anchor="none", block="event"):
    row = dict(probe_version=config["version"], probe_family=block,
               case_id=case["id"], scene_cluster_id=case["id"], event_family=case["family"],
               event_subtype=case.get("subtype", ""), evidence_format=evidence,
               entity_pair_id=entity["id"], target_entity=entity["target"], reference_entity=entity["reference"],
               target_entity_order=order, context_entity_order=context_order, facing_direction=facing,
               anchor=anchor, reference_frame="observer_relative", length_band="compact",
               definition_present=False, answer_bridge_present=False, headline_eligible=block == "event",
               source_seed_ids=case.get("source_seed_ids", []), facing_fixed=block != "observer_turn",
               initial_relation_explicit=False, final_relation_explicit=block == "control",
               geometry_precision="exact_positions" if evidence == "numeric" else "qualitative",
               pair_type="opposite_initial_positions" if block == "event" else block)
    for i, word in enumerate(("front", "behind"), 1):
        row[f"Target{i}"] = answer(entity["target"], entity["reference"], word, order)
        row[f"target{i}_relation_word"] = word
    row["probe_id"] = "__".join(str(row[k]) for k in (
        "probe_family", "case_id", "evidence_format", "entity_pair_id", "target_entity_order",
        "context_entity_order", "facing_direction", "anchor"))
    return row


def add_geometry(row, context, scene, facing_start, facing_end):
    prefix = f"context{context}_"
    for entity in ("target", "reference"):
        for index, endpoint in enumerate(("start", "end")):
            row[prefix + entity + "_" + endpoint] = scene[entity][index]
    row[prefix + "reference_facing_start"] = facing_start
    row[prefix + "reference_facing_end"] = facing_end
    row[prefix + "reference_motion_relative_to_initial_facing"] = motion(*scene["reference"], facing_start)
    initial = relation(scene["target"][0], scene["reference"][0], facing_start)
    gold = relation(scene["target"][1], scene["reference"][1], facing_end)
    row[prefix + "initial_relation"] = initial
    row[prefix + "gold_relation"] = gold
    row[prefix + "relation_changed"] = initial != gold
    row[f"correct_target_for_context{context}"] = "Target1" if gold == "front" else "Target2"


def numeric_context(scene, entity, facing, context_order):
    clauses = {}
    for role in ("target", "reference"):
        name = entity[role]
        start, end = scene[role]
        if start == end:
            clauses[role] = f"{name} stays at {start}."
        else:
            ending = " without turning." if role == "reference" else "."
            clauses[role] = f"{name} moves from {start} to {end}" + ending
    sequence = ("target", "reference") if context_order == "target_first" else ("reference", "target")
    intro = f"{entity['reference']} faces {facing} position numbers."
    return " ".join((intro, *(clauses[role] for role in sequence)))


def qualitative_context(case, scene, entity, anchor="door", facing=1):
    a, b = entity["target"], entity["reference"]
    initial = relation(scene["target"][0], scene["reference"][0], facing)
    faces = f"{b} faces {a}." if initial == "front" else f"{b} faces away from {a}."
    family = case["family"]
    if family == "static_placement":
        face = f"{b} faces the {anchor}." if facing == 1 else f"{b} faces away from the {anchor}."
        between = (f"{a} stands between {b} and the {anchor}." if scene["target"][0] > scene["reference"][0]
                   else f"{b} stands between {a} and the {anchor}.")
        return face + " " + between
    if family == "target_crosses":
        phrase = "in front of" if initial == "front" else "behind"
        return f"{a} starts {phrase} {b}, then walks straight past {b}. {b} stays still without turning."
    if family == "reference_crosses":
        action = "walks straight" if motion(*scene["reference"], facing) == "forward" else "steps backward"
        return f"{faces[:-1]}, then {action} past {a} without turning. {a} stays still."
    if family == "target_moves_without_crossing":
        action = (f"{a} approaches {b} but stops before reaching {b}." if case["subtype"] == "approach"
                  else f"{a} walks straight away from {b}.")
        return f"{faces} {action} {b} stays still without turning."
    if case["subtype"] == "same_direction_preserved":
        phrase = "in front of" if initial == "front" else "behind"
        direction = f"in {b}'s facing direction" if motion(*scene["reference"], facing) == "forward" else f"opposite {b}'s facing direction"
        return f"{a} starts {phrase} {b}. Both move equal distances {direction}, without turning."
    if case["subtype"] == "opposite_direction_preserved":
        return f"{faces} They move directly away from each other without turning."
    if case["subtype"] == "order_reversed":
        return f"{faces} They move straight past each other without turning."
    raise ValueError(f"Unknown qualitative case: {case['id']}")


def build(config):
    if config["lengths"] != ["compact"] or config["definition_present"] or config["answer_bridge_present"]:
        raise ValueError("This probe is compact-only, without a definition prefix or answer bridge")
    cases = canonical_cases(config)
    rows = []
    for case in cases:
        for entity, co, to, facing in itertools.product(config["entity_pairs"], config["context_entity_orders"],
                                                       config["target_entity_orders"], config["facings"]):
            row = base_row(config, case, entity, "numeric", to, facing, co)
            direction = 1 if facing == "increasing" else -1
            for i, scene in enumerate((case, case["mirror"]), 1):
                row[f"Context{i}"] = numeric_context(scene, entity, facing, co)
                add_geometry(row, i, scene, direction, direction)
            rows.append(row)
        # The three static distance/offset cases collapse to one qualitative
        # between-scene: do not inflate it with duplicate nonnumeric renderings.
        if case["family"] == "static_placement" and case["id"] != "static_adjacent":
            continue
        variants = itertools.product(config["minimal_anchors"], (1, -1)) if case["family"] == "static_placement" else [("none", 1)]
        for anchor, direction in variants:
            for entity, to in itertools.product(config["entity_pairs"], config["target_entity_orders"]):
                facing = ("toward_anchor" if direction == 1 else "away_from_anchor") if anchor != "none" else "implicit_initial_facing"
                row = base_row(config, case, entity, "nonnumeric", to, facing, anchor=anchor)
                row["initial_relation_explicit"] = case["family"] == "target_crosses" or case["subtype"] == "same_direction_preserved"
                for i, scene in enumerate((case, case["mirror"]), 1):
                    row[f"Context{i}"] = qualitative_context(case, scene, entity, anchor, direction)
                    add_geometry(row, i, scene, direction, direction)
                rows.append(row)
    rows.extend(build_diagnostics(config))
    for row in rows:
        for field in ("Context1", "Context2", "Target1", "Target2"):
            row[field + "_word_count"] = len(row[field].split())
    validate_rows(rows, config["max_context_words"])
    return rows, variant_matches(rows), cases


def build_diagnostics(config):
    rows = []
    for entity, to in itertools.product(config["entity_pairs"], config["target_entity_orders"]):
        a, b = entity["target"], entity["reference"]
        case = dict(id="direct_labels", family="direct_relation")
        row = base_row(config, case, entity, "direct_label", to, "not_applicable", block="control")
        row["initial_relation_explicit"] = True
        for i, word in enumerate(("front", "behind"), 1):
            row[f"Context{i}"] = answer(a, b, word, "target_first")
            row[f"correct_target_for_context{i}"] = f"Target{i}"
            row[f"context{i}_gold_relation"] = word
        rows.append(row)
        for initial in ("front", "behind"):
            case = dict(id="observer_turn_" + initial, family="observer_turn")
            row = base_row(config, case, entity, "initial_relation_then_action", to, "implicit_initial_facing", block="observer_turn")
            row.update(scene_cluster_id="observer_turn_front_behind", initial_relation_explicit=True, pair_type="no_turn_vs_half_turn")
            scene = {"target": [2, 2] if initial == "front" else [0, 0], "reference": [1, 1]}
            for i, direction in ((1, 1), (2, -1)):
                action = f"{b} does not turn." if i == 1 else f"{b} turns halfway around in place."
                row[f"Context{i}"] = f"{answer(a, b, initial, 'target_first')} {action} Neither person changes location."
                add_geometry(row, i, scene, 1, direction)
            rows.append(row)
    return rows


def validate_rows(rows, max_context_words=24):
    if len({r["probe_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate probe IDs")
    for row in rows:
        if row["length_band"] != "compact" or row["definition_present"] or row["answer_bridge_present"]:
            raise ValueError("Detailed renderings are not allowed")
        for i in (1, 2):
            if len(row[f"Context{i}"].split()) > max_context_words:
                raise ValueError(f"Context exceeds {max_context_words} words: {row['probe_id']}")
            prefix = f"context{i}_"
            if prefix + "target_end" in row:
                gold = relation(row[prefix + "target_end"], row[prefix + "reference_end"], row[prefix + "reference_facing_end"])
                if gold != row[prefix + "gold_relation"]:
                    raise ValueError("Geometry/gold disagreement")
            expected = "Target1" if row[prefix + "gold_relation"] == "front" else "Target2"
            if expected != row[f"correct_target_for_context{i}"]:
                raise ValueError("Wrong answer key")
        if row["correct_target_for_context1"] == row["correct_target_for_context2"]:
            raise ValueError("Expected opposing gold relations within pair")


def variant_matches(rows):
    factors = ("context_entity_order", "target_entity_order", "facing_direction", "entity_pair_id", "anchor")
    links = []
    for factor in factors:
        groups = defaultdict(list)
        for row in rows:
            fields = ["probe_family", "case_id", "evidence_format", *(f for f in factors if f != factor)]
            groups[tuple(row[f] for f in fields)].append(row)
        for group in groups.values():
            group.sort(key=lambda r: str(r[factor]))
            for other in group[1:]:
                first = group[0]
                changes = factor == "facing_direction"
                if any((first[f"correct_target_for_context{i}"] != other[f"correct_target_for_context{i}"]) != changes for i in (1, 2)):
                    raise ValueError(f"Unexpected gold change for {factor}")
                links.append(dict(match_type=factor, probe_id_a=first["probe_id"], probe_id_b=other["probe_id"],
                                  expected_gold_change=changes, scene_cluster_id=first["scene_cluster_id"]))
    return links


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    config = json.loads((ROOT / "components.json").read_text())
    rows, links, cases = build(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (args.out_dir / "probes.jsonl").write_text(text)
    write_csv(args.out_dir / "probes.csv", rows)
    write_csv(args.out_dir / "variant_matches.csv", links)
    counts = [row[f"Context{i}_word_count"] for row in rows for i in (1, 2)]
    manifest = dict(name=config["name"], version=config["version"], paired_rows=len(rows),
                    proposed_seed_scenes=len(config["cases"]), distinct_numeric_geometries=len(cases),
                    blocks=dict(Counter(r["probe_family"] for r in rows)),
                    event_formats=dict(Counter(r["evidence_format"] for r in rows if r["probe_family"] == "event")),
                    case_seed_mapping={c["id"]: c["source_seed_ids"] for c in cases},
                    matched_variants=dict(Counter(m["match_type"] for m in links)),
                    context_words=dict(min=min(counts), mean=sum(counts)/len(counts), max=max(counts)),
                    probes_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    components_sha256=hashlib.sha256((ROOT / "components.json").read_bytes()).hexdigest(),
                    scoring_note="Use explicit correct_target_for_context fields; gold is not always diagonal.",
                    replacement_note="Compact-only replacement of v1.0. Detailed numeric/named-location renderings, long tiers, definition prefixes and answer bridges removed.",
                    benchmark_status="Generated, not evaluated or included in the scored six-category composite.")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    examples = ["# Compact front/behind pairs", "",
                "Compact-only, with no definition prefix or answer bridge. Not evaluated.", ""]
    selected = set()
    # Put the easy-to-read natural-language scenes first.
    for row in sorted(rows, key=lambda r: (r["evidence_format"] != "nonnumeric", r["probe_family"] != "event")):
        key = (row["probe_family"], row["case_id"], row["evidence_format"])
        if key in selected or row["entity_pair_id"] != "a_b" or row["target_entity_order"] != "target_first":
            continue
        selected.add(key)
        examples += [f"## {row['case_id']} / {row['evidence_format']}", "",
                     f"Context 1: {row['Context1']}", "", f"Context 2: {row['Context2']}", "",
                     f"Answers: {row['Target1']} / {row['Target2']}", "",
                     f"Gold: {row['correct_target_for_context1']}, {row['correct_target_for_context2']}.", ""]
    (args.out_dir / "examples.md").write_text("\n".join(examples))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
