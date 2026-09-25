#!/usr/bin/env python3
"""Build additive C0 baselines for the selected spatial matched pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RELATION_WORDS = (
    "above", "below", "left", "right", "north", "south", "east", "west",
    "front", "behind", "close", "far", "closer", "farther", "unchanged",
)
HASH_CACHE = {}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path):
    path = Path(path)
    if path not in HASH_CACHE:
        HASH_CACHE[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return HASH_CACHE[path]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def words(text):
    return re.findall(r"\b[\w'-]+\b", text.lower())


def edit_distance(a, b):
    a, b = words(a), words(b)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def common_sentence_prefix(a, b):
    limit = 0
    for x, y in zip(a, b):
        if x != y:
            break
        limit += 1
    common = a[:limit]
    boundary = common.rfind(". ")
    return common[:boundary + 2] if boundary >= 0 else ""


def prefix_before_entities(context, entities):
    """Return framing text before the first entity clause."""
    positions = []
    for entity in entities:
        for needle in (entity + " ", ". " + entity + " "):
            index = context.find(needle)
            if index >= 0:
                positions.append(index + (2 if needle.startswith(". ") else 0))
    if not positions:
        raise ValueError(f"Could not locate entity clause in: {context}")
    return context[:min(positions)]


def sentence(text):
    return text[0].upper() + text[1:] if text else text


def ordered_roles(row):
    return (("target", "reference") if row.get("context_entity_order") == "target_first"
            else ("reference", "target"))


def join_context(prefix, clauses, bridge=""):
    parts = [prefix.strip(), *(clause.strip() for clause in clauses), bridge.strip()]
    return " ".join(part for part in parts if part)


def above_label(evidence, value):
    shelves = {1: "the bottom shelf", 2: "the lower shelf", 3: "the middle shelf",
               4: "the upper shelf", 5: "the top shelf"}
    if evidence == "named_shelves":
        return shelves[int(value)]
    if evidence == "numbered_steps":
        return f"step {value}"
    if evidence == "numbered_floors":
        return f"floor {value}"
    unit = "foot" if int(value) == 1 else "feet"
    return f"a height of {value} {unit} from the floor"


def above_c0(row):
    prefix = prefix_before_entities(row["Context1"],
                                    (f"The {row['target_object']}", f"The {row['reference_object']}"))
    family, evidence = row["event_family"], row["evidence_type"]
    names = {"target": f"the {row['target_object']}", "reference": f"the {row['reference_object']}"}
    order = ordered_roles(row)
    if family == "static_placement":
        positions = sorted({int(row["target_end_above"]), int(row["reference_end_above"])})
        labels = [above_label(evidence, p) for p in positions]
        clause = (f"{sentence(names[order[0]])} and {names[order[1]]} occupy {labels[0]} and "
                  f"{labels[1]}, one object at each position.")
        return join_context(prefix, [clause]), "unordered_position_assignment", False, []
    if family in ("target_crosses", "reference_crosses"):
        mover = "target" if family == "target_crosses" else "reference"
        still = "reference" if mover == "target" else "target"
        endpoints = sorted({int(row[f"{mover}_start_above"]), int(row[f"{mover}_end_above"])})
        fixed = int(row[f"{still}_end_above"])
        clauses = {
            mover: (f"{sentence(names[mover])} moves between {above_label(evidence, endpoints[0])} and "
                    f"{above_label(evidence, endpoints[1])}, from one to the other."),
            still: (f"{sentence(names[still])} remains at {above_label(evidence, fixed)}."
                    if evidence == "measured_height" else
                    f"{sentence(names[still])} remains on {above_label(evidence, fixed)}."),
        }
        return join_context(prefix, [clauses[r] for r in order]), "direction_unspecified_crossing", False, []
    if family == "target_moves_without_crossing":
        direction = row["target_motion_above"]
        clauses = {
            "target": (f"{sentence(names['target'])} moves {direction} without reaching or passing "
                       f"{names['reference']}'s level."),
            "reference": f"{sentence(names['reference'])} remains in place.",
        }
        return join_context(prefix, [clauses[r] for r in order]), "side_unspecified_no_crossing", True, []
    if row["event_subtype"] == "order_reversed":
        clause = (f"{sentence(names[order[0]])} and {names[order[1]]} begin at two different heights "
                  "and exchange heights.")
        return join_context(prefix, [clause]), "unordered_exchange", False, []
    direction = row["target_motion_above"]
    clause = (f"{sentence(names[order[0]])} and {names[order[1]]} each move {direction} by the same "
              "amount and finish at different heights.")
    return join_context(prefix, [clause]), "unordered_equal_comotion", False, []


def left_label(evidence, value):
    value = int(value)
    if evidence == "named_positions":
        return ("far-left", "inner-left", "center", "inner-right", "far-right")[value - 1]
    if evidence == "lettered_positions":
        return "ABCDE"[value - 1]
    if evidence == "numbered_right_to_left":
        return str(6 - value)
    return str(value)


def left_c0(row):
    prefix = prefix_before_entities(row["Context1"],
                                    (f"The {row['target_object']} icon", f"The {row['reference_object']} icon"))
    family, evidence = row["event_family"], row["evidence_type"]
    names = {"target": f"the {row['target_object']} icon", "reference": f"the {row['reference_object']} icon"}
    order = ordered_roles(row)
    if family == "static_placement":
        positions = sorted({int(row["target_end_left"]), int(row["reference_end_left"])})
        labels = [left_label(evidence, p) for p in positions]
        clause = (f"{sentence(names[order[0]])} and {names[order[1]]} occupy {labels[0]} and {labels[1]}, "
                  "one icon in each slot.")
        return join_context(prefix, [clause]), "unordered_position_assignment", False, []
    if family in ("target_crosses", "reference_crosses"):
        mover = "target" if family == "target_crosses" else "reference"
        still = "reference" if mover == "target" else "target"
        endpoints = sorted({int(row[f"{mover}_start_left"]), int(row[f"{mover}_end_left"])})
        fixed = int(row[f"{still}_end_left"])
        clauses = {
            mover: (f"{sentence(names[mover])} moves between {left_label(evidence, endpoints[0])} and "
                    f"{left_label(evidence, endpoints[1])}, from one to the other."),
            still: f"{sentence(names[still])} stays at {left_label(evidence, fixed)}.",
        }
        return join_context(prefix, [clauses[r] for r in order]), "direction_unspecified_crossing", False, []
    if family == "target_moves_without_crossing":
        direction = row["target_motion_left"]
        clauses = {
            "target": (f"{sentence(names['target'])} moves one slot {direction} without reaching or passing "
                       f"{names['reference']}."),
            "reference": f"{sentence(names['reference'])} stays in place.",
        }
        return join_context(prefix, [clauses[r] for r in order]), "side_unspecified_no_crossing", True, []
    if row["event_subtype"] == "order_reversed":
        clause = (f"{sentence(names[order[0]])} and {names[order[1]]} begin in two different slots and "
                  "exchange slots.")
        return join_context(prefix, [clause]), "unordered_exchange", False, []
    direction = row["target_motion_left"]
    clause = (f"{sentence(names[order[0]])} and {names[order[1]]} each move one slot {direction} and "
              "finish in different slots.")
    return join_context(prefix, [clause]), "unordered_equal_comotion", False, []


def cardinal_c0(row):
    prefix = prefix_before_entities(row["Context1"], ("A", "B", "Marker A", "Marker B"))
    bridge, family = row["answer_bridge"], row["event_family"]
    names = {"target": row["target_entity"], "reference": row["reference_entity"]}
    order = ordered_roles(row)
    def label(role, point="end"):
        return row[f"context1_{role}_{point}_label"]
    if family == "static_placement":
        labels = sorted({label("target"), label("reference")})
        clause = f"{names[order[0]]} and {names[order[1]]} occupy {labels[0]} and {labels[1]}, one marker at each."
        return join_context(prefix, [clause], bridge), "unordered_position_assignment", False, []
    if family in ("target_crosses", "reference_crosses"):
        mover = "target" if family == "target_crosses" else "reference"
        still = "reference" if mover == "target" else "target"
        endpoints = sorted({label(mover, "start"), label(mover, "end")})
        clauses = {
            mover: f"{names[mover]} moves between {endpoints[0]} and {endpoints[1]}, from one to the other.",
            still: f"{names[still]} stays at {label(still)}.",
        }
        return join_context(prefix, [clauses[r] for r in order], bridge), "direction_unspecified_crossing", False, []
    if family == "target_moves_without_crossing":
        direction = row["context1_target_motion"]
        clauses = {
            "target": f"{names['target']} moves one position {direction} without reaching or passing {names['reference']}.",
            "reference": f"{names['reference']} stays in place.",
        }
        return join_context(prefix, [clauses[r] for r in order], bridge), "side_unspecified_no_crossing", True, []
    if row["event_subtype"] == "swap_two_positions":
        clause = f"{names[order[0]]} and {names[order[1]]} begin at two different positions and exchange positions."
        return join_context(prefix, [clause], bridge), "unordered_exchange", False, []
    direction = row["context1_target_motion"]
    clause = (f"{names[order[0]]} and {names[order[1]]} each move one position {direction} and finish "
              "at different positions.")
    return join_context(prefix, [clause], bridge), "unordered_equal_comotion", False, []


def close_c0(row):
    if row["condition_id"].startswith("distance_phrase"):
        c0 = re.sub(r"\ba short distance\b", "a distance", row["Context1"])
        return c0, "unspecified_distance_magnitude", True, []
    first = row["Context1"].split(". ", 1)[0] + "."
    c0 = f"{first} The {row['object']} rests somewhere in the {row['setting']}."
    return c0, "unspecified_setting_position", True, ["larger_edit", "location_unspecified"]


def neutralize_toward(context, destination):
    if "directly toward it" in context:
        return context.replace("directly toward it", "along the straight line containing both of them")
    phrase = "directly toward " + destination
    if phrase not in context:
        raise ValueError(f"Could not neutralize movement direction: {context}")
    return context.replace(phrase, "along the straight line shared with " + destination)


def initial_distance_phrase(context, name):
    patterns = (
        rf"were (.+?) apart\.",
        rf"Starting (.+?) apart,",
        rf", (.+?) from {re.escape(name)},",
        rf"began (.+?) from {re.escape(name)}",
    )
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(1)
    return "some distance"


def closer_c0(row, first_item=None):
    if row["family"] != "object_moves_vs_both_move_cross_family":
        destination = ("it" if row["source"] == "person_movement_v1_2" and "toward it" in row["Context1"]
                       else f"the {row['object']}" if row["source"] == "person_movement_v1_2"
                       else row["name"])
        c0 = neutralize_toward(row["Context1"], destination)
        return c0, "unspecified_linear_motion_direction", True, []
    distance = initial_distance_phrase(row["Context1"], row["name"])
    if row["context_entity_order"] == "object_first":
        entities = f"the {row['object']} and {row['name']}"
    else:
        entities = f"{row['name']} and the {row['object']}"
    c0 = f"Initially, {entities} were {distance} apart. Their positions were recorded again after the interval."
    return c0, "unspecified_before_after_distance", True, ["contrast_level", "larger_edit"]


def front_c0(row):
    a, b = row["target_entity"], row["reference_entity"]
    family, subtype = row["event_family"], row["event_subtype"]
    if row["evidence_format"] == "nonnumeric":
        if family == "static_placement":
            c0 = f"{b} faces the {row['anchor']}. {a} and {b} stand at different positions along the line to the {row['anchor']}."
            return c0, "facing_preserved_position_unspecified", True, ["qualitative_position"]
        if family == "target_crosses":
            c0 = f"{a} starts on one side of {b}, then walks straight past {b}. {b} stays still without turning."
        elif family == "reference_crosses":
            c0 = f"{b} moves straight past {a} without turning. {a} stays still."
        elif subtype == "approach":
            c0 = f"{a} approaches {b} but stops before reaching {b}. {b} stays still without turning."
        elif subtype == "recede":
            c0 = f"{a} walks straight away from {b}. {b} stays still without turning."
        elif subtype == "same_direction_preserved":
            c0 = f"{a} and {b} start at different positions. Both move equal distances in the same direction, without turning."
        elif subtype == "opposite_direction_preserved":
            c0 = f"{a} and {b} start at different positions and move directly away from each other without turning."
        else:
            c0 = f"{a} and {b} start at different positions and move straight past each other without turning."
        return c0, "reference_facing_unspecified", True, ["facing_value_removed"]
    prefix = row["Context1"].split(". ", 1)[0] + ". "
    order = ordered_roles(row)
    names = {"target": a, "reference": b}
    if family == "static_placement":
        sets = [{row[f"context{i}_target_end"], row[f"context{i}_reference_end"]} for i in (1, 2)]
        if sets[0] == sets[1]:
            p, q = sorted(sets[0])
            clause = f"{names[order[0]]} and {names[order[1]]} stay at {p} and {q}, one at each position."
            return join_context(prefix, [clause]), "unordered_position_assignment", False, []
        clause = f"{names[order[0]]} and {names[order[1]]} stay at different positions."
        return join_context(prefix, [clause]), "position_unspecified", True, ["exact_positions_removed"]
    if family in ("target_crosses", "reference_crosses"):
        mover = "target" if family == "target_crosses" else "reference"
        still = "reference" if mover == "target" else "target"
        endpoints = sorted({row[f"context1_{mover}_start"], row[f"context1_{mover}_end"]})
        ending = " without turning" if mover == "reference" else ""
        clauses = {mover: f"{names[mover]} moves between {endpoints[0]} and {endpoints[1]}, from one to the other{ending}.",
                   still: f"{names[still]} stays at {row[f'context1_{still}_end']}."}
        return join_context(prefix, [clauses[r] for r in order]), "direction_unspecified_crossing", False, []
    if family == "target_moves_without_crossing":
        clauses = {"target": f"{a} moves without crossing {b}.", "reference": f"{b} stays at a different position without turning."}
        return join_context(prefix, [clauses[r] for r in order]), "side_unspecified_no_crossing", True, ["exact_positions_removed"]
    if subtype == "order_reversed":
        clause = f"{names[order[0]]} and {names[order[1]]} start at different positions and pass each other without turning."
        return join_context(prefix, [clause]), "unordered_exchange", False, []
    clause = f"{names[order[0]]} and {names[order[1]]} move without turning and finish at different positions."
    return join_context(prefix, [clause]), "unordered_comotion", True, ["exact_motion_removed"]


def minimal_c0(row):
    line = "north-south" if row["axis"] == "north_south" else "east-west"
    case, verb, prep = row["case_id"], row["movement_verb"], row["preposition"]
    parts = [f"A and B are at different positions on an {line} line." if line == "east-west"
             else f"A and B are at different positions on a {line} line."]
    if case in ("target_crosses", "reference_crosses"):
        mover, other = ("A", "B") if case == "target_crosses" else ("B", "A")
        parts.append(f"{mover} {verb} straight {prep} {other} and continues past {other}.")
    elif case == "toward_without_crossing":
        parts.append(f"A {verb} toward B but stops before reaching B.")
    elif case == "away":
        parts.append(f"A {verb} directly farther away from B.")
    elif case == "both_equal":
        plural = "move" if verb == "moves" else "head"
        parts.append(f"Both {plural} the same distance in the same direction.")
    else:
        parts.append("A and B swap positions.")
    explicit = {"target_crosses": "B stays still.", "reference_crosses": "A stays still.",
                "toward_without_crossing": "B stays still.", "away": "B stays still."}
    if row["persistence_cue"] == "explicit":
        parts.append(explicit[case])
    parts.append(row["answer_bridge"])
    return " ".join(parts), "initial_relation_unspecified", False, []


def target_relation(row, index):
    key = f"target{index}_relation_word"
    if row.get(key):
        return row[key]
    text = row[f"Target{index}"].lower()
    if "same distance" in text:
        return "unchanged"
    matches = [word for word in RELATION_WORDS if re.search(rf"\b{word}\b", text)]
    if "in front of" in text:
        return "front"
    if len(matches) != 1:
        raise ValueError(f"Cannot identify target relation: {row[f'Target{index}']}")
    return matches[0]


def witness_state(source, row, index, label):
    state = {
        "source_context": f"Context{index}",
        "supports_target": label,
        "target_relation": target_relation(row, int(label[-1])),
        "context_sha256": stable_hash(row[f"Context{index}"]),
    }
    if source == "above_below":
        truth = "above" if index == 1 else "below"
        state["geometry"] = {f"{role}_{point}": row[f"{role}_{point}_{truth}"]
                             for role in ("target", "reference") for point in ("start", "end")}
    elif source == "left_right":
        truth = "left" if index == 1 else "right"
        state["geometry"] = {f"{role}_{point}": row[f"{role}_{point}_{truth}"]
                             for role in ("target", "reference") for point in ("start", "end")}
    elif source in ("cardinal", "front_behind", "minimal_cardinal"):
        prefix = f"context{index}_"
        state["geometry"] = {key.removeprefix(prefix): value for key, value in row.items()
                             if key.startswith(prefix) and any(term in key for term in (
                                 "start", "end", "motion", "facing", "initial_relation",
                                 "gold_relation", "order_reversed"))}
    elif source == "close_far":
        state["geometry"] = {key: row.get(key) for key in (
            "condition_id", "setting_id", "endpoint_geometry", "evidence_type",
            "distance_magnitude_explicit")}
    else:
        state["geometry"] = {key: row.get(key) for key in (
            "family", "contrast", "mode", "name_id", "object_id", "numeric_case_id", "unit_id")}
    return state


def make_record(source, row, c0_info, metadata, source_path, additional_source_paths=()):
    c0, rule, extra, tradeoffs = c0_info
    original = [row[k] for k in ("Context1", "Context2", "Target1", "Target2")]
    gold = [row["correct_target_for_context1"], row["correct_target_for_context2"]]
    if sorted(gold) != ["Target1", "Target2"]:
        raise ValueError(f"Expected opposing gold labels: {metadata['source_pair_id']}")
    witnesses = {label: f"Context{gold.index(label) + 1}" for label in ("Target1", "Target2")}
    witness_states = {label: witness_state(source, row, gold.index(label) + 1, label)
                      for label in ("Target1", "Target2")}
    overlap = {word: len(re.findall(rf"\b{re.escape(word)}\b", c0.lower()))
               for word in sorted({target_relation(row, 1), target_relation(row, 2)})}
    if c0 in original[:2] or not c0.strip():
        raise ValueError(f"Invalid neutral context: {metadata['source_pair_id']}")
    canonical = sorted(original[2:], key=lambda value: (value.casefold(), value))
    result = {
        "neutral_probe_id": source + "__" + stable_hash(metadata["source_pair_id"])[:20],
        "source_dataset": source,
        "source_pair_id": metadata.pop("source_pair_id"),
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "source_inputs_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                 for path in (source_path, *additional_source_paths)},
        "original_pair_sha256": stable_hash(original),
        "Context1": original[0], "Context2": original[1], "Context0": c0,
        "Target1": original[2], "Target2": original[3],
        "correct_target_for_context1": gold[0], "correct_target_for_context2": gold[1],
        "correct_target_for_context0": None,
        "target1_relation": target_relation(row, 1), "target2_relation": target_relation(row, 2),
        "canonical_target1": canonical[0], "canonical_target2": canonical[1],
        "canonical_pair_id": stable_hash(canonical),
        "neutralization_rule_id": rule, "neutralization_rule_version": "1.0",
        "neutralization_scope": "contrast_level" if "contrast_level" in tradeoffs else "template_matched",
        "neutral_allowed_targets": ["Target1", "Target2"],
        "neutral_additional_outcomes_possible": extra,
        "neutral_witnesses": witnesses,
        "neutral_witness_states": witness_states,
        "relation_word_overlap": overlap,
        "tradeoff_flags": tradeoffs,
        "manual_review_status": "pending",
        "manual_review_priority": "high" if tradeoffs or len(set(overlap.values())) > 1 else "standard",
        "Context0_word_count": len(words(c0)),
        "Context0_whitespace_token_count": len(c0.split()),
        "c0_c1_word_edit_distance": edit_distance(c0, original[0]),
        "c0_c2_word_edit_distance": edit_distance(c0, original[1]),
        **metadata,
    }
    return result


def load_regular():
    specs = [
        ("above_below", ROOT / "data/above_below_situation_probe/generated/probes.jsonl",
         lambda r: r["probe_family"] == "applied", above_c0),
        ("left_right", ROOT / "data/left_right_situation_probe/generated/probes.jsonl",
         lambda r: r["probe_family"] == "applied", left_c0),
        ("cardinal", ROOT / "data/cardinal_situation_probe/generated/probes.jsonl",
         lambda r: r["probe_family"] == "event", cardinal_c0),
        ("close_far", ROOT / "data/close_far_evidence_probe/matched_scenes/generated/probes.jsonl",
         lambda r: r["condition_id"] != "direct_label", close_c0),
        ("front_behind", ROOT / "data/front_behind_situation_probe/generated/probes.jsonl",
         lambda r: r["probe_family"] == "event", front_c0),
        ("minimal_cardinal", ROOT / "data/cardinal_situation_probe/minimal_relations/generated/probes.jsonl",
         lambda r: r["probe_family"] == "event", minimal_c0),
    ]
    output = []
    for source, path, select, renderer in specs:
        for raw in read_jsonl(path):
            if not select(raw):
                continue
            row = dict(raw)
            # The close/far generator fixes C1=close/Target1 and C2=far/Target2
            # but its source schema predates explicit per-context answer keys.
            row.setdefault("correct_target_for_context1", "Target1")
            row.setdefault("correct_target_for_context2", "Target2")
            if source == "above_below":
                meta = dict(category=source, benchmark_block="six_category_composite",
                    family=row["event_family"], case_id=row["case_id"], contrast="above_vs_below",
                    evidence=row["evidence_type"], length_band=row["length_band"],
                    context_entity_order=row["context_entity_order"], target_entity_order=row["target_entity_order"],
                    reference_frame="vertical_fixed_levels", answer_bridge="", source_pair_id=row["probe_id"],
                    aggregation_cell=[row["length_band"], row["context_entity_order"], row["target_entity_order"]])
            elif source == "left_right":
                meta = dict(category=source, benchmark_block="six_category_composite",
                    family=row["event_family"], case_id=row["case_id"], contrast="left_vs_right",
                    evidence=row["evidence_type"], length_band=row["length_band"],
                    context_entity_order=row["context_entity_order"], target_entity_order=row["target_entity_order"],
                    reference_frame=row["reference_frame"], answer_bridge="", source_pair_id=row["probe_id"],
                    aggregation_cell=[row["length_band"], row["context_entity_order"], row["target_entity_order"]])
            elif source == "cardinal":
                meta = dict(category=row["axis"], benchmark_block="six_category_composite",
                    family=row["event_family"], case_id=row["case_id"], contrast=row["axis"],
                    evidence=row["evidence_format"], length_band=row["length_band"],
                    context_entity_order=row["context_entity_order"], target_entity_order=row["target_entity_order"],
                    reference_frame=row["reference_frame"], answer_bridge=row["answer_bridge"], source_pair_id=row["probe_id"],
                    aggregation_cell=[row["length_band"], row["context_entity_order"], row["target_entity_order"],
                                      row["location_list_order"], row["numeric_label_order"]])
            elif source == "close_far":
                evidence = "distance_phrase" if row["condition_id"].startswith("distance_phrase") else "endpoint_placement"
                meta = dict(category=source, benchmark_block="six_category_composite", family="static_distance",
                    case_id=row["setting_id"], contrast="close_vs_far", evidence=evidence, length_band="not_applicable",
                    context_entity_order=row["context_entity_order"], target_entity_order=row["target_order"],
                    reference_frame="setting_relative_distance", answer_bridge="", source_pair_id=row["probe_id"],
                    aggregation_cell=[row["context_entity_order"], row["target_order"]])
            elif source == "front_behind":
                meta = dict(category=source, benchmark_block="supplement", family=row["event_family"],
                    case_id=row["case_id"], contrast="front_vs_behind", evidence=row["evidence_format"],
                    length_band=row["length_band"], context_entity_order=row["context_entity_order"],
                    target_entity_order=row["target_entity_order"], reference_frame=row["reference_frame"],
                    answer_bridge="", source_pair_id=row["probe_id"],
                    aggregation_cell=[row["context_entity_order"], row["target_entity_order"], row["facing_direction"], row["anchor"]])
            else:
                meta = dict(category=row["axis"], benchmark_block="minimal_cardinal_supplement",
                    family=row["event_family"], case_id=row["case_id"], contrast=row["axis"],
                    evidence=row["evidence_format"], length_band=row["length_band"],
                    context_entity_order="fixed", target_entity_order="fixed", reference_frame=row["reference_frame"],
                    answer_bridge=row["answer_bridge"], source_pair_id=row["probe_id"],
                    aggregation_cell=[row["movement_verb"], row["preposition"], row["persistence_cue"]])
            output.append(make_record(source, row, renderer(row), meta, path))
    return output


def load_closer_farther():
    result_dir = ROOT / "runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2"
    person_dir = ROOT / "runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_non_numeric_v1_2"
    overall_path = result_dir / "overall_applied_pairs.csv"
    overall = read_csv(overall_path)
    ext_pairs = {(r["pair_id"], r["context_entity_order"]): r
                 for r in read_csv(result_dir / "binary_pair_scores.csv")}
    items = {r["probe_id"]: r for r in read_jsonl(result_dir / "item_scores.jsonl")}
    person = {r["pair_id"]: r for r in read_csv(person_dir / "pair_scores.csv")}
    target_key = {"closer": "Target1", "farther": "Target2", "unchanged": "Target3"}
    output = []
    for selected in overall:
        if selected["source"] == "event_extension_v1_2":
            pair = ext_pairs[selected["pair_id"], selected["context_entity_order"]]
            a, b = items[pair["first_probe_id"]], items[pair["second_probe_id"]]
            keys = [target_key[pair["first_outcome"]], target_key[pair["second_outcome"]]]
            row = {
                "Context1": a["Context"], "Context2": b["Context"],
                "Target1": a[keys[0]], "Target2": a[keys[1]],
                "correct_target_for_context1": "Target1", "correct_target_for_context2": "Target2",
                "target1_relation_word": pair["first_outcome"], "target2_relation_word": pair["second_outcome"],
                "family": selected["family"], "source": selected["source"],
                "name": a["name"], "object": a["object"],
                **selected,
            }
            source_path = result_dir / "binary_pair_scores.csv"
            additional = (overall_path, result_dir / "item_scores.jsonl")
        else:
            pair = person[selected["pair_id"]]
            object_match = re.search(r"the ([\w-]+) than before", pair["Target1"])
            name = pair["Target1"].split(" is now", 1)[0]
            row = {
                "Context1": pair["Context1"], "Context2": pair["Context2"],
                "Target1": pair["Target1"], "Target2": pair["Target2"],
                "correct_target_for_context1": "Target1", "correct_target_for_context2": "Target2",
                "target1_relation_word": "closer", "target2_relation_word": "farther",
                "family": selected["family"], "source": selected["source"],
                "name": name, "object": object_match.group(1),
                **selected,
            }
            source_path = person_dir / "pair_scores.csv"
            additional = (overall_path,)
        meta = dict(category="closer_farther", benchmark_block="six_category_composite",
            family=selected["family"], case_id=selected["family"], contrast=selected["contrast"],
            evidence=selected["mode"], length_band=selected["length_band"],
            context_entity_order=selected["context_entity_order"], target_entity_order="person_first",
            reference_frame="relative_distance_change", answer_bridge="",
            source_pair_id=selected["source"] + "__" + selected["pair_id"] + "__" + selected["context_entity_order"],
            aggregation_cell=[selected["contrast"], selected["length_band"], selected["context_entity_order"]])
        output.append(make_record("closer_farther", row, closer_c0(row), meta, source_path, additional))
    return output


def validate(rows):
    expected = {
        "above_below": 2304, "left_right": 2304, "north_south": 648,
        "east_west": 648, "close_far": 1152, "closer_farther": 9324,
        "front_behind": 212,
    }
    counts = Counter(r["category"] for r in rows if r["benchmark_block"] != "minimal_cardinal_supplement")
    if counts != expected:
        raise ValueError(f"Unexpected category counts: {counts}")
    minimal = [r for r in rows if r["benchmark_block"] == "minimal_cardinal_supplement"]
    if len(minimal) != 38 or Counter(r["category"] for r in minimal) != {"north_south": 19, "east_west": 19}:
        raise ValueError("Unexpected minimal-cardinal inventory")
    if len(rows) != 16630 or len({r["neutral_probe_id"] for r in rows}) != len(rows):
        raise ValueError("Wrong total or duplicate neutral IDs")
    if any(r["correct_target_for_context0"] is not None or r["manual_review_status"] != "pending" for r in rows):
        raise ValueError("Neutral contexts must be unlabeled and pending review")
    if any(set(r["neutral_witnesses"]) != {"Target1", "Target2"} for r in rows):
        raise ValueError("Both targets need source-scene witnesses")
    if any(set(r["neutral_witness_states"]) != {"Target1", "Target2"} or
           any(not state["geometry"] for state in r["neutral_witness_states"].values()) for r in rows):
        raise ValueError("Both targets need structured witness states")
    if any(r["answer_bridge"] and not r["Context0"].endswith(r["answer_bridge"]) for r in rows):
        raise ValueError("Neutral context lost its answer bridge")
    return counts, minimal


def write_csv_rows(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def render_review(rows):
    selected, seen = [], set()
    for row in rows:
        key = (row["source_dataset"], row["family"], row["evidence"], row["neutralization_rule_id"])
        if key not in seen:
            seen.add(key); selected.append(row)
    lines = ["# Neutral-context review examples", "",
             "Every generated rule/family/evidence combination is represented. These examples are pending human review.", ""]
    for row in selected:
        lines += [f"## {row['category']} / {row['family']} / {row['evidence']}", "",
                  f"Rule: `{row['neutralization_rule_id']}`; source: `{row['source_pair_id']}`", "",
                  f"C1: {row['Context1']}", "", f"C2: {row['Context2']}", "",
                  f"C0: {row['Context0']}", "", f"T1: {row['Target1']}", "", f"T2: {row['Target2']}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    rows = load_regular() + load_closer_farther()
    rows.sort(key=lambda r: (r["benchmark_block"], r["category"], r["source_dataset"], r["source_pair_id"]))
    counts, minimal = validate(rows)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "neutral_probes.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_csv_rows(out / "neutral_probes.csv", rows)
    (out / "review_examples.md").write_text(render_review(rows))
    rules = Counter(r["neutralization_rule_id"] for r in rows)
    manifest = {
        "probe_name": "spatial_matched_neutral_contexts", "version": "1.0",
        "paired_rows": len(rows), "neutral_contexts": len(rows),
        "six_category_rows": sum(r["benchmark_block"] == "six_category_composite" for r in rows),
        "front_behind_supplement_rows": counts["front_behind"],
        "minimal_cardinal_supplement_rows": len(minimal),
        "by_category": dict(sorted(Counter(r["category"] for r in rows).items())),
        "by_source_dataset": dict(sorted(Counter(r["source_dataset"] for r in rows).items())),
        "by_rule": dict(sorted(rules.items())),
        "high_priority_manual_review": sum(r["manual_review_priority"] == "high" for r in rows),
        "source_files": {path: digest for r in rows for path, digest in r["source_inputs_sha256"].items()},
        "generator_sha256": sha256(Path(__file__)),
        "neutral_probes_sha256": sha256(jsonl),
        "validation": "passed: exact selected inventory; original fields copied; C0 unlabeled; both targets have source-scene witnesses",
        "evaluation_status": "not_scored",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
