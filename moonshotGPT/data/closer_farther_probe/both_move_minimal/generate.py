#!/usr/bin/env python3
"""Generate minimal within-family closer/farther/unchanged co-movement pairs."""

import csv
import hashlib
import json
import re
from collections import Counter
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTCOMES = ("closer", "farther", "unchanged")


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantity(value, unit):
    word = unit["singular"] if value == 1 else unit["plural"]
    return f"{value} {word}"


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def target_map(parent, name, obj):
    return {outcome["id"]: outcome["target_template"].format(
        name=name["text"], object=obj["text"])
        for outcome in parent["outcomes"]}


def numeric_state(case, outcome, slow):
    initial = int(case["initial_distance"])
    delta = int(case["movement_distance"])
    fast = slow + delta
    person_before, object_before = 0, initial
    if outcome == "closer":
        person_move, object_move = fast, slow
    elif outcome == "farther":
        person_move, object_move = slow, fast
    else:
        person_move = object_move = fast
    person_after = person_before + person_move
    object_after = object_before + object_move
    final = abs(object_after - person_after)
    computed = "closer" if final < initial else "farther" if final > initial else "unchanged"
    assert computed == outcome
    assert final == int(case[outcome + "_final_distance"])
    assert person_move > 0 and object_move > 0
    assert outcome != "closer" or person_after < object_after
    return {"initial_distance": initial, "final_distance": final,
        "distance_change": final - initial, "distance_change_magnitude": abs(final - initial),
        "movement_difference": abs(person_move - object_move),
        "person_before": person_before, "person_after": person_after,
        "object_before": object_before, "object_after": object_after,
        "person_movement": person_move, "object_movement": object_move}


def main():
    config_path = ROOT / "components.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = (ROOT / config["parent_components"]).resolve()
    cases_path = (ROOT / config["parent_numeric_cases"]).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cases = load_csv(cases_path)
    assert len(cases) == 12
    assert tuple(outcome["id"] for outcome in parent["outcomes"]) == OUTCOMES
    slow = int(config["slow_movement"])
    assert slow > 0

    contexts = []
    grouped = {}
    for name, obj, order in product(parent["names"], parent["objects"],
                                    ("person_first", "object_first")):
        targets = target_map(parent, name, obj)
        entity_id = f"{name['id']}__{obj['id']}"
        for outcome in OUTCOMES:
            probe_id = f"{entity_id}__non_numeric__{order}__{outcome}"
            context = config["non_numeric_templates"][order][outcome].format(
                name=name["text"], object=obj["text"])
            row = {"probe_id": probe_id, "probe_version": config["version"],
                "event_family": "both_entities_move", "event_subtype": "same_direction_relative_displacement",
                "outcome": outcome, "correct_target": targets[outcome],
                "numeric_information": "absent", "context_entity_order": order,
                "name_id": name["id"], "name": name["text"],
                "object_id": obj["id"], "object": obj["text"],
                "numeric_case_id": "", "unit_id": "", "unit": "",
                "initial_distance": "", "final_distance": "", "distance_change": "",
                "distance_change_magnitude": "", "movement_difference": "",
                "person_before": "", "person_after": "",
                "object_before": "", "object_after": "", "person_movement": "",
                "object_movement": "", "both_entities_move": True,
                "same_movement_direction": True, "passes_other_entity": False,
                "Context": context, "Context_word_count": word_count(context),
                **{f"target_{key}": value for key, value in targets.items()}}
            contexts.append(row)
            grouped.setdefault(("absent", entity_id, "", "", order), {})[outcome] = row

        for case, unit, outcome in product(cases, parent["units"], OUTCOMES):
            state = numeric_state(case, outcome, slow)
            fmt = {"name": name["text"], "object": obj["text"],
                "initial": quantity(state["initial_distance"], unit),
                "person_move": quantity(state["person_movement"], unit),
                "object_move": quantity(state["object_movement"], unit)}
            context = config["numeric_templates"][order].format(**fmt)
            probe_id = "__".join((entity_id, case["id"], unit["id"], order, outcome))
            row = {"probe_id": probe_id, "probe_version": config["version"],
                "event_family": "both_entities_move", "event_subtype": "same_direction_relative_displacement",
                "outcome": outcome, "correct_target": targets[outcome],
                "numeric_information": "specific", "context_entity_order": order,
                "name_id": name["id"], "name": name["text"],
                "object_id": obj["id"], "object": obj["text"],
                "numeric_case_id": case["id"], "unit_id": unit["id"], "unit": unit["unit"],
                **state, "both_entities_move": True, "same_movement_direction": True,
                "passes_other_entity": False, "Context": context,
                "Context_word_count": word_count(context),
                **{f"target_{key}": value for key, value in targets.items()}}
            contexts.append(row)
            grouped.setdefault(("specific", entity_id, case["id"], unit["id"], order), {})[outcome] = row

    assert len(contexts) == 2664
    assert len({row["probe_id"] for row in contexts}) == len(contexts)
    assert Counter(row["outcome"] for row in contexts) == Counter({outcome: 888 for outcome in OUTCOMES})
    assert Counter(row["numeric_information"] for row in contexts) == {"specific": 2592, "absent": 72}
    for row in contexts:
        lower = row["Context"].lower()
        person_at = re.search(r"\b" + re.escape(row["name"].lower()) + r"\b", lower).start()
        object_at = re.search(r"\b(?:the )?" + re.escape(row["object"].lower()) + r"\b", lower).start()
        expected = "person_first" if person_at < object_at else "object_first"
        assert expected == row["context_entity_order"]

    pairs = []
    for key, group in grouped.items():
        assert set(group) == set(OUTCOMES)
        for first, second in combinations(OUTCOMES, 2):
            a, b = group[first], group[second]
            pair = {"pair_id": f"{a['probe_id']}__vs__{b['probe_id']}",
                "probe_version": config["version"], "event_family": "both_entities_move",
                "event_subtype": "same_direction_relative_displacement",
                "contrast": f"{first}_vs_{second}",
                "first_outcome": first, "second_outcome": second,
                "first_probe_id": a["probe_id"], "second_probe_id": b["probe_id"],
                "numeric_information": key[0], "name_id": a["name_id"],
                "object_id": a["object_id"], "numeric_case_id": a["numeric_case_id"],
                "unit_id": a["unit_id"], "context_entity_order": a["context_entity_order"],
                "Context1": a["Context"], "Context2": b["Context"],
                "Target1": a[f"target_{first}"], "Target2": a[f"target_{second}"],
                "correct_target_for_context1": "Target1", "correct_target_for_context2": "Target2"}
            assert pair["Context1"] != pair["Context2"] and pair["Target1"] != pair["Target2"]
            pairs.append(pair)

    assert len(grouped) == 888 and len(pairs) == 2664
    assert Counter(row["contrast"] for row in pairs) == {
        "closer_vs_farther": 888, "closer_vs_unchanged": 888,
        "farther_vs_unchanged": 888}
    assert Counter(row["numeric_information"] for row in pairs) == {
        "specific": 2592, "absent": 72}

    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    write_csv(out / "contexts.csv", contexts)
    (out / "contexts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in contexts), encoding="utf-8")
    write_csv(out / "pairs.csv", pairs)
    (out / "pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    review_contexts = [row for row in contexts if row["name_id"] == "name_0"
        and row["object_id"] == "obj_cone" and (row["numeric_information"] == "absent"
        or row["numeric_case_id"] == "num_00" and row["unit_id"] == "unit_ft")]
    write_csv(out / "review_contexts.csv", review_contexts)
    review_ids = {row["probe_id"] for row in review_contexts}
    write_csv(out / "review_pairs.csv", [row for row in pairs
        if row["first_probe_id"] in review_ids and row["second_probe_id"] in review_ids])
    manifest = {"probe_name": config["probe_name"], "version": config["version"],
        "context_rows": len(contexts), "paired_rows": len(pairs),
        "binary_judgments": 2 * len(pairs), "underlying_outcome_groups": len(grouped),
        "numeric_contexts": 2592, "non_numeric_contexts": 72,
        "numeric_pairs": 2592, "non_numeric_pairs": 72,
        "event_families": ["both_entities_move"],
        "event_subtypes": ["same_direction_relative_displacement"],
        "contrasts": dict(Counter(row["contrast"] for row in pairs)),
        "context_entity_orders": dict(Counter(row["context_entity_order"] for row in contexts)),
        "evaluated": False, "validation": "passed",
        "source_sha256": {role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in (("components", config_path), ("parent_components", parent_path),
                              ("parent_numeric_cases", cases_path))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
