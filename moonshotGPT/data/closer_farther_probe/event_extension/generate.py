#!/usr/bin/env python3
"""Generate direct-label controls and matched reference/co-motion events."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    config_path = ROOT / "components.json"
    config = json.loads(config_path.read_text())
    parent_path = (ROOT / config["parent_components"]).resolve()
    cases_path = (ROOT / config["parent_numeric_cases"]).resolve()
    probes_path = (ROOT / config["parent_probes"]).resolve()
    parent = json.loads(parent_path.read_text())
    cases = load_csv(cases_path)
    original = [json.loads(line) for line in probes_path.read_text().splitlines()]
    assert len(cases) == 12 and len(original) == 7776
    original_index = {
        (r["name_id"], r["object_id"], r["numeric_case_id"], r["unit_id"],
         r["condition_id"], r["outcome"]): r
        for r in original if r["length_band"] == "standard"
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    target_ids = {o["id"]: o["target_key"] for o in parent["outcomes"]}
    rows, matches = [], []
    direct = {}
    nonnumeric = {}
    numeric = []
    length_groups = {}
    entity_order_groups = {}
    bands = ["compact", "standard", "expanded"]

    def order_forms(event, band, mode):
        template_key = mode if band == "standard" else band + "_" + mode
        baseline_order = ("object_first" if event["id"] == "reference_object_moves"
                          and band == "compact" else "person_first")
        alternate_order = "person_first" if baseline_order == "object_first" else "object_first"
        return [(baseline_order, template_key, False),
                (alternate_order, "alternate_" + template_key, True)]

    def add(row):
        for field in ["Context", "Target1", "Target2", "Target3"]:
            row[field + "_word_count"] = len(re.findall(r"\b\w+\b", row[field]))
            row[field + "_token_count"] = len(tokenizer.encode(row[field], add_special_tokens=False))
        row["probe_row_index"] = len(rows)
        rows.append(row)
        return row

    def targets(name, obj):
        return {o["target_key"]: o["target_template"].format(name=name["text"], object=obj["text"])
                for o in parent["outcomes"]}

    for name, obj in product(parent["names"], parent["objects"]):
        fmt = {"name": name["text"], "object": obj["text"]}
        base = {"probe_type": config["probe_name"], "probe_version": config["version"],
                "name_id": name["id"], "name": name["text"],
                "object_id": obj["id"], "object": obj["text"], **targets(name, obj)}
        entity_id = name["id"] + "__" + obj["id"]
        for outcome, form in product(target_ids, config["direct_label_templates"]):
            row = add({**base, "probe_id": "__".join([entity_id, form["id"], outcome]),
                "condition_id": "direct_label", "event_family": "direct_relation_control",
                "direct_label_match_id": "__".join([entity_id, outcome]),
                "evidence_type": "explicit_comparative_label", "numeric_information": "absent",
                "template_id": form["id"], "context_entity_order": form["context_entity_order"],
                "length_band": "direct", "length_match_id": "",
                "entity_order_match_id": "",
                "reference_order_id": "person_relative_to_object", "outcome": outcome,
                "correct_target": target_ids[outcome], "event_id": "",
                "distance_trajectory_id": "", "numeric_case_id": "", "unit_id": "",
                "initial_distance": "", "final_distance": "", "movement_distance": "",
                "person_before": "", "person_after": "", "object_before": "", "object_after": "",
                "person_moves": "", "object_moves": "", "object_stationary": "",
                "passes_reference": "", "target_relation_repeated": True,
                "Context": form[outcome].format(**fmt)})
            direct[name["id"], obj["id"], outcome, form["id"]] = row

        for event in config["event_templates"]:
            for outcome in event["outcomes"]:
                direction = "toward" if outcome == "closer" else "away from" if outcome == "farther" else ""
                event_id = "__".join([entity_id, event["id"], outcome, "non_numeric"])
                for band in bands:
                    for order, template_key, alternate in order_forms(event, band, "non_numeric"):
                        probe_id = (event_id if band == "standard" else event_id + "__" + band)
                        if alternate:
                            probe_id += "__" + order
                        row = add({**base, "probe_id": probe_id,
                            "condition_id": event["id"], "event_family": event["id"],
                            "direct_label_match_id": "__".join([entity_id, outcome]),
                            "evidence_type": "movement_situation", "numeric_information": "absent",
                            "template_id": event["id"] + "__non_numeric" +
                                ("" if band == "standard" else "__" + band) +
                                ("__" + order if alternate else ""),
                            "length_band": band, "length_match_id": event_id + "__" + order,
                            "entity_order_match_id": event_id + "__" + band,
                            "context_entity_order": order,
                            "reference_order_id": "person_relative_to_object",
                            "outcome": outcome, "correct_target": target_ids[outcome],
                            "event_id": event_id,
                            "distance_trajectory_id": "", "numeric_case_id": "", "unit_id": "",
                            "initial_distance": "", "final_distance": "", "movement_distance": "",
                            "person_before": "", "person_after": "", "object_before": "", "object_after": "",
                            "person_moves": event["id"] == "both_move_same_separation",
                            "object_moves": True, "object_stationary": False,
                            "passes_reference": False if outcome == "closer" else "",
                            "target_relation_repeated": False,
                            "Context": event[template_key].format(**fmt, direction=direction)})
                        nonnumeric[name["id"], obj["id"], event["id"], outcome, band, order] = row
                        length_groups.setdefault(row["length_match_id"], {})[band] = row
                        entity_order_groups.setdefault(row["entity_order_match_id"], {})[order] = row
                        label_id = "direct_" + order
                        label = direct[name["id"], obj["id"], outcome, label_id]
                        matches.append({"reference_probe_id": label["probe_id"],
                            "variant_probe_id": row["probe_id"], "control": "direct_label_vs_event",
                            "outcome": outcome, "numeric_case_id": "", "unit_id": ""})

        for case, unit, event in product(cases, parent["units"], config["event_templates"]):
            initial = int(case["initial_distance"])
            movement = int(case["movement_distance"])
            initial_unit = unit["singular"] if initial == 1 else unit["plural"]
            movement_unit = unit["singular"] if movement == 1 else unit["plural"]
            for outcome in event["outcomes"]:
                direction = "toward" if outcome == "closer" else "away from" if outcome == "farther" else ""
                person_before, object_before = 0, initial
                if event["id"] == "reference_object_moves":
                    person_after = 0
                    object_after = initial - movement if outcome == "closer" else initial + movement
                else:
                    person_after, object_after = movement, initial + movement
                final = abs(person_after - object_after)
                change = final - abs(person_before - object_before)
                computed = "closer" if change < 0 else "farther" if change > 0 else "unchanged"
                assert computed == outcome and final == int(case[outcome + "_final_distance"])
                assert event["id"] != "reference_object_moves" or object_after > person_after
                numeric_id = "__".join([entity_id, case["id"], unit["id"], event["id"], outcome])
                for band in bands:
                    for order, template_key, alternate in order_forms(event, band, "numeric"):
                        probe_id = (numeric_id if band == "standard" else numeric_id + "__" + band)
                        if alternate:
                            probe_id += "__" + order
                        row = add({**base, "probe_id": probe_id,
                            "condition_id": event["id"], "event_family": event["id"],
                            "direct_label_match_id": "__".join([entity_id, outcome]),
                            "evidence_type": "movement_situation", "numeric_information": "specific",
                            "template_id": event["id"] + "__numeric" +
                                ("" if band == "standard" else "__" + band) +
                                ("__" + order if alternate else ""),
                            "length_band": band, "length_match_id": numeric_id + "__" + order,
                            "entity_order_match_id": numeric_id + "__" + band,
                            "context_entity_order": order,
                            "reference_order_id": "person_relative_to_object",
                            "outcome": outcome, "correct_target": target_ids[outcome],
                            "event_id": numeric_id,
                            "distance_trajectory_id": "__".join([entity_id, case["id"], unit["id"], outcome]),
                            "numeric_case_id": case["id"], "unit_id": unit["id"], "unit": unit["unit"],
                            "initial_distance": initial, "final_distance": final, "movement_distance": movement,
                            "person_before": person_before, "person_after": person_after,
                            "object_before": object_before, "object_after": object_after,
                            "person_moves": event["id"] == "both_move_same_separation",
                            "object_moves": True, "object_stationary": False,
                            "passes_reference": False if outcome == "closer" else "",
                            "target_relation_repeated": False,
                            "Context": event[template_key].format(**fmt, initial=f"{initial} {initial_unit}",
                                movement=f"{movement} {movement_unit}", direction=direction)})
                        numeric.append(row)
                        length_groups.setdefault(row["length_match_id"], {})[band] = row
                        entity_order_groups.setdefault(row["entity_order_match_id"], {})[order] = row
                        label = direct[name["id"], obj["id"], outcome, "direct_" + order]
                        matches.append({"reference_probe_id": label["probe_id"],
                            "variant_probe_id": row["probe_id"], "control": "direct_label_vs_event",
                            "outcome": outcome, "numeric_case_id": case["id"], "unit_id": unit["id"]})
                        counterpart = nonnumeric[name["id"], obj["id"], event["id"], outcome, band, order]
                        assert all(row[k] == counterpart[k] for k in ["Target1", "Target2", "Target3", "outcome", "length_band", "context_entity_order"])
                        matches.append({"reference_probe_id": row["probe_id"],
                            "variant_probe_id": counterpart["probe_id"],
                            "control": "numeric_vs_non_numeric", "outcome": outcome,
                            "numeric_case_id": case["id"], "unit_id": unit["id"]})

                        if band == "standard":
                            for condition, control in [
                                ("explicit_distance", "same_distances_different_event"),
                                (("movement_description" if outcome != "unchanged" else "orientation_control"),
                                 "same_distances_different_mover_or_action")
                            ]:
                                old = original_index[name["id"], obj["id"], case["id"], unit["id"], condition, outcome]
                                assert all(row[k] == old[k] for k in ["Target1", "Target2", "Target3", "initial_distance", "final_distance"])
                                if outcome != "unchanged":
                                    assert row["movement_distance"] == old["movement_distance"]
                                matches.append({"reference_probe_id": old["probe_id"],
                                    "variant_probe_id": row["probe_id"], "control": control,
                                    "outcome": outcome, "numeric_case_id": case["id"], "unit_id": unit["id"]})

    for name, obj, outcome in product(parent["names"], parent["objects"], target_ids):
        a = direct[name["id"], obj["id"], outcome, "direct_person_first"]
        b = direct[name["id"], obj["id"], outcome, "direct_object_first"]
        assert all(a[k] == b[k] for k in ["Target1", "Target2", "Target3", "correct_target"])
        matches.append({"reference_probe_id": a["probe_id"], "variant_probe_id": b["probe_id"],
            "control": "direct_entity_order", "outcome": outcome,
            "numeric_case_id": "", "unit_id": ""})

    assert len(length_groups) == 2664
    for group in length_groups.values():
        assert set(group) == set(bands)
        compact, standard, expanded = (group[band] for band in bands)
        assert all(all(row[k] == standard[k] for k in [
            "event_id", "name_id", "object_id", "outcome", "correct_target",
            "Target1", "Target2", "Target3", "numeric_information",
            "initial_distance", "final_distance", "movement_distance",
            "person_before", "person_after", "object_before", "object_after"
        ]) for row in [compact, expanded])
        assert compact["Context_word_count"] < standard["Context_word_count"] < expanded["Context_word_count"]
        assert compact["Context_token_count"] < standard["Context_token_count"] < expanded["Context_token_count"]
        for variant in [compact, expanded]:
            matches.append({"reference_probe_id": standard["probe_id"],
                "variant_probe_id": variant["probe_id"], "control": "length_variant",
                "outcome": standard["outcome"], "numeric_case_id": standard["numeric_case_id"],
                "unit_id": standard["unit_id"]})

    assert len(entity_order_groups) == 3996
    for group in entity_order_groups.values():
        assert set(group) == {"person_first", "object_first"}
        a, b = group["person_first"], group["object_first"]
        assert all(a[k] == b[k] for k in [
            "event_id", "event_family", "name_id", "object_id", "outcome", "correct_target",
            "Target1", "Target2", "Target3", "numeric_information", "length_band",
            "initial_distance", "final_distance", "movement_distance",
            "person_before", "person_after", "object_before", "object_after"
        ])
        assert a["Context"] != b["Context"]
        matches.append({"reference_probe_id": a["probe_id"],
            "variant_probe_id": b["probe_id"], "control": "event_entity_order",
            "outcome": a["outcome"], "numeric_case_id": a["numeric_case_id"],
            "unit_id": a["unit_id"]})

    assert len(direct) == 72 and len(nonnumeric) == 216 and len(numeric) == 7776
    assert len(rows) == 8064 and len({r["probe_id"] for r in rows}) == 8064
    assert Counter(r["outcome"] for r in rows) == {"closer": 2688, "farther": 2688, "unchanged": 2688}
    assert Counter(r["condition_id"] for r in rows) == {
        "direct_label": 72, "reference_object_moves": 5328, "both_move_same_separation": 2664}
    assert Counter(r["context_entity_order"] for r in rows if r["condition_id"] != "direct_label") == {
        "person_first": 3996, "object_first": 3996}
    assert len(matches) == 30312
    assert Counter(r["control"] for r in matches) == {
        "numeric_vs_non_numeric": 7776, "same_distances_different_event": 2592,
        "same_distances_different_mover_or_action": 2592,
        "direct_label_vs_event": 7992, "direct_entity_order": 36,
        "event_entity_order": 3996, "length_variant": 5328}
    for row in rows:
        assert not row["target_relation_repeated"] or row["condition_id"] == "direct_label"
        context = row["Context"].lower()
        first_person = re.search(r"\b" + re.escape(row["name"].lower()) + r"\b", context)
        first_object = re.search(r"\bthe " + re.escape(row["object"].lower()) + r"\b", context)
        assert first_person and first_object
        expected = "person_first" if first_person.start() < first_object.start() else "object_first"
        assert row["context_entity_order"] == expected, row["probe_id"]
    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    write_csv(out / "probes.csv", rows)
    (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_csv(out / "matches.csv", matches)
    write_csv(out / "direct_examples.csv", [r for r in rows if r["condition_id"] == "direct_label"])
    write_csv(out / "event_examples.csv", [r for r in rows if r["name_id"] == "name_0" and r["object_id"] == "obj_cone"
              and (r["numeric_information"] == "absent" or
                   r["numeric_case_id"] == "num_00" and r.get("unit_id") == "unit_ft")])
    write_csv(out / "length_examples.csv", [r for r in rows if r["name_id"] == "name_0" and r["object_id"] == "obj_cone"
              and r["condition_id"] != "direct_label" and
              (r["numeric_information"] == "absent" or
               r["numeric_case_id"] == "num_00" and r.get("unit_id") == "unit_ft")])
    write_csv(out / "event_templates.csv", config["event_templates"])
    write_csv(out / "direct_label_templates.csv", config["direct_label_templates"])
    manifest = {"probe_name": config["probe_name"], "version": config["version"],
        "context_rows": len(rows), "direct_label_rows": len(direct),
        "numeric_event_rows": len(numeric), "non_numeric_event_rows": len(nonnumeric),
        "event_families": 2, "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "length_bands": dict(Counter(r["length_band"] for r in rows)),
        "length_groups": len(length_groups),
        "entity_order_groups": len(entity_order_groups),
        "comparison_links": len(matches), "unique_context_texts": len({r["Context"] for r in rows}),
        "default_score_reduction": "mean", "new_forms_evaluated": False,
        "tokenizer_source": args.tokenizer,
        "source_sha256": {role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in [("extension_components", config_path),
                ("parent_components", parent_path), ("parent_numeric_cases", cases_path),
                ("parent_probes", probes_path)]},
        "validation": "passed"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
