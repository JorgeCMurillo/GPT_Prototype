#!/usr/bin/env python3
"""Generate setting-matched close/far evidence conditions."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    config_path = ROOT / "components.json"
    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []

    for setting, name, obj, evidence, order in product(
        config["settings"], config["names"], config["objects"],
        config["evidence_conditions"], config["target_orders"]
    ):
        fmt = {
            "setting": setting["text"], "person_endpoint": setting["person_endpoint"],
            "far_endpoint": setting["far_endpoint"], "name": name["text"],
            "object": obj["text"]
        }
        scene_id = "__".join([setting["id"], name["id"], obj["id"]])
        row = {
            "probe_id": "__".join([scene_id, evidence["id"], order["id"]]),
            "probe_type": config["probe_name"], "probe_version": config["version"],
            "scene_id": scene_id, "evidence_match_id": scene_id + "__" + order["id"],
            "target_order_match_id": scene_id + "__" + evidence["id"],
            "setting_match_id": "__".join([name["id"], obj["id"], evidence["id"], order["id"]]),
            "setting_id": setting["id"], "setting": setting["text"],
            "endpoint_geometry": setting["geometry"], "name_id": name["id"],
            "name": name["text"], "object_id": obj["id"], "object": obj["text"],
            "condition_id": evidence["id"], "evidence_type": evidence["evidence_type"],
            "context_structure": evidence["context_structure"],
            "context_entity_order": evidence["context_entity_order"], "target_order": order["id"],
            "reverses_context_entity_order": evidence["context_entity_order"] in ["object_first", "person_first"] and evidence["context_entity_order"] != order["id"],
            "same_setting_in_both_contexts": True,
            "close_requires_body_part_inference": evidence["id"] == "endpoint_placement",
            "far_requires_endpoint_inference": evidence["id"] == "endpoint_placement",
            "relation_words_explicit": evidence["id"] == "direct_label",
            "distance_magnitude_explicit": evidence["id"].startswith("distance_phrase"),
            "target_relation_clause_repeated": evidence["id"] == "direct_label" and order["id"] == "object_first",
            "Context1": evidence["close"].format(**fmt),
            "Context2": evidence["far"].format(**fmt),
            "Target1": order["close"].format(**fmt),
            "Target2": order["far"].format(**fmt)
        }
        for field in ["Context1", "Context2", "Target1", "Target2"]:
            row[field + "_word_count"] = len(re.findall(r"\b\w+\b", row[field]))
            row[field + "_token_count"] = len(tokenizer.encode(row[field], add_special_tokens=False))
        rows.append(row)

    expected_scenes = len(config["settings"]) * len(config["names"]) * len(config["objects"])
    expected_rows = expected_scenes * len(config["evidence_conditions"]) * len(config["target_orders"])
    assert len(rows) == expected_rows == 1536
    assert len({r["probe_id"] for r in rows}) == len(rows)
    assert len({r["scene_id"] for r in rows}) == expected_scenes == 192
    assert Counter(r["condition_id"] for r in rows) == {
        "direct_label": 384, "distance_phrase": 384,
        "distance_phrase_object_first": 384, "endpoint_placement": 384
    }

    evidence_groups, order_groups, setting_groups = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for row in rows:
        evidence_groups[row["evidence_match_id"]][row["condition_id"]] = row
        order_groups[row["target_order_match_id"]][row["target_order"]] = row
        setting_groups[row["setting_match_id"]][row["setting_id"]] = row

    matches = []
    for group in evidence_groups.values():
        assert set(group) == {"direct_label", "distance_phrase", "distance_phrase_object_first", "endpoint_placement"}
        direct = group["direct_label"]
        for reference, variant, label in [
            (direct, group["distance_phrase"], "direct_vs_distance"),
            (direct, group["endpoint_placement"], "direct_vs_placement"),
            (group["distance_phrase"], group["endpoint_placement"], "distance_vs_placement")
        ]:
            assert all(reference[k] == variant[k] for k in [
                "setting_id", "name_id", "object_id", "target_order", "Target1", "Target2"
            ])
            matches.append({"reference_probe_id": reference["probe_id"],
                "variant_probe_id": variant["probe_id"], "control": "evidence_type",
                "comparison": label, "setting_id": reference["setting_id"],
                "scene_id": reference["scene_id"], "target_order": reference["target_order"]})
        person_first_distance = group["distance_phrase"]
        object_first_distance = group["distance_phrase_object_first"]
        assert all(person_first_distance[k] == object_first_distance[k] for k in [
            "setting_id", "name_id", "object_id", "target_order", "Target1", "Target2"
        ])
        for field in ["Context1", "Context2"]:
            assert Counter(re.findall(r"\b\w+\b", person_first_distance[field].lower())) == Counter(
                re.findall(r"\b\w+\b", object_first_distance[field].lower()))
        matches.append({"reference_probe_id": person_first_distance["probe_id"],
            "variant_probe_id": object_first_distance["probe_id"], "control": "context_entity_order",
            "comparison": "distance_person_first_vs_object_first",
            "setting_id": person_first_distance["setting_id"],
            "scene_id": person_first_distance["scene_id"],
            "target_order": person_first_distance["target_order"]})

    for group in order_groups.values():
        assert set(group) == {"object_first", "person_first"}
        a, b = group["object_first"], group["person_first"]
        assert a["Context1"] == b["Context1"] and a["Context2"] == b["Context2"]
        matches.append({"reference_probe_id": a["probe_id"], "variant_probe_id": b["probe_id"],
            "control": "target_order", "comparison": "object_first_vs_person_first",
            "setting_id": a["setting_id"], "scene_id": a["scene_id"], "target_order": "varied"})

    for group in setting_groups.values():
        assert set(group) == {s["id"] for s in config["settings"]}
        reference = group["gym"]
        for setting_id, variant in group.items():
            if setting_id == "gym":
                continue
            assert all(reference[k] == variant[k] for k in ["name_id", "object_id", "condition_id", "target_order", "Target1", "Target2"])
            matches.append({"reference_probe_id": reference["probe_id"],
                "variant_probe_id": variant["probe_id"], "control": "setting",
                "comparison": "gym_vs_" + setting_id, "setting_id": setting_id,
                "scene_id": variant["scene_id"], "target_order": variant["target_order"]})

    assert len(matches) == 3584
    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    write_csv(out / "probes.csv", rows)
    (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_csv(out / "matches.csv", matches)
    for key in ["settings", "names", "objects", "evidence_conditions", "target_orders"]:
        write_csv(out / (key + ".csv"), config[key])
    write_csv(out / "review_examples.csv", [
        r for r in rows if r["name_id"] == "name_maya" and r["object_id"] == "obj_ball"
        and r["setting_id"] in ["gym", "courtyard"]
    ])
    write_csv(out / "scene_overview.csv", [
        r for r in rows if r["name_id"] == "name_maya" and r["object_id"] == "obj_ball"
        and r["target_order"] == "object_first"
    ])

    manifest = {
        "probe_name": config["probe_name"], "version": config["version"],
        "underlying_scenes": expected_scenes, "settings": len(config["settings"]),
        "names": len(config["names"]), "objects": len(config["objects"]),
        "evidence_conditions": len(config["evidence_conditions"]),
        "target_orders": len(config["target_orders"]), "matched_pairs": len(rows),
        "context_judgments": 2 * len(rows), "pairs_per_evidence_condition": 384,
        "pairs_per_evidence_condition_and_target_order": 192,
        "comparison_links": len(matches),
        "unique_conditional_sequences": len({
            (r[f"Context{c}"], r[f"Target{t}"]) for r in rows for c in [1, 2] for t in [1, 2]
        }),
        "default_score_reduction": "mean", "new_forms_evaluated": False,
        "tokenizer_source": args.tokenizer,
        "components_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "validation": "passed"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
