import copy
import json
import re
from collections import Counter

import pytest

from data.front_behind_situation_probe.generate import (
    ROOT, answer, build, canonical_cases, reflect, relation, validate_rows,
)


@pytest.fixture(scope="module")
def config():
    return json.loads((ROOT / "components.json").read_text())


@pytest.fixture(scope="module")
def generated(config):
    return build(config)


def test_twelve_seeds_still_covered_in_numeric_cases(config, generated):
    _, _, cases = generated
    assert len(cases) == 10
    assert sorted(s for c in cases for s in c["source_seed_ids"]) == sorted(c["id"] for c in config["cases"])
    assert Counter(c["family"] for c in cases) == {
        "static_placement": 3, "target_crosses": 1, "reference_crosses": 1,
        "target_moves_without_crossing": 2, "both_move": 3}
    for seed in config["cases"]:
        cluster = next(c for c in cases if seed["id"] in c["source_seed_ids"])
        assert any(all(seed[k] == variant[k] for k in ("target", "reference"))
                   for variant in (cluster, cluster["mirror"]))


def test_counts_unique_ids_and_balanced_pairs(generated):
    rows, _, _ = generated
    assert len({r["probe_id"] for r in rows}) == len(rows) == 224
    assert Counter(r["probe_family"] for r in rows) == dict(event=212, control=4, observer_turn=8)
    assert Counter(r["evidence_format"] for r in rows if r["probe_family"] == "event") == dict(numeric=160, nonnumeric=52)
    for row in rows:
        assert {row["correct_target_for_context1"], row["correct_target_for_context2"]} == {"Target1", "Target2"}
        assert row["headline_eligible"] == (row["probe_family"] == "event")
    numeric = [r for r in rows if r["evidence_format"] == "numeric"]
    assert Counter(r["correct_target_for_context1"] for r in numeric) == {"Target1": 80, "Target2": 80}


def test_compact_only_and_no_long_scaffolding(config, generated):
    assert config["lengths"] == ["compact"]
    for row in generated[0]:
        assert row["length_band"] == "compact"
        assert not row["definition_present"] and not row["answer_bridge_present"]
        for i in (1, 2):
            text = row[f"Context{i}"]
            assert len(text.split()) <= config["max_context_words"]
            assert row[f"Context{i}_word_count"] == len(text.split())
            for old_phrase in ("red sign", "blue sign", "marked places", "to summarize", "means on the", "throughout the scene"):
                assert old_phrase not in text


def test_numeric_gold_from_rendered_positions(generated):
    for row in generated[0]:
        if row["evidence_format"] != "numeric":
            continue
        for i in (1, 2):
            text = row[f"Context{i}"]
            b = re.escape(row["reference_entity"])
            face = re.search(b + r" faces (increasing|decreasing) position numbers\.", text)
            assert face
            direction = 1 if face[1] == "increasing" else -1
            parsed = {}
            for role in ("target", "reference"):
                name = re.escape(row[role + "_entity"])
                static = re.search(name + r" stays at (\d+)\.", text)
                moving = re.search(name + r" moves from (\d+) to (\d+)( without turning)?\.", text)
                assert bool(static) != bool(moving)
                if static:
                    start = end = int(static[1])
                else:
                    start, end = int(moving[1]), int(moving[2])
                    if role == "reference":
                        assert moving[3] == " without turning"
                parsed[role] = (start, end)
                assert start == row[f"context{i}_{role}_start"]
                assert end == row[f"context{i}_{role}_end"]
            expected = "Target1" if (parsed["target"][1] - parsed["reference"][1]) * direction > 0 else "Target2"
            assert row[f"correct_target_for_context{i}"] == expected


def test_nonnumeric_gold_from_language_not_coordinate_metadata(generated):
    for row in generated[0]:
        if row["evidence_format"] != "nonnumeric":
            continue
        a, b = row["target_entity"], row["reference_entity"]
        for i in (1, 2):
            text = row[f"Context{i}"]
            if "stands between" in text:
                between = f"{a} stands between {b}" in text
                away = f"{b} faces away from the" in text
                front = between != away
                assert not row["initial_relation_explicit"]
            else:
                if f"{a} starts in front of {b}" in text:
                    initially_front = True
                    assert row["initial_relation_explicit"]
                elif f"{a} starts behind {b}" in text:
                    initially_front = False
                    assert row["initial_relation_explicit"]
                else:
                    assert f"{b} faces {a}" in text or f"{b} faces away from {a}" in text
                    initially_front = f"{b} faces away from {a}" not in text
                    assert not row["initial_relation_explicit"]
                crosses = "past" in text
                front = initially_front != crosses
                assert "without turning" in text
                if row["event_family"] in ("target_crosses", "reference_crosses"):
                    assert crosses and "stays still" in text
                if row["event_family"] == "target_moves_without_crossing":
                    assert not crosses
                if row["event_subtype"] == "same_direction_preserved":
                    assert "Both move equal distances" in text
                    assert "B's facing direction" in text or "Eli's facing direction" in text
            assert row[f"correct_target_for_context{i}"] == ("Target1" if front else "Target2")


def test_nonnumeric_static_cases_not_duplicated_for_unstated_distances(generated):
    static = [r for r in generated[0] if r["evidence_format"] == "nonnumeric" and r["event_family"] == "static_placement"]
    assert len(static) == 24
    assert {r["case_id"] for r in static} == {"static_adjacent"}
    assert {r["anchor"] for r in static} == {"door", "window", "gate"}
    for r in static:
        assert r["geometry_precision"] == "qualitative"


def test_variant_links_and_facing_reversal(generated):
    rows, links, _ = generated
    lookup = {r["probe_id"]: r for r in rows}
    assert {m["match_type"] for m in links} == {
        "context_entity_order", "target_entity_order", "facing_direction", "entity_pair_id", "anchor"}
    for link in links:
        a, b = lookup[link["probe_id_a"]], lookup[link["probe_id_b"]]
        for i in (1, 2):
            assert (a[f"correct_target_for_context{i}"] != b[f"correct_target_for_context{i}"]) == link["expected_gold_change"]
            if link["match_type"] == "target_entity_order":
                assert a[f"Context{i}"] == b[f"Context{i}"]
            if link["match_type"] == "facing_direction":
                for role in ("target", "reference"):
                    for endpoint in ("start", "end"):
                        key = f"context{i}_{role}_{endpoint}"
                        assert a[key] == b[key]
                assert a[f"context{i}_reference_facing_end"] == -b[f"context{i}_reference_facing_end"]


def test_reference_frame_preserved_in_answer_order_variants():
    assert answer("A", "B", "front", "reference_first") == "In front of B is A."
    assert answer("A", "B", "behind", "reference_first") == "Behind B is A."


def test_observer_turn_positions_fixed(generated):
    for row in generated[0]:
        if row["probe_family"] != "observer_turn":
            continue
        assert not row["headline_eligible"] and row["initial_relation_explicit"]
        for role in ("target", "reference"):
            assert len({row[f"context{i}_{role}_{point}"] for i in (1, 2) for point in ("start", "end")}) == 1
        assert not row["context1_relation_changed"]
        assert row["context2_relation_changed"]
        assert "does not turn" in row["Context1"]
        assert "turns halfway around in place" in row["Context2"]


def test_direct_labels_stay_separate(generated):
    for row in generated[0]:
        if row["probe_family"] == "control":
            assert row["final_relation_explicit"] and not row["headline_eligible"]
            for i in (1, 2):
                word = row[f"context{i}_gold_relation"]
                assert answer(row["target_entity"], row["reference_entity"], word, "target_first") == row[f"Context{i}"]


def test_invalid_geometry_rejected(config):
    bad = copy.deepcopy(config)
    bad["cases"][0]["target"] = [2, 2]
    with pytest.raises(ValueError, match="distinct positions"):
        canonical_cases(bad)
    bad = copy.deepcopy(config)
    bad["cases"][0]["target"] = [0, 3]
    with pytest.raises(ValueError, match="endpoints"):
        canonical_cases(bad)
    with pytest.raises(ValueError):
        relation(2, 1, 0)


def test_wrong_gold_detected(generated):
    bad = copy.deepcopy(generated[0][:1])
    bad[0]["correct_target_for_context1"] = "Target2"
    with pytest.raises(ValueError, match="answer key"):
        validate_rows(bad)


def test_long_contexts_and_old_tiers_rejected(config, generated):
    bad = copy.deepcopy(generated[0][:1])
    bad[0]["Context1"] += " Extra words." * 20
    with pytest.raises(ValueError, match="exceeds"):
        validate_rows(bad)
    bad_config = copy.deepcopy(config)
    bad_config["lengths"] = ["compact", "expanded"]
    with pytest.raises(ValueError, match="compact-only"):
        build(bad_config)


def test_reflection_and_repeatability(config, generated):
    for case in config["cases"]:
        restored = reflect(reflect(case))
        assert all(restored[k] == case[k] for k in ("target", "reference"))
    assert build(config) == generated


def test_generated_artifacts_are_current(generated):
    saved = [json.loads(line) for line in (ROOT / "generated/probes.jsonl").read_text().splitlines()]
    assert saved == generated[0]
    manifest = json.loads((ROOT / "generated/manifest.json").read_text())
    assert manifest["version"] == "2.0"
    assert manifest["paired_rows"] == len(saved)
