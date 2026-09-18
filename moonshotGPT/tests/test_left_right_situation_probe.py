"""Semantic and matched-variant checks for fixed-frame spatial scenes."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "data/left_right_situation_probe"
SPEC = importlib.util.spec_from_file_location("left_right_generator", ROOT / "generate.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def dataset():
    return probe.build(json.loads((ROOT / "components.json").read_text()))


def test_answer_semantics_and_rendered_positions(dataset):
    rows, _ = dataset
    for row in rows:
        if row["probe_family"] == "control":
            continue
        for i, truth in enumerate(("left", "right"), 1):
            a, b = row[f"target_end_{truth}"], row[f"reference_end_{truth}"]
            expected = "left" if a < b else "right"
            if row["target_entity_order"] == "reference_first":
                expected = "right" if expected == "left" else "left"
            assert f"to the {expected} of" in row[f"Target{i}"]
            labels = row["displayed_slot_labels_left_to_right"].split(", ")
            for entity in ("target", "reference"):
                label = labels[row[f"{entity}_end_{truth}"] - 1]
                assert row[f"{entity}_end_label_{truth}"] == label
                assert row[f"Context{i}"].startswith(f"Fixed screen slots, left to right: {', '.join(labels)}. ")
                subject = re.escape(f"The {row[entity + '_object']} icon")
                if row["length_band"] != "expanded":
                    rendered = re.search(subject + r" (?:(?:is|stays) at (\S+)|moves from (\S+) to (\S+))\.", row[f"Context{i}"])
                    assert rendered
                    first = rendered[1] or rendered[2]
                    last = rendered[1] or rendered[3]
                else:
                    rendered = re.search(subject + r" starts at (\S+?)(?: and stays there|, then moves to (\S+))\.", row[f"Context{i}"])
                    assert rendered
                    first = rendered[1]
                    last = rendered[2] or first
                assert labels.index(first) + 1 == row[f"{entity}_start_{truth}"]
                assert labels.index(last) + 1 == row[f"{entity}_end_{truth}"]


def test_numbering_reversal_preserves_scene_but_reverses_numeric_comparison(dataset):
    rows, links = dataset
    lookup = {r["probe_id"]: r for r in rows}
    pairs = [m for m in links if m["control"] == "numbering_direction"]
    assert len(pairs) == 576
    for link in pairs:
        a, b = lookup[link["base_probe_id"]], lookup[link["variant_probe_id"]]
        for truth in ("left", "right"):
            diff_a = int(a[f"target_end_label_{truth}"]) - int(a[f"reference_end_label_{truth}"])
            diff_b = int(b[f"target_end_label_{truth}"]) - int(b[f"reference_end_label_{truth}"])
            assert diff_a == -diff_b
        assert a["Target1"] == b["Target1"] and a["Target2"] == b["Target2"]


def test_no_crossing_pairs_keep_direction_and_reverse_answer(dataset):
    rows, _ = dataset
    for row in rows:
        if row["event_family"] != "target_moves_without_crossing":
            continue
        assert row["target_motion_left"] == row["target_motion_right"] != "still"
        for truth in ("left", "right"):
            assert row[f"initial_relation_{truth}"] == row[f"final_relation_{truth}"] == truth
            assert not row[f"order_reversed_{truth}"]


def test_bad_case_rejected():
    case = json.loads((ROOT / "components.json").read_text())["cases"][3]
    case["left"]["target"] = [5, 4]
    with pytest.raises(AssertionError):
        probe.validate_case(case)


def test_length_and_mention_order_matches(dataset):
    rows, links = dataset
    lookup = {r["probe_id"]: r for r in rows}
    for link in links:
        a, b = lookup[link["base_probe_id"]], lookup[link["variant_probe_id"]]
        if link["control"] == "length_band":
            for i in (1, 2):
                assert a[f"Context{i}_word_count"] < b[f"Context{i}_word_count"]
        if link["control"] == "context_entity_order":
            assert a["last_mentioned_entity"] != b["last_mentioned_entity"]
            for row in (a, b):
                noun = row[row["last_mentioned_entity"] + "_object"]
                for i in (1, 2):
                    context = row[f"Context{i}"]
                    other = row["target_object"] if noun == row["reference_object"] else row["reference_object"]
                    assert context.rfind(noun) > context.rfind(other)


@pytest.mark.parametrize("band,minimum,average,maximum", [
    ("compact", 23, 25, 27), ("standard", 27, 29, 31), ("expanded", 35, 36, 37)])
def test_word_budgets(dataset, band, minimum, average, maximum):
    rows, _ = dataset
    counts = [len(r[f"Context{i}"].split()) for r in rows
              if r["probe_family"] == "applied" and r["length_band"] == band for i in (1, 2)]
    assert len(counts) == 1536
    assert min(counts) == minimum and max(counts) == maximum
    assert sum(counts) / len(counts) == average


def test_only_applied_wording_changed_from_v1(dataset):
    baseline = ROOT.parents[1] / "runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_v1/input_probes.jsonl"
    if not baseline.exists():
        pytest.skip("Archived v1 input snapshot not available")
    old = {r["probe_id"]: r for r in map(json.loads, baseline.read_text().splitlines())}
    rows, _ = dataset
    assert {r["probe_id"] for r in rows} == set(old)
    changed = 0
    for row in rows:
        allowed = {"probe_version"}
        previous = old[row["probe_id"]]
        if row["probe_family"] == "applied":
            allowed.update(("Context1", "Context2", "Context1_word_count", "Context2_word_count"))
            assert all(row[f"Context{i}"] != previous[f"Context{i}"] for i in (1, 2))
            changed += 1
        assert {k: v for k, v in row.items() if k not in allowed} == {k: v for k, v in previous.items() if k not in allowed}
    assert changed == 2304
