import importlib.util
import gzip
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/spatial_neutral_context_probe/generate.py"
SPEC = importlib.util.spec_from_file_location("neutral_spatial", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@lru_cache(maxsize=1)
def rows():
    # Validate the published snapshot without depending on local model run files.
    path = ROOT / "data/spatial_neutral_context_probe/generated/neutral_probes.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_exact_inventory_and_no_gold_for_c0():
    data = rows()
    MODULE.validate(data)
    assert len(data) == 16630
    assert all(row["correct_target_for_context0"] is None for row in data)
    assert all(row["Context0"] not in (row["Context1"], row["Context2"]) for row in data)


def test_scope_excludes_controls_and_observer_turns():
    data = rows()
    assert not any(row["family"] in {"direct_label_control", "direct_relation", "observer_turn"}
                   for row in data)
    blocks = Counter(row["benchmark_block"] for row in data)
    assert blocks == {"six_category_composite": 16380, "supplement": 212,
                      "minimal_cardinal_supplement": 38}


def test_both_targets_have_witnesses_and_originals_are_retained():
    for row in rows():
        assert row["original_pair_sha256"] == MODULE.stable_hash(
            [row[key] for key in ("Context1", "Context2", "Target1", "Target2")])
        assert row["neutral_witnesses"] == {
            row["correct_target_for_context1"]: "Context1",
            row["correct_target_for_context2"]: "Context2",
        }
        assert all(state["geometry"] for state in row["neutral_witness_states"].values())
        assert {row["canonical_target1"], row["canonical_target2"]} == {
            row["Target1"], row["Target2"]}


def test_representative_neutralization_rules():
    data = rows()
    above_static = next(row for row in data if row["source_dataset"] == "above_below"
                        and row["family"] == "static_placement")
    left_static = next(row for row in data if row["source_dataset"] == "left_right"
                       and row["family"] == "static_placement")
    close = next(row for row in data if row["source_dataset"] == "close_far")
    assert "one object at each position" in above_static["Context0"]
    assert "one icon in each slot" in left_static["Context0"]
    assert close["neutralization_rule_id"] in {
        "unspecified_distance_magnitude", "unspecified_setting_position"}
    assert any(row["neutralization_scope"] == "contrast_level" for row in data)
    assert all(row["manual_review_status"] == "pending" for row in data)


def test_persisted_manifest_and_output_hashes():
    generated = ROOT / "data/spatial_neutral_context_probe/generated"
    manifest = json.loads((generated / "manifest.json").read_text())
    output = generated / "neutral_probes.jsonl.gz"
    assert manifest["paired_rows"] == 16630
    assert manifest["evaluation_status"] == "not_scored"
    assert manifest["generator_sha256"] == MODULE.sha256(PATH)
    with gzip.open(output, "rb") as stream:
        payload = stream.read()
    assert manifest["neutral_probes_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(payload.splitlines()) == 16630
