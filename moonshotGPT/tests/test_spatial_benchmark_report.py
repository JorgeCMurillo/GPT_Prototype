import copy
import json

import pytest

from data.spatial_benchmark.report import ROOT, collapse, load_manifest, normalize, probability, summarize


def row(category="a", family="f", case="one", evidence="numeric", score=0.5, group=None, cell=()):
    return dict(category=category, family=family, case=case, evidence=evidence,
                accuracy=score, both_correct=float(score == 1), group=group or category, cell=cell)


def test_balances_categories_families_evidence_and_variant_cells():
    # Numeric evidence has 100 rows but half the evidence weight. One category
    # has two families; category b still receives exactly half the headline.
    rows = [row(score=1)] * 100 + [row(evidence="named", score=0)]
    rows += [row(family="g", score=0)]
    rows += [row(category="b", score=1)]
    result = summarize(rows, 100)
    assert result["categories"]["a"]["accuracy"]["estimate"] == 0.25
    assert result["overall"]["accuracy"]["estimate"] == 0.625
    rows = [row(score=1, cell=("short",))] * 100 + [row(score=0, cell=("long",))]
    assert summarize(rows, 100)["overall"]["accuracy"]["estimate"] == 0.5


def test_duplicate_renderings_do_not_increase_bootstrap_precision():
    rows = [row(case="one", score=0), row(case="two", score=1)]
    a = summarize(rows, 2000)
    b = summarize(rows * 100, 2000)
    assert a["overall"] == b["overall"]
    assert b["n_unique_cluster_units"] == 2


def test_paired_contexts_remain_one_unit():
    # A constant answer gets one of each pair right, never both.
    result = summarize([row(case=str(i)) for i in range(5)], 100)
    assert result["overall"]["accuracy"]["ci95"] == [0.5, 0.5]
    assert result["overall"]["both_correct"]["estimate"] == 0


def test_shared_axis_draws_preserve_covariance():
    rows = [row(category=c, case=str(i), score=float(i == j), group="shared")
            for c, j in [("north_south", 0), ("east_west", 1)] for i in range(2)]
    result = summarize(rows, 1000)
    assert result["overall"]["accuracy"]["ci95"] == [0.5, 0.5]
    assert result["n_unique_cluster_units"] == 2


def test_shared_groups_require_identical_case_sets():
    with pytest.raises(ValueError, match="Unmatched case sets"):
        summarize([row(group="shared"), row(category="b", case="different", group="shared")], 100)


def test_incomplete_evidence_crossing_rejected():
    with pytest.raises(ValueError, match="Incomplete evidence"):
        collapse([row(), row(evidence="named"), row(case="two")])


def test_reproducible_and_input_order_independent():
    rows = [row(case=str(i), score=float(i % 2)) for i in range(8)]
    assert summarize(rows, 500, 7) == summarize(list(reversed(rows)), 500, 7)


def test_singleton_is_flagged_not_claimed_as_precise():
    result = summarize([row()], 100)
    assert result["categories"]["a"]["all_strata_singleton"]
    assert any("cannot be estimated" in s for s in result["warnings"])


@pytest.mark.parametrize("value", ["nan", "inf", -0.1, 1.1])
def test_rejects_invalid_scores(value):
    with pytest.raises(ValueError):
        probability(value)


def test_controls_excluded_and_duplicate_pairs_rejected():
    spec = dict(category="a", adapter="directional", select={"probe_family": "applied"},
                cluster_group="a", evidence_field="evidence", accuracy_field="accuracy")
    item = dict(probe_id="p", probe_family="applied", event_family="f", case_id="c",
                evidence="numeric", accuracy=0.5, both_correct=False)
    assert len(normalize([item, {**item, "probe_family": "control"}], spec)) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        normalize([item, item], spec)


def test_checkpoint_mismatch_rejected(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps(dict(
        probe_id="p", event_family="f", case_id="c", evidence="numeric",
        accuracy=0.5, both_correct=False)) + "\n")
    sources = []
    for cat in ("a", "b"):
        (tmp_path / f"{cat}.json").write_text(json.dumps(dict(model=cat, score_reduction="mean")))
        sources.append(dict(category=cat, path="rows.jsonl", metadata_paths=[f"{cat}.json"],
                            adapter="directional", cluster_group=cat,
                            evidence_field="evidence", accuracy_field="accuracy"))
    with pytest.raises(ValueError, match="Expected one checkpoint"):
        load_manifest({"sources": sources}, tmp_path)


def test_saved_results_reproduce_existing_category_macros():
    manifest = json.loads((ROOT / "data/spatial_benchmark/qwen3_step19500.json").read_text())
    if not all((ROOT / s["path"]).exists() for s in manifest["sources"]):
        pytest.skip("Local scored artifacts not available")
    rows, _, _ = load_manifest(manifest, ROOT)
    result = summarize(rows, 100)
    assert len(result["categories"]) == 6
    assert len(rows) == 16380
    for spec in manifest["sources"]:
        cat = spec["category"]
        directory = (ROOT / spec["path"]).parent
        if cat in ("above_below", "left_right"):
            expected = json.loads((directory / "summary.json").read_text())["overall_applied"]["equal_family_binary_accuracy"]
        elif cat in ("north_south", "east_west"):
            expected = json.loads((directory / "summary.json").read_text())["results"][cat]["balanced_events"]["accuracy"]
        elif cat == "close_far":
            expected = json.loads((directory / "applied_score_summary.json").read_text())["overall_applied_mean"]["raw_accuracy"]
        else:
            expected = json.loads((directory / "overall_summary.json").read_text())["overall_applied_mean"]["raw_accuracy"]
        assert result["categories"][cat]["accuracy"]["estimate"] == pytest.approx(expected, abs=1e-12)
    # Category filtering doesn't silently retain the two distance categories.
    directional = ["above_below", "left_right", "north_south", "east_west"]
    filtered, _, _ = load_manifest(manifest, ROOT, directional)
    assert {r["category"] for r in filtered} == set(directional)
    wrong_count = copy.deepcopy(manifest)
    wrong_count["sources"][0]["expected_pairs"] = 1
    with pytest.raises(ValueError, match="expected 1 pairs"):
        load_manifest(wrong_count, ROOT)
