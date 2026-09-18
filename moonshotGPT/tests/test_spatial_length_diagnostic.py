import json

import numpy as np
import pytest

from data.spatial_benchmark.length_diagnostic import (
    analyze, decode, describe, features, hierarchy_weights, load_comparisons, unique_index,
)
from data.spatial_benchmark.report import ROOT


def row(delta, choice, gold=1, case="one", family="f", category="a", weight=1):
    return dict(category=category, family=family, case=case, group=category,
                evidence="numeric", cell=(), contrast="test", comparison_hash=f"{case}:{delta}:{choice}",
                length_difference=delta, length1=10 + delta, length2=10, gold=gold, weight=weight,
                mean_choice=choice, sum_choice=choice,
                mean_correct=choice == gold, sum_correct=choice == gold)


def test_point_biserial_matches_pearson():
    rows = [row(x, c) for x, c in [(-2, 1), (-1, 1), (0, 2), (1, 1), (2, 2)]]
    result = describe(rows, "mean")["raw"]
    expected = np.corrcoef([r["length_difference"] for r in rows], [r["mean_choice"] == 1 for r in rows])[0, 1]
    assert result["point_biserial_r"] == pytest.approx(expected)


def test_perfect_shorter_preference_negative_r():
    result = describe([row(-2, 1), row(2, 2)], "mean")["raw"]
    assert result["point_biserial_r"] == pytest.approx(-1)
    assert result["shorter_choice_rate"] == 1


def test_equal_lengths_undefined_not_zero():
    result = describe([row(0, 1), row(0, 2)], "mean")["raw"]
    assert result["point_biserial_r"] is None
    assert result["shorter_choice_rate"] is None
    assert "length" in result["correlation_unavailable_reason"]


def test_constant_choices_undefined():
    assert describe([row(-2, 1), row(2, 1)], "mean")["raw"]["point_biserial_r"] is None


def test_ties_excluded_from_choice_rate_but_accuracy_includes_them():
    result = describe([row(-2, 1), row(2, 2, gold=2), row(-2, 0)], "mean")["raw"]
    assert result["point_biserial_r"] == pytest.approx(-1)
    assert result["shorter_choice_rate"] == 1
    assert result["accuracy"] == pytest.approx(2/3)
    assert result["accuracy_correct_shorter"] == pytest.approx(2/3)
    assert result["tie_rate"] == pytest.approx(1/3)


def test_weighted_correlation_matches_integer_replication():
    rows = [row(-1, 1, weight=3), row(0, 2, weight=2), row(1, 1)]
    replicated = [rows[0]] * 3 + [rows[1]] * 2 + [rows[2]]
    assert describe(rows, "mean")["balanced"] == describe(replicated, "mean")["raw"]


def test_hierarchy_does_not_overweight_numeric_renderings():
    rows = [row(0, 1)] * 100 + [{**row(0, 2), "evidence": "named"}] + [row(0, 1, category="b")]
    weights = hierarchy_weights(rows)
    assert sum(weights[:100]) == pytest.approx(.25)
    assert weights[-2:] == pytest.approx([.25, .5])


def test_singletons_do_not_get_spurious_length_confidence_interval():
    rows = [row(-2, 1, family="f"), row(0, 2, family="g")]
    result = analyze(rows, 100)
    scope = result["scopes"]["overall"]
    assert scope["length_informative_clusters"] == 1
    assert not scope["length_informative_clusters_resampleable"]
    assert scope["mean"]["balanced_cluster_intervals"]["point_biserial_r"]["ci95"] is None


def test_cluster_intervals_reproducible_and_ignore_uniform_replication():
    rows = [row(-2, 1, case="one", weight=.5), row(2, 2, case="two", weight=.5)]
    a = analyze(rows, 200, 42)
    b = analyze(rows, 200, 42)
    assert a == b
    replicated = [{**r, "weight": r["weight"]/10} for r in rows for _ in range(10)]
    c = analyze(replicated, 200, 42)
    for metric in ("point_biserial_r", "shorter_choice_rate", "accuracy"):
        assert c["scopes"]["overall"]["mean"]["balanced_cluster_intervals"][metric] == a["scopes"]["overall"]["mean"]["balanced_cluster_intervals"][metric]


def test_same_candidate_swap_preserves_correlation_and_shorter_rate():
    rows = [row(-2, 1), row(0, 2), row(2, 2)]
    swapped = [row(-r["length_difference"], 3-r["mean_choice"], gold=2) for r in rows]
    a, b = describe(rows, "mean")["raw"], describe(swapped, "mean")["raw"]
    assert a["point_biserial_r"] == pytest.approx(b["point_biserial_r"])
    assert a["shorter_choice_rate"] == b["shorter_choice_rate"]


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        unique_index([{"id": 1}, {"id": 1}], lambda r: r["id"])


def test_saved_results_lengths_choices_and_balanced_accuracy():
    manifest = json.loads((ROOT / "data/spatial_benchmark/qwen3_step19500.json").read_text())
    if not all((ROOT / s["path"]).exists() for s in manifest["sources"]):
        pytest.skip("Local scored artifacts unavailable")
    rows, _ = load_comparisons(manifest, ROOT)
    assert len(rows) == 32760
    assert sum(r["length_difference"] != 0 for r in rows) == 10656
    assert {r["category"] for r in rows if r["length_difference"]} == {"closer_farther"}
    assert {(r["length1"], r["length2"]) for r in rows if r["length_difference"]} == {(10, 13)}
    assert all(r["mean_choice"] == r["sum_choice"] for r in rows if not r["length_difference"])
    assert all(r["mean_choice"] == 2 for r in rows if r["length_difference"])
    assert sum(r["mean_choice"] != r["sum_choice"] for r in rows) == 8770
    assert describe(rows, "mean")["balanced"]["accuracy"] == pytest.approx(0.5121578682270234)
    assert describe(rows, "sum")["balanced"]["accuracy"] == pytest.approx(0.5143280071159122)
    # All close/far targets have seven CONDITIONAL tokens, even when standalone
    # token metadata in the original rows reports eight.
    assert {(r["length1"], r["length2"]) for r in rows if r["category"] == "close_far"} == {(7, 7)}
