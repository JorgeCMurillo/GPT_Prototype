#!/usr/bin/env python3
"""Compute an equal-evidence applied close/far score from saved matched scenes."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = {
    "distance_phrase": "distance_phrase",
    "distance_phrase_object_first": "distance_phrase",
    "endpoint_placement": "endpoint_placement",
    "direct_label": "direct_label_control",
}


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.results_dir
    with (out / "item_scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1536
    assert Counter(r["condition_id"] for r in rows) == {k: 384 for k in CONDITIONS}
    assert len({r["scene_id"] for r in rows}) == 192
    buckets = defaultdict(list)
    for row in rows:
        assert row["condition_id"] in CONDITIONS
        buckets[row["scene_id"], CONDITIONS[row["condition_id"]]].append(row)
    scene_scores = []
    metrics = [method + "_" + outcome + "_correct"
               for method in ["raw", "pmi", "context"]
               for outcome in ["close", "far"]]
    for (scene_id, family), subset in buckets.items():
        expected = 4 if family == "distance_phrase" else 2
        assert len(subset) == expected
        assert Counter(r["target_order"] for r in subset) == {
            "person_first": expected//2, "object_first": expected//2}
        if family == "distance_phrase":
            assert Counter(r["context_entity_order"] for r in subset) == {
                "person_first": 2, "object_first": 2}
        scene_scores.append({"scene_id": scene_id, "evidence_family": family,
            "n_wording_rows": expected,
            **{metric: mean([r[metric] == "True" for r in subset]) for metric in metrics}})
    assert Counter(r["evidence_family"] for r in scene_scores) == {
        "distance_phrase": 192, "endpoint_placement": 192,
        "direct_label_control": 192}
    families = []
    for family in ["distance_phrase", "endpoint_placement", "direct_label_control"]:
        subset = [r for r in scene_scores if r["evidence_family"] == family]
        result = {"evidence_family": family, "n_scenes": len(subset),
            "n_probe_rows": sum(r["n_wording_rows"] for r in subset)}
        for method in ["raw", "pmi", "context"]:
            close = mean([r[method + "_close_correct"] for r in subset])
            far = mean([r[method + "_far_correct"] for r in subset])
            result.update({method + "_close_accuracy": close,
                method + "_far_accuracy": far,
                method + "_accuracy": (close + far)/2})
        families.append(result)
    applied = [r for r in families if r["evidence_family"] != "direct_label_control"]
    overall = {method + "_" + key: mean([r[method + "_" + key] for r in applied])
        for method in ["raw", "pmi", "context"]
        for key in ["accuracy", "close_accuracy", "far_accuracy"]}
    summary = {"name": "matched_scene_applied_close_far",
        "scope": "Two applied evidence families in the matched-scene probe only; other close/far probes excluded.",
        "scoring": "mean full-target token log likelihood; binary close/far choice",
        "weighting": "Within each scene, average balanced target and context orders; then average 192 scenes; then give distance phrase and endpoint placement equal weight.",
        "family_scores": families, "overall_applied_mean": overall,
        "direct_label_control_excluded": True}
    (out / "applied_score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Close/far matched-scene applied score", "",
        "Qwen3 359M at step 19,500. Each choice compares exactly two full close/far target sentences using mean target-token log likelihood. Raw choice is primary; PMI and fixed-target context sensitivity are secondary.",
        "The distance family averages its two context entity orders and two target orders within each scene. Endpoint placement averages its two target orders. The 192 scenes are averaged within each family, then the two applied families receive equal weight. Direct labels are a separate control.", "",
        "| Evidence family | Probe rows | Raw choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in families:
        report.append(f"| {row['evidence_family']} | {row['n_probe_rows']} | {row['raw_accuracy']:.2%} | {row['pmi_accuracy']:.2%} | {row['context_accuracy']:.2%} | {row['raw_close_accuracy']:.2%} | {row['raw_far_accuracy']:.2%} |")
    report += [f"| **Matched-scene applied mean** | 1,152 | **{overall['raw_accuracy']:.2%}** | {overall['pmi_accuracy']:.2%} | {overall['context_accuracy']:.2%} | {overall['raw_close_accuracy']:.2%} | {overall['raw_far_accuracy']:.2%} |",
        "", "The direct-label row is excluded from the applied mean. The separate physical-arrangement, synonym, and literal-definition probes are also excluded, so this is not yet a full close/far benchmark composite. The distance and endpoint families differ in evidence and difficulty, and the repeated settings, entities, and target orders are not independent scene samples.", ""]
    (out / "applied_score_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
