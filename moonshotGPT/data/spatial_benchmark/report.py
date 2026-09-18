#!/usr/bin/env python3
"""Balanced spatial-situation scores and conditional scene-cluster intervals."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "both_correct")


def read_rows(path):
    with path.open() as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return list(csv.DictReader(stream))


def probability(value):
    if value in ("True", "False"):
        return float(value == "True")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"Invalid score: {value!r}")
    return result


def normalize(rows, spec):
    """Adapters use raw binary scores only; never PMI or three-way accuracy."""
    result = []
    seen = set()
    for row in rows:
        if any(row.get(k) != v for k, v in spec.get("select", {}).items()):
            continue
        adapter = spec["adapter"]
        if adapter == "directional":
            family = row["event_family"]
            case = row["case_id"]
            case = spec.get("case_aliases", {}).get(case, case)
            evidence = row[spec["evidence_field"]]
            accuracy = probability(row[spec["accuracy_field"]])
            both = probability(row["both_correct"])
            identity = row["probe_id"]
        elif adapter == "close_far":
            condition = row["condition_id"]
            if condition == "direct_label":
                continue
            mapping = {"distance_phrase": "distance_phrase",
                       "distance_phrase_object_first": "distance_phrase",
                       "endpoint_placement": "endpoint_placement"}
            evidence = mapping[condition]
            family, case = "static_distance", row["setting_id"]
            a, b = (probability(row[f"raw_{side}_correct"])
                    for side in ("close", "far"))
            accuracy, both = (a + b) / 2, a * b
            identity = row["probe_id"]
        elif adapter == "closer_farther":
            family = row["family"]
            # Numbers, entities, units and wording reuse the same event structure.
            # Keep the two cross-family contrasts together: they share contexts.
            case = family
            evidence = row["mode"]
            if evidence not in ("absent", "specific"):
                raise ValueError(f"Unexpected movement evidence: {evidence}")
            accuracy = probability(row["raw_accuracy"])
            both = probability(row["raw_both_correct"])
            identity = (row["source"], row["contrast"],
                        row["context_entity_order"], row["pair_id"])
        else:
            raise ValueError(f"Unknown adapter: {adapter}")
        if identity in seen:
            raise ValueError(f"Duplicate selected pair in {spec['category']}: {identity}")
        seen.add(identity)
        if both > accuracy:
            raise ValueError("Both-correct score exceeds accuracy")
        cell = tuple(row[k] for k in spec.get("balance_fields", []))
        result.append({"category": spec["category"], "group": spec["cluster_group"],
                       "family": family, "case": case, "evidence": evidence,
                       "cell": cell, "accuracy": accuracy, "both_correct": both})
    if not result:
        raise ValueError(f"Empty selection: {spec['category']}")
    expected = spec.get("expected_pairs")
    if expected is not None and len(result) != expected:
        raise ValueError(f"{spec['category']}: expected {expected} pairs, got {len(result)}")
    return result


def collapse(rows):
    """Variant cells -> evidence -> case; leave cases/families for bootstrap."""
    cells = defaultdict(list)
    groups = {}
    for row in rows:
        cat = row["category"]
        if cat in groups and groups[cat] != row["group"]:
            raise ValueError(f"Multiple cluster groups for {cat}")
        groups[cat] = row["group"]
        key = tuple(row[k] for k in ("category", "family", "case", "evidence", "cell"))
        cells[key].append([row[m] for m in METRICS])
    evidence = defaultdict(list)
    for key, values in cells.items():
        evidence[key[:-1]].append(np.mean(values, axis=0))
    cases = defaultdict(dict)
    for key, values in evidence.items():
        cases[key[:-1]][key[-1]] = np.mean(values, axis=0)
    families = defaultdict(dict)
    signatures = {}
    for (cat, family, case), formats in cases.items():
        signature = frozenset(formats)
        if (cat, family) in signatures and signatures[cat, family] != signature:
            raise ValueError(f"Incomplete evidence crossing: {cat}/{family}/{case}")
        signatures[cat, family] = signature
        families[cat, family][case] = np.mean(list(formats.values()), axis=0)
    return families, groups


def interval(point, draws):
    result = {}
    for i, metric in enumerate(METRICS):
        lo, hi = np.quantile(draws[:, i], [0.025, 0.975])
        result[metric] = {"estimate": float(point[i]), "ci95": [float(lo), float(hi)],
                          "degenerate": bool(np.isclose(lo, hi, atol=1e-14, rtol=0))}
    return result


def summarize(rows, n_bootstrap=20000, seed=42):
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if not rows:
        raise ValueError("No rows")
    families, groups = collapse(rows)
    rng = np.random.default_rng(seed)
    # Matched axes use identical draws. Different source groups are independent.
    shared_draws = {}
    category_points, category_draws, details = defaultdict(list), defaultdict(list), {}
    for (cat, family), cases in sorted(families.items()):
        case_ids = tuple(sorted(cases))
        cluster_key = (groups[cat], family)
        if cluster_key not in shared_draws:
            indices = rng.integers(len(case_ids), size=(n_bootstrap, len(case_ids)))
            shared_draws[cluster_key] = (case_ids, indices)
        other_ids, indices = shared_draws[cluster_key]
        if case_ids != other_ids:
            raise ValueError(f"Unmatched case sets in shared cluster group {cluster_key}")
        values = np.array([cases[c] for c in case_ids])
        point, draws = values.mean(axis=0), values[indices].mean(axis=1)
        category_points[cat].append(point)
        category_draws[cat].append(draws)
        details.setdefault(cat, {})[family] = {
            "cases": list(case_ids), "n_clusters": len(case_ids),
            "singleton_stratum": len(case_ids) == 1, **interval(point, draws)}
    categories = {}
    points, draws_list = [], []
    for cat in sorted(category_points):
        point = np.mean(category_points[cat], axis=0)
        draws = np.mean(category_draws[cat], axis=0)
        points.append(point)
        draws_list.append(draws)
        n_pairs = sum(r["category"] == cat for r in rows)
        categories[cat] = {"n_pairs": n_pairs, "n_binary_judgments": 2 * n_pairs,
                           "cluster_group": groups[cat],
                           "n_clusters": sum(f["n_clusters"] for f in details[cat].values()),
                           "all_strata_singleton": all(f["singleton_stratum"] for f in details[cat].values()),
                           "families": details[cat], **interval(point, draws)}
    warnings = [
        "Intervals are conditional percentile scene-cluster bootstrap intervals, not binomial/Wilson intervals.",
        "These hand-built cases are not a random sample of spatial reasoning. Intervals do not cover new templates, domains, model seeds, or training uncertainty.",
        "A zero-width interval means no observed cluster-score variation, not certainty about generalization.",
        "Binary judgments include reused contexts in distance contrasts; counts are not independent sample sizes."]
    for cat, family_details in details.items():
        singleton = [f for f, v in family_details.items() if v["singleton_stratum"]]
        if singleton:
            warnings.append(f"{cat}: singleton strata {', '.join(singleton)} are fixed in resampling; their scene uncertainty cannot be estimated.")
    return {"method": "equal-category/family/evidence weighting; stratified paired scene-cluster bootstrap",
            "bootstrap_replicates": n_bootstrap, "seed": seed,
            "confidence_level": 0.95, "chance_accuracy": 0.5,
            "categories": categories,
            "overall": interval(np.mean(points, axis=0), np.mean(draws_list, axis=0)),
            "n_unique_cluster_units": sum(len(ids) for ids, _ in shared_draws.values()),
            "warnings": warnings}


def load_manifest(manifest, root, selected=None):
    rows, provenance, models = [], [], set()
    specifications = manifest["sources"]
    names = [s["category"] for s in specifications]
    if len(set(names)) != len(names):
        raise ValueError("Each category must have exactly one source specification")
    if selected and set(selected) - set(names):
        raise ValueError(f"Unknown categories: {set(selected) - set(names)}")
    for spec in specifications:
        if selected and spec["category"] not in selected:
            continue
        path = root / spec["path"]
        metadata_paths = [root / p for p in spec["metadata_paths"]]
        for metadata_path in metadata_paths:
            meta = json.loads(metadata_path.read_text())
            models.add(meta["model"])
            reduction = meta.get("score_reduction")
            # Cardinal evaluator records its score contract as prose.
            if reduction != "mean" and not (
                reduction is None and meta.get("score", "").startswith("raw mean full-target")):
                raise ValueError(f"Not verified mean-token scoring: {metadata_path}")
        raw_rows = read_rows(path)
        normalized = normalize(raw_rows, spec)
        rows.extend(normalized)
        provenance.append({**spec, "total_source_rows": len(raw_rows),
                           "included_pairs": len(normalized),
                           "excluded_rows": len(raw_rows) - len(normalized),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "metadata_sha256": {str(p.relative_to(root)):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata_paths}})
    if len(models) != 1:
        raise ValueError(f"Expected one checkpoint; found {sorted(models)}")
    return rows, provenance, models.pop()


def render_report(result):
    def display(metric):
        lo, hi = metric["ci95"]
        return f"{metric['estimate']:.2%} | {lo:.2%}–{hi:.2%}"
    lines = ["# Custom spatial situation benchmark", "",
             "Raw mean full-target token likelihood; no PMI. Equal weight per included relation category, then event family; evidence formats and declared wording cells are balanced within scene cases. Definitions, direct-label controls and observer-turn diagnostics are excluded.", "",
             "| Relation category | Paired rows | Scene clusters | Accuracy | Conditional 95% interval | Both contexts correct |",
             "|---|---:|---:|---:|---:|---:|"]
    for cat, info in result["categories"].items():
        score = (f"{info['accuracy']['estimate']:.2%} | Not estimable (singleton strata)"
                 if info["all_strata_singleton"] else display(info["accuracy"]))
        lines.append(f"| {cat.replace('_', '/')} | {info['n_pairs']:,} | {info['n_clusters']} | {score} | {info['both_correct']['estimate']:.2%} |")
    overall = result["overall"]
    lines += [f"| **Balanced overall** | — | — | {display(overall['accuracy'])} | {overall['both_correct']['estimate']:.2%} |", "",
              f"Bootstrap: {result['bootstrap_replicates']:,} replicates, seed {result['seed']}; {result['n_unique_cluster_units']} unique cluster units across shared-axis groups. These are not independent template families.", "",
              "All variants of a selected case travel together. Matched above/below–left/right cases share draws; north/south–east/west cases share draws. Event-family weights remain fixed.", "",
              "## Interpretation limits", ""]
    lines += [f"- {warning}" for warning in result["warnings"]]
    lines += ["", "## Included sources", ""]
    for source in result["sources"]:
        lines.append(f"- `{source['category']}`: `{source['path']}`. {source.get('note', '')}")
    lines += ["", "The JSON report records checkpoint, source hashes, category/family intervals, and the exact scope. No model inference was rerun.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("qwen3_step19500.json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, provenance, model = load_manifest(manifest, args.root, args.categories)
    result = summarize(rows, args.bootstrap, args.seed)
    result.update(model=model, sources=provenance, scope=manifest["name"],
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    for source in provenance:
        if source.get("uncertainty_note"):
            result["warnings"].append(source["uncertainty_note"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = render_report(result)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
