#!/usr/bin/env python3
"""Answer-length/choice associations from the six-category saved-score manifest."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.spatial_benchmark import report as benchmark


class Inputs:
    def __init__(self):
        self.hashes, self.cache = {}, {}

    def read(self, path):
        path = Path(path)
        self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return benchmark.read_rows(path)

    def tokens(self, directory):
        path = directory / "token_scores.jsonl"
        if path not in self.cache:
            lookup = {}
            for row in self.read(path):
                values = row.get("token_log_probs", row.get("target_token_log_probs"))
                if not values or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Invalid saved token scores: {path}")
                key = row["context"], row["target"]
                value = (math.fsum(values), len(values))
                if key in lookup and lookup[key] != value:
                    raise ValueError("Inconsistent repeated token scores")
                lookup[key] = value
            self.cache[path] = lookup
        return self.cache[path]


def unique_index(rows, key):
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate score lookup keys")
    return indexed


def hierarchy_weights(rows):
    """Same category/family/case/evidence/wording-cell weights as the benchmark."""
    children, leaves = defaultdict(set), Counter()
    paths = []
    for row in rows:
        path = tuple(row[k] for k in ("category", "family", "case", "evidence", "cell"))
        paths.append(path)
        leaves[path] += 1
        for i, child in enumerate(path):
            children[path[:i]].add(child)
    weights = [1 / (leaves[p] * math.prod(len(children[p[:i]]) for i in range(len(p)))) for p in paths]
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights do not sum to one")
    return weights


def load_comparisons(manifest, root):
    normalized, sources, model = benchmark.load_manifest(manifest, root)
    weights = hierarchy_weights(normalized)
    by_category = defaultdict(list)
    for row, weight in zip(normalized, weights):
        by_category[row["category"]].append((row, weight))
    inputs, comparisons = Inputs(), []
    for spec in manifest["sources"]:
        path = root / spec["path"]
        raw = [row for row in inputs.read(path)
               if all(row.get(k) == v for k, v in spec.get("select", {}).items())
               and not (spec["adapter"] == "close_far" and row["condition_id"] == "direct_label")]
        if len(raw) != len(by_category[spec["category"]]):
            raise ValueError("Selection differs from benchmark")
        if spec["adapter"] == "closer_farther":
            parent = (root / spec["metadata_paths"][1]).parent
            ext = unique_index(inputs.read(path.parent / "binary_pair_scores.csv"), lambda r: (r["pair_id"], r["context_entity_order"]))
            items = unique_index(inputs.read(path.parent / "item_scores.jsonl"), lambda r: r["probe_id"])
            person = unique_index(inputs.read(parent / "pair_scores.csv"), lambda r: r["pair_id"])
        for row, (norm, weight) in zip(raw, by_category[spec["category"]]):
            contrast = row.get("contrast", spec["category"])
            directory = path.parent
            if spec["adapter"] != "closer_farther":
                contexts = [row["Context1"], row["Context2"]]
                targets = [row["Target1"], row["Target2"]]
                means = [[float(row[f"S{c}{t}"]) for t in (1, 2)] for c in (1, 2)]
                gold = [int(row.get(f"correct_target_for_context{c}", f"Target{c}")[-1]) for c in (1, 2)]
                pair_id = row["probe_id"]
            elif row["source"] == "event_extension_v1_2":
                pair = ext[row["pair_id"], row["context_entity_order"]]
                a, b = items[pair["first_probe_id"]], items[pair["second_probe_id"]]
                keys = [{"closer": "Target1", "farther": "Target2", "unchanged": "Target3"}[pair[k]]
                        for k in ("first_outcome", "second_outcome")]
                if any(a[k] != b[k] for k in keys):
                    raise ValueError("Unmatched targets")
                contexts, targets = [a["Context"], b["Context"]], [a[k] for k in keys]
                means = [[float(item[f"raw_{k}_mean"]) for k in keys] for item in (a, b)]
                gold, pair_id = [1, 2], row["pair_id"] + "__" + row["context_entity_order"]
            else:
                if row["source"] != "person_movement_v1_2":
                    raise ValueError("Unknown movement source")
                directory = parent
                pair = person[row["pair_id"]]
                contexts, targets = [pair["Context1"], pair["Context2"]], [pair["Target1"], pair["Target2"]]
                lookup = inputs.tokens(directory)
                means = [[lookup[c, t][0] / lookup[c, t][1] for t in targets] for c in contexts]
                for c, field in enumerate(("raw_a_margin", "raw_b_margin")):
                    reconstructed = (means[c][0] - means[c][1]) * (1 if c == 0 else -1)
                    if abs(reconstructed - float(pair[field])) > 2e-6:
                        raise ValueError("Parent mean reconstruction mismatch")
                gold, pair_id = [1, 2], row["pair_id"]
            correct = []
            for c, context in enumerate(contexts):
                # Only these two probes have saved CONDITIONAL token counts.
                # Standalone Target*_token_count fields elsewhere are not used.
                if "S11_target_token_count" in row:
                    lengths = [int(row[f"S{c+1}{t}_target_token_count"]) for t in (1, 2)]
                    sums = [m * n for m, n in zip(means[c], lengths)]
                    sum_source = "saved_mean_times_conditional_count"
                else:
                    lookup = inputs.tokens(directory)
                    values = [lookup[context, target] for target in targets]
                    sums, lengths = [v[0] for v in values], [v[1] for v in values]
                    if any(abs(s / n - m) > 2e-6 for s, n, m in zip(sums, lengths, means[c])):
                        raise ValueError("Token means differ from saved means")
                    sum_source = "saved_conditional_token_log_probs"
                if any(n <= 0 for n in lengths):
                    raise ValueError("Empty answer")
                record = {**norm, "weight": weight / 2, "pair_id": pair_id, "context_index": c + 1,
                          "contrast": contrast, "gold": gold[c], "target1": targets[0], "target2": targets[1],
                          "length1": lengths[0], "length2": lengths[1], "length_difference": lengths[0] - lengths[1],
                          "sum_source": sum_source,
                          "comparison_hash": hashlib.sha256(json.dumps([context, sorted(targets)]).encode()).hexdigest()}
                for method, scores in (("mean", means[c]), ("sum", sums)):
                    delta = scores[0] - scores[1]
                    record[method + "_choice"] = 0 if delta == 0 else 1 if delta > 0 else 2
                    record[method + "_correct"] = record[method + "_choice"] == gold[c]
                    record[method + "_margin"] = delta * (1 if gold[c] == 1 else -1)
                correct.append(record["mean_correct"])
                comparisons.append(record)
            if sum(correct) / 2 != norm["accuracy"]:
                raise ValueError("Reconstructed choices differ from benchmark")
    return comparisons, dict(model=model, sources=sources, auxiliary_sha256=inputs.hashes)


# Sufficient statistics allow cluster resampling without resampling 32k rows.
FIELDS = "total correct nontie x y xx xy unequal shorter gold_short correct_short gold_long correct_long gold_equal correct_equal ties".split()


def features(row, method):
    x, choice = row["length_difference"], row[method + "_choice"]
    live, correct = choice != 0, row[method + "_correct"]
    gold_gap = x if row["gold"] == 1 else -x
    short = (choice == 1 and x < 0) or (choice == 2 and x > 0)
    return np.array([1, correct, live, x if live else 0, choice == 1, x*x if live else 0,
                     x if choice == 1 else 0, live and x != 0, short,
                     gold_gap < 0, correct and gold_gap < 0, gold_gap > 0, correct and gold_gap > 0,
                     gold_gap == 0, correct and gold_gap == 0, not live], dtype=float)


def decode(stats):
    s = dict(zip(FIELDS, stats))
    def ratio(a, b):
        return float(s[a] / s[b]) if s[b] > 0 else None
    r, reason = None, "no non-tied comparisons"
    if s["nontie"] > 0:
        mx, my = s["x"] / s["nontie"], s["y"] / s["nontie"]
        vx, vy = s["xx"] / s["nontie"] - mx*mx, my*(1-my)
        if vx <= 1e-14:
            reason = "no variation in token-length difference"
        elif vy <= 1e-14:
            reason = "no variation in answer choice"
        else:
            r = float(np.clip((s["xy"] / s["nontie"] - mx*my) / math.sqrt(vx*vy), -1, 1))
            reason = None
    return {"point_biserial_r": r, "correlation_unavailable_reason": reason,
            "shorter_choice_rate": ratio("shorter", "unequal"),
            "accuracy_correct_shorter": ratio("correct_short", "gold_short"),
            "accuracy_correct_longer": ratio("correct_long", "gold_long"),
            "accuracy_equal_length": ratio("correct_equal", "gold_equal"),
            "accuracy": ratio("correct", "total"), "tie_rate": ratio("ties", "total")}


def describe(rows, method):
    matrix = np.array([features(r, method) for r in rows])
    weights = np.array([r["weight"] for r in rows])
    unequal = [r for r in rows if r["length_difference"] != 0]
    result = {"judgments": len(rows), "unique_comparisons": len({r["comparison_hash"] for r in rows}),
              "unequal_length_judgments": len(unequal),
              "length_differences": dict(sorted(Counter(r["length_difference"] for r in rows).items())),
              "raw": decode(matrix.sum(axis=0)), "balanced": decode((matrix * weights[:, None]).sum(axis=0))}
    if unequal:
        result["unequal_only_raw"] = decode(np.array([features(r, method) for r in unequal]).sum(axis=0))
    return result


def analyze(rows, replicates=10000, seed=42):
    if not rows or replicates < 100:
        raise ValueError("Need rows and at least 100 bootstrap replicates")
    categories = sorted({r["category"] for r in rows})
    scopes = {cat: [r for r in rows if r["category"] == cat] for cat in categories}
    scopes["overall"] = rows
    output = {name: {method: describe(subset, method) for method in ("mean", "sum")} for name, subset in scopes.items()}
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["category"], row["family"], row["case"]].append(row)
    strata = defaultdict(dict)
    for (cat, family, case), subset in by_cluster.items():
        strata[cat, family][case] = subset
    rng, shared = np.random.default_rng(seed), {}
    draws = {method: {cat: np.zeros((replicates, len(FIELDS))) for cat in categories} for method in ("mean", "sum")}
    for (cat, family), cases in sorted(strata.items()):
        ids = tuple(sorted(cases))
        group = cases[ids[0]][0]["group"]
        key = group, family
        if key not in shared:
            shared[key] = ids, rng.integers(len(ids), size=(replicates, len(ids)))
        shared_ids, indices = shared[key]
        if shared_ids != ids:
            raise ValueError("Unmatched shared-axis cluster sets")
        for method in ("mean", "sum"):
            values = np.array([sum((features(r, method) * r["weight"] for r in cases[case]), np.zeros(len(FIELDS))) for case in ids])
            draws[method][cat] += values[indices].sum(axis=1)
    for method in ("mean", "sum"):
        draws[method]["overall"] = sum(draws[method].values())
    length_metrics = {"point_biserial_r", "shorter_choice_rate", "accuracy_correct_shorter", "accuracy_correct_longer"}
    for name, subset in scopes.items():
        informative = {(r["category"], r["family"], r["case"]) for r in subset if r["length_difference"]}
        resampleable = any(len(strata[cat, family]) > 1 for cat, family, _ in informative)
        output[name]["length_informative_clusters"] = len(informative)
        output[name]["length_informative_clusters_resampleable"] = resampleable
        for method in ("mean", "sum"):
            decoded = [decode(s) for s in draws[method][name]]
            intervals = {}
            for metric, point in output[name][method]["balanced"].items():
                if metric.endswith("reason"):
                    continue
                if point is None:
                    intervals[metric] = {"ci95": None, "reason": "metric undefined"}
                elif metric in length_metrics and not resampleable:
                    intervals[metric] = {"ci95": None, "reason": "no resampleable length-informative scene strata"}
                else:
                    values = [d[metric] for d in decoded if d[metric] is not None]
                    intervals[metric] = {"ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
                                         "valid_replicates": len(values), "total_replicates": replicates}
            output[name][method]["balanced_cluster_intervals"] = intervals
    grouped = []
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["category"], row["contrast"], row["length_difference"]].append(row)
    for (cat, contrast, delta), subset in sorted(buckets.items()):
        record = dict(category=cat, contrast=contrast, length_difference=delta, judgments=len(subset),
                      conditional_token_lengths="; ".join(f"{a}/{b}" for a, b in sorted({(r['length1'], r['length2']) for r in subset})))
        for method in ("mean", "sum"):
            live = [r for r in subset if r[method + "_choice"]]
            record[method + "_answer1_rate"] = sum(r[method + "_choice"] == 1 for r in live) / len(live) if live else None
            record[method + "_balanced_answer1_rate"] = (sum(r["weight"] for r in live if r[method + "_choice"] == 1) / sum(r["weight"] for r in live)) if live else None
        grouped.append(record)
    return dict(scopes=output, by_length_gap=grouped, bootstrap_replicates=replicates, seed=seed,
                method="Point-biserial r = Pearson(delta conditional answer tokens, answer1 chosen); ties excluded from r and shorter-choice rate, counted incorrect for accuracy.",
                warnings=["Association is not causation; length is confounded with answer identity and contrast.",
                          "Raw r weights rendered judgments equally. Balanced r uses the benchmark hierarchy; it is weighted Pearson with a binary outcome.",
                          "Counts include repeated contexts and wording variants, not independent samples. No iid p-values are reported.",
                          "Bootstrap resamples scene cases within fixed families, sharing draws across matched axes. Singleton strata stay fixed.",
                          "No length-association CI is reported when all length-informative strata are singletons; a degenerate bootstrap would imply unjustified precision.",
                          "Scores describe evaluated left/right v1.0, not the shortened v1.2 wording. Front/behind has not been evaluated and is excluded."])


def render(result):
    def fmt(value, percent=False):
        return "N/A" if value is None else f"{value:.2%}" if percent else f"{value:+.4f}"
    lines = ["# Answer token length versus model choice", "", result["method"], "",
             "Delta length is answer 1 minus answer 2. Negative r means a larger relative length for answer 1 is associated with less frequent selection of answer 1. This is NOT the same statistic as the fraction choosing the shorter answer. Counts are rendered binary judgments. The primary table is an ordinary, unweighted point-biserial correlation; balanced estimates follow separately.", "",
             "| Category | Judgments | Unequal lengths | Mean r | Sum r | Mean: shorter chosen | Sum: shorter chosen |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cat, info in result["scopes"].items():
        a, b = info["mean"], info["sum"]
        lines.append(f"| {cat} | {a['judgments']:,} | {a['unequal_length_judgments']:,} | {fmt(a['raw']['point_biserial_r'])} | {fmt(b['raw']['point_biserial_r'])} | {fmt(a['raw']['shorter_choice_rate'], True)} | {fmt(b['raw']['shorter_choice_rate'], True)} |")
    lines += ["", "## Benchmark-balanced diagnostics", "",
              "Equal category/family/case/evidence/wording-cell weighting, conditional on each column's eligible comparisons. Shorter/longer accuracy includes ties as errors. These are not unweighted count percentages.", "",
              "| Scoring | Weighted r | Shorter chosen | Correct shorter: accuracy | Correct longer: accuracy | Equal length: accuracy |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in ("mean", "sum"):
        stats = result["scopes"]["overall"][method]["balanced"]
        lines.append(f"| {method} | {fmt(stats['point_biserial_r'])} | {fmt(stats['shorter_choice_rate'], True)} | {fmt(stats['accuracy_correct_shorter'], True)} | {fmt(stats['accuracy_correct_longer'], True)} | {fmt(stats['accuracy_equal_length'], True)} |")
    lines += ["", "## Length gaps and answer identity", "",
              "| Category | Contrast | Tokens: answer 1/2 | Token difference | Judgments | Mean: answer 1 chosen | Sum: answer 1 chosen |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in result["by_length_gap"]:
        lines.append(f"| {row['category']} | {row['contrast']} | {row['conditional_token_lengths']} | {row['length_difference']:+d} | {row['judgments']:,} | {fmt(row['mean_answer1_rate'], True)} | {fmt(row['sum_answer1_rate'], True)} |")
    lines += ["", "Equal-length categories have undefined length correlation, not zero correlation. A correlation can also be undefined if the model always chooses the same candidate.", "",
              "For this saved dataset, unequal lengths occur only in closer-versus-unchanged and farther-versus-unchanged contrasts. The unchanged answer is three tokens longer. Within these contrasts the gap is constant, so the effect of token length cannot be separated from the meaning/wording of the answer. No causal length effect can be estimated here.", "",
              "Why summed scoring can choose the shorter answer frequently yet have near-zero r within closer/farther: answer 1 is also chosen frequently in the equal-length closer-versus-farther contrast. The correlation compares these contrast types, whereas the shorter-choice rate considers only unequal lengths. Pooled cross-category r adds further category-composition effects.", "",
              "## Uncertainty and scope", ""]
    lines += ["- " + w for w in result["warnings"]]
    lines += ["", f"Conditional scene bootstrap: {result['bootstrap_replicates']:,} replicates, seed {result['seed']}. Detailed metric availability and eligible cluster counts are in summary.json. The current length-informative strata are all singletons, so their length-association CIs are not estimable.", "",
              "Lengths are the conditional target tokens actually scored—not whitespace words, standalone-tokenizer counts, or context-plus-answer length. Sums use saved token log probabilities where available, otherwise saved mean times conditional token count (subject to float rounding). No model inference was rerun.", "",
              "![Choice rates by length gap](choice_by_length_gap.png)", ""]
    return "\n".join(lines)


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [r for r in result["by_length_gap"] if r["category"] == "closer_farther"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(values))
    for offset, method in ((-0.18, "mean"), (0.18, "sum")):
        ax.bar(x + offset, [100*r[method + "_answer1_rate"] for r in values], width=.36, label=method)
    ax.set_xticks(x, [r["contrast"].replace("_vs_", " vs\n") + f"\nΔ tokens = {r['length_difference']:+d}" for r in values])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Answer 1 chosen (% of non-tied judgments)")
    ax.set_title("Closer/farther: answer length and choice\nLength and answer meaning are confounded")
    ax.legend(title="Scoring")
    fig.tight_layout()
    fig.savefig(out / "choice_by_length_gap.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("qwen3_step19500.json"))
    parser.add_argument("--root", type=Path, default=benchmark.ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, provenance = load_comparisons(json.loads(args.manifest.read_text()), args.root)
    result = analyze(rows, args.bootstrap, args.seed)
    result.update(provenance=provenance, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "comparisons.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    with (args.output_dir / "by_length_gap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["by_length_gap"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["by_length_gap"])
    plot(result, args.output_dir)
    report = render(result)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
