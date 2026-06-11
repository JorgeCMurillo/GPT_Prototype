import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.bos_aligned_proto.analysis.linear_probe.activations import (
    ActivationCache,
    save_activation_cache,
)
from research.bos_aligned_proto.analysis.linear_probe.data import (
    EWOKProbePair,
    build_ewok_probe_pairs,
    build_probe_pairs_from_dataframe,
)
from research.bos_aligned_proto.analysis.linear_probe.evaluation import compute_probe_lm_cases
from research.bos_aligned_proto.analysis.linear_probe.probes import (
    fit_validation_selected_probe,
    run_shuffle_controls,
)
from research.bos_aligned_proto.analysis.linear_probe.plot_ewok_linear_probe import plot_run
from research.bos_aligned_proto.analysis.linear_probe.splits import (
    SplitConfig,
    assign_grouped_splits,
    pair_split_labels,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Domain": "spatial-relations",
                "ConceptA": "cat",
                "ConceptB": "box",
                "Context1": "The cat is inside",
                "Context2": "The box contains",
                "Target1": "the box",
                "Target2": "the cat",
                "ContextType": "direct",
                "ContextDiff": "variable_swap",
                "TargetDiff": "variable_swap",
            }
        ]
    )


def _pairs_for_rows(n_rows: int) -> tuple[EWOKProbePair, ...]:
    frames = []
    for idx in range(n_rows):
        row = _toy_df().iloc[0].copy()
        row["ConceptA"] = f"cat{idx}"
        row["ConceptB"] = f"box{idx}"
        row["Context1"] = f"The cat{idx} is inside"
        row["Context2"] = f"The box{idx} contains"
        frames.append(row)
    return build_probe_pairs_from_dataframe(pd.DataFrame(frames), variant="fast")


def test_ewok_pair_construction_labels_roles_and_metadata() -> None:
    pairs = build_probe_pairs_from_dataframe(_toy_df(), variant="fast")

    assert [pair.role for pair in pairs] == ["c1t1", "c1t2", "c2t2", "c2t1"]
    assert [pair.label for pair in pairs] == [1, 0, 1, 0]
    assert len({pair.row_index for pair in pairs}) == 1
    assert pairs[0].domain == "spatial-relations"
    assert pairs[0].context_diff == "variable swap"
    assert pairs[0].text == "The cat is inside the box"


def test_grouped_split_is_deterministic_and_keeps_rows_together() -> None:
    pairs = _pairs_for_rows(20)
    config = SplitConfig(seed=123)
    first = assign_grouped_splits(pairs, config)
    second = assign_grouped_splits(pairs, config)

    assert first == second
    seen = {}
    for pair in pairs:
        split = first[pair.row_index]
        seen.setdefault(pair.row_index, split)
        assert seen[pair.row_index] == split
    assert set(first.values()) == {"train", "val", "test"}


def test_context_sensitivity_bucket_assignment_uses_strict_row_correctness() -> None:
    pairs = _pairs_for_rows(2)
    split_labels = np.asarray(["test"] * len(pairs), dtype=object)
    # Row 0: probe correct, LM wrong. Row 1: probe wrong, LM correct.
    probe_scores = np.asarray([2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    lm_scores = np.asarray([-1.0, 2.0, -1.0, 2.0, 2.0, -1.0, 2.0, -1.0])

    cases, bucket_counts, domain_counts = compute_probe_lm_cases(
        pairs,
        probe_scores=probe_scores,
        lm_scores=lm_scores,
        split_labels=split_labels,
    )

    assert [row["bucket"] for row in cases] == [
        "probe_correct_lm_wrong",
        "probe_wrong_lm_correct",
    ]
    assert bucket_counts["probe_correct_lm_wrong"] == 1
    assert bucket_counts["probe_wrong_lm_correct"] == 1
    assert sum(row["count"] for row in domain_counts) == 2


def test_validation_selection_uses_validation_not_test() -> None:
    pairs = _pairs_for_rows(9)
    row_splits = {idx: "train" for idx in range(3)}
    row_splits.update({idx: "val" for idx in range(3, 6)})
    row_splits.update({idx: "test" for idx in range(6, 9)})
    split_labels = pair_split_labels(pairs, row_splits)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)

    features = np.zeros((len(pairs), 2, 2), dtype=np.float32)
    features[:, 1, 0] = labels * 2.0 - 1.0
    cache = ActivationCache(
        features=features,
        layer_indices=(0, 1),
        pair_ids=tuple(pair.pair_id for pair in pairs),
        row_indices=np.asarray([pair.row_index for pair in pairs], dtype=np.int64),
        roles=tuple(pair.role for pair in pairs),
        labels=labels,
        lm_score_mean=np.zeros(len(pairs), dtype=np.float32),
        lm_score_sum=np.zeros(len(pairs), dtype=np.float32),
    )

    selected = fit_validation_selected_probe(
        cache=cache,
        pairs=pairs,
        split_labels=split_labels,
        c_grid=(1.0,),
        seed=0,
    )

    assert selected.layer_index == 1
    assert selected.test_metrics["row_strict_accuracy"] == 1.0
    assert all("test_pair_accuracy" not in row for row in selected.validation_table)


def test_shuffle_controls_preserve_true_validation_and_test_labels() -> None:
    pairs = _pairs_for_rows(12)
    row_splits = assign_grouped_splits(pairs, SplitConfig(seed=7))
    split_labels = pair_split_labels(pairs, row_splits)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    features = np.zeros((len(pairs), 1, 2), dtype=np.float32)
    features[:, 0, 0] = labels * 2.0 - 1.0
    cache = ActivationCache(
        features=features,
        layer_indices=(0,),
        pair_ids=tuple(pair.pair_id for pair in pairs),
        row_indices=np.asarray([pair.row_index for pair in pairs], dtype=np.int64),
        roles=tuple(pair.role for pair in pairs),
        labels=labels,
        lm_score_mean=np.zeros(len(pairs), dtype=np.float32),
        lm_score_sum=np.zeros(len(pairs), dtype=np.float32),
    )

    controls = run_shuffle_controls(
        cache=cache,
        pairs=pairs,
        split_labels=split_labels,
        c_grid=(1.0,),
        repeats=2,
        seed=99,
    )

    assert len(controls) == 2
    assert all("test_row_strict_accuracy" in row for row in controls)
    assert all(row["selected_layer_index"] == 0 for row in controls)


def test_cli_smoke_reuses_fake_activation_cache(tmp_path) -> None:
    pairs = build_ewok_probe_pairs(variant="fast", max_targets=20)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    features = np.zeros((len(pairs), 2, 3), dtype=np.float32)
    features[:, 1, 0] = labels * 2.0 - 1.0
    lm_scores = np.asarray([1.0 if pair.label else -1.0 for pair in pairs], dtype=np.float32)
    cache = ActivationCache(
        features=features.astype(np.float16),
        layer_indices=(0, 1),
        pair_ids=tuple(pair.pair_id for pair in pairs),
        row_indices=np.asarray([pair.row_index for pair in pairs], dtype=np.int64),
        roles=tuple(pair.role for pair in pairs),
        labels=labels,
        lm_score_mean=lm_scores,
        lm_score_sum=lm_scores,
    )
    save_activation_cache(tmp_path / "activation_cache.fp16.npz", cache)

    cmd = [
        sys.executable,
        "-m",
        "research.bos_aligned_proto.analysis.linear_probe.run_ewok_linear_probe",
        "--model",
        "not-loaded-because-cache-exists",
        "--output-dir",
        str(tmp_path),
        "--ewok-variant",
        "fast",
        "--max-targets",
        "20",
        "--shuffle-repeats",
        "0",
        "--plot-formats",
        "png",
    ]
    subprocess.run(cmd, cwd="/home/jorge/tokenPred/moonshotGPT", check=True)

    expected = [
        "items.jsonl",
        "split_assignments.csv",
        "layer_validation_table.csv",
        "selected_probe_summary.json",
        "test_metrics.json",
        "probe_vs_lm_bucket_counts.csv",
        "domain_bucket_counts.csv",
        "shuffle_controls.csv",
    ]
    for name in expected:
        assert (tmp_path / name).exists()
    metrics = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["score_view"] == "ewok_context_sensitivity"
    assert metrics["selected_layer_index"] == 1
    assert (tmp_path / "plots" / "layer_validation_curve.png").exists()
    assert (tmp_path / "plots" / "layer_domain_directional_avg_4x3.png").exists()
    assert (tmp_path / "plots" / "plot_manifest.json").exists()


def test_linear_probe_plot_smoke_from_cli_artifacts(tmp_path) -> None:
    pairs = build_ewok_probe_pairs(variant="fast", max_targets=20)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    features = np.zeros((len(pairs), 2, 3), dtype=np.float32)
    features[:, 1, 0] = labels * 2.0 - 1.0
    lm_scores = np.asarray([1.0 if pair.label else -1.0 for pair in pairs], dtype=np.float32)
    cache = ActivationCache(
        features=features.astype(np.float16),
        layer_indices=(0, 1),
        pair_ids=tuple(pair.pair_id for pair in pairs),
        row_indices=np.asarray([pair.row_index for pair in pairs], dtype=np.int64),
        roles=tuple(pair.role for pair in pairs),
        labels=labels,
        lm_score_mean=lm_scores,
        lm_score_sum=lm_scores,
    )
    save_activation_cache(tmp_path / "activation_cache.fp16.npz", cache)

    cmd = [
        sys.executable,
        "-m",
        "research.bos_aligned_proto.analysis.linear_probe.run_ewok_linear_probe",
        "--model",
        "not-loaded-because-cache-exists",
        "--output-dir",
        str(tmp_path),
        "--ewok-variant",
        "fast",
        "--max-targets",
        "20",
        "--shuffle-repeats",
        "0",
        "--no-plots",
    ]
    subprocess.run(cmd, cwd="/home/jorge/tokenPred/moonshotGPT", check=True)

    created = plot_run(tmp_path, formats=("png",), dpi=80)
    names = {path.name for path in created}
    assert {
        "layer_validation_curve.png",
        "probe_lm_test_summary.png",
        "probe_vs_lm_bucket_counts.png",
        "domain_bucket_heatmap.png",
        "probe_correct_lm_wrong_by_domain.png",
        "layer_domain_directional_avg_4x3.png",
    }.issubset(names)
    assert (tmp_path / "plots" / "layer_domain_scores.csv").exists()
    assert (tmp_path / "plots" / "plot_manifest.json").exists()
