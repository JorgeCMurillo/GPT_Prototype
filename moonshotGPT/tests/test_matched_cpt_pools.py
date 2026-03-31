import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.bos_aligned_proto.analysis.attribution.build_matched_cpt_pools import (
    build_matched_cpt_pools,
    load_candidate_score_frame,
    match_control_candidates,
)


def _write_u16(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype=np.uint16).tofile(path)


def _make_row_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "row_data"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"format": "bos_row_packed_bestfit", "row_tokens": 4, "seq_len": 3}),
        encoding="utf-8",
    )
    _write_u16(data_dir / "train_000000.bin", [10, 11, 12, 13, 20, 21, 22, 23])
    _write_u16(data_dir / "train_000001.bin", [30, 31, 32, 33, 40, 41, 42, 43])
    return data_dir


def _make_stream_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "stream_data"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"format": "token_stream"}),
        encoding="utf-8",
    )
    _write_u16(data_dir / "train_000000.bin", [10, 11, 12, 13, 14, 15, 16, 17])
    _write_u16(data_dir / "train_000001.bin", [30, 31, 32, 33, 34, 35, 36, 37])
    return data_dir


def _make_attribution_output(tmp_path: Path, data_dir: Path) -> Path:
    out_dir = tmp_path / "attrib"
    out_dir.mkdir()
    row_summary = pd.DataFrame(
        [
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 0,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 0,
                "local_row_idx": 0,
                "mean_score": -0.10,
                "mean_abs_score": 0.10,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.20,
                "max_abs_score": 0.20,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 1,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 1,
                "token_offset_start": 4,
                "token_offset_end": 8,
                "row_id": 1,
                "local_row_idx": 1,
                "mean_score": 0.35,
                "mean_abs_score": 0.35,
                "positive_score_sum": 0.70,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.40,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 2,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 2,
                "local_row_idx": 0,
                "mean_score": 0.05,
                "mean_abs_score": 0.05,
                "positive_score_sum": 0.10,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.10,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 3,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 1,
                "token_offset_start": 4,
                "token_offset_end": 8,
                "row_id": 3,
                "local_row_idx": 1,
                "mean_score": -0.20,
                "mean_abs_score": 0.20,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.40,
                "max_abs_score": 0.25,
                "target_count": 2,
            },
        ]
    )
    row_summary.to_csv(out_dir / "row_summary_step00001000.csv", index=False)
    np.save(
        out_dir / "dense_scores_step00001000.npy",
        np.asarray(
            [
                [-0.10, 0.80, 0.20, -0.30],
                [0.00, -0.10, 0.05, 0.40],
            ],
            dtype=np.float64,
        ),
    )
    with (out_dir / "target_items.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"target_id": "q-alpha"}) + "\n")
        handle.write(json.dumps({"target_id": "q-beta"}) + "\n")
    return out_dir


def _make_stream_attribution_output(tmp_path: Path, data_dir: Path) -> Path:
    out_dir = tmp_path / "attrib_stream"
    out_dir.mkdir()
    row_summary = pd.DataFrame(
        [
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 0,
                "candidate_kind": "stream_window",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 0,
                "local_row_idx": 0,
                "mean_score": -0.10,
                "mean_abs_score": 0.10,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.20,
                "max_abs_score": 0.20,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 1,
                "candidate_kind": "stream_window",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 1,
                "token_offset_start": 3,
                "token_offset_end": 7,
                "row_id": 1,
                "local_row_idx": 1,
                "mean_score": 0.35,
                "mean_abs_score": 0.35,
                "positive_score_sum": 0.70,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.40,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 2,
                "candidate_kind": "stream_window",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 2,
                "local_row_idx": 0,
                "mean_score": 0.05,
                "mean_abs_score": 0.05,
                "positive_score_sum": 0.10,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.10,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 3,
                "candidate_kind": "stream_window",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 1,
                "token_offset_start": 3,
                "token_offset_end": 7,
                "row_id": 3,
                "local_row_idx": 1,
                "mean_score": -0.20,
                "mean_abs_score": 0.20,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.40,
                "max_abs_score": 0.25,
                "target_count": 2,
            },
        ]
    )
    row_summary.to_csv(out_dir / "row_summary_step00001000.csv", index=False)
    return out_dir


def test_load_candidate_score_frame_supports_positive_pooled_net_pooled_and_per_query(tmp_path: Path) -> None:
    data_dir = _make_row_data(tmp_path)
    attrib_dir = _make_attribution_output(tmp_path, data_dir)

    pooled = load_candidate_score_frame(
        attribution_dir=attrib_dir,
        step=1000,
        score_mode="positive_pooled",
    )
    net_pooled = load_candidate_score_frame(
        attribution_dir=attrib_dir,
        step=1000,
        score_mode="net_pooled",
    )
    per_query = load_candidate_score_frame(
        attribution_dir=attrib_dir,
        step=1000,
        score_mode="per_query",
        target_id="q-alpha",
    )

    assert pooled["selection_score"].tolist() == [0.0, 0.7, 0.1, 0.0]
    assert net_pooled["selection_score"].tolist() == pytest.approx([-0.1, 0.7, 0.25, 0.1])
    assert net_pooled.attrs["selection_source"] == {"kind": "dense_scores", "pooling": "sum_signed"}
    assert per_query["selection_score"].tolist() == [-0.1, 0.8, 0.2, -0.3]
    assert per_query["selection_target_id"].tolist() == ["q-alpha"] * 4


def test_match_control_candidates_prefers_same_shard_and_low_scores(tmp_path: Path) -> None:
    data_dir = _make_row_data(tmp_path)
    attrib_dir = _make_attribution_output(tmp_path, data_dir)
    scored = load_candidate_score_frame(
        attribution_dir=attrib_dir,
        step=1000,
        score_mode="positive_pooled",
    )
    treated = scored.sort_values("selection_score", ascending=False).head(2).reset_index(drop=True)

    pairings, control = match_control_candidates(
        treated=treated,
        candidates=scored,
        max_control_score=0.0,
        allow_relaxed_shard_match=True,
    )

    assert pairings["treated_candidate_id"].tolist() == [1, 2]
    assert pairings["control_candidate_id"].tolist() == [0, 3]
    assert pairings["match_level"].tolist() == [
        "exact_shard_below_threshold",
        "exact_shard_below_threshold",
    ]
    assert control["candidate_id"].tolist() == [0, 3]


def test_build_matched_cpt_pools_materializes_trainable_bos_row_sets(tmp_path: Path) -> None:
    data_dir = _make_row_data(tmp_path)
    attrib_dir = _make_attribution_output(tmp_path, data_dir)
    out_dir = tmp_path / "matched"

    artifacts = build_matched_cpt_pools(
        attribution_dir=attrib_dir,
        data_dir=data_dir,
        step=1000,
        output_dir=out_dir,
        score_mode="positive_pooled",
        num_treated=2,
        max_control_score=0.0,
        allow_relaxed_shard_match=True,
        rows_per_shard=8,
    )

    treated_meta = json.loads((artifacts["treated_data_dir"] / "meta.json").read_text(encoding="utf-8"))
    control_meta = json.loads((artifacts["control_data_dir"] / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    treated_tokens = np.memmap(
        artifacts["treated_data_dir"] / "train_000000.bin",
        dtype=np.uint16,
        mode="r",
    )
    control_tokens = np.memmap(
        artifacts["control_data_dir"] / "train_000000.bin",
        dtype=np.uint16,
        mode="r",
    )

    assert treated_meta["row_tokens"] == 4
    assert control_meta["row_tokens"] == 4
    assert treated_meta["num_rows"] == 2
    assert control_meta["num_rows"] == 2
    assert summary["selection_source"] == {"kind": "row_summary", "column": "positive_score_sum"}
    assert summary["balance_report"]["token_count_balance_exact"] is True
    assert summary["balance_report"]["same_shard_pair_fraction"] == 1.0
    assert treated_tokens.tolist() == [20, 21, 22, 23, 30, 31, 32, 33]
    assert control_tokens.tolist() == [10, 11, 12, 13, 40, 41, 42, 43]


def test_build_matched_cpt_pools_records_net_pooled_selection_source(tmp_path: Path) -> None:
    data_dir = _make_row_data(tmp_path)
    attrib_dir = _make_attribution_output(tmp_path, data_dir)
    out_dir = tmp_path / "matched_net"

    artifacts = build_matched_cpt_pools(
        attribution_dir=attrib_dir,
        data_dir=data_dir,
        step=1000,
        output_dir=out_dir,
        score_mode="net_pooled",
        num_treated=2,
        max_control_score=0.0,
        allow_relaxed_shard_match=True,
        rows_per_shard=8,
    )

    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))

    assert summary["selection_source"] == {"kind": "dense_scores", "pooling": "sum_signed"}
    assert summary["score_mode"] == "net_pooled"


def test_build_matched_cpt_pools_supports_stream_window_candidates(tmp_path: Path) -> None:
    data_dir = _make_stream_data(tmp_path)
    attrib_dir = _make_stream_attribution_output(tmp_path, data_dir)
    out_dir = tmp_path / "matched_stream"

    artifacts = build_matched_cpt_pools(
        attribution_dir=attrib_dir,
        data_dir=data_dir,
        step=1000,
        output_dir=out_dir,
        score_mode="positive_pooled",
        num_treated=2,
        max_control_score=0.0,
        allow_relaxed_shard_match=True,
        rows_per_shard=8,
    )

    treated_meta = json.loads((artifacts["treated_data_dir"] / "meta.json").read_text(encoding="utf-8"))
    control_meta = json.loads((artifacts["control_data_dir"] / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    treated_tokens = np.memmap(
        artifacts["treated_data_dir"] / "train_000000.bin",
        dtype=np.uint16,
        mode="r",
    )
    control_tokens = np.memmap(
        artifacts["control_data_dir"] / "train_000000.bin",
        dtype=np.uint16,
        mode="r",
    )

    assert treated_meta["format"] == "exact_window_row_packed"
    assert treated_meta["candidate_kind"] == "stream_window"
    assert control_meta["candidate_kind"] == "stream_window"
    assert treated_meta["row_tokens"] == 4
    assert treated_meta["seq_len"] == 3
    assert summary["source_candidate_kind"] == "stream_window"
    assert summary["materialized_format"] == "exact_window_row_packed"
    assert treated_tokens.tolist() == [13, 14, 15, 16, 30, 31, 32, 33]
    assert control_tokens.tolist() == [10, 11, 12, 13, 33, 34, 35, 36]
