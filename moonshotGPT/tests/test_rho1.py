import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from training_utils.rho1 import (
    Rho1Config,
    Rho1GapOptStepAccumulator,
    Rho1OptStepAccumulator,
    Rho1RefLossLoader,
    compute_rho1_batch_result,
    compute_rho1_loss_from_reference,
    validate_rho_ref_loss_alignment,
)


class _FakeRefLossLoader:
    def __init__(self, ref_values, valid_mask) -> None:
        self.ref_values = np.asarray(ref_values, dtype=np.float32)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        self.calls = []

    def load_batch(self, shard_meta: dict, expected_bt: int):
        self.calls.append((dict(shard_meta), int(expected_bt)))
        assert expected_bt == int(self.ref_values.size)
        return self.ref_values.copy(), self.valid_mask.copy()


def _write_alignment_fixture(tmp_path):
    data_dir = tmp_path / "data"
    ref_dir = tmp_path / "ref"
    data_dir.mkdir()
    ref_dir.mkdir()

    shard_path = data_dir / "train_000000.bin"
    shard_tokens = np.asarray([10, 11, 12, 13, 14], dtype=np.uint16)
    shard_tokens.tofile(shard_path)

    ref_path = ref_dir / "train_000000.ref_loss.f32.bin"
    ref_values = np.asarray([0.0, 0.25, np.nan, 0.5, 0.75], dtype=np.float32)
    ref_values.tofile(ref_path)

    meta_path = ref_dir / "train_000000.ref_loss.f32.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_shard_basename": shard_path.name,
                "source_num_tokens": int(shard_tokens.size),
                "seq_len": 2,
                "batch_size": 2,
                "stride_tokens": 4,
                "block_tokens": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_dir, ref_dir, shard_path


def test_rho1_config_validation_and_suffix(tmp_path) -> None:
    config = Rho1Config(
        ref_loss_dir=str(tmp_path),
        keep_frac=0.25,
        warmup_steps=12,
        mode="student_only",
        ref_loss_cap=3.0,
    )
    config.validate()
    assert config.enabled is True
    assert config.run_name_suffix() == "_rhostudentfocus_k0250_wu12"


def test_rho1_config_sequence_suffix(tmp_path) -> None:
    config = Rho1Config(
        ref_loss_dir=str(tmp_path),
        keep_frac=0.5,
        mode="delta",
        granularity="sequence",
    )
    config.validate()
    assert config.run_name_suffix() == "_rhodeltaseq_k0500_wu0"


def test_rho1_config_rejects_invalid_keep_frac(tmp_path) -> None:
    config = Rho1Config(ref_loss_dir=str(tmp_path), keep_frac=0.0)
    with pytest.raises(ValueError, match="rho_keep_frac"):
        config.validate()


def test_validate_rho_ref_loss_alignment_and_loader(tmp_path) -> None:
    data_dir, ref_dir, shard_path = _write_alignment_fixture(tmp_path)

    checked = validate_rho_ref_loss_alignment(
        data_dir=str(data_dir),
        ref_loss_dir=str(ref_dir),
        seq_len=2,
        micro_batch_size=2,
    )
    assert checked == 1

    loader = Rho1RefLossLoader(str(ref_dir))
    ref_arr, ref_valid = loader.load_batch(
        shard_meta={"shard_path": str(shard_path), "start": 0, "end": 5},
        expected_bt=4,
    )
    assert ref_arr.tolist() == [0.25, 1.0e9, 0.5, 0.75]
    assert ref_valid.tolist() == [True, False, True, True]


def test_compute_rho1_loss_from_reference_delta_topk() -> None:
    token_loss = torch.tensor([[4.0, 1.0, 3.0, 2.0]], dtype=torch.float32)
    ref_loss = torch.tensor([[1.0, 0.2, 2.0, 0.5]], dtype=torch.float32)
    ref_valid_mask = torch.ones_like(token_loss, dtype=torch.bool)
    config = Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, mode="delta")

    result = compute_rho1_loss_from_reference(
        token_loss=token_loss,
        ref_loss=ref_loss,
        ref_valid_mask=ref_valid_mask,
        config=config,
    )

    assert torch.isclose(result.loss, torch.tensor(3.0))
    assert result.kept_tokens == 2
    assert result.keep_frac == 0.5
    assert result.kept_sequences is None
    assert result.keep_seq_frac is None
    assert pytest.approx(result.ref_loss_mean) == 0.925


def test_compute_rho1_loss_from_reference_sequence_topk() -> None:
    token_loss = torch.tensor([[4.0, 1.0], [3.0, 2.0]], dtype=torch.float32)
    ref_loss = torch.tensor([[1.0, 0.2], [2.0, 0.5]], dtype=torch.float32)
    ref_valid_mask = torch.ones_like(token_loss, dtype=torch.bool)
    config = Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, mode="delta", granularity="sequence")

    result = compute_rho1_loss_from_reference(
        token_loss=token_loss,
        ref_loss=ref_loss,
        ref_valid_mask=ref_valid_mask,
        config=config,
    )

    assert torch.isclose(result.loss, torch.tensor(2.5))
    assert result.kept_tokens == 2
    assert result.keep_frac == 0.5
    assert result.kept_sequences == 1
    assert result.keep_seq_frac == 0.5
    assert pytest.approx(result.ref_loss_mean) == 0.925


def test_rho1_gap_accumulator_collects_all_and_kept_stats() -> None:
    gap = Rho1GapOptStepAccumulator()
    gap.update(
        token_loss=torch.tensor([[4.0, 1.0, 3.0, 2.0]], dtype=torch.float32),
        ref_loss=torch.tensor([[1.0, 0.2, 2.0, 0.5]], dtype=torch.float32),
        ref_valid_mask=torch.tensor([[True, True, True, True]]),
        keep_mask=torch.tensor([[True, False, True, False]]),
    )

    summary = gap.finalize()
    assert summary.delta_mean_all == pytest.approx(1.575)
    assert summary.delta_median_all == pytest.approx(1.25)
    assert summary.delta_p90_all == pytest.approx(2.55)
    assert summary.delta_pos_frac_all == pytest.approx(1.0)
    assert summary.delta_pos_mean_all == pytest.approx(1.575)
    assert summary.delta_mean_kept == pytest.approx(2.0)
    assert summary.delta_pos_frac_kept == pytest.approx(1.0)
    assert summary.delta_mean_seq_all == pytest.approx(1.575)
    assert summary.delta_median_seq_all == pytest.approx(1.575)
    assert summary.delta_p90_seq_all == pytest.approx(1.575)
    assert summary.delta_pos_frac_seq_all == pytest.approx(1.0)
    assert summary.delta_pos_mean_seq_all == pytest.approx(1.575)
    assert summary.delta_mean_seq_kept == pytest.approx(2.0)
    assert summary.delta_pos_frac_seq_kept == pytest.approx(1.0)


def test_compute_rho1_batch_result_warmup_skips_loader() -> None:
    token_loss = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)
    loader = _FakeRefLossLoader([1.0, 1.0, 1.0], [True, True, True])
    config = Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, warmup_steps=10)

    result = compute_rho1_batch_result(
        token_loss=token_loss,
        shard_meta=None,
        opt_step=3,
        config=config,
        ref_loss_loader=loader,
    )

    assert torch.isclose(result.loss, torch.tensor(4.0))
    assert result.kept_tokens == 3
    assert result.keep_frac == 1.0
    assert result.ref_loss_mean is None
    assert loader.calls == []


def test_compute_rho1_batch_result_warmup_collects_diagnostics_when_requested() -> None:
    token_loss = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)
    loader = _FakeRefLossLoader([1.0, 3.0, 10.0], [True, True, True])
    config = Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, warmup_steps=10)
    gap = Rho1GapOptStepAccumulator()

    result = compute_rho1_batch_result(
        token_loss=token_loss,
        shard_meta={"shard_path": "train_000000.bin", "start": 0, "end": 4},
        opt_step=3,
        config=config,
        ref_loss_loader=loader,
        gap_accumulator=gap,
    )

    summary = gap.finalize()
    assert torch.isclose(result.loss, torch.tensor(4.0))
    assert result.kept_tokens == 3
    assert result.keep_frac == 1.0
    assert result.ref_loss_mean == pytest.approx((1.0 + 3.0 + 10.0) / 3.0)
    assert loader.calls == [({"shard_path": "train_000000.bin", "start": 0, "end": 4}, 3)]
    assert summary.delta_mean_all == pytest.approx(-2.0 / 3.0)
    assert summary.delta_mean_kept == pytest.approx(-2.0 / 3.0)
    assert summary.delta_pos_frac_all == pytest.approx(2.0 / 3.0)
    assert summary.delta_pos_frac_kept == pytest.approx(2.0 / 3.0)
    assert summary.delta_mean_seq_all == pytest.approx(-2.0 / 3.0)
    assert summary.delta_median_seq_all == pytest.approx(-2.0 / 3.0)
    assert summary.delta_p90_seq_all == pytest.approx(-2.0 / 3.0)
    assert summary.delta_pos_frac_seq_all == pytest.approx(0.0)
    assert summary.delta_pos_mean_seq_all is None
    assert summary.delta_mean_seq_kept == pytest.approx(-2.0 / 3.0)
    assert summary.delta_pos_frac_seq_kept == pytest.approx(0.0)


def test_rho1_opt_step_accumulator_tracks_means() -> None:
    stats = Rho1OptStepAccumulator()
    stats.update(
        compute_rho1_batch_result(
            token_loss=torch.tensor([[1.0, 5.0]], dtype=torch.float32),
            shard_meta={"shard_path": "train_000000.bin", "start": 0, "end": 3},
            opt_step=20,
            config=Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, mode="student_only"),
            ref_loss_loader=_FakeRefLossLoader([0.0, 0.0], [True, True]),
        )
    )
    stats.update(
        compute_rho1_batch_result(
            token_loss=torch.tensor([[9.0, 3.0]], dtype=torch.float32),
            shard_meta={"shard_path": "train_000001.bin", "start": 0, "end": 3},
            opt_step=20,
            config=Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, mode="student_only"),
            ref_loss_loader=_FakeRefLossLoader([0.0, 0.0], [True, True]),
        )
    )

    summary = stats.finalize()
    assert summary.keep_frac_mean == 0.5
    assert summary.kept_tokens_mean == 1
    assert summary.keep_seq_frac_mean is None
    assert summary.kept_sequences_mean is None
    assert summary.ref_loss_mean == 0.0


def test_rho1_opt_step_accumulator_tracks_sequence_means() -> None:
    stats = Rho1OptStepAccumulator()
    stats.update(
        compute_rho1_loss_from_reference(
            token_loss=torch.tensor([[5.0, 5.0], [1.0, 1.0]], dtype=torch.float32),
            ref_loss=torch.tensor([[1.0, 1.0], [0.5, 0.5]], dtype=torch.float32),
            ref_valid_mask=torch.tensor([[True, True], [True, True]]),
            config=Rho1Config(ref_loss_dir="enabled", keep_frac=0.5, mode="delta", granularity="sequence"),
        )
    )

    summary = stats.finalize()
    assert summary.keep_frac_mean == 0.5
    assert summary.kept_tokens_mean == 2
    assert summary.keep_seq_frac_mean == 0.5
    assert summary.kept_sequences_mean == 1
    assert summary.ref_loss_mean == 0.75
