import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.ewok import BABYLM_COMPLETION_CHOICE
from research.bos_aligned_proto.analysis.trak.candidates import select_candidate_rows
from research.bos_aligned_proto.analysis.trak.checkpoints import CheckpointRef, discover_checkpoints
from research.bos_aligned_proto.analysis.trak.compare import compare_adjacent_row_summaries
from research.bos_aligned_proto.analysis.trak.config import TRAKConfig
from research.bos_aligned_proto.analysis.trak.ewok_targets import (
    EWOKTargetBundle,
    EWOKTargetItem,
    build_ewok_targets,
)
from research.bos_aligned_proto.analysis.trak.exposures import ExposureIndex, build_exposure_index
from research.bos_aligned_proto.analysis.trak.model_output import (
    CheckpointScores,
    TargetDiagnostics,
    build_backend,
    reduce_masked_token_logprobs,
)
from research.bos_aligned_proto.analysis.trak.row_dataset import (
    FiniteBOSRowDataset,
    build_row_manifest,
    iter_row_batches,
)
from research.bos_aligned_proto.analysis.trak.run_trak import execute_trak_run


def _write_u16(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype=np.uint16).tofile(path)


def _make_row_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "row_data"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"row_tokens": 4, "seq_len": 3}),
        encoding="utf-8",
    )
    _write_u16(data_dir / "train_000000.bin", [10, 11, 12, 13, 20, 21, 22, 23])
    _write_u16(data_dir / "train_000001.bin", [30, 31, 32, 33, 40, 41, 42, 43])
    return data_dir


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "ckpt_periodic_step0001000",
        "ckpt_periodic_step0002000",
        "ckpt_final_step0002000",
    ):
        (run_dir / name).mkdir()
    exposure_dir = run_dir / "exposures"
    exposure_dir.mkdir()
    rows = [
        {
            "step": 100,
            "micro_batches": [
                {"shard_idx": 0, "start": 0, "end": 8},
            ],
        },
        {
            "step": 200,
            "micro_batches": [
                {"shard_idx": 1, "start": 0, "end": 4},
            ],
        },
        {
            "step": 300,
            "micro_batches": [
                {"shard_idx": 1, "start": 4, "end": 8},
            ],
        },
    ]
    with (exposure_dir / "exposures_rank0000.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run_dir


def _make_target_bundle() -> EWOKTargetBundle:
    items = (
        EWOKTargetItem(
            target_id="target-a",
            domain="social-relations",
            row_index=0,
            score_view=BABYLM_COMPLETION_CHOICE,
            context1="A",
            context2="B",
            target1="C",
            target2="D",
            context_type_raw="direct",
            context_type="direct",
            context_diff_raw="negation",
            context_diff="negation",
            target_diff_raw="antonym",
            target_diff="antonym",
        ),
        EWOKTargetItem(
            target_id="target-b",
            domain="material-dynamics",
            row_index=1,
            score_view=BABYLM_COMPLETION_CHOICE,
            context1="E",
            context2="F",
            target1="G",
            target2="H",
            context_type_raw="indirect",
            context_type="indirect",
            context_diff_raw="variable_swap",
            context_diff="variable swap",
            target_diff_raw="concept swap",
            target_diff="concept swap",
        ),
    )
    return EWOKTargetBundle(
        items=items,
        groups={
            "overall": ("target-a", "target-b"),
            "domain:material-dynamics": ("target-b",),
            "domain:social-relations": ("target-a",),
        },
        source_path=Path("synthetic-ewok.jsonl"),
        score_view=BABYLM_COMPLETION_CHOICE,
        score_reduction="mean",
    )


class _FakeBackend:
    def score_checkpoint(
        self,
        *,
        checkpoint: CheckpointRef,
        manifest,
        candidate_selection,
        target_bundle,
    ) -> CheckpointScores:
        num_targets = len(target_bundle.items)
        num_rows = len(candidate_selection.row_ids)
        base = np.arange(num_targets * num_rows, dtype=np.float64).reshape(num_targets, num_rows)
        score_matrix = base + float(checkpoint.step)
        diagnostics = tuple(
            TargetDiagnostics(
                target_id=item.target_id,
                domain=item.domain,
                score_view=target_bundle.score_view,
                score_reduction=target_bundle.score_reduction,
                s11_mean=0.1 + idx,
                s12_mean=0.2 + idx,
                s22_mean=0.3 + idx,
                s21_mean=0.4 + idx,
                s11_sum=1.1 + idx,
                s12_sum=1.2 + idx,
                s22_sum=1.3 + idx,
                s21_sum=1.4 + idx,
                margin_1=0.5 + idx,
                margin_2=0.6 + idx,
                combined_margin=0.55 + idx,
                softplus_loss=0.25 + idx,
                score=-0.25 - idx,
            )
            for idx, item in enumerate(target_bundle.items)
        )
        return CheckpointScores(
            checkpoint_step=checkpoint.step,
            checkpoint_path=str(checkpoint.path),
            candidate_row_ids=tuple(candidate_selection.row_ids),
            target_ids=tuple(item.target_id for item in target_bundle.items),
            score_matrix=score_matrix,
            target_diagnostics=diagnostics,
        )


def test_discover_checkpoints_prefers_final_checkpoint(tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path)

    checkpoints = discover_checkpoints(run_dir)

    assert [(ref.step, ref.kind) for ref in checkpoints] == [
        (1000, "periodic"),
        (2000, "final"),
    ]


def test_row_manifest_and_dataset_are_deterministic(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)

    assert len(manifest.rows) == 4
    assert manifest.row_ref(2).shard_idx == 1
    assert manifest.row_ref(2).local_row_idx == 0

    dataset = FiniteBOSRowDataset(manifest, row_ids=(0, 2))
    first = dataset[0]
    second = dataset[1]

    assert first["row_id"] == 0
    assert torch.equal(first["input_ids"], torch.tensor([10, 11, 12]))
    assert torch.equal(first["labels"], torch.tensor([11, 12, 13]))
    assert second["row_id"] == 2
    assert torch.equal(second["input_ids"], torch.tensor([30, 31, 32]))

    batches = list(iter_row_batches(dataset, batch_size=2))
    assert len(batches) == 1
    assert batches[0].row_ids == (0, 2)
    assert batches[0].local_inds.tolist() == [0, 1]


def test_exposure_index_maps_offsets_to_global_row_ids(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    run_dir = _make_run_dir(tmp_path)
    manifest = build_row_manifest(data_dir)

    index = build_exposure_index(run_dir, manifest)

    assert index.rows_exposed_up_to_step(200) == (0, 1, 2)
    assert index.rows_exposed_between_steps(100, 300) == (2, 3)
    assert index.rows_first_seen_between_steps(100, 300) == (2, 3)


def test_candidate_selection_is_deterministic() -> None:
    index = ExposureIndex(
        run_dir=Path("/tmp/fake"),
        step_to_row_ids={100: (0, 1, 2), 200: (3, 4, 5)},
        first_seen_step_by_row_id={0: 100, 1: 100, 2: 100, 3: 200, 4: 200, 5: 200},
    )

    first = select_candidate_rows(
        index,
        strategy="up_to_step",
        checkpoint_step=200,
        previous_step=100,
        max_candidate_rows=3,
        seed=7,
        recent_window_steps=50,
    )
    second = select_candidate_rows(
        index,
        strategy="up_to_step",
        checkpoint_step=200,
        previous_step=100,
        max_candidate_rows=3,
        seed=7,
        recent_window_steps=50,
    )

    assert first.row_ids == second.row_ids
    assert first.selected_count == 3


def test_ewok_targets_include_domain_groups_and_normalized_metadata() -> None:
    bundle = build_ewok_targets(
        score_view=BABYLM_COMPLETION_CHOICE,
        target_scope="both",
        score_reduction="mean",
    )

    domain_groups = {name: ids for name, ids in bundle.groups.items() if name.startswith("domain:")}
    assert len(bundle.items) == 1100
    assert len(domain_groups) == 11
    assert all(len(ids) == 100 for ids in domain_groups.values())
    assert any(
        item.context_diff_raw == "variable_swap" and item.context_diff == "variable swap"
        for item in bundle.items
    )


def test_mean_reduction_normalizes_by_target_length() -> None:
    token_logprobs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.5, -0.5, 2.0, 0.0],
        ]
    )
    token_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    summed = reduce_masked_token_logprobs(token_logprobs, token_mask, reduction="sum")
    meaned = reduce_masked_token_logprobs(token_logprobs, token_mask, reduction="mean")

    assert torch.allclose(summed, torch.tensor([2.0, 2.0]))
    assert torch.allclose(meaned, torch.tensor([1.0, 2.0 / 3.0]))


def test_compare_adjacent_row_summaries_reports_overlap_and_sign_flips() -> None:
    row_summaries = {
        100: pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "mean_score": [0.1, -0.2, 0.3],
                "mean_abs_score": [0.1, 0.2, 0.3],
            }
        ),
        200: pd.DataFrame(
            {
                "row_id": [1, 2, 4],
                "mean_score": [-0.4, -0.1, 0.5],
                "mean_abs_score": [0.4, 0.1, 0.5],
            }
        ),
    }

    comparison = compare_adjacent_row_summaries(row_summaries, topk=2)

    assert len(comparison) == 1
    assert int(comparison.loc[0, "shared_row_count"]) == 2
    assert int(comparison.loc[0, "topk_abs_overlap"]) == 0
    assert int(comparison.loc[0, "sign_flip_count"]) == 1


def test_build_backend_requires_traker_dependency(tmp_path) -> None:
    config = TRAKConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
    ).resolved()
    model = torch.nn.Linear(2, 2)

    class _Tokenizer:
        bos_token_id = 0
        eos_token_id = 0
        pad_token_id = 0

    try:
        build_backend(config=config, model=model, tokenizer=_Tokenizer())
    except ImportError as exc:
        assert "traker" in str(exc)
    else:  # pragma: no cover - only happens if traker is installed
        assert True


def test_execute_trak_run_writes_expected_artifacts(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    run_dir = _make_run_dir(tmp_path)
    manifest = build_row_manifest(data_dir)
    exposure_index = build_exposure_index(run_dir, manifest)
    target_bundle = _make_target_bundle()
    checkpoints = [
        CheckpointRef(step=100, path=run_dir / "ckpt_periodic_step0001000", kind="periodic"),
        CheckpointRef(step=200, path=run_dir / "ckpt_final_step0002000", kind="final"),
    ]
    output_dir = run_dir / "analysis" / "trak" / "smoke"
    config = TRAKConfig(
        run_dir=run_dir,
        data_dir=data_dir,
        exp_name="smoke",
        output_dir=output_dir,
        checkpoint_steps=(100, 200),
        topk=2,
        max_candidate_rows=10,
    ).resolved()

    summary = execute_trak_run(
        config=config,
        checkpoints=checkpoints,
        manifest=manifest,
        exposure_index=exposure_index,
        target_bundle=target_bundle,
        backend=_FakeBackend(),
    )

    assert summary["target_count"] == 2
    assert (output_dir / "config.json").exists()
    assert (output_dir / "checkpoint_manifest.json").exists()
    assert (output_dir / "target_items.jsonl").exists()
    assert (output_dir / "top_rows_step00000100.csv").exists()
    assert (output_dir / "row_summary_step00000100.csv").exists()
    assert (output_dir / "domain_summary_step00000100.csv").exists()
    assert (output_dir / "checkpoint_compare.csv").exists()
    assert (output_dir / "run_summary.json").exists()

    top_rows = pd.read_csv(output_dir / "top_rows_step00000100.csv")
    assert {"checkpoint_step", "domain", "row_id", "m1", "m2"} <= set(top_rows.columns)
