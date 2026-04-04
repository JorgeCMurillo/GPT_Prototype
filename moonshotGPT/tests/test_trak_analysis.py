import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.ewok import BABYLM_COMPLETION_CHOICE
from research.bos_aligned_proto.analysis.attribution.trackstar.backend import (
    BergsonAttributionBackend,
    BergsonShardResult,
    _patched_candidate_forward_for_external_shift,
    assemble_sharded_scores,
    build_backend as build_trackstar_backend,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_datasets import (
    BergsonCandidateDataset,
    build_candidate_index_fingerprint,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_queries import (
    _select_gradient_modules,
)
from research.bos_aligned_proto.analysis.attribution.common.candidates import select_candidate_rows
from research.bos_aligned_proto.analysis.attribution.common.checkpoints import (
    CheckpointRef,
    align_state_dict_to_model,
    discover_checkpoints,
)
from research.bos_aligned_proto.analysis.attribution.common.compare import compare_adjacent_row_summaries
from research.bos_aligned_proto.analysis.attribution.common.ewok_targets import (
    CheckpointScores,
    EWOKTargetBundle,
    EWOKTargetItem,
    TargetDiagnostics,
    build_ewok_targets,
    reduce_masked_token_logprobs,
)
from research.bos_aligned_proto.analysis.attribution.common.export import (
    build_bottom_rows_frame,
    write_checkpoint_outputs,
)
from research.bos_aligned_proto.analysis.attribution.common.ewok_filters import (
    load_ewok_target_filter_spec,
)
from research.bos_aligned_proto.analysis.attribution.common.exposures import (
    ExposureIndex,
    build_exposure_index,
)
from research.bos_aligned_proto.analysis.attribution.common.row_dataset import (
    FiniteBOSRowDataset,
    build_row_manifest,
    iter_row_batches,
)
from research.bos_aligned_proto.analysis.attribution.run_trak import (
    RunExecutionContext,
    build_previous_checkpoint_step_map,
    execute_attribution_run,
    resolve_candidate_window,
    resolve_execution_context,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.config import build_arg_parser as build_trackstar_arg_parser
from research.bos_aligned_proto.analysis.attribution.trackstar.config import TrackstarConfig
from research.bos_aligned_proto.analysis.attribution.trak.backend import build_backend
from research.bos_aligned_proto.analysis.attribution.trak.config import TRAKConfig


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
            concept_a="trust",
            concept_b="distrust",
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
            concept_a="melt",
            concept_b="freeze",
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
            candidate_ids=tuple(candidate_selection.row_ids),
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
        step_to_example_ids={100: (0, 1, 2), 200: (3, 4, 5)},
        first_seen_step_by_example_id={0: 100, 1: 100, 2: 100, 3: 200, 4: 200, 5: 200},
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


def test_candidate_selection_supports_explicit_upper_window() -> None:
    index = ExposureIndex(
        run_dir=Path("/tmp/fake"),
        step_to_example_ids={
            12000: (0, 1),
            16000: (2,),
            18000: (3, 4),
            20000: (5, 6),
            22000: (7,),
        },
        first_seen_step_by_example_id={
            0: 12000,
            1: 12000,
            2: 16000,
            3: 18000,
            4: 18000,
            5: 20000,
            6: 20000,
            7: 22000,
        },
    )

    selection = select_candidate_rows(
        index,
        strategy="between_checkpoints",
        checkpoint_step=16000,
        previous_step=16000,
        candidate_to_step=20000,
        max_candidate_rows=20,
        seed=7,
        recent_window_steps=50,
    )

    assert selection.checkpoint_step == 16000
    assert selection.previous_step == 16000
    assert selection.candidate_to_step == 20000
    assert selection.row_ids == (3, 4, 5, 6)


def test_resolve_candidate_window_uses_real_previous_discovered_checkpoint() -> None:
    checkpoints = [
        CheckpointRef(step=4000, path=Path("/tmp/ckpt4000"), kind="periodic"),
        CheckpointRef(step=8000, path=Path("/tmp/ckpt8000"), kind="periodic"),
        CheckpointRef(step=12000, path=Path("/tmp/ckpt12000"), kind="periodic"),
        CheckpointRef(step=16000, path=Path("/tmp/ckpt16000"), kind="periodic"),
        CheckpointRef(step=20000, path=Path("/tmp/ckpt20000"), kind="periodic"),
    ]
    previous_by_step = build_previous_checkpoint_step_map(checkpoints)
    config = TrackstarConfig(run_dir=Path("/tmp/run"), data_dir=Path("/tmp/data")).resolved()

    candidate_from_step, candidate_to_step = resolve_candidate_window(
        checkpoint_step=20000,
        previous_checkpoint_step=previous_by_step[20000],
        config=config,
    )

    assert candidate_from_step == 16000
    assert candidate_to_step == 20000


def test_resolve_candidate_window_allows_explicit_window_overrides(tmp_path) -> None:
    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        candidate_from_step=16000,
        candidate_to_step=20000,
    ).resolved()

    candidate_from_step, candidate_to_step = resolve_candidate_window(
        checkpoint_step=16000,
        previous_checkpoint_step=12000,
        config=config,
    )

    assert candidate_from_step == 16000
    assert candidate_to_step == 20000


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


def test_ewok_filter_spec_restricts_targets_by_multiple_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "ewok_filter.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "social_indirect_variable_swap",
                "variant": "fast",
                "domains": ["social-relations"],
                "context_types": ["indirect"],
                "context_diffs": ["variable swap"],
            }
        ),
        encoding="utf-8",
    )

    spec = load_ewok_target_filter_spec(spec_path)
    assert spec.domains == ("social-relations",)
    assert spec.context_types == ("indirect",)
    assert spec.context_diffs == ("variable swap",)

    bundle = build_ewok_targets(
        score_view=BABYLM_COMPLETION_CHOICE,
        target_scope="both",
        score_reduction="mean",
        filter_spec_path=spec_path,
    )

    assert len(bundle.items) > 0
    assert all(item.domain == "social-relations" for item in bundle.items)
    assert all(item.context_type == "indirect" for item in bundle.items)
    assert all(item.context_diff == "variable swap" for item in bundle.items)
    assert set(bundle.groups) == {"overall", "domain:social-relations"}


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


def test_align_state_dict_to_model_materializes_tied_weight_aliases() -> None:
    class _ToyTiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(7, 5)
            self.lm_head = torch.nn.Linear(5, 7, bias=False)
            self.lm_head.weight = self.embed.weight

    model = _ToyTiedModel()
    source = model.state_dict()
    partial = {"embed.weight": source["embed.weight"].clone()}

    aligned = align_state_dict_to_model(partial, model)

    assert "embed.weight" in aligned
    assert "lm_head.weight" in aligned
    assert torch.equal(aligned["embed.weight"], aligned["lm_head.weight"])


def test_trak_config_accepts_backend_device_and_distributed(tmp_path) -> None:
    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        device="auto",
        distributed="ddp",
    ).resolved()

    assert config.backend == "trackstar"
    assert config.device == "auto"
    assert config.distributed == "ddp"


def test_trackstar_arg_parser_defaults_to_projected_indexing(tmp_path) -> None:
    parser = build_trackstar_arg_parser()

    ns = parser.parse_args(["--run_dir", str(tmp_path / "run"), "--data_dir", str(tmp_path / "data")])

    assert ns.use_fast_jl is True
    assert ns.proj_dim == 16
    assert ns.projection_layout == "module"
    assert ns.paper_block_features == 4096
    assert ns.bottomk == 0
    assert ns.show_progress is True


def test_trackstar_arg_parser_accepts_candidate_window_overrides(tmp_path) -> None:
    parser = build_trackstar_arg_parser()

    ns = parser.parse_args(
        [
            "--run_dir",
            str(tmp_path / "run"),
            "--data_dir",
            str(tmp_path / "data"),
            "--candidate_from_step",
            "16000",
            "--candidate_to_step",
            "20000",
        ]
    )

    assert ns.candidate_from_step == 16000
    assert ns.candidate_to_step == 20000


def test_trackstar_config_resolves_paper_block_projection_fields(tmp_path) -> None:
    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        use_fast_jl=True,
        projection_layout="paper_blocks",
        paper_block_features=4096,
    ).resolved()

    assert config.projection_layout == "paper_blocks"
    assert config.paper_block_features == 4096
    assert config.paper_block_side == 64


def test_build_bottom_rows_frame_returns_lowest_scoring_rows(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)
    bundle = _make_target_bundle()
    result = CheckpointScores(
        checkpoint_step=1000,
        checkpoint_path=str(tmp_path / "ckpt"),
        candidate_ids=(0, 1, 2, 3),
        target_ids=("target-a", "target-b"),
        score_matrix=np.asarray(
            [
                [0.4, -0.3, 0.2, -0.5],
                [0.1, -0.2, 0.9, -0.4],
            ],
            dtype=np.float64,
        ),
        target_diagnostics=(
            TargetDiagnostics(
                target_id="target-a",
                domain="social-relations",
                score_view=BABYLM_COMPLETION_CHOICE,
                score_reduction="mean",
                s11_mean=0.0,
                s12_mean=0.0,
                s22_mean=0.0,
                s21_mean=0.0,
                s11_sum=0.0,
                s12_sum=0.0,
                s22_sum=0.0,
                s21_sum=0.0,
                margin_1=0.0,
                margin_2=0.0,
                combined_margin=0.0,
                softplus_loss=0.0,
                score=0.0,
            ),
            TargetDiagnostics(
                target_id="target-b",
                domain="material-dynamics",
                score_view=BABYLM_COMPLETION_CHOICE,
                score_reduction="mean",
                s11_mean=0.0,
                s12_mean=0.0,
                s22_mean=0.0,
                s21_mean=0.0,
                s11_sum=0.0,
                s12_sum=0.0,
                s22_sum=0.0,
                s21_sum=0.0,
                margin_1=0.0,
                margin_2=0.0,
                combined_margin=0.0,
                softplus_loss=0.0,
                score=0.0,
            ),
        ),
    )

    bottom_rows = build_bottom_rows_frame(result, bundle, manifest, bottomk=2)

    target_a = bottom_rows.loc[bottom_rows["target_id"] == "target-a"].sort_values("rank")
    target_b = bottom_rows.loc[bottom_rows["target_id"] == "target-b"].sort_values("rank")
    assert target_a["row_id"].tolist() == [3, 1]
    assert target_b["row_id"].tolist() == [3, 1]


def test_write_checkpoint_outputs_writes_bottom_rows_when_requested(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)
    bundle = _make_target_bundle()
    backend_result = _FakeBackend().score_checkpoint(
        checkpoint=CheckpointRef(step=1000, path=tmp_path / "ckpt", kind="periodic"),
        manifest=manifest,
        candidate_selection=type("Selection", (), {"row_ids": (0, 1, 2, 3)})(),
        target_bundle=bundle,
    )

    paths = write_checkpoint_outputs(
        output_dir=tmp_path / "artifacts",
        result=backend_result,
        bundle=bundle,
        manifest=manifest,
        topk=2,
        bottomk=2,
        write_dense_scores=False,
    )

    assert paths["top_rows"] is not None and Path(paths["top_rows"]).exists()
    assert paths["bottom_rows"] is not None and Path(paths["bottom_rows"]).exists()
    bottom_rows = pd.read_csv(paths["bottom_rows"])
    assert set(bottom_rows["rank"]) == {1, 2}


def test_resolve_execution_context_fails_loudly_for_missing_cuda(monkeypatch, tmp_path) -> None:
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_NUM_PROCESSES",
        "ACCELERATE_LOCAL_PROCESS_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        device="cuda",
        distributed="none",
    ).resolved()

    with pytest.raises(RuntimeError, match="--device cuda was requested"):
        resolve_execution_context(config)


def test_resolve_execution_context_uses_accelerate_env(monkeypatch, tmp_path) -> None:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ACCELERATE_PROCESS_INDEX", "1")
    monkeypatch.setenv("ACCELERATE_NUM_PROCESSES", "4")
    monkeypatch.setenv("ACCELERATE_LOCAL_PROCESS_INDEX", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        device="cuda",
        distributed="ddp",
    ).resolved()

    context = resolve_execution_context(config)

    assert context.launcher == "accelerate"
    assert context.rank == 1
    assert context.world_size == 4
    assert context.local_rank == 2
    assert context.resolved_device == "cuda:2"


def test_bergson_candidate_dataset_preserves_metadata(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)
    dataset = BergsonCandidateDataset(manifest, candidate_ids=(1, 3))

    sample = dataset[0]

    assert sample["row_id"] == 1
    assert sample["candidate_idx"] == 0
    assert sample["shard_idx"] == 0
    assert sample["local_row_idx"] == 1
    assert torch.equal(sample["input_ids"], torch.tensor([20, 21, 22, 23]))
    assert torch.equal(sample["labels"], sample["input_ids"])
    assert torch.equal(sample["attention_mask"], torch.ones_like(sample["input_ids"]))


def test_bergson_candidate_dataset_supports_batched_indexing(tmp_path) -> None:
    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)
    dataset = BergsonCandidateDataset(manifest, candidate_ids=(1, 3))

    batch = dataset[[0, 1]]

    assert batch["row_id"] == [1, 3]
    assert batch["candidate_idx"] == [0, 1]
    assert batch["input_ids"] == [[20, 21, 22, 23], [40, 41, 42, 43]]
    assert batch["labels"] == [[20, 21, 22, 23], [40, 41, 42, 43]]


def test_bergson_candidate_dataset_supports_dataset_like_teardown(tmp_path) -> None:
    datasets = pytest.importorskip("datasets")

    data_dir = _make_row_data(tmp_path)
    manifest = build_row_manifest(data_dir)
    dataset = BergsonCandidateDataset(manifest, candidate_ids=(1, 3))

    view = dataset.remove_columns(["input_ids"]).add_column("loss", np.asarray([0.25, 0.75], dtype=np.float32))
    out_dir = tmp_path / "candidate_view.hf"
    view.save_to_disk(str(out_dir))
    restored = datasets.load_from_disk(str(out_dir))

    assert "input_ids" not in restored.column_names
    assert "loss" in restored.column_names
    assert restored["row_id"] == [1, 3]
    assert restored["loss"] == pytest.approx([0.25, 0.75])


def test_trackstar_candidate_forward_patch_handles_one_token_overflow() -> None:
    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(n_positions=4)
            self.seen_lengths: list[int] = []

        def forward(self, input_ids, attention_mask=None):
            del attention_mask
            self.seen_lengths.append(int(input_ids.shape[1]))
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, seq_len, 7), dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    model = _ToyModel()
    original_forward = model.forward

    with _patched_candidate_forward_for_external_shift(model):
        normal = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        overflow = model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

    restored = model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

    assert model.forward == original_forward
    assert normal.logits.shape == (1, 4, 7)
    assert overflow.logits.shape == (1, 5, 7)
    assert restored.logits.shape == (1, 5, 7)
    assert model.seen_lengths == [4, 4, 5]


def test_trackstar_candidate_forward_patch_rejects_longer_overflow() -> None:
    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(n_positions=4)

        def forward(self, input_ids):
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 5), dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    model = _ToyModel()

    with _patched_candidate_forward_for_external_shift(model):
        with pytest.raises(ValueError, match="one token longer"):
            model(torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long))


def test_candidate_index_fingerprint_changes_with_candidate_ids(tmp_path) -> None:
    checkpoint = CheckpointRef(step=100, path=tmp_path / "ckpt", kind="periodic")
    first = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_ids=(1, 2, 3),
        projection_dim=0,
        use_fast_jl=False,
        adam_second_moment_correction=False,
    )
    second = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_ids=(1, 3, 4),
        projection_dim=0,
        use_fast_jl=False,
        adam_second_moment_correction=False,
    )

    assert first != second


def test_trackstar_backend_resolves_partial_gradient_dir(tmp_path) -> None:
    index_dir = tmp_path / "index_deadbeef"
    partial_dir = Path(str(index_dir) + ".part")
    partial_dir.mkdir()
    (partial_dir / "info.json").write_text(json.dumps({"num_grads": 0, "dtype": []}), encoding="utf-8")
    (partial_dir / "gradients.bin").write_bytes(b"")

    resolved = BergsonAttributionBackend._resolve_gradient_dir(index_dir)

    assert resolved == partial_dir


def test_trackstar_backend_rejects_zero_width_gradient_dir(tmp_path) -> None:
    gradient_dir = tmp_path / "index_deadbeef"
    gradient_dir.mkdir()
    (gradient_dir / "info.json").write_text(
        json.dumps(
            {
                "num_grads": 2,
                "dtype": {"names": ["mod"], "formats": ["(0,)<f4"], "itemsize": 0},
                "grad_sizes": {"mod": 0},
                "base_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    (gradient_dir / "gradients.bin").write_bytes(b"")

    assert BergsonAttributionBackend._gradient_dir_has_nonzero_sizes(gradient_dir) is False


def test_trackstar_backend_normalizes_structured_memmap_like_gradients() -> None:
    loaded = np.zeros(3, dtype=[("module_a", np.float32, (2,)), ("module_b", np.float32, (1,))])
    loaded["module_a"] = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    loaded["module_b"] = np.asarray([[7.0], [8.0], [9.0]], dtype=np.float32)

    normalized = BergsonAttributionBackend._normalize_loaded_gradients(loaded)

    assert set(normalized) == {"module_a", "module_b"}
    assert normalized["module_a"].shape == (3, 2)
    assert normalized["module_b"].shape == (3, 1)


def test_select_gradient_modules_accepts_base_model_relative_names() -> None:
    class _Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2)

    class _Wrapped(torch.nn.Module):
        base_model_prefix = "transformer"

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Inner()

    model = _Wrapped()

    selected = _select_gradient_modules(model, ("proj",))

    assert tuple(selected) == ("proj",)
    assert selected["proj"] is model.transformer.proj


def test_assemble_sharded_scores_preserves_target_order() -> None:
    diagnostics = (
        TargetDiagnostics(
            target_id="target-a",
            domain="social-relations",
            score_view="babylm_completion_choice",
            score_reduction="mean",
            s11_mean=0.1,
            s12_mean=0.2,
            s22_mean=0.3,
            s21_mean=0.4,
            s11_sum=1.1,
            s12_sum=1.2,
            s22_sum=1.3,
            s21_sum=1.4,
            margin_1=0.5,
            margin_2=0.6,
            combined_margin=0.55,
            softplus_loss=0.25,
            score=-0.25,
        ),
        TargetDiagnostics(
            target_id="target-b",
            domain="material-dynamics",
            score_view="babylm_completion_choice",
            score_reduction="mean",
            s11_mean=1.1,
            s12_mean=1.2,
            s22_mean=1.3,
            s21_mean=1.4,
            s11_sum=2.1,
            s12_sum=2.2,
            s22_sum=2.3,
            s21_sum=2.4,
            margin_1=1.5,
            margin_2=1.6,
            combined_margin=1.55,
            softplus_loss=1.25,
            score=-1.25,
        ),
    )
    shard_results = [
        BergsonShardResult(
            rank=1,
            target_indices=(1,),
            score_matrix=np.asarray([[20.0, 21.0]], dtype=np.float64),
            diagnostics=(diagnostics[1],),
        ),
        BergsonShardResult(
            rank=0,
            target_indices=(0,),
            score_matrix=np.asarray([[10.0, 11.0]], dtype=np.float64),
            diagnostics=(diagnostics[0],),
        ),
    ]

    score_matrix, merged_diagnostics = assemble_sharded_scores(
        num_targets=2,
        num_candidates=2,
        shard_results=shard_results,
    )

    assert score_matrix.tolist() == [[10.0, 11.0], [20.0, 21.0]]
    assert tuple(diag.target_id for diag in merged_diagnostics) == ("target-a", "target-b")


def test_build_bergson_backend_requires_dependency(tmp_path) -> None:
    config = TrackstarConfig(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
    ).resolved()
    model = torch.nn.Linear(2, 2)
    context = RunExecutionContext.single_process(
        backend="trackstar",
        requested_device="cpu",
        resolved_device="cpu",
    )

    class _Tokenizer:
        bos_token_id = 0
        eos_token_id = 0
        pad_token_id = 0

    try:
        backend = build_trackstar_backend(
            config=config,
            model=model,
            tokenizer=_Tokenizer(),
            execution_context=context,
        )
    except ImportError as exc:
        assert "bergson" in str(exc)
    else:  # pragma: no cover - only happens if bergson is installed
        assert backend is not None


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
    output_dir = run_dir / "analysis" / "attribution" / "smoke"
    config = TRAKConfig(
        run_dir=run_dir,
        data_dir=data_dir,
        exp_name="smoke",
        output_dir=output_dir,
        checkpoint_steps=(100, 200),
        topk=2,
        max_candidate_rows=10,
    ).resolved()

    summary = execute_attribution_run(
        config=config,
        checkpoints=checkpoints,
        previous_checkpoint_step_by_step=build_previous_checkpoint_step_map(checkpoints),
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
