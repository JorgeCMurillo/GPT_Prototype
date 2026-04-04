import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from research.bos_aligned_proto.analysis.attribution.trackstar import cpt_ablation
from research.bos_aligned_proto.analysis.attribution.trackstar import plot_cpt_ablation
from research.bos_aligned_proto.analysis.attribution.trackstar import run_cpt_ablation


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_u16(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.uint16).tofile(path)


def _make_checkpoint_dir(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt_final_step0000006"
    ckpt.mkdir()
    _write_json(
        ckpt / "config.json",
        {
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_ctx": 1024,
            "n_embd": 1024,
            "n_head": 16,
            "n_layer": 24,
        },
    )
    return ckpt


def _make_bos_row_dataset(root: Path, name: str, *, num_rows: int = 40) -> Path:
    data_dir = root / name
    data_dir.mkdir(parents=True, exist_ok=True)
    row_tokens = 1025
    shard0 = np.arange((num_rows // 2) * row_tokens, dtype=np.uint16)
    shard1 = np.arange((num_rows // 2) * row_tokens, 2 * (num_rows // 2) * row_tokens, dtype=np.uint16)
    _write_u16(data_dir / "train_000000.bin", shard0)
    _write_u16(data_dir / "train_000001.bin", shard1)
    _write_json(
        data_dir / "meta.json",
        {
            "format": "bos_row_packed_bestfit",
            "seq_len": 1024,
            "row_tokens": row_tokens,
            "num_rows": num_rows,
        },
    )
    _write_jsonl(
        data_dir / "rows.jsonl",
        [{"candidate_id": idx, "pool_role": name} for idx in range(num_rows)],
    )
    return data_dir


def _make_matched_pool_root(tmp_path: Path) -> Path:
    root = tmp_path / "matched"
    _make_bos_row_dataset(root, "treated_dataset")
    _make_bos_row_dataset(root, "control_dataset")
    return root


def _make_run_dir(root: Path, arm: str, lr: float, seed: int, values_by_step: dict[int, list[float]]) -> dict:
    run_dir = root / f"{arm}_lr_{lr:.0e}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    step_metrics = []
    ewok_rows = []
    for step, values in sorted(values_by_step.items()):
        timestamp = f"2026-01-01T00:00:{step:02d}"
        step_metrics.append(
            {
                "step": int(step),
                "timestamp": timestamp,
                "final": bool(step == max(values_by_step)),
                "eval_official_mean": {"average": 0.0},
            }
        )
        item_type = "ewok_item_final_mean" if step == max(values_by_step) else "ewok_item_mean"
        for row_index, value in enumerate(values):
            ewok_rows.append(
                {
                    "type": item_type,
                    "step": int(step),
                    "timestamp": timestamp,
                    "row_index": int(row_index),
                    "domain": "physical-relations",
                    "babylm_completion_choice_margin_combined": float(value),
                }
            )
    _write_json(run_dir / "step_metrics.json", step_metrics)
    _write_jsonl(run_dir / "ewok_items.jsonl", ewok_rows)
    return {
        "arm": arm,
        "learning_rate": float(lr),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "step_metrics_path": str(run_dir / "step_metrics.json"),
        "ewok_items_path": str(run_dir / "ewok_items.jsonl"),
        "steps_per_epoch": 2,
        "max_train_steps": 6,
    }


def _make_baseline_summary(path: Path) -> Path:
    _write_json(
        path,
        {
            "metric_name": "babylm_completion_choice_margin_combined",
            "reductions": {
                "mean": {
                    "average": {"average": 0.4},
                    "domain": {"physical-relations": 0.4},
                },
                "sum": {
                    "average": {"average": 0.4},
                    "domain": {"physical-relations": 0.4},
                },
            },
        },
    )
    return path


def test_prepare_bos_row_training_view_and_build_specs_use_requested_defaults(tmp_path: Path) -> None:
    matched_root = _make_matched_pool_root(tmp_path)
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    output_dir = tmp_path / "ablation"

    training_views, specs = cpt_ablation.build_ablation_run_specs(
        base_ckpt=ckpt_dir,
        matched_pool_dir=matched_root,
        output_dir=output_dir,
        learning_rates=(4e-5,),
        seeds=(42,),
        micro_batch_size=4,
        total_batch_tokens=32768,
        num_epochs=3,
        num_processes=1,
        ewok_batch_size=4,
        num_workers=0,
    )

    assert set(training_views) == {"treated", "control"}
    assert len(specs) == 2
    treated_view = training_views["treated"]
    treated_meta = json.loads((treated_view / "meta.json").read_text(encoding="utf-8"))
    assert treated_meta["cpt_ablation"]["synthetic_val_split"] == "concatenate_train_shards"
    assert (treated_view / "val_000000.bin").exists()
    assert specs[0].budget.grad_accum_steps == 8
    assert specs[0].budget.effective_global_batch_seqs == 32
    assert specs[0].budget.steps_per_epoch == 2
    assert specs[0].budget.max_train_steps == 6
    assert "--loader_kind" in specs[0].command
    assert "bos_row" in specs[0].command
    assert "--micro_batch_size" in specs[0].command
    assert "4" in specs[0].command
    assert "--total_batch_tokens" in specs[0].command
    assert "32768" in specs[0].command
    assert "--ewok_every" in specs[0].command
    assert "2" in specs[0].command
    assert "--ewok_reductions" in specs[0].command
    assert "mean" in specs[0].command
    assert "--hellaswag_every" in specs[0].command
    assert "--core_every" in specs[0].command
    assert "--no-save_final_checkpoint" in specs[0].command


def test_build_bos_trainer_command_uses_accelerate_for_multi_process(tmp_path: Path) -> None:
    matched_root = _make_matched_pool_root(tmp_path)
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    output_dir = tmp_path / "ablation_mp"

    _, specs = cpt_ablation.build_ablation_run_specs(
        base_ckpt=ckpt_dir,
        matched_pool_dir=matched_root,
        output_dir=output_dir,
        learning_rates=(4e-5,),
        seeds=(42,),
        micro_batch_size=4,
        total_batch_tokens=32768,
        num_epochs=3,
        num_processes=2,
        ewok_batch_size=4,
        num_workers=0,
    )

    assert specs
    assert specs[0].command[:4] == ("accelerate", "launch", "--num_processes", "2")


def test_run_ablation_aggregation_computes_deltas_and_final_effect(tmp_path: Path) -> None:
    baseline_path = _make_baseline_summary(tmp_path / "baseline" / "baseline_summary.json")
    runs_root = tmp_path / "runs"
    run_records = [
        _make_run_dir(runs_root, "treated", 4e-5, 42, {2: [0.6, 0.8], 6: [1.0, 1.2]}),
        _make_run_dir(runs_root, "control", 4e-5, 42, {2: [0.3, 0.5], 6: [0.5, 0.7]}),
    ]

    outputs = cpt_ablation.run_ablation_aggregation(
        output_dir=tmp_path / "ablation_outputs",
        run_records=run_records,
        baseline_summary_path=baseline_path,
        metric_name="babylm_completion_choice_margin_combined",
    )

    curves = [json.loads(line) for line in Path(outputs["curves_path"]).read_text(encoding="utf-8").splitlines() if line]
    summary = json.loads(Path(outputs["summary_path"]).read_text(encoding="utf-8"))

    average_curves = [
        row
        for row in curves
        if row["group_by"] == "average" and row["group_name"] == "average" and row["reduction"] == "mean"
    ]
    treated_final = next(
        row for row in average_curves if row["arm"] == "treated" and int(row["step"]) == 6
    )
    assert treated_final["value"] == pytest.approx(1.1)
    assert treated_final["baseline_value"] == pytest.approx(0.4)
    assert treated_final["delta_from_baseline"] == pytest.approx(0.7)

    final_record = next(
        row
        for row in summary["final_records"]
        if row["group_by"] == "average" and row["group_name"] == "average" and row["reduction"] == "mean"
    )
    assert final_record["treated_value"] == pytest.approx(1.1)
    assert final_record["control_value"] == pytest.approx(0.6)
    assert final_record["treated_minus_control"] == pytest.approx(0.5)


def test_generate_ablation_plots_writes_pngs(tmp_path: Path) -> None:
    if plot_cpt_ablation.plt is None:
        pytest.skip("matplotlib is not available in this test environment")

    baseline_path = _make_baseline_summary(tmp_path / "baseline" / "baseline_summary.json")
    runs_root = tmp_path / "runs"
    run_records = [
        _make_run_dir(runs_root, "treated", 4e-5, 42, {2: [0.6, 0.8], 6: [1.0, 1.2]}),
        _make_run_dir(runs_root, "control", 4e-5, 42, {2: [0.3, 0.5], 6: [0.5, 0.7]}),
        _make_run_dir(runs_root, "treated", 8e-5, 43, {2: [0.5, 0.7], 6: [0.9, 1.1]}),
        _make_run_dir(runs_root, "control", 8e-5, 43, {2: [0.2, 0.4], 6: [0.4, 0.6]}),
    ]
    ablation_dir = tmp_path / "ablation_outputs"
    cpt_ablation.run_ablation_aggregation(
        output_dir=ablation_dir,
        run_records=run_records,
        baseline_summary_path=baseline_path,
        metric_name="babylm_completion_choice_margin_combined",
    )
    matched_root = tmp_path / "matched_positive_pooled_top1000"
    matched_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        matched_root / "summary.json",
        {
            "score_mode": "positive_pooled",
            "target_id": None,
            "selection_source": {"kind": "row_summary", "column": "positive_score_sum"},
        },
    )
    _write_json(
        ablation_dir / "ablation_manifest.json",
        {
            "matched_pool_dir": str(matched_root),
        },
    )

    plot_outputs = plot_cpt_ablation.generate_ablation_plots(
        ablation_dir=ablation_dir,
        output_dir=ablation_dir / "plots",
        group_by="average",
        reduction="mean",
        metric_name="babylm_completion_choice_margin_combined",
        x_axis="epoch",
        dpi=80,
    )

    manifest = json.loads(Path(plot_outputs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["selection_label"] == "selection=positive_pooled"
    assert manifest["selection_metadata"]["score_mode"] == "positive_pooled"
    assert manifest["selection_filename_tag"] == "selection_positive_pooled"
    assert len(manifest["plots"]) == 2
    for plot_record in manifest["plots"]:
        assert Path(plot_record["path"]).exists()
        assert "selection_positive_pooled" in Path(plot_record["path"]).name


def test_run_cpt_ablation_dry_run_writes_manifest_and_plans_runs(tmp_path: Path) -> None:
    matched_root = _make_matched_pool_root(tmp_path)
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    output_dir = tmp_path / "ablation_cli"

    exit_code = run_cpt_ablation.main(
        [
            "--base_ckpt",
            str(ckpt_dir),
            "--matched_pool_dir",
            str(matched_root),
            "--output_dir",
            str(output_dir),
            "--learning_rates",
            "4e-5",
            "--seeds",
            "42",
            "--dry_run",
        ]
    )

    manifest = json.loads((output_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["dry_run"] is True
    assert set(manifest["training_views"]) == {"treated", "control"}
    assert len(manifest["planned_runs"]) == 2
    assert manifest["defaults"]["micro_batch_size"] == 4
    assert manifest["defaults"]["total_batch_tokens"] == 32768
    assert manifest["defaults"]["save_final_checkpoint"] is False


def test_run_cpt_ablation_parser_supports_progress_toggle() -> None:
    parser = run_cpt_ablation.build_arg_parser()

    default_args = parser.parse_args(
        [
            "--base_ckpt",
            "/tmp/ckpt",
            "--matched_pool_dir",
            "/tmp/matched",
            "--output_dir",
            "/tmp/out",
        ]
    )
    quiet_args = parser.parse_args(
        [
            "--base_ckpt",
            "/tmp/ckpt",
            "--matched_pool_dir",
            "/tmp/matched",
            "--output_dir",
            "/tmp/out",
            "--no_progress",
        ]
    )

    assert default_args.show_progress is True
    assert quiet_args.show_progress is False
    assert default_args.save_final_checkpoint is False


def test_run_cpt_ablation_auto_plots_include_all_grouped_views() -> None:
    expected = ("average", "domain", "ContextDiff", "TargetDiff", "ContextType")
    assert run_cpt_ablation._auto_plot_group_bys("average") == expected
    assert run_cpt_ablation._auto_plot_group_bys("domain") == expected
    assert run_cpt_ablation._auto_plot_group_bys("ContextDiff") == expected
    assert run_cpt_ablation._auto_plot_group_bys("TargetDiff") == expected
