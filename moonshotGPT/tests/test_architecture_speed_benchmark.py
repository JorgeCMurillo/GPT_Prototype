import json
from pathlib import Path

import pytest

from research.bos_aligned_proto.experiments import run_architecture_speed_benchmark as bench


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_param_matched_variant_rounds_intermediate_size() -> None:
    args = bench.build_parser().parse_args(
        [
            "--data_dir",
            "/tmp/data",
            "--source_data_dir",
            "/tmp/source",
            "--launcher",
            "python",
            "--variants",
            "gpt2,llama_shape,llama_param",
            "--n_embd",
            "1024",
        ]
    )

    variants = bench.build_variants(args)
    assert [variant.label for variant in variants] == ["gpt2", "llama_shape", "llama_param_i2816"]
    assert variants[1].llama_intermediate_size == 0
    assert variants[2].llama_intermediate_size == 2816


def test_build_trainer_command_disables_noisy_hooks(tmp_path: Path) -> None:
    args = bench.build_parser().parse_args(
        [
            "--data_dir",
            "/tmp/data",
            "--source_data_dir",
            "/tmp/source",
            "--launcher",
            "python",
            "--num_processes",
            "1",
            "--variants",
            "llama_param",
            "--n_embd",
            "768",
        ]
    )
    variant = bench.build_variants(args)[0]
    cmd = bench.build_trainer_command(args, variant, experiments_dir=tmp_path / "runs")

    assert cmd[0].endswith("python") or "python" in Path(cmd[0]).name
    assert "--model_arch" in cmd
    assert cmd[cmd.index("--model_arch") + 1] == "llama"
    assert "--mixed_precision" in cmd
    assert cmd[cmd.index("--mixed_precision") + 1] == "bf16"
    assert "--llama_intermediate_size" in cmd
    assert cmd[cmd.index("--llama_intermediate_size") + 1] == "2048"
    assert "--eval_every" in cmd
    assert cmd[cmd.index("--eval_every") + 1] == "0"
    assert "--no-save_final_checkpoint" in cmd
    assert "--skip_final_ewok" in cmd
    assert "--no-use_liger_kernel" in cmd


def test_build_trainer_command_allows_liger_llama_variant(tmp_path: Path) -> None:
    args = bench.build_parser().parse_args(
        [
            "--data_dir",
            "/tmp/data",
            "--launcher",
            "python",
            "--num_processes",
            "1",
            "--variants",
            "llama_param",
            "--liger_modes",
            "off,on",
        ]
    )
    variant = bench.build_variants(args)[0]
    kernel_variants = bench.build_kernel_variants(args, variant)

    assert [item.label for item in kernel_variants] == ["noliger", "liger"]

    cmd = bench.build_trainer_command(
        args,
        variant,
        experiments_dir=tmp_path / "runs",
        kernel_variant=kernel_variants[1],
    )
    assert "--use_liger_kernel" in cmd
    assert "--no-use_liger_kernel" not in cmd


def test_build_trainer_command_allows_learning_probe_hooks(tmp_path: Path) -> None:
    args = bench.build_parser().parse_args(
        [
            "--data_dir",
            "/tmp/data",
            "--launcher",
            "python",
            "--variants",
            "gpt2",
            "--max_train_steps",
            "5000",
            "--eval_every",
            "250",
            "--ewok_every",
            "250",
            "--save_every",
            "1000",
            "--save_final_checkpoint",
            "--no-skip_final_ewok",
        ]
    )
    variant = bench.build_variants(args)[0]
    cmd = bench.build_trainer_command(args, variant, experiments_dir=tmp_path / "runs")

    assert cmd[cmd.index("--max_train_steps") + 1] == "5000"
    assert cmd[cmd.index("--eval_every") + 1] == "250"
    assert cmd[cmd.index("--ewok_every") + 1] == "250"
    assert cmd[cmd.index("--save_every") + 1] == "1000"
    assert "--save_final_checkpoint" in cmd
    assert "--no-save_final_checkpoint" not in cmd
    assert "--skip_final_ewok" not in cmd


def test_build_trainer_command_allows_muon_optimizer_variant(tmp_path: Path) -> None:
    args = bench.build_parser().parse_args(
        [
            "--data_dir",
            "/tmp/data",
            "--launcher",
            "python",
            "--variants",
            "gpt2",
            "--optimizers",
            "adamw,muon_pe",
            "--muon_lr",
            "0.03",
            "--muon_ns_steps",
            "5",
            "--profile_optimizer_steps",
        ]
    )
    variant = bench.build_variants(args)[0]
    optimizer_variant = bench.build_optimizer_variants(args)[1]
    cmd = bench.build_trainer_command(
        args,
        variant,
        experiments_dir=tmp_path / "runs",
        optimizer_variant=optimizer_variant,
    )

    assert cmd[cmd.index("--optimizer") + 1] == "muon_pe"
    assert cmd[cmd.index("--muon_lr") + 1] == "0.03"
    assert cmd[cmd.index("--muon_ns_steps") + 1] == "5"
    assert "--muon_nesterov" in cmd
    assert "--muon_split_qkv" in cmd
    assert "--muon_batch_updates" in cmd
    assert "--profile_optimizer_steps" in cmd


def test_summarize_run_uses_post_warmup_throughput(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "run_config.json",
        {
            "config": {
                "model_arch": "llama",
                "n_embd": 32,
                "n_head": 4,
                "n_layer": 2,
                "llama_intermediate_size": 128,
                "llama_num_key_value_heads": 4,
                "micro_batch_size": 2,
                "seq_len": 8,
                "use_liger_kernel": True,
            }
        },
    )
    _append_jsonl(
        run_dir / "scalars.jsonl",
        [
            {
                "type": "scalars",
                "step": 1,
                "timestamp": "2026-05-21T00:00:00",
                "tokens_seen_global_approx": 100,
                "optimizer_step_ms": 1.0,
                "step_wall_ms": 1000.0,
                "cuda_max_memory_allocated_mb": 101.0,
                "cuda_max_memory_reserved_mb": 151.0,
                "world_size": 1,
                "micro_batch_size": 2,
                "seq_len": 8,
                "train_loss_opt_step_mean": 7.0,
            },
            {
                "type": "scalars",
                "step": 2,
                "timestamp": "2026-05-21T00:00:10",
                "tokens_seen_global_approx": 300,
                "optimizer_step_ms": 2.0,
                "step_wall_ms": 200.0,
                "cuda_max_memory_allocated_mb": 202.0,
                "cuda_max_memory_reserved_mb": 252.0,
                "world_size": 1,
                "micro_batch_size": 2,
                "seq_len": 8,
                "train_loss_opt_step_mean": 6.5,
            },
            {
                "type": "scalars",
                "step": 3,
                "timestamp": "2026-05-21T00:00:20",
                "tokens_seen_global_approx": 500,
                "optimizer_step_ms": 3.0,
                "step_wall_ms": 300.0,
                "cuda_max_memory_allocated_mb": 303.0,
                "cuda_max_memory_reserved_mb": 353.0,
                "world_size": 1,
                "micro_batch_size": 2,
                "seq_len": 8,
                "train_loss_opt_step_mean": 6.0,
            },
        ],
    )

    summary = bench.summarize_run(run_dir, label="probe", warmup_steps=1, compile_window_steps=2)
    assert summary.label == "probe"
    assert summary.model_arch == "llama"
    assert summary.use_liger_kernel is True
    assert summary.measured_steps == 2
    assert summary.tokens_delta == 200
    assert summary.elapsed_seconds == 10.0
    assert summary.tokens_per_sec == 20.0
    assert summary.median_step_tokens_per_sec == 20.0
    assert summary.compile_window_steps == 2
    assert summary.compile_window_measured_steps == 2
    assert summary.compile_window_total_step_wall_seconds == pytest.approx(1.2)
    assert summary.compile_window_mean_step_wall_ms == 600.0
    assert summary.compile_window_mean_optimizer_step_ms == 1.5
    assert summary.post_warmup_measured_steps == 2
    assert summary.post_warmup_total_step_wall_seconds == pytest.approx(0.5)
    assert summary.post_warmup_mean_step_wall_ms == 250.0
    assert summary.max_cuda_memory_allocated_mb == 303.0
    assert summary.max_cuda_memory_reserved_mb == 353.0
    assert summary.final_train_loss == 6.0
