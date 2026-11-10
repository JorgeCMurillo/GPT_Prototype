"""Short full-size ClimbMix training and numerical resume checks on idle GPUs.

Runs the actual trainer with 778,114,048 parameters, BF16, Liger, Muon,
microbatch 2, and T=1024. It is a software integration test, not a quality pilot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def trace_map(folder):
    result = {}
    for path in sorted(folder.glob("data_trace_rank*.jsonl")):
        for row in rows(path):
            key = (row["rank"], row["global_step"], row["micro_step_in_step"], row["batch_position"])
            require(key not in result, f"Duplicate trace key: {key}")
            result[key] = row["sha1"]
    return result


def weight_difference(reference, actual):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    maximum, difference_squared, reference_squared, count = 0.0, 0.0, 0.0, 0
    with safe_open(reference, framework="pt", device="cpu") as a, safe_open(actual, framework="pt", device="cpu") as b:
        require(set(a.keys()) == set(b.keys()), "Resume changed parameter names")
        for key in a.keys():
            left, right = a.get_tensor(key).flatten(), b.get_tensor(key).flatten()
            require(left.shape == right.shape, f"Resume shape mismatch: {key}")
            for start in range(0, left.numel(), 262144):
                x, y = left[start:start + 262144].float(), right[start:start + 262144].float()
                require(torch.isfinite(y).all().item(), f"Nonfinite resumed weights: {key}")
                difference = y - x
                maximum = max(maximum, difference.abs().max().item())
                difference_squared += difference.double().square().sum().item()
                reference_squared += x.double().square().sum().item()
                count += x.numel()
    relative_rms = math.sqrt(difference_squared / max(reference_squared, 1e-30))
    require(maximum < 5e-4 and relative_rms < 1e-4,
            f"Resume weights diverged: max={maximum}, relative_rms={relative_rms}")
    return {"max_absolute_difference": maximum, "relative_rms_difference": relative_rms,
            "serialized_parameter_elements_compared": count,
            "max_abs_limit": 5e-4, "relative_rms_limit": 1e-4}


def gpu_preflight(gpus):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    devices = {int(fields[0]): (int(fields[1]), int(fields[2]))
               for fields in (line.split(",") for line in output.splitlines())}
    for gpu in gpus:
        free, utilization = devices[gpu]
        require(free >= 29000 and utilization <= 10,
                f"GPU {gpu} is busy: {free} MiB free, {utilization}% utilization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--modes", nargs="+", choices=["sdpa", "flash_varlen", "flash_varlen_document"],
                        default=["sdpa", "flash_varlen", "flash_varlen_document"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--checkpoint-step", type=int, default=3)
    parser.add_argument("--grad-accum-steps", type=int, default=30)
    args = parser.parse_args()
    require(0 < args.checkpoint_step < args.steps < 2 * args.checkpoint_step,
            "Choose one periodic checkpoint strictly before the final step")
    require(args.grad_accum_steps > 0 and len(set(args.gpus)) == len(args.gpus), "Invalid batch/GPU configuration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/processed/climbmix_gpt2_10b"
    global_tokens = len(args.gpus) * 2 * 1024 * args.grad_accum_steps
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "EWOK_VARIANT": "fast"}
    report = {"kind": "integration_test_not_quality_pilot", "data_dir": str(data),
              "physical_gpus": args.gpus, "world_size": len(args.gpus),
              "parameters": 778114048, "sequence_length": 1024, "micro_batch_size": 2,
              "grad_accum_steps": args.grad_accum_steps, "tokens_per_step": global_tokens,
              "steps": args.steps, "resume_from_step": args.checkpoint_step, "modes": {}}
    write(args.output_dir / "report.json", report)
    reference_trace = None
    for mode in args.modes:
        gpu_preflight(args.gpus)
        mode_root = args.output_dir / mode
        mode_root.mkdir()
        common = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(len(args.gpus)),
            str(ROOT / "research/bos_aligned_proto/training/trainer.py"),
            "--loader_kind", "stream", "--data_dir", str(data), "--tokenizer_name_or_path", "gpt2",
            "--experiments_dir", str(mode_root / "runs"), "--seed", "42", "--mixed_precision", "bf16",
            "--micro_batch_size", "2", "--total_batch_tokens", str(global_tokens), "--seq_len", "1024",
            "--model_arch", "qwen3", "--n_embd", "1280", "--n_head", "20", "--n_layer", "36",
            "--qwen_intermediate_size", "4096", "--qwen_num_key_value_heads", "5", "--qwen_head_dim", "64",
            "--qwen_tie_word_embeddings", "--rope_theta", "1000000", "--use_liger_kernel",
            "--attention_mode", mode, "--num_workers", "0", "--learning_rate", "0.0006",
            "--warmup_iters", "700", "--learning_rate_decay_frac", "0.0", "--optimizer", "muon_pe",
            "--weight_decay", "0.1", "--beta1", "0.9", "--beta2", "0.95", "--muon_lr", "0.02",
            "--muon_momentum", "0.95", "--muon_weight_decay", "0.1", "--muon_ns_steps", "5",
            "--muon_nesterov", "--muon_split_qkv", "--muon_batch_updates", "--grad_clip", "1.0",
            "--max_train_steps", str(args.steps), "--eval_every", str(args.steps),
            "--hellaswag_every", "0", "--core_every", "0", "--ewok_every", "0", "--skip_final_ewok",
            "--exposure_every", "0", "--profile_optimizer_steps", "--debug_trace_data",
            "--debug_trace_steps", str(args.steps),
        ]

        def launch(phase, extra):
            command = common + extra
            write(mode_root / f"{phase}_command.json", {"argv": command, "environment_overrides": {
                key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS")}})
            print(f"START {mode} {phase}; log={mode_root / (phase + '.log')}", flush=True)
            start = time.monotonic()
            with (mode_root / f"{phase}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            require(result.returncode == 0, f"{mode} {phase} failed; see {mode_root / (phase + '.log')}")
            print(f"DONE {mode} {phase} in {time.monotonic() - start:.1f}s", flush=True)

        launch("uninterrupted", ["--save_every", str(args.checkpoint_step)])
        run_dirs = list((mode_root / "runs").iterdir())
        require(len(run_dirs) == 1, f"Expected one trainer run for {mode}")
        run = run_dirs[0]
        scalar_path = run / "scalars.jsonl"
        original_rows = rows(scalar_path)
        (mode_root / "uninterrupted_scalars.jsonl").write_text(scalar_path.read_text())
        original_steps = {r["step"]: r for r in original_rows if r.get("type") == "scalars"}
        require(set(original_steps) == set(range(1, args.steps + 1)), "Incomplete optimizer steps")
        fields = ["train_loss_opt_step_mean", "grad_norm_l2_preclip", "param_norm_l2", "step_wall_ms"]
        for r in original_steps.values():
            require(all(math.isfinite(float(r[key])) for key in fields), "Nonfinite training metric")
            require(r["attention_mode"] == mode, "Wrong recorded attention mode")
        original_trace = trace_map(run / "debug_data_traces/from_step0000000")
        require(len(original_trace) == args.steps * args.grad_accum_steps * 2 * len(args.gpus),
                "Incomplete training data traces")
        if reference_trace is None:
            reference_trace = original_trace
        require(original_trace == reference_trace, "Training tokens/order changed between attention arms")
        final = run / f"ckpt_final_step{args.steps:07d}"
        reference = mode_root / "uninterrupted_final"
        final.rename(reference)
        checkpoint = run / f"ckpt_periodic_step{args.checkpoint_step:07d}"
        launch("resumed", ["--save_every", "0", "--resume_from_run", str(checkpoint)])
        resumed_trace = trace_map(run / f"debug_data_traces/from_step{args.checkpoint_step:07d}")
        expected_trace = {key: value for key, value in original_trace.items() if key[1] > args.checkpoint_step}
        require(resumed_trace == expected_trace, "Resume changed training tokens/order")
        resumed_rows = rows(scalar_path)
        resumed_steps = {r["step"]: r for r in resumed_rows if r.get("type") == "scalars"}
        loss_difference = max(abs(resumed_steps[step]["train_loss_opt_step_mean"]
                                  - original_steps[step]["train_loss_opt_step_mean"])
                              for step in range(args.checkpoint_step + 1, args.steps + 1))
        require(loss_difference < 1e-4, f"Resume loss diverged by {loss_difference}")
        state = json.loads((final / "trainer_state.json").read_text())
        require(state["attention_mode"] == mode and state["opt_step"] == args.steps,
                "Resumed checkpoint metadata mismatch")
        require(state["micro_steps_seen"] == args.steps * args.grad_accum_steps, "Wrong resume microstep count")
        require(state["world_size"] == len(args.gpus), "Wrong resume world size")
        require((final / "optimizer.pt").is_file(), "Missing optimizer state")
        config = json.loads((final / "config.json").read_text())
        require(config["num_hidden_layers"] == 36 and config["hidden_size"] == 1280,
                "Integration run did not use the full model shape")
        differences = weight_difference(str(reference / "model.safetensors"), str(final / "model.safetensors"))
        require(differences["serialized_parameter_elements_compared"] == 778114048,
                "Did not compare all 778,114,048 model parameters")
        validation = [r for r in resumed_rows if r.get("type") == "val_loss"][-1]
        require(math.isfinite(validation["val_loss"]), "Nonfinite ordinary validation loss")
        if mode != "sdpa":
            require(math.isfinite(validation["document_isolated_val_loss"]), "Nonfinite isolated validation loss")
        median_ms = statistics.median(original_steps[step]["step_wall_ms"] for step in range(2, args.steps + 1))
        report["modes"][mode] = {
            "passed": True, "run_dir": str(run), "checkpoint": str(final),
            "steady_step_median_seconds": median_ms / 1000,
            "global_training_tokens_per_second": global_tokens / (median_ms / 1000),
            "rank0_peak_training_memory_mib": max(r["cuda_max_memory_allocated_mb"] for r in original_steps.values()),
            "final_train_loss": original_steps[args.steps]["train_loss_opt_step_mean"],
            "validation": validation, "data_traces_match_across_arms": True,
            "resumed_data_trace_matches": True, "resume_max_train_loss_difference": loss_difference,
            "resume_weight_comparison": differences,
        }
        write(args.output_dir / "report.json", report)
        print(json.dumps(report["modes"][mode]), flush=True)
    report["passed"] = all(r["passed"] for r in report["modes"].values())
    report["timing_scope"] = "Two-GPU short integration run; median of steps 2-5, includes metadata, model, backward, DDP, and optimizer; excludes validation and checkpoint I/O."
    write(args.output_dir / "report.json", report)
    print(f"ALL PASSED; report={args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
