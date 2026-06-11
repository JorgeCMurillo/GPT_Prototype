"""Run Muon batching speed checks with timing and GPU memory summaries.

The suite is intentionally small enough to run in a tmux session:

1. Optimizer-only synthetic Llama-shape benchmark:
   scalar Muon and exact-shape batching.
2. Real short training with exact-shape batching.
3. Real short training with scalar Muon.
4. Optional real short AdamW baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER = REPO_ROOT / "research" / "bos_aligned_proto" / "training" / "trainer.py"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "fineweb_edu_100B"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "research" / "bos_aligned_proto"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class SuiteRow:
    kind: str
    label: str
    output_path: str
    returncode: int | None = None
    measured_steps: int | None = None
    median_step_ms: float | None = None
    mean_step_ms: float | None = None
    p95_step_ms: float | None = None
    min_step_ms: float | None = None
    max_step_ms: float | None = None
    median_optimizer_step_ms: float | None = None
    mean_optimizer_step_ms: float | None = None
    median_step_seconds: float | None = None
    mean_step_seconds: float | None = None
    median_tokens_per_sec: float | None = None
    final_train_loss: float | None = None
    max_cuda_allocated_mb: float | None = None
    max_cuda_reserved_mb: float | None = None
    max_gpu_memory_used_mb: float | None = None


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def write_rows_csv(path: Path, rows: Sequence[SuiteRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(SuiteRow.__dataclass_fields__.keys())
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    os.replace(tmp_path, path)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(float(value) for value in values))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    index = int(round(0.95 * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _query_gpu_memory_mb(gpu: str) -> tuple[float, float] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class MemorySampler:
    def __init__(self, *, path: Path, gpu: str, label: str, interval_seconds: float) -> None:
        self.path = path
        self.gpu = str(gpu)
        self.label = label
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(["timestamp", "label", "gpu", "memory_used_mb", "memory_total_mb"])
        self._thread = threading.Thread(target=self._run, name=f"memory-sampler-{self.label}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _query_gpu_memory_mb(self.gpu)
            if sample is not None:
                used_mb, total_mb = sample
                with self.path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([datetime.now().isoformat(), self.label, self.gpu, used_mb, total_mb])
            self._stop.wait(self.interval_seconds)


def memory_max_by_label(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    max_by_label: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("label") or "")
            used = _float_or_none(row.get("memory_used_mb"))
            if not label or used is None:
                continue
            max_by_label[label] = max(float(used), max_by_label.get(label, 0.0))
    return max_by_label


def build_synthetic_llama_params(
    *,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    device: str,
    dtype,
) -> list:
    import torch

    shapes: list[tuple[int, int]] = []
    for _ in range(int(layers)):
        shapes.extend([(hidden_size, hidden_size)] * 4)
        shapes.extend([(intermediate_size, hidden_size)] * 2)
        shapes.append((hidden_size, intermediate_size))

    params = [
        torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype) * 0.02)
        for shape in shapes
    ]
    for param in params:
        param.grad = torch.randn_like(param)
    return params


def run_optimizer_microbench(args: argparse.Namespace, output_dir: Path, memory_csv: Path) -> SuiteRow:
    import torch
    from training_utils.muon_polar_express import MuonWithAuxAdamPE

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[str(args.microbench_dtype)]
    mode_results: list[dict] = []
    modes = ("scalar", "exact_shape")
    with MemorySampler(
        path=memory_csv,
        gpu=str(args.gpu),
        label="optimizer_microbench",
        interval_seconds=float(args.memory_sample_interval),
    ):
        for mode in modes:
            torch.manual_seed(int(args.seed))
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            params = build_synthetic_llama_params(
                hidden_size=int(args.n_embd),
                intermediate_size=int(args.llama_intermediate_size),
                layers=int(args.n_layer),
                device=device,
                dtype=dtype,
            )
            optimizer = MuonWithAuxAdamPE(
                [
                    {
                        "params": params,
                        "use_muon": True,
                        "lr": float(args.muon_lr),
                        "base_lr": float(args.muon_lr),
                        "momentum": float(args.muon_momentum),
                        "weight_decay": float(args.muon_weight_decay),
                        "ns_steps": int(args.muon_ns_steps),
                        "nesterov": bool(args.muon_nesterov),
                    }
                ],
                qkv_split_dims={},
                batch_muon_updates=(mode != "scalar"),
            )

            for _ in range(int(args.microbench_warmup_steps)):
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()

            timings_ms: list[float] = []
            for _ in range(int(args.microbench_steps)):
                if device == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - started) * 1000.0)

            peak_allocated_mb = None
            peak_reserved_mb = None
            if device == "cuda":
                peak_allocated_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
                peak_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024**2))
            mode_results.append(
                {
                    "mode": mode,
                    "measured_steps": len(timings_ms),
                    "median_step_ms": _median(timings_ms),
                    "mean_step_ms": _mean(timings_ms),
                    "p95_step_ms": _p95(timings_ms),
                    "min_step_ms": min(timings_ms) if timings_ms else None,
                    "max_step_ms": max(timings_ms) if timings_ms else None,
                    "max_cuda_allocated_mb": peak_allocated_mb,
                    "max_cuda_reserved_mb": peak_reserved_mb,
                }
            )
            del optimizer, params
            if device == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "created_at": datetime.now().isoformat(),
        "device": device,
        "dtype": str(args.microbench_dtype),
        "hidden_size": int(args.n_embd),
        "intermediate_size": int(args.llama_intermediate_size),
        "layers": int(args.n_layer),
        "warmup_steps": int(args.microbench_warmup_steps),
        "measured_steps": int(args.microbench_steps),
        "modes": mode_results,
    }
    out_path = output_dir / "optimizer_microbench.json"
    atomic_write_json(out_path, payload)

    return SuiteRow(
        kind="optimizer_microbench",
        label="optimizer_microbench",
        output_path=str(out_path),
        measured_steps=int(args.microbench_steps),
    )


def trainer_base_command(args: argparse.Namespace, experiments_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--loader_kind",
        str(args.loader_kind),
        "--data_dir",
        str(args.data_dir),
        "--experiments_dir",
        str(experiments_dir),
        "--seed",
        str(args.seed),
        "--mixed_precision",
        str(args.mixed_precision),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--total_batch_tokens",
        str(args.total_batch_tokens),
        "--max_train_steps",
        str(args.training_steps),
        "--seq_len",
        str(args.seq_len),
        "--vocab_size",
        str(args.vocab_size),
        "--model_arch",
        "llama",
        "--n_embd",
        str(args.n_embd),
        "--n_head",
        str(args.n_head),
        "--n_layer",
        str(args.n_layer),
        "--llama_intermediate_size",
        str(args.llama_intermediate_size),
        "--llama_num_key_value_heads",
        str(args.llama_num_key_value_heads),
        "--rope_theta",
        str(args.rope_theta),
        "--num_workers",
        str(args.num_workers),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_iters",
        str(args.warmup_iters),
        "--learning_rate_decay_frac",
        str(args.learning_rate_decay_frac),
        "--weight_decay",
        str(args.weight_decay),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--muon_lr",
        str(args.muon_lr),
        "--muon_momentum",
        str(args.muon_momentum),
        "--muon_weight_decay",
        str(args.muon_weight_decay),
        "--muon_ns_steps",
        str(args.muon_ns_steps),
        "--grad_clip",
        str(args.grad_clip),
        "--eval_every",
        "0",
        "--hellaswag_every",
        "0",
        "--core_every",
        "0",
        "--ewok_every",
        "0",
        "--save_every",
        "0",
        "--exposure_every",
        "0",
        "--no-save_final_checkpoint",
        "--skip_final_ewok",
        "--profile_optimizer_steps",
    ]


def build_training_command(args: argparse.Namespace, *, label: str, experiments_dir: Path) -> list[str]:
    cmd = trainer_base_command(args, experiments_dir)
    if args.source_data_dir:
        cmd.extend(["--source_data_dir", str(args.source_data_dir)])
    if args.tokenizer_name_or_path:
        cmd.extend(["--tokenizer_name_or_path", str(args.tokenizer_name_or_path)])

    if label == "train_adamw":
        cmd.extend(["--optimizer", "adamw"])
    else:
        cmd.extend(
            [
                "--optimizer",
                "muon_pe",
                "--muon_nesterov" if args.muon_nesterov else "--no-muon_nesterov",
                "--muon_split_qkv" if args.muon_split_qkv else "--no-muon_split_qkv",
            ]
        )
        cmd.append("--muon_batch_updates" if label == "train_muon_batched" else "--no-muon_batch_updates")
    return cmd


def find_single_run_dir(experiments_dir: Path) -> Path | None:
    if not experiments_dir.is_dir():
        return None
    children = sorted(
        (path for path in experiments_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not children:
        return None
    return children[-1].resolve()


def load_scalars(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "scalars":
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("step", 0)))
    return rows


def summarize_training_run(run_dir: Path, *, label: str, warmup_steps: int, returncode: int) -> SuiteRow:
    scalars_path = run_dir / "scalars.jsonl"
    if not scalars_path.exists():
        return SuiteRow(
            kind="training",
            label=label,
            output_path=str(run_dir),
            returncode=returncode,
        )
    rows = load_scalars(scalars_path)
    measured = [row for row in rows if int(row.get("step", 0)) > int(warmup_steps)]
    if len(measured) < 2:
        measured = rows

    optimizer_ms = [
        value
        for value in (_float_or_none(row.get("optimizer_step_ms")) for row in measured)
        if value is not None
    ]

    step_seconds: list[float] = []
    tokens_per_sec: list[float] = []
    for prev, curr in zip(measured, measured[1:]):
        dt = max((_parse_timestamp(curr["timestamp"]) - _parse_timestamp(prev["timestamp"])).total_seconds(), 0.0)
        if dt <= 0:
            continue
        step_seconds.append(float(dt))
        prev_tokens = _float_or_none(prev.get("tokens_seen_global_approx"))
        curr_tokens = _float_or_none(curr.get("tokens_seen_global_approx"))
        if prev_tokens is not None and curr_tokens is not None:
            tokens_per_sec.append(float((curr_tokens - prev_tokens) / dt))

    final_loss = _float_or_none(rows[-1].get("train_loss_opt_step_mean")) if rows else None
    return SuiteRow(
        kind="training",
        label=label,
        output_path=str(run_dir),
        returncode=returncode,
        measured_steps=len(measured),
        median_optimizer_step_ms=_median(optimizer_ms),
        mean_optimizer_step_ms=_mean(optimizer_ms),
        median_step_seconds=_median(step_seconds),
        mean_step_seconds=_mean(step_seconds),
        median_tokens_per_sec=_median(tokens_per_sec),
        final_train_loss=final_loss,
    )


def run_training_variant(
    args: argparse.Namespace,
    *,
    label: str,
    output_dir: Path,
    memory_csv: Path,
) -> SuiteRow:
    experiments_dir = output_dir / "runs" / label
    cmd = build_training_command(args, label=label, experiments_dir=experiments_dir)
    log_path = output_dir / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(shlex.join(cmd) + "\n\n")
        log_handle.flush()
        if args.dry_run:
            return SuiteRow(
                kind="training",
                label=label,
                output_path=str(log_path),
                returncode=None,
            )
        with MemorySampler(
            path=memory_csv,
            gpu=str(args.gpu),
            label=label,
            interval_seconds=float(args.memory_sample_interval),
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = int(proc.wait())

    run_dir = find_single_run_dir(experiments_dir)
    if run_dir is None:
        return SuiteRow(
            kind="training",
            label=label,
            output_path=str(log_path),
            returncode=returncode,
        )
    return summarize_training_run(
        run_dir,
        label=label,
        warmup_steps=int(args.warmup_steps),
        returncode=returncode,
    )


def attach_memory_summaries(rows: Sequence[SuiteRow], memory_csv: Path) -> list[SuiteRow]:
    memory_by_label = memory_max_by_label(memory_csv)
    updated: list[SuiteRow] = []
    microbench_payload: dict | None = None
    for row in rows:
        data = asdict(row)
        data["max_gpu_memory_used_mb"] = memory_by_label.get(row.label)
        if row.kind == "optimizer_microbench":
            data["max_gpu_memory_used_mb"] = memory_by_label.get("optimizer_microbench")
            microbench_path = Path(row.output_path)
            if microbench_path.exists():
                with microbench_path.open("r", encoding="utf-8") as handle:
                    microbench_payload = json.load(handle)
        updated.append(SuiteRow(**data))

    if microbench_payload:
        for mode in microbench_payload.get("modes", []):
            updated.append(
                SuiteRow(
                    kind="optimizer_microbench_mode",
                    label=str(mode.get("mode")),
                    output_path=str(rows[0].output_path) if rows else "",
                    measured_steps=mode.get("measured_steps"),
                    median_step_ms=mode.get("median_step_ms"),
                    mean_step_ms=mode.get("mean_step_ms"),
                    p95_step_ms=mode.get("p95_step_ms"),
                    min_step_ms=mode.get("min_step_ms"),
                    max_step_ms=mode.get("max_step_ms"),
                    max_cuda_allocated_mb=mode.get("max_cuda_allocated_mb"),
                    max_cuda_reserved_mb=mode.get("max_cuda_reserved_mb"),
                    max_gpu_memory_used_mb=memory_by_label.get("optimizer_microbench"),
                )
            )
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / f"muon_batch_speed_suite_{timestamp_slug()}"),
    )
    parser.add_argument("--gpu", type=str, default=os.environ.get("MUON_BENCH_GPU", "4"))
    parser.add_argument("--memory_sample_interval", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_microbench", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--include_adamw_training", action="store_true")

    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--source_data_dir", type=str, default="")
    parser.add_argument("--loader_kind", type=str, default="stream", choices=("stream", "bos_row", "bos_packed_index"))
    parser.add_argument("--tokenizer_name_or_path", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--total_batch_tokens", type=int, default=81920)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=0)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--llama_intermediate_size", type=int, default=2816)
    parser.add_argument("--llama_num_key_value_heads", type=int, default=0)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_iters", type=int, default=700)
    parser.add_argument("--learning_rate_decay_frac", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.1)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon_split_qkv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--microbench_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--microbench_warmup_steps", type=int, default=5)
    parser.add_argument("--microbench_steps", type=int, default=20)
    return parser


def run_suite(args: argparse.Namespace) -> int:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_csv = output_dir / "memory_samples.csv"

    manifest = {
        "created_at": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "args": vars(args),
        "commands": {},
    }
    rows: list[SuiteRow] = []

    if not args.skip_microbench:
        if args.dry_run:
            rows.append(
                SuiteRow(
                    kind="optimizer_microbench",
                    label="optimizer_microbench",
                    output_path=str(output_dir / "optimizer_microbench.json"),
                    returncode=None,
                )
            )
        else:
            rows.append(run_optimizer_microbench(args, output_dir, memory_csv))

    if not args.skip_training:
        labels = ["train_muon_batched", "train_muon_scalar"]
        if args.include_adamw_training:
            labels.append("train_adamw")
        for label in labels:
            cmd = build_training_command(args, label=label, experiments_dir=output_dir / "runs" / label)
            manifest["commands"][label] = shlex.join(cmd)
            rows.append(run_training_variant(args, label=label, output_dir=output_dir, memory_csv=memory_csv))

    rows = attach_memory_summaries(rows, memory_csv)
    summary_json = output_dir / "muon_batch_speed_suite_summary.json"
    summary_csv = output_dir / "muon_batch_speed_suite_summary.csv"
    atomic_write_json(summary_json, {"runs": [asdict(row) for row in rows]})
    write_rows_csv(summary_csv, rows)
    manifest["summary_json"] = str(summary_json)
    manifest["summary_csv"] = str(summary_csv)
    manifest["memory_csv"] = str(memory_csv)
    atomic_write_json(output_dir / "muon_batch_speed_suite_manifest.json", manifest)

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote manifest: {output_dir / 'muon_batch_speed_suite_manifest.json'}")
    print(f"Wrote memory samples: {memory_csv}")

    failed = [row for row in rows if row.returncode not in (None, 0)]
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
