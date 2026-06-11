"""Run and summarize GPT-2-vs-Llama architecture throughput probes.

This harness keeps the training data/tokenizer fixed and varies only the
from-scratch model architecture exposed by the BOS research trainer.
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
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER = REPO_ROOT / "research" / "bos_aligned_proto" / "training" / "trainer.py"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "research" / "bos_aligned_proto" / "architecture_speed"


@dataclass(frozen=True)
class Variant:
    label: str
    model_arch: str
    llama_intermediate_size: int = 0
    llama_num_key_value_heads: int = 0
    llama_tie_word_embeddings: bool = True
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class OptimizerVariant:
    label: str
    optimizer: str
    muon_lr: float
    muon_momentum: float
    muon_weight_decay: float
    muon_ns_steps: int
    muon_nesterov: bool
    muon_split_qkv: bool
    muon_batch_updates: bool


@dataclass(frozen=True)
class KernelVariant:
    label: str
    use_liger_kernel: bool


@dataclass(frozen=True)
class ThroughputSummary:
    label: str
    run_dir: str
    model_arch: str
    optimizer: str
    use_liger_kernel: bool | None
    muon_lr: float | None
    muon_momentum: float | None
    muon_weight_decay: float | None
    muon_ns_steps: int | None
    n_embd: int | None
    n_head: int | None
    n_layer: int | None
    llama_intermediate_size: int | None
    llama_num_key_value_heads: int | None
    world_size: int | None
    micro_batch_size: int | None
    seq_len: int | None
    max_step: int | None
    measured_steps: int
    warmup_steps: int
    compile_window_steps: int
    elapsed_seconds: float | None
    tokens_delta: int | None
    tokens_per_sec: float | None
    median_step_tokens_per_sec: float | None
    mean_step_seconds: float | None
    mean_optimizer_step_ms: float | None
    max_cuda_memory_allocated_mb: float | None
    max_cuda_memory_reserved_mb: float | None
    compile_window_measured_steps: int
    compile_window_total_step_wall_seconds: float | None
    compile_window_mean_step_wall_ms: float | None
    compile_window_median_step_wall_ms: float | None
    compile_window_p95_step_wall_ms: float | None
    compile_window_mean_optimizer_step_ms: float | None
    post_warmup_measured_steps: int
    post_warmup_total_step_wall_seconds: float | None
    post_warmup_mean_step_wall_ms: float | None
    post_warmup_median_step_wall_ms: float | None
    post_warmup_p95_step_wall_ms: float | None
    final_train_loss: float | None


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_scalars(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if row.get("type") == "scalars":
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("step", 0)))
    return rows


def config_from_run(run_dir: Path) -> dict:
    run_config = run_dir / "run_config.json"
    if not run_config.exists():
        return {}
    payload = load_json(run_config)
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        return cfg
    args = payload.get("args")
    if isinstance(args, dict):
        return args
    return {}


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(v) for v in values) / len(values))


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    index = int(round(0.95 * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return None


def _step_wall_stats(rows: Sequence[dict]) -> dict[str, float | int | None]:
    step_wall_ms = [
        value
        for value in (_float_or_none(row.get("step_wall_ms")) for row in rows)
        if value is not None
    ]
    optimizer_step_ms = [
        value
        for value in (_float_or_none(row.get("optimizer_step_ms")) for row in rows)
        if value is not None
    ]
    return {
        "measured_steps": len(step_wall_ms),
        "total_step_wall_seconds": (
            float(sum(step_wall_ms) / 1000.0) if step_wall_ms else None
        ),
        "mean_step_wall_ms": _mean(step_wall_ms),
        "median_step_wall_ms": _median(step_wall_ms),
        "p95_step_wall_ms": _p95(step_wall_ms),
        "mean_optimizer_step_ms": _mean(optimizer_step_ms),
    }


def _max_scalar(rows: Sequence[dict], key: str) -> float | None:
    values = [
        value
        for value in (_float_or_none(row.get(key)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return float(max(values))


def summarize_run(
    run_dir: str | Path,
    *,
    label: str | None = None,
    warmup_steps: int = 20,
    compile_window_steps: int = 20,
) -> ThroughputSummary:
    run_path = Path(run_dir).expanduser().resolve()
    scalars_path = run_path / "scalars.jsonl"
    if not scalars_path.exists():
        raise FileNotFoundError(f"Missing scalars.jsonl: {scalars_path}")

    rows = load_scalars(scalars_path)
    measured = [row for row in rows if int(row.get("step", 0)) > int(warmup_steps)]
    if len(measured) < 2:
        measured = rows

    cfg = config_from_run(run_path)
    resolved_label = label or str(cfg.get("model_arch") or run_path.name)
    compile_window_rows = [
        row for row in rows if int(row.get("step", 0)) <= int(compile_window_steps)
    ]
    post_warmup_rows = [row for row in rows if int(row.get("step", 0)) > int(warmup_steps)]
    compile_window_stats = _step_wall_stats(compile_window_rows)
    post_warmup_stats = _step_wall_stats(post_warmup_rows)

    elapsed_seconds = None
    tokens_delta = None
    tokens_per_sec = None
    mean_step_seconds = None
    mean_optimizer_step_ms = None
    median_step_tokens_per_sec = None

    if len(measured) >= 2:
        first = measured[0]
        last = measured[-1]
        t0 = parse_iso_timestamp(first["timestamp"])
        t1 = parse_iso_timestamp(last["timestamp"])
        elapsed_seconds = max((t1 - t0).total_seconds(), 0.0)
        tok0 = _int_or_none(first.get("tokens_seen_global_approx"))
        tok1 = _int_or_none(last.get("tokens_seen_global_approx"))
        if tok0 is not None and tok1 is not None:
            tokens_delta = int(tok1 - tok0)
            if elapsed_seconds > 0:
                tokens_per_sec = float(tokens_delta / elapsed_seconds)

        interval_tps: list[float] = []
        interval_seconds: list[float] = []
        for prev, curr in zip(measured, measured[1:]):
            prev_t = parse_iso_timestamp(prev["timestamp"])
            curr_t = parse_iso_timestamp(curr["timestamp"])
            dt = max((curr_t - prev_t).total_seconds(), 0.0)
            prev_tok = _int_or_none(prev.get("tokens_seen_global_approx"))
            curr_tok = _int_or_none(curr.get("tokens_seen_global_approx"))
            if dt > 0:
                interval_seconds.append(dt)
                if prev_tok is not None and curr_tok is not None:
                    interval_tps.append(float((curr_tok - prev_tok) / dt))
        median_step_tokens_per_sec = _median(interval_tps)
        if interval_seconds:
            mean_step_seconds = float(sum(interval_seconds) / len(interval_seconds))
        optimizer_step_ms_values = [
            value
            for value in (_float_or_none(row.get("optimizer_step_ms")) for row in measured)
            if value is not None
        ]
        if optimizer_step_ms_values:
            mean_optimizer_step_ms = float(
                sum(optimizer_step_ms_values) / len(optimizer_step_ms_values)
            )

    last_row = rows[-1] if rows else {}
    return ThroughputSummary(
        label=resolved_label,
        run_dir=str(run_path),
        model_arch=str(cfg.get("model_arch") or last_row.get("model_arch") or ""),
        optimizer=str(cfg.get("optimizer") or "adamw"),
        use_liger_kernel=_bool_or_none(cfg.get("use_liger_kernel")),
        muon_lr=_float_or_none(cfg.get("muon_lr")),
        muon_momentum=_float_or_none(cfg.get("muon_momentum")),
        muon_weight_decay=_float_or_none(cfg.get("muon_weight_decay")),
        muon_ns_steps=_int_or_none(cfg.get("muon_ns_steps")),
        n_embd=_int_or_none(cfg.get("n_embd")),
        n_head=_int_or_none(cfg.get("n_head")),
        n_layer=_int_or_none(cfg.get("n_layer")),
        llama_intermediate_size=_int_or_none(cfg.get("llama_intermediate_size")),
        llama_num_key_value_heads=_int_or_none(cfg.get("llama_num_key_value_heads")),
        world_size=_int_or_none(last_row.get("world_size")),
        micro_batch_size=_int_or_none(last_row.get("micro_batch_size") or cfg.get("micro_batch_size")),
        seq_len=_int_or_none(last_row.get("seq_len") or cfg.get("seq_len")),
        max_step=_int_or_none(last_row.get("step")),
        measured_steps=len(measured),
        warmup_steps=int(warmup_steps),
        compile_window_steps=int(compile_window_steps),
        elapsed_seconds=elapsed_seconds,
        tokens_delta=tokens_delta,
        tokens_per_sec=tokens_per_sec,
        median_step_tokens_per_sec=median_step_tokens_per_sec,
        mean_step_seconds=mean_step_seconds,
        mean_optimizer_step_ms=mean_optimizer_step_ms,
        max_cuda_memory_allocated_mb=_max_scalar(rows, "cuda_max_memory_allocated_mb"),
        max_cuda_memory_reserved_mb=_max_scalar(rows, "cuda_max_memory_reserved_mb"),
        compile_window_measured_steps=int(compile_window_stats["measured_steps"] or 0),
        compile_window_total_step_wall_seconds=_float_or_none(
            compile_window_stats["total_step_wall_seconds"]
        ),
        compile_window_mean_step_wall_ms=_float_or_none(
            compile_window_stats["mean_step_wall_ms"]
        ),
        compile_window_median_step_wall_ms=_float_or_none(
            compile_window_stats["median_step_wall_ms"]
        ),
        compile_window_p95_step_wall_ms=_float_or_none(
            compile_window_stats["p95_step_wall_ms"]
        ),
        compile_window_mean_optimizer_step_ms=_float_or_none(
            compile_window_stats["mean_optimizer_step_ms"]
        ),
        post_warmup_measured_steps=int(post_warmup_stats["measured_steps"] or 0),
        post_warmup_total_step_wall_seconds=_float_or_none(
            post_warmup_stats["total_step_wall_seconds"]
        ),
        post_warmup_mean_step_wall_ms=_float_or_none(
            post_warmup_stats["mean_step_wall_ms"]
        ),
        post_warmup_median_step_wall_ms=_float_or_none(
            post_warmup_stats["median_step_wall_ms"]
        ),
        post_warmup_p95_step_wall_ms=_float_or_none(
            post_warmup_stats["p95_step_wall_ms"]
        ),
        final_train_loss=_float_or_none(last_row.get("train_loss_opt_step_mean")),
    )


def write_summary(output_dir: Path, summaries: Sequence[ThroughputSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(summary) for summary in summaries]
    atomic_write_json(output_dir / "architecture_speed_summary.json", {"runs": payload})
    csv_path = output_dir / "architecture_speed_summary.csv"
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(summaries[0]).keys()) if summaries else list(ThroughputSummary.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in payload:
            writer.writerow(summary)
    os.replace(tmp_path, csv_path)


def rounded_param_matched_intermediate(n_embd: int, *, multiple: int) -> int:
    raw = (8.0 / 3.0) * int(n_embd)
    if int(multiple) <= 1:
        return int(round(raw))
    return int(round(raw / int(multiple)) * int(multiple))


def build_variants(args: argparse.Namespace) -> list[Variant]:
    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    variants: list[Variant] = []
    for name in requested:
        if name == "gpt2":
            variants.append(Variant(label="gpt2", model_arch="gpt2"))
        elif name == "llama_shape":
            variants.append(
                Variant(
                    label="llama_shape",
                    model_arch="llama",
                    llama_intermediate_size=int(args.llama_shape_intermediate_size),
                    llama_num_key_value_heads=int(args.llama_num_key_value_heads),
                    llama_tie_word_embeddings=bool(args.llama_tie_word_embeddings),
                    rope_theta=float(args.rope_theta),
                )
            )
        elif name == "llama_param":
            intermediate = int(args.llama_param_intermediate_size)
            if intermediate <= 0:
                intermediate = rounded_param_matched_intermediate(
                    int(args.n_embd),
                    multiple=int(args.llama_param_round_multiple),
                )
            variants.append(
                Variant(
                    label=f"llama_param_i{intermediate}",
                    model_arch="llama",
                    llama_intermediate_size=intermediate,
                    llama_num_key_value_heads=int(args.llama_num_key_value_heads),
                    llama_tie_word_embeddings=bool(args.llama_tie_word_embeddings),
                    rope_theta=float(args.rope_theta),
                )
            )
        else:
            raise ValueError(f"Unknown variant {name!r}; expected gpt2,llama_shape,llama_param.")
    return variants


def build_optimizer_variants(args: argparse.Namespace) -> list[OptimizerVariant]:
    requested = [value.strip() for value in args.optimizers.split(",") if value.strip()]
    variants: list[OptimizerVariant] = []
    for name in requested:
        if name not in {"adamw", "muon_pe"}:
            raise ValueError(f"Unknown optimizer {name!r}; expected adamw,muon_pe.")
        variants.append(
            OptimizerVariant(
                label=name,
                optimizer=name,
                muon_lr=float(args.muon_lr),
                muon_momentum=float(args.muon_momentum),
                muon_weight_decay=float(args.muon_weight_decay),
                muon_ns_steps=int(args.muon_ns_steps),
                muon_nesterov=bool(args.muon_nesterov),
                muon_split_qkv=bool(args.muon_split_qkv),
                muon_batch_updates=bool(args.muon_batch_updates),
            )
        )
    return variants


def build_kernel_variants(args: argparse.Namespace, variant: Variant) -> list[KernelVariant]:
    requested = [value.strip().lower() for value in args.liger_modes.split(",") if value.strip()]
    variants: list[KernelVariant] = []
    for name in requested:
        if name in {"off", "false", "0", "none", "no"}:
            variants.append(KernelVariant(label="noliger", use_liger_kernel=False))
        elif name in {"on", "true", "1", "liger", "yes"}:
            if variant.model_arch == "llama":
                variants.append(KernelVariant(label="liger", use_liger_kernel=True))
        else:
            raise ValueError(f"Unknown Liger mode {name!r}; expected off,on.")
    if not variants:
        variants.append(KernelVariant(label="noliger", use_liger_kernel=False))
    deduped: list[KernelVariant] = []
    seen: set[bool] = set()
    for item in variants:
        if item.use_liger_kernel in seen:
            continue
        seen.add(item.use_liger_kernel)
        deduped.append(item)
    return deduped


def launcher_prefix(args: argparse.Namespace) -> list[str]:
    if args.launcher == "python":
        return [sys.executable]
    accelerate_bin = shutil.which("accelerate")
    if accelerate_bin is None:
        raise FileNotFoundError("Could not find 'accelerate' on PATH for --launcher=accelerate.")
    prefix = [accelerate_bin, "launch", "--num_processes", str(args.num_processes)]
    if args.main_process_port:
        prefix.extend(["--main_process_port", str(args.main_process_port)])
    return prefix


def build_trainer_command(
    args: argparse.Namespace,
    variant: Variant,
    *,
    experiments_dir: Path,
    optimizer_variant: OptimizerVariant | None = None,
    kernel_variant: KernelVariant | None = None,
) -> list[str]:
    if optimizer_variant is None:
        optimizer_variant = build_optimizer_variants(args)[0]
    if kernel_variant is None:
        kernel_variant = KernelVariant(label="noliger", use_liger_kernel=False)
    cmd = launcher_prefix(args)
    cmd.append(str(TRAINER))
    cmd.extend(
        [
            "--loader_kind",
            args.loader_kind,
            "--data_dir",
            args.data_dir,
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
            str(args.max_train_steps),
            "--seq_len",
            str(args.seq_len),
            "--vocab_size",
            str(args.vocab_size),
            "--model_arch",
            variant.model_arch,
            "--n_embd",
            str(args.n_embd),
            "--n_head",
            str(args.n_head),
            "--n_layer",
            str(args.n_layer),
            "--llama_intermediate_size",
            str(variant.llama_intermediate_size),
            "--llama_num_key_value_heads",
            str(variant.llama_num_key_value_heads),
            "--rope_theta",
            str(variant.rope_theta),
            "--num_workers",
            str(args.num_workers),
            "--learning_rate",
            str(args.learning_rate),
            "--warmup_iters",
            str(args.warmup_iters),
            "--learning_rate_decay_frac",
            str(args.learning_rate_decay_frac),
            "--optimizer",
            optimizer_variant.optimizer,
            "--weight_decay",
            str(args.weight_decay),
            "--beta1",
            str(args.beta1),
            "--beta2",
            str(args.beta2),
            "--muon_lr",
            str(optimizer_variant.muon_lr),
            "--muon_momentum",
            str(optimizer_variant.muon_momentum),
            "--muon_weight_decay",
            str(optimizer_variant.muon_weight_decay),
            "--muon_ns_steps",
            str(optimizer_variant.muon_ns_steps),
            "--grad_clip",
            str(args.grad_clip),
            "--eval_every",
            str(args.eval_every),
            "--hellaswag_every",
            str(args.hellaswag_every),
            "--core_every",
            str(args.core_every),
            "--ewok_every",
            str(args.ewok_every),
            "--save_every",
            str(args.save_every),
            "--exposure_every",
            str(args.exposure_every),
        ]
    )
    cmd.append("--save_final_checkpoint" if args.save_final_checkpoint else "--no-save_final_checkpoint")
    if args.skip_final_ewok:
        cmd.append("--skip_final_ewok")
    if args.source_data_dir:
        cmd.extend(["--source_data_dir", args.source_data_dir])
    if args.tokenizer_name_or_path:
        cmd.extend(["--tokenizer_name_or_path", args.tokenizer_name_or_path])
    if not variant.llama_tie_word_embeddings:
        cmd.append("--no-llama_tie_word_embeddings")
    cmd.append("--use_liger_kernel" if kernel_variant.use_liger_kernel else "--no-use_liger_kernel")
    cmd.append("--muon_nesterov" if optimizer_variant.muon_nesterov else "--no-muon_nesterov")
    cmd.append("--muon_split_qkv" if optimizer_variant.muon_split_qkv else "--no-muon_split_qkv")
    cmd.append("--muon_batch_updates" if optimizer_variant.muon_batch_updates else "--no-muon_batch_updates")
    cmd.append("--profile_optimizer_steps" if args.profile_optimizer_steps else "--no-profile_optimizer_steps")
    cmd.extend(args.trainer_extra_arg)
    return cmd


def find_single_run_dir(experiments_dir: Path) -> Path | None:
    if not experiments_dir.is_dir():
        return None
    children = sorted((path for path in experiments_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not children:
        return None
    return children[-1].resolve()


def run_command(cmd: Sequence[str], *, cwd: Path, label: str) -> int:
    print(f"[{label}] launching:")
    print("  " + shlex.join(cmd))
    proc = subprocess.run(list(cmd), cwd=str(cwd), check=False)
    print(f"[{label}] exit code: {proc.returncode}")
    return int(proc.returncode)


def _parse_labeled_run(value: str) -> tuple[str | None, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip() or None, path.strip()
    return None, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="", help="Training artifact data_dir passed to the BOS trainer.")
    parser.add_argument("--source_data_dir", type=str, default="", help="Raw token shard dir for bos_packed_index tokenizer metadata.")
    parser.add_argument("--loader_kind", type=str, default="bos_packed_index", choices=("stream", "bos_row", "bos_packed_index"))
    parser.add_argument("--tokenizer_name_or_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_RUN_ROOT / timestamp_slug()))
    parser.add_argument("--variants", type=str, default="gpt2,llama_shape,llama_param")
    parser.add_argument("--optimizers", type=str, default="adamw", help="Comma-separated optimizer variants: adamw,muon_pe.")
    parser.add_argument(
        "--liger_modes",
        type=str,
        default="off",
        help=(
            "Comma-separated Liger modes for Llama variants: off,on. "
            "Use off,on to compare startup-inclusive Triton compile cost and steady-state speed."
        ),
    )
    parser.add_argument("--launcher", type=str, default="accelerate", choices=("accelerate", "python"))
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--main_process_port", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Write commands/manifest without launching training.")
    parser.add_argument("--run_dir", action="append", default=[], help="Summarize an existing run dir, optionally LABEL=PATH.")
    parser.add_argument("--summarize_only", action="store_true", help="Only summarize --run_dir entries; do not launch training.")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument(
        "--compile_window_steps",
        type=int,
        default=20,
        help="First N optimizer steps summarized separately, including Triton first-use compile when present.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--total_batch_tokens", type=int, default=491520)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=0)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--llama_shape_intermediate_size", type=int, default=0, help="0 means LlamaConfig default in trainer: 4*n_embd.")
    parser.add_argument("--llama_param_intermediate_size", type=int, default=0, help="0 means rounded 8/3*n_embd.")
    parser.add_argument("--llama_param_round_multiple", type=int, default=256)
    parser.add_argument("--llama_num_key_value_heads", type=int, default=0)
    parser.add_argument("--llama_tie_word_embeddings", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--muon_batch_updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile_optimizer_steps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--hellaswag_every", type=int, default=0)
    parser.add_argument("--core_every", type=int, default=0)
    parser.add_argument("--ewok_every", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_final_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exposure_every", type=int, default=0)
    parser.add_argument("--skip_final_ewok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trainer_extra_arg", action="append", default=[], help="Extra raw arg token appended to each trainer command.")
    return parser


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "args": vars(args),
        "variants": [],
        "existing_runs": [],
    }

    summaries: list[ThroughputSummary] = []
    for run_arg in args.run_dir:
        label, path = _parse_labeled_run(run_arg)
        summary = summarize_run(
            path,
            label=label,
            warmup_steps=int(args.warmup_steps),
            compile_window_steps=int(args.compile_window_steps),
        )
        summaries.append(summary)
        manifest["existing_runs"].append(asdict(summary))

    if not args.summarize_only:
        if not args.data_dir:
            raise ValueError("--data_dir is required unless --summarize_only is set.")
        variants = build_variants(args)
        optimizer_variants = build_optimizer_variants(args)
        for variant in variants:
            for optimizer_variant in optimizer_variants:
                kernel_variants = build_kernel_variants(args, variant)
                for kernel_variant in kernel_variants:
                    label_parts = [variant.label]
                    if len(optimizer_variants) > 1:
                        label_parts.append(optimizer_variant.label)
                    if len(kernel_variants) > 1 or kernel_variant.use_liger_kernel:
                        label_parts.append(kernel_variant.label)
                    label = "_".join(label_parts)
                    variant_experiments_dir = output_dir / "runs" / label
                    cmd = build_trainer_command(
                        args,
                        variant,
                        experiments_dir=variant_experiments_dir,
                        optimizer_variant=optimizer_variant,
                        kernel_variant=kernel_variant,
                    )
                    record = {
                        "variant": asdict(variant),
                        "optimizer_variant": asdict(optimizer_variant),
                        "kernel_variant": asdict(kernel_variant),
                        "label": label,
                        "experiments_dir": str(variant_experiments_dir),
                        "command": cmd,
                        "command_text": shlex.join(cmd),
                        "returncode": None,
                        "run_dir": None,
                    }
                    if args.dry_run:
                        print(f"[{label}] dry run:")
                        print("  " + record["command_text"])
                        record["returncode"] = 0
                    else:
                        record["returncode"] = run_command(cmd, cwd=REPO_ROOT, label=label)
                        run_dir = find_single_run_dir(variant_experiments_dir)
                        record["run_dir"] = str(run_dir) if run_dir is not None else None
                        if int(record["returncode"]) == 0 and run_dir is not None:
                            summaries.append(
                                summarize_run(
                                    run_dir,
                                    label=label,
                                    warmup_steps=int(args.warmup_steps),
                                    compile_window_steps=int(args.compile_window_steps),
                                )
                            )
                    manifest["variants"].append(record)

    if summaries:
        write_summary(output_dir, summaries)
        manifest["summary_csv"] = str(output_dir / "architecture_speed_summary.csv")
        manifest["summary_json"] = str(output_dir / "architecture_speed_summary.json")
    atomic_write_json(output_dir / "architecture_speed_manifest.json", manifest)

    if summaries:
        print(f"Wrote summary: {output_dir / 'architecture_speed_summary.csv'}")
    print(f"Wrote manifest: {output_dir / 'architecture_speed_manifest.json'}")
    return 0 if all((item.get("returncode") in (None, 0)) for item in manifest["variants"]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
