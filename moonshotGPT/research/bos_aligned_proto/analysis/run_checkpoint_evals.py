#!/usr/bin/env python3
"""Run post-hoc BOS benchmark evaluations for a resolved checkpoint.

This script is meant for the "analysis" layer of the BOS prototype: given a
finished BOS run directory or a specific checkpoint directory, resolve one
checkpoint, load the model/tokenizer, and run standalone evaluations in a
deliberate order:

1. CORE
2. HellaSwag
3. EWoK
4. BLiMP

Artifacts are written under:

  <run_dir>/posthoc_eval/<checkpoint_name>/

or, for Hugging Face models:

  <repo>/runs/research/bos_aligned_proto/posthoc_hf_eval/<model_slug>/

The script is rerunnable. By default it skips task outputs that already exist,
so a second invocation can continue after an interrupted run. Use --force to
rerun completed tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST as _HF_CACHED_NO_EXIST
    from huggingface_hub.utils import enable_progress_bars as _hf_enable_progress_bars
except Exception:
    try_to_load_from_cache = None
    _HF_CACHED_NO_EXIST = object()
    _hf_enable_progress_bars = None


_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROTO_ROOT = os.path.dirname(_THIS_DIR)
_RESEARCH_ROOT = os.path.dirname(_PROTO_ROOT)
_REPO_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from evaluation import core as shared_core
from evaluation import blimp as shared_blimp
from evaluation import ewok as shared_ewok
from evaluation import hellaswag as shared_hellaswag
from research.bos_aligned_proto.training.config import TrainConfig


DEFAULT_TASK_ORDER = ("core", "hellaswag", "ewok", "blimp")
DEFAULT_BLIMP_BATCH_SIZE = 8
DEFAULT_BLIMP_DATA_DIR = ""
DEFAULT_BLIMP_MAX_EXAMPLES_PER_SUBSET = 0
DEFAULT_HF_EVAL_DIR = Path(_REPO_ROOT) / "runs" / "research" / "bos_aligned_proto" / "posthoc_hf_eval"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim > 0 else value.item()
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_jsonable(record)) + "\n")
    os.replace(tmp, path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is not available in this Python environment. "
                "If you normally evaluate from the repo root, try launching with "
                "`conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals ... --device cuda`, "
                "and verify that the host NVIDIA driver is available."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_torch_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def _hf_cached_file(repo_id: str, filename: str, revision: Optional[str]) -> Optional[Path]:
    if try_to_load_from_cache is None:
        return None
    try:
        cached = try_to_load_from_cache(repo_id, filename, revision=revision)
    except Exception:
        return None
    if cached is None or cached is _HF_CACHED_NO_EXIST:
        return None
    try:
        return Path(cached)
    except Exception:
        return None


def _maybe_enable_hf_progress_bars() -> None:
    if _hf_enable_progress_bars is None:
        return
    try:
        _hf_enable_progress_bars()
    except Exception:
        pass


def _announce_hf_download_status(
    model_ref: str,
    *,
    revision: Optional[str],
    local_files_only: bool,
    tokenizer_fallback: str,
    show_status: bool,
) -> None:
    if not show_status:
        return

    if local_files_only:
        _log_progress("hf", "local-files-only mode enabled; the evaluator will not download from the Hub.")
    else:
        _maybe_enable_hf_progress_bars()
        _log_progress("hf", "Hugging Face download progress bars are enabled.")

    model_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    cached_model = next(
        (path for path in (_hf_cached_file(model_ref, name, revision) for name in model_files) if path is not None),
        None,
    )
    cached_tokenizer = next(
        (path for path in (_hf_cached_file(model_ref, name, revision) for name in tokenizer_files) if path is not None),
        None,
    )

    if cached_model is not None:
        _log_progress("hf", f"found cached model file: {cached_model}")
    elif local_files_only:
        _log_progress("hf", f"no cached model weights detected for '{model_ref}'; offline load will fail.")
    else:
        _log_progress("hf", f"no cached model weights detected for '{model_ref}'; starting Hub download if needed.")

    if cached_tokenizer is not None:
        _log_progress("hf", f"found cached tokenizer file: {cached_tokenizer}")
    elif tokenizer_fallback and tokenizer_fallback != model_ref:
        _log_progress(
            "hf",
            f"no cached tokenizer file detected for '{model_ref}'; will fall back to tokenizer '{tokenizer_fallback}' if needed.",
        )
    elif local_files_only:
        _log_progress("hf", f"no cached tokenizer file detected for '{model_ref}'; offline tokenizer load may fail.")
    else:
        _log_progress("hf", f"no cached tokenizer file detected for '{model_ref}'; tokenizer may download from the Hub.")


def _checkpoint_step_from_dirname(path: Path) -> Optional[int]:
    match = re.match(r"^ckpt_[^/]*_step(\d+)$", path.name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _checkpoint_priority(path: Path) -> int:
    name = path.name
    if name.startswith("ckpt_final_"):
        return 2
    if name.startswith("ckpt_periodic_"):
        return 1
    return 0


def _is_model_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    if (path / "model.safetensors").exists():
        return True
    if (path / "pytorch_model.bin").exists():
        return True
    if any(path.glob("model-*.safetensors")):
        return True
    return False


def _checkpoint_integrity_error(path: Path) -> Optional[str]:
    if not _is_model_checkpoint_dir(path):
        return "missing model checkpoint files"

    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt") as handle:
                next(iter(handle.keys()), None)
            return None
        except Exception as exc:
            return f"model.safetensors is unreadable: {exc}"

    sharded = sorted(path.glob("model-*.safetensors"))
    if sharded:
        try:
            from safetensors import safe_open

            with safe_open(str(sharded[0]), framework="pt") as handle:
                next(iter(handle.keys()), None)
            return None
        except Exception as exc:
            return f"{sharded[0].name} is unreadable: {exc}"

    if (path / "pytorch_model.bin").exists():
        return None
    return "no supported weight file found"


def _select_best_checkpoint(candidates: Sequence[Path], *, context_label: str) -> Tuple[Path, Optional[int]]:
    valid: List[Tuple[int, int, float, Path]] = []
    invalid: List[Tuple[Path, str]] = []

    for candidate in candidates:
        error = _checkpoint_integrity_error(candidate)
        if error is None:
            valid.append(
                (
                    _checkpoint_step_from_dirname(candidate) or -1,
                    _checkpoint_priority(candidate),
                    float(candidate.stat().st_mtime),
                    candidate,
                )
            )
        else:
            invalid.append((candidate, error))

    if not valid:
        lines = [f"  - {str(path)}: {error}" for path, error in invalid[:5]]
        if len(invalid) > 5:
            lines.append(f"  - ... and {len(invalid) - 5} more invalid checkpoint candidates.")
        detail = "\n".join(lines) if lines else "  - No checkpoint candidates were found."
        raise RuntimeError(
            f"No loadable checkpoint was found for {context_label}.\n{detail}"
        )

    valid.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))
    selected = valid[-1][3]
    return selected, _checkpoint_step_from_dirname(selected)


def _resolve_checkpoint_target(
    path_str: str,
    *,
    checkpoint_name: Optional[str],
    step: Optional[int],
) -> Tuple[Path, Path, Optional[int]]:
    raw = Path(path_str).expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"Target path does not exist or is not a directory: {raw}")

    if _is_model_checkpoint_dir(raw):
        error = _checkpoint_integrity_error(raw)
        if error is not None:
            raise RuntimeError(f"Checkpoint exists but is not loadable: {raw}\n  - {error}")
        return raw.parent, raw, _checkpoint_step_from_dirname(raw)

    candidates: List[Path] = []
    for child in raw.iterdir():
        if _is_model_checkpoint_dir(child):
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directories found under '{raw}'. "
            "Expected folders named like ckpt_<tag>_step0001234."
        )

    if checkpoint_name:
        matched = [c for c in candidates if c.name == checkpoint_name]
        if not matched:
            raise FileNotFoundError(
                f"No checkpoint named '{checkpoint_name}' found under '{raw}'."
            )
        selected, resolved_step = _select_best_checkpoint(
            matched,
            context_label=f"checkpoint_name='{checkpoint_name}' under {raw}",
        )
        return raw, selected, resolved_step

    if step is not None:
        matched = [c for c in candidates if _checkpoint_step_from_dirname(c) == int(step)]
        if not matched:
            raise FileNotFoundError(
                f"No checkpoint with step {int(step)} found under '{raw}'."
            )
        selected, resolved_step = _select_best_checkpoint(
            matched,
            context_label=f"step={int(step)} under {raw}",
        )
        return raw, selected, resolved_step

    selected, resolved_step = _select_best_checkpoint(
        candidates,
        context_label=f"latest checkpoint under {raw}",
    )
    return raw, selected, resolved_step


def _parse_tasks(raw_tasks: str) -> List[str]:
    parsed = []
    seen = set()
    for token in str(raw_tasks).split(","):
        task = token.strip().lower()
        if not task:
            continue
        if task not in DEFAULT_TASK_ORDER:
            raise ValueError(
                f"Unknown task '{task}'. Supported tasks: {', '.join(DEFAULT_TASK_ORDER)}"
            )
        if task in seen:
            continue
        seen.add(task)
        parsed.append(task)
    if not parsed:
        raise ValueError("No evaluation tasks were selected.")
    priority = {name: idx for idx, name in enumerate(DEFAULT_TASK_ORDER)}
    parsed.sort(key=lambda name: priority[name])
    return parsed


def _resolve_optional_str(cli_value: Optional[str], config_value: Any, default: str = "") -> str:
    if cli_value is not None:
        return str(cli_value)
    if config_value is not None:
        return str(config_value)
    return str(default)


def _resolve_optional_int(cli_value: Optional[int], config_value: Any, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(default)


def _resolve_optional_bool(cli_value: Optional[bool], config_value: Any, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if config_value is not None:
        return bool(config_value)
    return bool(default)


def _resolve_settings(
    args: argparse.Namespace,
    run_config: Dict[str, Any],
    *,
    tokenizer_default_source: str,
) -> Dict[str, Any]:
    defaults = TrainConfig(data_dir="")
    return {
        "core_max_per_task": _resolve_optional_int(
            args.core_max_per_task,
            run_config.get("core_max_per_task"),
            defaults.core_max_per_task,
        ),
        "core_bundle_dir": _resolve_optional_str(
            args.core_bundle_dir,
            run_config.get("core_bundle_dir"),
            defaults.core_bundle_dir,
        ),
        "core_local_files_only": _resolve_optional_bool(
            args.core_local_files_only,
            run_config.get("core_local_files_only"),
            defaults.core_local_files_only,
        ),
        "hellaswag_batch_size": _resolve_optional_int(
            args.hellaswag_batch_size,
            run_config.get("hellaswag_batch_size"),
            defaults.hellaswag_batch_size,
        ),
        "hellaswag_max_examples": _resolve_optional_int(
            args.hellaswag_max_examples,
            run_config.get("hellaswag_max_examples"),
            defaults.hellaswag_max_examples,
        ),
        "hellaswag_dataset": _resolve_optional_str(
            args.hellaswag_dataset,
            run_config.get("hellaswag_dataset"),
            defaults.hellaswag_dataset,
        ),
        "hellaswag_dataset_config": (
            None
            if _resolve_optional_str(
                args.hellaswag_dataset_config,
                run_config.get("hellaswag_dataset_config"),
                "",
            )
            == ""
            else _resolve_optional_str(
                args.hellaswag_dataset_config,
                run_config.get("hellaswag_dataset_config"),
                "",
            )
        ),
        "hellaswag_split": _resolve_optional_str(
            args.hellaswag_split,
            run_config.get("hellaswag_split"),
            defaults.hellaswag_split,
        ),
        "hellaswag_local_files_only": _resolve_optional_bool(
            args.hellaswag_local_files_only,
            run_config.get("hellaswag_local_files_only"),
            defaults.hellaswag_local_files_only,
        ),
        "ewok_batch_size": _resolve_optional_int(
            args.ewok_batch_size,
            run_config.get("ewok_batch_size"),
            defaults.ewok_batch_size,
        ),
        "blimp_batch_size": _resolve_optional_int(
            args.blimp_batch_size,
            run_config.get("blimp_batch_size"),
            DEFAULT_BLIMP_BATCH_SIZE,
        ),
        "blimp_data_dir": _resolve_optional_str(
            args.blimp_data_dir,
            run_config.get("blimp_data_dir"),
            DEFAULT_BLIMP_DATA_DIR,
        ),
        "blimp_max_examples_per_subset": _resolve_optional_int(
            args.blimp_max_examples_per_subset,
            run_config.get("blimp_max_examples_per_subset"),
            DEFAULT_BLIMP_MAX_EXAMPLES_PER_SUBSET,
        ),
        "tokenizer_fallback": (
            args.tokenizer
            if args.tokenizer
            else str(run_config.get("tokenizer") or tokenizer_default_source or "gpt2")
        ),
    }


def _load_model_and_tokenizer(
    model_ref: str,
    *,
    device: torch.device,
    dtype,
    revision: Optional[str],
    trust_remote_code: bool,
    model_local_files_only: bool,
    tokenizer_fallback: str,
    tokenizer_fallback_local_files_only: bool,
    show_hf_download_status: bool = False,
) -> Tuple[Any, Any, str]:
    if show_hf_download_status:
        _log_progress("hf", f"loading model weights from '{model_ref}'")
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=model_local_files_only,
    )
    model.to(device)
    model.eval()
    if show_hf_download_status:
        _log_progress("hf", f"model load finished for '{model_ref}'")

    tokenizer_source = str(model_ref)
    try:
        if show_hf_download_status:
            _log_progress("hf", f"loading tokenizer from '{model_ref}'")
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=model_local_files_only,
        )
    except Exception:
        tokenizer_source = str(tokenizer_fallback)
        tokenizer_path = Path(tokenizer_fallback).expanduser()
        if show_hf_download_status:
            _log_progress(
                "hf",
                f"tokenizer load from '{model_ref}' failed; falling back to '{tokenizer_fallback}'",
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_fallback,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=(tokenizer_fallback_local_files_only or tokenizer_path.exists()),
        )
    if show_hf_download_status:
        _log_progress("hf", f"tokenizer load finished from '{tokenizer_source}'")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if getattr(model, "get_input_embeddings", None) is not None:
        embed = model.get_input_embeddings()
        if embed is not None and embed.num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, tokenizer_source


def _release_eval_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_progress(scope: str, message: str) -> None:
    print(f"[{scope}] {message}", flush=True)


def _write_core_csv(path: Path, core_results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        results = dict(core_results.get("results", {}))
        centered = dict(core_results.get("centered_results", {}))
        for label in results:
            handle.write(
                f"{label:<35}, {float(results[label]):<10.6f}, {float(centered[label]):<10.6f}\n"
            )
        handle.write(
            f"{'CORE':<35}, {'':<10}, {float(core_results['core_metric']):<10.6f}\n"
        )
    os.replace(tmp, path)


def _task_paths(output_dir: Path, task: str) -> Dict[str, Path]:
    if task == "core":
        return {
            "primary": output_dir / "core.json",
            "csv": output_dir / "core.csv",
        }
    if task == "hellaswag":
        return {
            "primary": output_dir / "hellaswag_metrics.json",
            "predictions": output_dir / "hellaswag_predictions.json",
        }
    if task == "ewok":
        return {
            "primary": output_dir / "ewok_metrics.json",
        }
    if task == "blimp":
        return {
            "primary": output_dir / "blimp_metrics.json",
        }
    raise KeyError(f"Unexpected task: {task}")


def _read_existing_summary(task: str, paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    if not paths["primary"].exists():
        return None
    payload = _load_json(paths["primary"], {})
    if task == "core":
        return {
            "core_metric": payload.get("core_metric"),
            "num_tasks": payload.get("num_tasks"),
            "max_per_task": payload.get("max_per_task"),
        }
    if task == "hellaswag":
        return {
            "accuracy": payload.get("accuracy"),
            "accuracy_norm": payload.get("accuracy_norm"),
            "num_examples": payload.get("num_examples"),
            "max_seq_len": payload.get("max_seq_len"),
        }
    if task == "ewok":
        return payload.get("summary")
    if task == "blimp":
        return payload.get("summary")
    return None


def _run_core_task(
    *,
    model,
    tokenizer,
    device: torch.device,
    settings: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    _log_progress("core", "starting")
    _log_progress(
        "core",
        f"running DCLM CORE (max_per_task={int(settings['core_max_per_task'])})",
    )
    start_time = time.time()
    core_results = shared_core.evaluate_core(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_per_task=int(settings["core_max_per_task"]),
        bundle_dir=(settings["core_bundle_dir"] or None),
        local_files_only=bool(settings["core_local_files_only"]),
        distributed=False,
        show_progress=True,
    )
    elapsed = time.time() - start_time
    core_results["elapsed_seconds"] = float(elapsed)
    _atomic_write_json(output_dir / "core.json", core_results)
    _write_core_csv(output_dir / "core.csv", core_results)
    summary = {
        "core_metric": float(core_results["core_metric"]),
        "num_tasks": int(core_results["num_tasks"]),
        "max_per_task": int(core_results["max_per_task"]),
        "bundle_dir": str(core_results["bundle_dir"]),
        "elapsed_seconds": float(elapsed),
    }
    _log_progress(
        "core",
        f"finished core_metric={summary['core_metric']:.4f}, tasks={summary['num_tasks']}",
    )
    return summary


def _run_hellaswag_task(
    *,
    model,
    tokenizer,
    device: torch.device,
    dtype,
    settings: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    _log_progress("hellaswag", "starting")
    _log_progress(
        "hellaswag",
        f"loading dataset split='{settings['hellaswag_split']}' from '{settings['hellaswag_dataset']}'",
    )
    dataset = shared_hellaswag.load_dataset_compat(
        dataset_name=settings["hellaswag_dataset"],
        dataset_config=settings["hellaswag_dataset_config"],
        split=settings["hellaswag_split"],
        local_files_only=bool(settings["hellaswag_local_files_only"]),
    )
    max_examples = int(settings["hellaswag_max_examples"])
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    max_seq_len = shared_hellaswag.infer_max_seq_len(model, tokenizer)
    _log_progress(
        "hellaswag",
        f"scoring {len(dataset)} examples (batch_size={int(settings['hellaswag_batch_size'])}, max_seq_len={int(max_seq_len)})",
    )
    start_time = time.time()
    metrics = shared_hellaswag.evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=int(settings["hellaswag_batch_size"]),
        device=device,
        max_seq_len=max_seq_len,
        show_progress=True,
    )
    elapsed = time.time() - start_time

    metrics_payload = {
        "dataset": settings["hellaswag_dataset"],
        "dataset_config": settings["hellaswag_dataset_config"],
        "split": settings["hellaswag_split"],
        "batch_size": int(settings["hellaswag_batch_size"]),
        "dtype": str(dtype),
        "device": str(device),
        "max_seq_len": int(max_seq_len),
        "num_examples": int(metrics["num_examples"]),
        "accuracy": float(metrics["accuracy"]),
        "accuracy_norm": float(metrics["accuracy_norm"]),
        "elapsed_seconds": float(elapsed),
    }
    predictions_payload = {
        "labels": list(metrics["labels"]),
        "pred_raw": list(metrics["pred_raw"]),
        "pred_norm": list(metrics["pred_norm"]),
    }
    _atomic_write_json(output_dir / "hellaswag_metrics.json", metrics_payload)
    _atomic_write_json(output_dir / "hellaswag_predictions.json", predictions_payload)

    summary = {
        "accuracy": float(metrics["accuracy"]),
        "accuracy_norm": float(metrics["accuracy_norm"]),
        "num_examples": int(metrics["num_examples"]),
        "max_seq_len": int(max_seq_len),
        "elapsed_seconds": float(elapsed),
    }
    _log_progress(
        "hellaswag",
        f"finished acc={summary['accuracy']:.4f}, acc_norm={summary['accuracy_norm']:.4f}",
    )
    return summary


def _ewok_official_average(metric_block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metric_block, dict):
        return None
    domain_scores_full = metric_block.get("domain_scores_full")
    if not isinstance(domain_scores_full, dict):
        return None
    avg = domain_scores_full.get("average")
    if isinstance(avg, (list, tuple)) and avg:
        return float(avg[0])
    return None


def _run_ewok_task(
    *,
    model,
    tokenizer,
    settings: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    _log_progress("ewok", "starting")
    _log_progress(
        "ewok",
        "running mean/full metrics for BabyLM completion choice and context sensitivity",
    )
    start_time = time.time()
    metrics_by_method_mean = shared_ewok.evaluate(
        model,
        tokenizer,
        batch_size=int(settings["ewok_batch_size"]),
        return_per_item=False,
        score_reduction="mean",
        return_all_methods=True,
        show_progress=True,
    )
    elapsed = time.time() - start_time

    babylm_mean = metrics_by_method_mean.get(shared_ewok.BABYLM_COMPLETION_CHOICE)
    context_mean = metrics_by_method_mean.get(shared_ewok.EWOK_CONTEXT_SENSITIVITY)

    summary = {
        "babylm_completion_choice_full_mean": (
            babylm_mean.get("domain_scores_full") if isinstance(babylm_mean, dict) else None
        ),
        "ewok_context_sensitivity_full_mean": (
            context_mean.get("domain_scores_full") if isinstance(context_mean, dict) else None
        ),
        "babylm_completion_choice_official_mean_average": _ewok_official_average(babylm_mean),
        "ewok_context_sensitivity_official_mean_average": _ewok_official_average(context_mean),
        "num_items": int(len(shared_ewok.ewok_df)),
        "elapsed_seconds": float(elapsed),
    }

    payload = {
        "ewok_source": str(shared_ewok.SRC),
        "batch_size": int(settings["ewok_batch_size"]),
        "elapsed_seconds": float(elapsed),
        "mean": {
            "metrics_by_method": metrics_by_method_mean,
            "num_items": int(len(shared_ewok.ewok_df)),
        },
        "summary": summary,
    }

    _atomic_write_json(output_dir / "ewok_metrics.json", payload)

    babylm_mean_avg = summary["babylm_completion_choice_official_mean_average"]
    context_mean_avg = summary["ewok_context_sensitivity_official_mean_average"]
    if babylm_mean_avg is not None and context_mean_avg is not None:
        _log_progress(
            "ewok",
            f"finished babylm_mean_avg={float(babylm_mean_avg):.4f}, "
            f"context_mean_avg={float(context_mean_avg):.4f}",
        )
    elif babylm_mean_avg is not None:
        _log_progress("ewok", f"finished babylm_mean_avg={float(babylm_mean_avg):.4f}")
    else:
        _log_progress("ewok", "finished")
    return summary


def _run_blimp_task(
    *,
    model,
    tokenizer,
    device: torch.device,
    settings: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    _log_progress("blimp", "starting")
    _log_progress(
        "blimp",
        "loading BLiMP-fast records"
        + (
            f" from '{settings['blimp_data_dir']}'"
            if settings["blimp_data_dir"]
            else " from auto-resolved source"
        ),
    )
    records, source_path = shared_blimp.load_blimp_records(
        data_dir=(settings["blimp_data_dir"] or None),
        max_examples_per_subset=int(settings["blimp_max_examples_per_subset"]),
    )
    _log_progress(
        "blimp",
        f"scoring {len(records)} sentence pairs (batch_size={int(settings['blimp_batch_size'])})",
    )
    start_time = time.time()
    metrics_sum = shared_blimp.evaluate(
        model,
        tokenizer,
        batch_size=int(settings["blimp_batch_size"]),
        return_per_item=False,
        score_reduction="sum",
        records=records,
        source_path=source_path,
        device=device,
        show_progress=True,
    )
    metrics_mean = shared_blimp.evaluate(
        model,
        tokenizer,
        batch_size=int(settings["blimp_batch_size"]),
        return_per_item=False,
        score_reduction="mean",
        records=records,
        source_path=source_path,
        device=device,
        show_progress=True,
    )
    elapsed = time.time() - start_time

    summary = {
        "sum_accuracy": float(metrics_sum["accuracy"]),
        "sum_accuracy_macro_uid": float(metrics_sum["accuracy_macro_uid"]),
        "mean_accuracy": float(metrics_mean["accuracy"]),
        "mean_accuracy_macro_uid": float(metrics_mean["accuracy_macro_uid"]),
        "num_examples": int(metrics_sum["num_examples"]),
        "num_subsets": int(metrics_sum["num_subsets"]),
        "elapsed_seconds": float(elapsed),
    }

    payload = {
        "source": str(source_path),
        "batch_size": int(settings["blimp_batch_size"]),
        "max_examples_per_subset": int(settings["blimp_max_examples_per_subset"]),
        "elapsed_seconds": float(elapsed),
        "sum": metrics_sum,
        "mean": metrics_mean,
        "summary": summary,
    }
    _atomic_write_json(output_dir / "blimp_metrics.json", payload)
    _log_progress(
        "blimp",
        f"finished sum_acc={summary['sum_accuracy']:.4f}, mean_acc={summary['mean_accuracy']:.4f}",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc BOS evaluations for a resolved checkpoint, always ordering "
            "tasks as CORE -> HellaSwag -> EWoK -> BLiMP."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Run directory containing ckpt_* folders, or a direct checkpoint directory.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="Optional Hugging Face model id to evaluate instead of a local run/checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="",
        help="Optional exact checkpoint folder name when passing a run directory.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional checkpoint step to resolve when passing a run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Optional override for the output directory. "
            "Default: <run_dir>/posthoc_eval/<checkpoint_name> for local checkpoints, "
            "or runs/research/bos_aligned_proto/posthoc_hf_eval/<model_slug> for --hf-model."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="core,hellaswag,ewok,blimp",
        help="Comma-separated subset of tasks to run. Order is normalized to CORE first.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun tasks even if their output files already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first task failure instead of continuing to later tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve checkpoint, output dir, and settings without loading the model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Execution device (default: cuda). Use --device auto to allow CPU fallback.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Fallback tokenizer path/name if the checkpoint does not include tokenizer files (default: gpt2).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Request offline/local-cache loading for tokenizer fallback and datasets when possible.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (branch/tag/commit) for --hf-model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code when loading --hf-model.",
    )
    parser.add_argument(
        "--show-hf-download-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When using --hf-model, print cache/download status and enable Hugging Face "
            "progress bars. Disable with --no-show-hf-download-status."
        ),
    )
    parser.add_argument(
        "--core-max-per-task",
        type=int,
        default=None,
        help="Override max examples per CORE task.",
    )
    parser.add_argument(
        "--core-bundle-dir",
        type=str,
        default=None,
        help="Override CORE eval bundle directory.",
    )
    parser.add_argument(
        "--core-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CORE bundle local-files-only behavior.",
    )
    parser.add_argument(
        "--hellaswag-batch-size",
        type=int,
        default=None,
        help="Override HellaSwag candidate batch size.",
    )
    parser.add_argument(
        "--hellaswag-max-examples",
        type=int,
        default=None,
        help="Override HellaSwag max examples.",
    )
    parser.add_argument(
        "--hellaswag-dataset",
        type=str,
        default=None,
        help="Override HellaSwag dataset name/path.",
    )
    parser.add_argument(
        "--hellaswag-dataset-config",
        type=str,
        default=None,
        help="Override HellaSwag dataset config name.",
    )
    parser.add_argument(
        "--hellaswag-split",
        type=str,
        default=None,
        help="Override HellaSwag split.",
    )
    parser.add_argument(
        "--hellaswag-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override HellaSwag local-files-only behavior.",
    )
    parser.add_argument(
        "--ewok-batch-size",
        type=int,
        default=None,
        help="Override EWoK batch size.",
    )
    parser.add_argument(
        "--blimp-batch-size",
        type=int,
        default=None,
        help="Override BLiMP batch size.",
    )
    parser.add_argument(
        "--blimp-data-dir",
        type=str,
        default=None,
        help="Override BLiMP-fast JSONL directory.",
    )
    parser.add_argument(
        "--blimp-max-examples-per-subset",
        type=int,
        default=None,
        help="Override BLiMP cap per subset (0 uses all).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)

    if bool(args.target) == bool(args.hf_model):
        raise SystemExit("Provide exactly one of: positional <target> or --hf-model <model_id>.")

    source_type: str
    run_dir: Optional[Path]
    checkpoint_dir: Optional[Path]
    checkpoint_name: str
    checkpoint_step: Optional[int]
    model_ref: str
    run_config: Dict[str, Any]

    if args.hf_model:
        source_type = "hf_model"
        run_dir = None
        checkpoint_dir = None
        checkpoint_step = None
        model_ref = str(args.hf_model).strip()
        if not model_ref:
            raise SystemExit("--hf-model must be a non-empty Hugging Face model id.")
        checkpoint_name = _sanitize_name(
            model_ref if args.revision is None else f"{model_ref}@{args.revision}"
        )
        run_config = {}
    else:
        source_type = "local_checkpoint"
        run_dir, checkpoint_dir, checkpoint_step = _resolve_checkpoint_target(
            args.target,
            checkpoint_name=(args.checkpoint_name or None),
            step=args.step,
        )
        checkpoint_name = checkpoint_dir.name
        model_ref = str(checkpoint_dir)
        run_config = _load_json(run_dir / "run_config.json", {})
        if not isinstance(run_config, dict):
            run_config = {}

    settings = _resolve_settings(
        args,
        run_config,
        tokenizer_default_source=(model_ref if args.hf_model else "gpt2"),
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif source_type == "hf_model":
        output_dir = DEFAULT_HF_EVAL_DIR / checkpoint_name
    else:
        output_dir = run_dir / "posthoc_eval" / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    manifest = {
        "target": (str(Path(args.target).expanduser().resolve()) if args.target else model_ref),
        "source_type": source_type,
        "run_dir": (str(run_dir) if run_dir is not None else None),
        "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
        "hf_model": (model_ref if source_type == "hf_model" else None),
        "hf_revision": args.revision,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "tasks": tasks,
        "started_at": _utc_now_iso(),
        "settings": settings,
        "task_status": {},
    }

    if run_dir is not None:
        print(f"Resolved run_dir:        {run_dir}")
    if checkpoint_dir is not None:
        print(f"Resolved checkpoint_dir: {checkpoint_dir}")
    if source_type == "hf_model":
        print(f"Resolved hf_model:       {model_ref}")
        if args.revision is not None:
            print(f"Resolved revision:       {args.revision}")
    print(f"Resolved output_dir:     {output_dir}")
    print(f"Task order:              {', '.join(tasks)}")
    if checkpoint_step is not None:
        print(f"Checkpoint step:         {checkpoint_step}")

    if args.dry_run:
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(
            summary_path,
            {
                "source_type": source_type,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "hf_model": (model_ref if source_type == "hf_model" else None),
                "checkpoint_step": checkpoint_step,
                "tasks": {},
                "dry_run": True,
            },
        )
        print("Dry run complete. No model was loaded.")
        return

    device = _get_device(args.device)
    dtype = _get_torch_dtype(args.dtype, device)
    if source_type == "hf_model":
        print(f"Loading Hugging Face model on {device} with dtype={dtype}")
        _announce_hf_download_status(
            model_ref,
            revision=args.revision,
            local_files_only=bool(args.local_files_only),
            tokenizer_fallback=str(settings["tokenizer_fallback"]),
            show_status=bool(args.show_hf_download_status),
        )
    else:
        print(f"Loading checkpoint on {device} with dtype={dtype}")
    model, tokenizer, tokenizer_source = _load_model_and_tokenizer(
        model_ref,
        device=device,
        dtype=dtype,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        model_local_files_only=bool(source_type != "hf_model" or args.local_files_only),
        tokenizer_fallback=settings["tokenizer_fallback"],
        tokenizer_fallback_local_files_only=bool(args.local_files_only),
        show_hf_download_status=bool(source_type == "hf_model" and args.show_hf_download_status),
    )
    manifest["device"] = str(device)
    manifest["dtype"] = str(dtype)
    manifest["tokenizer_source"] = str(tokenizer_source)
    manifest["model_ref"] = str(model_ref)
    _atomic_write_json(manifest_path, manifest)

    summary_payload = {
        "source_type": source_type,
        "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
        "hf_model": (model_ref if source_type == "hf_model" else None),
        "checkpoint_step": checkpoint_step,
        "tokenizer_source": str(tokenizer_source),
        "tasks": {},
    }

    task_fns = {
        "core": lambda: _run_core_task(
            model=model,
            tokenizer=tokenizer,
            device=device,
            settings=settings,
            output_dir=output_dir,
        ),
        "hellaswag": lambda: _run_hellaswag_task(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            settings=settings,
            output_dir=output_dir,
        ),
        "ewok": lambda: _run_ewok_task(
            model=model,
            tokenizer=tokenizer,
            settings=settings,
            output_dir=output_dir,
        ),
        "blimp": lambda: _run_blimp_task(
            model=model,
            tokenizer=tokenizer,
            device=device,
            settings=settings,
            output_dir=output_dir,
        ),
    }

    for task in tasks:
        paths = _task_paths(output_dir, task)
        if paths["primary"].exists() and not args.force:
            summary = _read_existing_summary(task, paths)
            manifest["task_status"][task] = {
                "status": "skipped_existing",
                "finished_at": _utc_now_iso(),
                "output_files": [str(p) for p in paths.values() if p.exists()],
                "summary": summary,
            }
            if summary is not None:
                summary_payload["tasks"][task] = summary
            _atomic_write_json(manifest_path, manifest)
            _atomic_write_json(summary_path, summary_payload)
            _log_progress(task, f"skipping existing output at {paths['primary']}")
            continue

        manifest["task_status"][task] = {
            "status": "running",
            "started_at": _utc_now_iso(),
        }
        _atomic_write_json(manifest_path, manifest)
        _log_progress(task, "task started")

        try:
            task_summary = task_fns[task]()
            manifest["task_status"][task] = {
                "status": "completed",
                "finished_at": _utc_now_iso(),
                "output_files": [str(p) for p in paths.values() if p.exists()],
                "summary": task_summary,
            }
            summary_payload["tasks"][task] = task_summary
            _atomic_write_json(manifest_path, manifest)
            _atomic_write_json(summary_path, summary_payload)
            _release_eval_memory()
        except Exception as exc:
            manifest["task_status"][task] = {
                "status": "failed",
                "finished_at": _utc_now_iso(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_write_json(manifest_path, manifest)
            _atomic_write_json(summary_path, summary_payload)
            _log_progress(task, f"failed: {exc}")
            if args.fail_fast:
                raise

    print("\nPost-hoc evaluation summary")
    for task in tasks:
        task_status = manifest["task_status"].get(task, {})
        status = task_status.get("status", "unknown")
        print(f"  {task}: {status}")
        task_summary = summary_payload["tasks"].get(task)
        if task == "core" and isinstance(task_summary, dict) and task_summary.get("core_metric") is not None:
            print(f"    core_metric={float(task_summary['core_metric']):.4f}")
        elif task == "hellaswag" and isinstance(task_summary, dict) and task_summary.get("accuracy") is not None:
            print(
                f"    accuracy={float(task_summary['accuracy']):.4f}, "
                f"accuracy_norm={float(task_summary['accuracy_norm']):.4f}"
            )
        elif (
            task == "ewok"
            and isinstance(task_summary, dict)
            and task_summary.get("babylm_completion_choice_official_mean_average") is not None
        ):
            parts = [
                "    babylm_mean_avg="
                f"{float(task_summary['babylm_completion_choice_official_mean_average']):.4f}"
            ]
            if task_summary.get("ewok_context_sensitivity_official_mean_average") is not None:
                parts.append(
                    "context_mean_avg="
                    f"{float(task_summary['ewok_context_sensitivity_official_mean_average']):.4f}"
                )
            print(", ".join(parts))
        elif task == "blimp" and isinstance(task_summary, dict) and task_summary.get("mean_accuracy") is not None:
            print(
                f"    sum_acc={float(task_summary['sum_accuracy']):.4f}, "
                f"mean_acc={float(task_summary['mean_accuracy']):.4f}"
            )

    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved summary:  {summary_path}")


if __name__ == "__main__":
    main()
