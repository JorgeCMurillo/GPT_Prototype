from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import fcntl
import fnmatch
import gc
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_CONFIG_PATH = THIS_DIR / "model_queue.example.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "moonshotGPT/runs/hf_ewok_eval/raw"
DEFAULT_DOWNLOADS_ROOT = Path(os.environ.get("HF_EWOK_DOWNLOADS_ROOT", str(Path.home() / ".cache/moonshotGPT/hf_models")))
LEGACY_DOWNLOADS_ROOT = THIS_DIR / "downloads"
SYSTEM_HF_HUB_ROOT = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub")))
KNOWN_DOWNLOADS_ROOTS = (
    DEFAULT_DOWNLOADS_ROOT,
    LEGACY_DOWNLOADS_ROOT,
)
DEFAULT_LIBRARY_NAME = "hf_ewok_eval"
DEFAULT_LIBRARY_VERSION = "0.1"
DEFAULT_DISABLE_XET = True
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_QUEUE_LOG_NAME = "queue_run.log"
DEFAULT_MODEL_LOG_NAME = "run.log"
QUEUE_SUMMARY_FIELDS = [
    "model_id",
    "revision",
    "param_count_b",
    "status",
    "load_strategy",
    "num_gpus_used",
    "quantization",
    "ewok_babylm_completion_full_mean_avg",
    "ewok_context_sensitivity_full_mean_avg",
    "ewok_pmi_completion_full_mean_avg",
    "ewok_pmi_completion_full_mean_clean_avg",
    "ewok_babylm_completion_official_mean_avg",
    "ewok_context_sensitivity_official_mean_avg",
    "ewok_pmi_completion_official_mean_avg",
    "ewok_pmi_completion_symmetric_mean_avg",
    "batch_size_final",
    "elapsed_seconds",
    "error",
]
TOKENIZER_ALLOW_PATTERNS = [
    "added_tokens.json",
    "chat_template.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
]
WEIGHT_FAMILY_IGNORE_PATTERNS = {
    "safetensors": ["*.bin", "*.pt", "*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*"],
    "pytorch_bin": ["*.safetensors", "*.pt", "*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*"],
    "pt": ["*.safetensors", "*.bin", "*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*"],
    "flax_msgpack": ["*.safetensors", "*.bin", "*.pt", "*.h5", "*.ot", "*.onnx", "onnx/*"],
}
MULTIMODAL_MODEL_PATTERNS = (
    "smolvlm",
    "qwen2.5-vl",
    "vision",
    "internvl",
    "gemma-3",
)
_FLASH_ATTN_BROKEN_CACHE: bool | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    param_count_b: float
    revision: str | None = None
    trust_remote_code: bool = False
    tokenizer_id: str | None = None

    @property
    def model_slug(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return _sanitize_name(f"{self.model_id}{suffix}")


class _ProcessorTextTokenizerAdapter:
    """Bare text tokenizer shim for processors that do not expose a tokenizer."""

    def __init__(self, processor: Any):
        self._processor = processor
        self._tokenizer = getattr(processor, "tokenizer", None)

    def __getattr__(self, name: str) -> Any:
        if self._tokenizer is not None and hasattr(self._tokenizer, name):
            return getattr(self._tokenizer, name)
        return getattr(self._processor, name)

    def __len__(self) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer)
        raise TypeError("Processor-backed text adapter does not expose a vocabulary length.")

    @property
    def pad_token(self) -> Any:
        if self._tokenizer is not None:
            return getattr(self._tokenizer, "pad_token", None)
        return getattr(self._processor, "pad_token", None)

    @pad_token.setter
    def pad_token(self, value: Any) -> None:
        if self._tokenizer is not None:
            setattr(self._tokenizer, "pad_token", value)
            return
        setattr(self._processor, "pad_token", value)

    @property
    def eos_token(self) -> Any:
        if self._tokenizer is not None:
            return getattr(self._tokenizer, "eos_token", None)
        return getattr(self._processor, "eos_token", None)

    def add_special_tokens(self, special_tokens_dict: dict[str, Any]) -> Any:
        if self._tokenizer is not None and hasattr(self._tokenizer, "add_special_tokens"):
            return self._tokenizer.add_special_tokens(special_tokens_dict)
        raise RuntimeError(
            "Processor-backed text adapter cannot add special tokens because no tokenizer is available."
        )

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        if self._tokenizer is not None and hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

        encoded = self(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
            padding=False,
            **kwargs,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor text call did not return input_ids.")
        if torch.is_tensor(input_ids):
            if input_ids.ndim == 0:
                return [int(input_ids.item())]
            if input_ids.ndim > 1:
                return [int(value) for value in input_ids[0].tolist()]
            return [int(value) for value in input_ids.tolist()]
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(value) for value in input_ids[0]]
            return [int(value) for value in input_ids]
        raise RuntimeError("Processor text call returned unsupported input_ids shape.")

    def __call__(
        self,
        input_texts: str | list[str],
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = "pt",
        padding: bool | str = True,
        **kwargs: Any,
    ) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer(
                input_texts,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs,
            )

        batch = self._processor(
            text=input_texts,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )
        if "input_ids" not in batch:
            raise RuntimeError(
                "Processor text call did not return input_ids; cannot perform bare text likelihood evaluation."
            )
        if "attention_mask" not in batch:
            input_ids = batch["input_ids"]
            if torch.is_tensor(input_ids):
                batch["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
            elif isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    batch["attention_mask"] = [[1] * len(row) for row in input_ids]
                else:
                    batch["attention_mask"] = [1] * len(input_ids)
        return batch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None


def _looks_multimodal_model_id(model_id: str) -> bool:
    normalized = str(model_id).strip().lower()
    return any(pattern in normalized for pattern in MULTIMODAL_MODEL_PATTERNS)


def _is_unrecognized_auto_model_config_error(exc: Exception) -> bool:
    message = str(exc)
    return "Unrecognized configuration class" in message and "AutoModel" in message


def _is_flash_attn_import_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return (
        "flash_attn_2_cuda" in lowered
        or ("flash_attn" in lowered and "undefined symbol" in lowered)
        or ("failed to import" in lowered and "flash_attn" in lowered)
    )


def _flash_attn_is_broken() -> bool:
    global _FLASH_ATTN_BROKEN_CACHE
    if _FLASH_ATTN_BROKEN_CACHE is not None:
        return bool(_FLASH_ATTN_BROKEN_CACHE)

    if importlib.util.find_spec("flash_attn") is None:
        _FLASH_ATTN_BROKEN_CACHE = False
        return False
    try:
        importlib.import_module("flash_attn")
        _FLASH_ATTN_BROKEN_CACHE = False
    except Exception:
        _FLASH_ATTN_BROKEN_CACHE = True
    return bool(_FLASH_ATTN_BROKEN_CACHE)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim > 0 else value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row)) + "\n")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_jsonable(row.get(key)) for key in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _sanitize_log_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    return str(detail).replace("\n", "\\n")


def _make_phase_logger(path: Path, *, model_slug: str | None = None) -> Callable[[str, str | None], None]:
    def _callback(phase: str, detail: str | None = None) -> None:
        parts = [_utc_now_iso()]
        if model_slug:
            parts.append(f"model={model_slug}")
        parts.append(f"phase={phase}")
        safe_detail = _sanitize_log_detail(detail)
        if safe_detail:
            parts.append(safe_detail)
        _append_text_line(path, " | ".join(parts))

    return _callback


def _compose_progress_callbacks(
    *callbacks: Optional[Callable[[str, str | None], None]],
) -> Optional[Callable[[str, str | None], None]]:
    active_callbacks = [callback for callback in callbacks if callback is not None]
    if not active_callbacks:
        return None

    def _callback(phase: str, detail: str | None = None) -> None:
        for callback in active_callbacks:
            callback(str(phase), detail)

    return _callback


def _progress_enabled() -> bool:
    if tqdm is None:
        return False
    stream = getattr(sys, "stderr", None)
    if stream is None:
        return False
    is_tty = getattr(stream, "isatty", None)
    return bool(is_tty and is_tty())


@contextmanager
def _temporary_hf_xet_mode(disable_xet: bool):
    env_key = "HF_HUB_DISABLE_XET"
    previous = os.environ.get(env_key)
    os.environ[env_key] = "1" if disable_xet else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


@contextmanager
def _model_download_lock(download_root: Path, spec: ModelSpec):
    lock_dir = Path(download_root).expanduser().resolve() / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{spec.model_slug}.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_queue_bar(total: int):
    if not _progress_enabled():
        return None
    return tqdm(
        total=int(total),
        desc="HF EWoK queue",
        unit="model",
        dynamic_ncols=True,
        leave=True,
    )


def _set_queue_progress(queue_bar: Any, *, model_slug: str, phase: str, detail: str | None = None) -> None:
    if queue_bar is None:
        return
    postfix = f"model={model_slug} | phase={phase}"
    if detail:
        postfix = f"{postfix} | {detail}"
    queue_bar.set_postfix_str(postfix, refresh=True)


def _notify_progress(
    progress_callback: Optional[Callable[[str, str | None], None]],
    phase: str,
    detail: str | None = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(str(phase), detail)
    except Exception:
        pass


def load_queue_config(config_path: str | os.PathLike[str]) -> list[ModelSpec]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Queue config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults: dict[str, Any] = {}
    if isinstance(raw, list):
        model_rows = raw
    elif isinstance(raw, dict):
        model_rows = raw.get("models", [])
        defaults = raw.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("Queue config 'defaults' must be a mapping if provided.")
    else:
        raise ValueError("Queue config must be a list or a mapping with a 'models' key.")

    if not isinstance(model_rows, list):
        raise ValueError("Queue config 'models' must be a list.")

    models: list[ModelSpec] = []
    for idx, row in enumerate(model_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Model entry at index {idx} must be a mapping.")

        merged = dict(defaults)
        merged.update(row)

        model_id = _optional_str(merged.get("model_id"))
        if not model_id:
            raise ValueError(f"Model entry at index {idx} is missing 'model_id'.")
        if "param_count_b" not in merged:
            raise ValueError(f"Model entry '{model_id}' is missing 'param_count_b'.")

        param_count_b = float(merged["param_count_b"])
        if param_count_b <= 0:
            raise ValueError(f"Model entry '{model_id}' must have param_count_b > 0.")

        models.append(
            ModelSpec(
                model_id=model_id,
                revision=_optional_str(merged.get("revision")),
                param_count_b=param_count_b,
                trust_remote_code=bool(merged.get("trust_remote_code", False)),
                tokenizer_id=_optional_str(merged.get("tokenizer_id")),
            )
        )

    models.sort(key=lambda spec: (float(spec.param_count_b), spec.model_id, spec.revision or ""))
    return models


def _detect_hardware_with_nvidia_smi() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue

        try:
            index = int(parts[0])
        except ValueError:
            continue

        total_mb = None
        free_mb = None
        try:
            total_mb = int(parts[2])
        except ValueError:
            pass
        try:
            free_mb = int(parts[3])
        except ValueError:
            pass

        gpus.append(
            {
                "index": index,
                "name": parts[1] or None,
                "total_memory_bytes": None if total_mb is None else int(total_mb * 1024 * 1024),
                "free_memory_bytes": None if free_mb is None else int(free_mb * 1024 * 1024),
            }
        )

    if not gpus:
        return None

    return {
        "cuda_available": True,
        "num_gpus": len(gpus),
        "gpus": gpus,
        "probe_method": "nvidia-smi",
    }


def _parse_cuda_visible_devices() -> tuple[bool, list[int] | None]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return False, None
    value = raw.strip()
    if value == "":
        return True, []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return True, []
    indices: list[int] = []
    for token in tokens:
        if token.isdigit():
            indices.append(int(token))
        else:
            return True, None
    return True, indices


def _apply_cuda_visible_devices(
    hardware: dict[str, Any],
    visible_indices: list[int],
) -> dict[str, Any] | None:
    if not visible_indices:
        return {
            "cuda_available": False,
            "num_gpus": 0,
            "gpus": [],
            "probe_method": f"{hardware.get('probe_method', 'unknown')}+masked",
        }
    gpus_by_physical = {int(gpu["index"]): gpu for gpu in hardware.get("gpus", [])}
    remapped: list[dict[str, Any]] = []
    for ordinal, physical_index in enumerate(visible_indices):
        gpu = gpus_by_physical.get(int(physical_index))
        if gpu is None:
            continue
        entry = dict(gpu)
        entry["physical_index"] = entry.get("index")
        entry["index"] = ordinal
        remapped.append(entry)
    if not remapped:
        return None
    return {
        "cuda_available": True,
        "num_gpus": len(remapped),
        "gpus": remapped,
        "probe_method": f"{hardware.get('probe_method', 'unknown')}+cuda_visible_devices",
    }


def _detect_hardware_with_torch() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "num_gpus": 0,
        "gpus": [],
        "probe_method": "torch_fallback",
    }
    if not hardware["cuda_available"]:
        return hardware

    try:
        count = int(torch.cuda.device_count())
    except Exception:
        return hardware

    hardware["num_gpus"] = count
    for idx in range(count):
        entry: dict[str, Any] = {
            "index": idx,
            "name": None,
            "total_memory_bytes": None,
            "free_memory_bytes": None,
        }
        try:
            props = torch.cuda.get_device_properties(idx)
            entry["name"] = props.name
            entry["total_memory_bytes"] = int(props.total_memory)
            # Avoid torch.cuda.mem_get_info() here because it can create a CUDA
            # context on each visible GPU just for bookkeeping.
            entry["free_memory_bytes"] = int(props.total_memory)
        except Exception:
            pass
        hardware["gpus"].append(entry)
    return hardware


def detect_hardware() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "num_gpus": 0,
            "gpus": [],
            "probe_method": "none",
        }

    has_cuda_env, visible = _parse_cuda_visible_devices()
    if has_cuda_env and visible == []:
        return {
            "cuda_available": False,
            "num_gpus": 0,
            "gpus": [],
            "probe_method": "cuda_visible_devices_empty",
        }

    if not has_cuda_env or visible is not None:
        hardware = _detect_hardware_with_nvidia_smi()
        if hardware is not None:
            if has_cuda_env and visible is not None:
                remapped = _apply_cuda_visible_devices(hardware, visible)
                if remapped is not None:
                    return remapped
            else:
                return hardware
    return _detect_hardware_with_torch()


def _dtype_bytes(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 4.0
    if dtype in (torch.float16, torch.bfloat16):
        return 2.0
    return 4.0


def _estimate_model_bytes(param_count_b: float, dtype: torch.dtype) -> float:
    return float(param_count_b) * 1_000_000_000.0 * _dtype_bytes(dtype)


def _memory_budget_bytes(gpu: dict[str, Any]) -> int:
    raw = gpu.get("free_memory_bytes") or gpu.get("total_memory_bytes") or 0
    if raw <= 0:
        return 0
    return int(raw * 0.85)


def _bytes_to_gib_string(num_bytes: int) -> str:
    gib = max(1, int(num_bytes // (1024 ** 3)))
    return f"{gib}GiB"


def _auto_dtype(dtype_name: str, hardware: dict[str, Any]) -> torch.dtype:
    resolved = str(dtype_name).strip().lower()
    if resolved == "float32":
        return torch.float32
    if resolved == "float16":
        return torch.float16
    if resolved == "bfloat16":
        return torch.bfloat16
    if hardware.get("cuda_available"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _bitsandbytes_available() -> bool:
    if BitsAndBytesConfig is None:
        return False
    return importlib.util.find_spec("bitsandbytes") is not None


def build_load_attempts(
    spec: ModelSpec,
    hardware: dict[str, Any],
    dtype: torch.dtype,
    *,
    bitsandbytes_available: bool,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    estimated_bytes = _estimate_model_bytes(spec.param_count_b, dtype)
    gpus = list(hardware.get("gpus", []))
    gpu_count = int(hardware.get("num_gpus", 0))

    if not hardware.get("cuda_available") or gpu_count <= 0:
        attempts.append(
            {
                "load_strategy": "cpu",
                "quantization": None,
                "device_mode": "cpu",
                "start_batch_size": 1,
            }
        )
        return attempts

    largest_budget = max((_memory_budget_bytes(gpu) for gpu in gpus), default=0)
    fits_single_gpu = estimated_bytes <= largest_budget if largest_budget > 0 else False

    if fits_single_gpu:
        attempts.append(
            {
                "load_strategy": "single_gpu",
                "quantization": None,
                "device_mode": "single_gpu",
                "start_batch_size": 4,
            }
        )

    if gpu_count > 1:
        attempts.append(
            {
                "load_strategy": "multi_gpu_shard",
                "quantization": None,
                "device_mode": "device_map_auto",
                "start_batch_size": 2,
            }
        )

    if bitsandbytes_available:
        attempts.append(
            {
                "load_strategy": "quantized_8bit",
                "quantization": "8bit",
                "device_mode": "device_map_auto",
                "start_batch_size": 2,
            }
        )
        attempts.append(
            {
                "load_strategy": "quantized_4bit",
                "quantization": "4bit",
                "device_mode": "device_map_auto",
                "start_batch_size": 2,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for attempt in attempts:
        key = (str(attempt["load_strategy"]), attempt.get("quantization"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(attempt)
    return deduped


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _is_capacity_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    capacity_markers = [
        "out of memory",
        "does not fit",
        "insufficient",
        "not enough memory",
        "modules are dispatched on the cpu or the disk",
        "offload",
    ]
    return any(marker in message for marker in capacity_markers)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _model_max_memory(hardware: dict[str, Any]) -> dict[int, str]:
    max_memory: dict[int, str] = {}
    for gpu in hardware.get("gpus", []):
        budget_bytes = _memory_budget_bytes(gpu)
        if budget_bytes <= 0:
            continue
        max_memory[int(gpu["index"])] = _bytes_to_gib_string(budget_bytes)
    return max_memory


def _list_model_repo_files(
    repo_id: str,
    *,
    revision: str | None,
    token: bool | str | None,
) -> list[str] | None:
    try:
        api = HfApi()
        return list(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
    except Exception:
        return None


def _preferred_weight_family(repo_files: Iterable[str] | None) -> str | None:
    if not repo_files:
        return None
    file_names = [Path(path).name for path in repo_files]

    if any(name.endswith(".safetensors") or name.endswith(".safetensors.index.json") for name in file_names):
        return "safetensors"
    if any(fnmatch.fnmatch(name, "pytorch_model*.bin") or name.endswith(".bin.index.json") for name in file_names):
        return "pytorch_bin"
    if any(name.endswith(".pt") for name in file_names):
        return "pt"
    if any(name.endswith(".msgpack") for name in file_names):
        return "flax_msgpack"
    return None


def _weight_family_ignore_patterns(repo_files: Iterable[str] | None) -> tuple[str, list[str] | None]:
    family = _preferred_weight_family(repo_files)
    if family is None:
        return "unfiltered", None
    return family, list(WEIGHT_FAMILY_IGNORE_PATTERNS.get(family, []))


def _resolve_existing_local_snapshot(
    spec: ModelSpec,
    *,
    download_root: Path,
) -> tuple[Path, Path, Path]:
    seen_roots: set[Path] = set()
    ordered_roots: list[Path] = []

    for root in (download_root, *KNOWN_DOWNLOADS_ROOTS):
        resolved_root = Path(root).expanduser().resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        ordered_roots.append(resolved_root)

    default_target_dir = ordered_roots[0] / spec.model_slug
    default_model_dir = default_target_dir / "model"

    for root in ordered_roots:
        target_dir = root / spec.model_slug
        model_dir = target_dir / "model"

        if _looks_like_complete_local_snapshot(model_dir):
            return target_dir, model_dir, model_dir

        # Older helper scripts downloaded snapshots directly into `target_dir`
        # rather than `target_dir/model`.
        if _looks_like_complete_local_snapshot(target_dir):
            return target_dir, target_dir, target_dir

    cached_snapshot_dir = _resolve_hf_hub_cached_snapshot(spec)
    if cached_snapshot_dir is not None:
        return cached_snapshot_dir, cached_snapshot_dir, cached_snapshot_dir

    return default_target_dir, default_model_dir, default_model_dir


def _weight_family_from_local_snapshot(snapshot_dir: Path) -> tuple[str, list[str] | None]:
    file_names = [path.name for path in snapshot_dir.iterdir() if path.is_file()]
    return _weight_family_ignore_patterns(file_names)


def _is_size_mismatch_error(message: str) -> bool:
    lowered = str(message).lower()
    return "consistency check failed" in lowered and "file should be of size" in lowered


def _extract_mismatch_filename(message: str) -> str | None:
    match = re.search(r"\(([^)]+)\)", str(message))
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


def _purge_partial_downloads(model_dir: Path, filename: str | None) -> None:
    if filename:
        direct_path = model_dir / filename
        if direct_path.exists():
            try:
                direct_path.unlink()
            except Exception:
                pass
        for path in model_dir.rglob(filename):
            try:
                if path.is_file():
                    path.unlink()
            except Exception:
                pass

    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    if cache_dir.exists():
        for path in cache_dir.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if path.name.endswith(".incomplete") or path.name.endswith(".lock"):
                    path.unlink()
                elif filename and filename in path.name:
                    path.unlink()
            except Exception:
                pass


def _verify_download_integrity(model_dir: Path) -> None:
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return
    incomplete = list(cache_dir.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"Incomplete download artifacts remain: {len(incomplete)} files.")


def _snapshot_has_complete_index(model_dir: Path) -> bool:
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in index_names:
        index_path = model_dir / index_name
        if not index_path.exists():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        expected_files = {str(value) for value in weight_map.values() if str(value).strip()}
        if not expected_files:
            return False
        return all((model_dir / filename).exists() for filename in expected_files)
    return False


def _has_tokenizer_files(snapshot_dir: Path) -> bool:
    tokenizer_names = {
        Path(pattern).name
        for pattern in TOKENIZER_ALLOW_PATTERNS
        if "*" not in pattern and "?" not in pattern
    }
    return any((snapshot_dir / name).exists() for name in tokenizer_names)


def _has_direct_weight_files(snapshot_dir: Path) -> bool:
    patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "*.pt",
        "*.msgpack",
    )
    for pattern in patterns:
        if any(snapshot_dir.glob(pattern)):
            return True
    return False


def _looks_like_complete_local_snapshot(snapshot_dir: Path) -> bool:
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return False
    if not (snapshot_dir / "config.json").exists():
        return False
    if not _has_tokenizer_files(snapshot_dir):
        return False
    try:
        _verify_download_integrity(snapshot_dir)
    except Exception:
        return False
    if _snapshot_has_complete_index(snapshot_dir):
        return True
    return _has_direct_weight_files(snapshot_dir)


def _hf_hub_cache_dir_name(model_id: str) -> str:
    return f"models--{str(model_id).strip().replace('/', '--')}"


def _candidate_hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.extend(
        [
            SYSTEM_HF_HUB_ROOT,
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _hf_hub_snapshot_candidates(spec: ModelSpec) -> list[Path]:
    candidates: list[Path] = []
    repo_dir_name = _hf_hub_cache_dir_name(spec.model_id)

    for hub_root in _candidate_hf_hub_roots():
        repo_dir = hub_root / repo_dir_name
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        if spec.revision:
            ref_path = repo_dir / "refs" / str(spec.revision)
            if ref_path.exists():
                try:
                    revision_hash = ref_path.read_text(encoding="utf-8").strip()
                except Exception:
                    revision_hash = ""
                if revision_hash:
                    candidates.append(snapshots_dir / revision_hash)
            candidates.append(snapshots_dir / str(spec.revision))
        else:
            main_ref = repo_dir / "refs" / "main"
            if main_ref.exists():
                try:
                    main_hash = main_ref.read_text(encoding="utf-8").strip()
                except Exception:
                    main_hash = ""
                if main_hash:
                    candidates.append(snapshots_dir / main_hash)

        try:
            snapshot_dirs = [
                item
                for item in snapshots_dir.iterdir()
                if item.exists() and item.is_dir()
            ]
        except Exception:
            snapshot_dirs = []
        candidates.extend(sorted(snapshot_dirs, key=lambda item: item.stat().st_mtime, reverse=True))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_hf_hub_cached_snapshot(spec: ModelSpec) -> Path | None:
    for snapshot_dir in _hf_hub_snapshot_candidates(spec):
        if _looks_like_complete_local_snapshot(snapshot_dir):
            return snapshot_dir
    return None


def _download_assets(
    spec: ModelSpec,
    *,
    download_root: Path,
    hf_token: str | None,
    disable_xet: bool = DEFAULT_DISABLE_XET,
    max_retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> dict[str, Any]:
    with _model_download_lock(download_root, spec):
        return _download_assets_unlocked(
            spec,
            download_root=download_root,
            hf_token=hf_token,
            disable_xet=disable_xet,
            max_retries=max_retries,
        )


def _download_assets_unlocked(
    spec: ModelSpec,
    *,
    download_root: Path,
    hf_token: str | None,
    disable_xet: bool = DEFAULT_DISABLE_XET,
    max_retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> dict[str, Any]:
    target_dir, model_dir, tokenizer_dir = _resolve_existing_local_snapshot(
        spec,
        download_root=download_root,
    )
    token_value: bool | str | None = hf_token if hf_token else None
    model_already_local = _looks_like_complete_local_snapshot(model_dir)

    if model_already_local:
        weight_family, ignore_patterns = _weight_family_from_local_snapshot(model_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        _purge_partial_downloads(model_dir, None)
        with _temporary_hf_xet_mode(disable_xet):
            repo_files = _list_model_repo_files(
                spec.model_id,
                revision=spec.revision,
                token=token_value,
            )
            weight_family, ignore_patterns = _weight_family_ignore_patterns(repo_files)

            snapshot_kwargs: dict[str, Any] = {
                "repo_id": spec.model_id,
                "revision": spec.revision,
                "local_dir": model_dir,
                "library_name": DEFAULT_LIBRARY_NAME,
                "library_version": DEFAULT_LIBRARY_VERSION,
                "token": token_value,
            }
            if ignore_patterns:
                snapshot_kwargs["ignore_patterns"] = ignore_patterns

            last_error: Exception | None = None
            retries = max(1, int(max_retries))
            for attempt in range(1, retries + 1):
                try:
                    if attempt > 1:
                        snapshot_kwargs["force_download"] = True
                    snapshot_download(
                        **snapshot_kwargs,
                    )
                    _verify_download_integrity(model_dir)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    message = str(exc)
                    if _is_size_mismatch_error(message):
                        filename = _extract_mismatch_filename(message)
                        _purge_partial_downloads(model_dir, filename)
                        if filename:
                            try:
                                snapshot_download(
                                    **snapshot_kwargs,
                                    allow_patterns=[filename],
                                    force_download=True,
                                )
                                _verify_download_integrity(model_dir)
                                last_error = None
                                break
                            except Exception as shard_exc:
                                last_error = shard_exc
                    if attempt < retries:
                        time.sleep(2.0 * attempt)
            if last_error is not None:
                raise last_error

            tokenizer_source = spec.tokenizer_id or spec.model_id
            tokenizer_mode = "model"
            tokenizer_override_dir = target_dir / "tokenizer"
            tokenizer_override_path = None

            if spec.tokenizer_id:
                tokenizer_path = Path(spec.tokenizer_id).expanduser()
                if tokenizer_path.exists():
                    tokenizer_dir = tokenizer_path.resolve()
                    tokenizer_mode = "local_path"
                elif spec.tokenizer_id != spec.model_id:
                    snapshot_download(
                        repo_id=spec.tokenizer_id,
                        revision=spec.revision,
                        local_dir=tokenizer_override_dir,
                        allow_patterns=TOKENIZER_ALLOW_PATTERNS,
                        library_name=DEFAULT_LIBRARY_NAME,
                        library_version=DEFAULT_LIBRARY_VERSION,
                        token=token_value,
                    )
                    tokenizer_dir = tokenizer_override_dir
                    tokenizer_mode = "downloaded_override"
                    tokenizer_override_path = tokenizer_override_dir
    if model_already_local:
        tokenizer_source = spec.tokenizer_id or spec.model_id
        tokenizer_mode = "model"
        tokenizer_override_path = None

    return {
        "download_dir": target_dir,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "tokenizer_source": tokenizer_source,
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_override_path": tokenizer_override_path,
        "weight_family": weight_family,
        "ignore_patterns": ignore_patterns,
    }


def _load_text_tokenizer_for_eval(
    tokenizer_dir: Path,
    *,
    trust_remote_code: bool,
) -> Any:
    errors: list[str] = []

    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            trust_remote_code=bool(trust_remote_code),
        )
    except Exception as exc:
        errors.append(f"AutoTokenizer: {str(exc)}")

    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                tokenizer_dir,
                local_files_only=True,
                trust_remote_code=bool(trust_remote_code),
            )
        except Exception as exc:
            errors.append(f"AutoProcessor: {str(exc)}")
        else:
            nested_tokenizer = getattr(processor, "tokenizer", None)
            if nested_tokenizer is not None:
                return nested_tokenizer
            return _ProcessorTextTokenizerAdapter(processor)

    raise RuntimeError(
        "Unable to construct a bare text tokenizer interface for evaluation. "
        "Tried AutoTokenizer first, then AutoProcessor without images or chat templates. "
        f"Errors: {' | '.join(errors) if errors else 'none recorded'}"
    )


@contextmanager
def _temporarily_disable_flash_attn_imports(enabled: bool):
    if not enabled:
        yield
        return

    patched: list[tuple[Any, str, Any]] = []
    module_names = (
        "transformers.utils.import_utils",
        "transformers.utils",
        "transformers.modeling_flash_attention_utils",
    )
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, "is_flash_attn_2_available"):
            continue
        patched.append((module, "is_flash_attn_2_available", getattr(module, "is_flash_attn_2_available")))
        setattr(module, "is_flash_attn_2_available", lambda: False)

    dropped_modules: dict[str, Any] = {}
    reload_prefixes = (
        "transformers.modeling_flash_attention_utils",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.smolvlm.modeling_smolvlm",
        "transformers.models.mllama.modeling_mllama",
        "transformers.models.llama.modeling_llama",
    )
    for module_name in list(sys.modules.keys()):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in reload_prefixes):
            dropped_modules[module_name] = sys.modules.pop(module_name)

    try:
        yield
    finally:
        for module, attr, original in reversed(patched):
            setattr(module, attr, original)
        for module_name, module in dropped_modules.items():
            if module_name not in sys.modules:
                sys.modules[module_name] = module


def _model_loader_candidates(spec: ModelSpec) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    if AutoModelForImageTextToText is not None and _looks_multimodal_model_id(spec.model_id):
        candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    return candidates


def _base_model_kwargs_for_attempt(
    spec: ModelSpec,
    *,
    attempt: dict[str, Any],
    dtype: torch.dtype,
    hardware: dict[str, Any],
    attn_implementation: str | None,
) -> dict[str, Any]:
    quantization = attempt.get("quantization")
    load_strategy = str(attempt["load_strategy"])
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "trust_remote_code": bool(spec.trust_remote_code),
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = str(attn_implementation)

    if load_strategy == "multi_gpu_shard":
        model_kwargs["device_map"] = "auto"
        max_memory = _model_max_memory(hardware)
        if max_memory:
            model_kwargs["max_memory"] = max_memory

    if quantization == "8bit":
        model_kwargs["device_map"] = "auto"
        max_memory = _model_max_memory(hardware)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if BitsAndBytesConfig is None:
            raise RuntimeError("8-bit loading requested but BitsAndBytesConfig is unavailable.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if quantization == "4bit":
        model_kwargs["device_map"] = "auto"
        max_memory = _model_max_memory(hardware)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if BitsAndBytesConfig is None:
            raise RuntimeError("4-bit loading requested but BitsAndBytesConfig is unavailable.")
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return model_kwargs


def _load_model_and_tokenizer_for_attempt(
    spec: ModelSpec,
    *,
    asset_paths: dict[str, Any],
    attempt: dict[str, Any],
    dtype: torch.dtype,
    hardware: dict[str, Any],
) -> tuple[Any, Any]:
    model_dir = Path(asset_paths["model_dir"])
    tokenizer_dir = asset_paths["tokenizer_dir"]
    load_strategy = str(attempt["load_strategy"])
    loader_errors: list[str] = []
    model = None
    eager_candidates: list[str | None] = [None]
    if _looks_multimodal_model_id(spec.model_id) or _flash_attn_is_broken():
        eager_candidates.append("eager")

    for loader_name, loader_cls in _model_loader_candidates(spec):
        for attn_implementation in eager_candidates:
            try:
                model_kwargs = _base_model_kwargs_for_attempt(
                    spec,
                    attempt=attempt,
                    dtype=dtype,
                    hardware=hardware,
                    attn_implementation=attn_implementation,
                )
                disable_flash = bool(attn_implementation == "eager" and _flash_attn_is_broken())
                with _temporarily_disable_flash_attn_imports(disable_flash):
                    model = loader_cls.from_pretrained(model_dir, **model_kwargs)
                setattr(model, "_hf_ewok_loader_name", loader_name)
                setattr(model, "_hf_ewok_attn_implementation", attn_implementation or "default")
                break
            except Exception as exc:
                loader_errors.append(
                    f"{loader_name}(attn={attn_implementation or 'default'}): {str(exc)}"
                )
                if _is_unrecognized_auto_model_config_error(exc):
                    break
                if _is_flash_attn_import_error(exc) and attn_implementation is None:
                    continue
                if attn_implementation == "eager":
                    continue
        if model is not None:
            break

    if model is None:
        raise RuntimeError(
            "All model loader candidates failed. "
            f"Tried: {' | '.join(loader_errors) if loader_errors else 'no loaders'}"
        )

    if load_strategy == "single_gpu":
        model.to(torch.device("cuda:0"))
    elif load_strategy == "cpu":
        model.to(torch.device("cpu"))
    model.eval()

    tokenizer = _load_text_tokenizer_for_eval(
        tokenizer_dir,
        trust_remote_code=bool(spec.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if getattr(model, "get_input_embeddings", None) is not None:
        embed = model.get_input_embeddings()
        tokenizer_size = None
        if embed is not None:
            try:
                tokenizer_size = len(tokenizer)
            except Exception:
                tokenizer_size = None
        if embed is not None and tokenizer_size is not None and embed.num_embeddings < tokenizer_size:
            model.resize_token_embeddings(tokenizer_size)

    return model, tokenizer


def _count_model_gpus(model: Any) -> int:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        gpu_ids: set[str] = set()
        for device in device_map.values():
            if isinstance(device, int):
                gpu_ids.add(str(device))
            elif isinstance(device, str) and device.startswith("cuda"):
                gpu_ids.add(device)
        if gpu_ids:
            return len(gpu_ids)
    try:
        device = next(model.parameters()).device
    except Exception:
        return 0
    return 1 if getattr(device, "type", "") == "cuda" else 0


def _load_shared_ewok_module() -> Any:
    os.environ["EWOK_VARIANT"] = "fast"
    module_name = "moonshotGPT.evaluation.ewok"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _ewok_official_average(metric_block: Optional[dict[str, Any]]) -> float | None:
    if not isinstance(metric_block, dict):
        return None
    domain_scores_full = metric_block.get("domain_scores_full")
    if not isinstance(domain_scores_full, dict):
        return None
    average = domain_scores_full.get("average")
    if isinstance(average, (list, tuple)) and average:
        return float(average[0])
    return None


def _ewok_full_mean_average(domain_scores_full: Optional[dict[str, Any]]) -> list[float] | None:
    if not isinstance(domain_scores_full, dict):
        return None
    average = domain_scores_full.get("average")
    if not isinstance(average, (list, tuple)) or not average:
        return None
    return [float(value) for value in average]


def _ewok_full_mean_clean_average(domain_scores_full: Optional[dict[str, Any]]) -> float | None:
    if not isinstance(domain_scores_full, dict):
        return None

    per_domain_scores: list[float] = []
    for domain_name, values in domain_scores_full.items():
        if str(domain_name) == "average":
            continue
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            per_domain_scores.append(0.5 * (float(values[0]) + float(values[1])))
        elif isinstance(values, (list, tuple)) and len(values) == 1:
            per_domain_scores.append(float(values[0]))
        elif isinstance(values, (int, float)):
            per_domain_scores.append(float(values))

    if per_domain_scores:
        return float(sum(per_domain_scores) / len(per_domain_scores))

    average = domain_scores_full.get("average")
    if isinstance(average, (list, tuple)) and average:
        return float(average[0])
    if isinstance(average, (int, float)):
        return float(average)
    return None


def _ewok_full_mean_average_member(domain_scores_full: Optional[dict[str, Any]], index: int) -> float | None:
    if not isinstance(domain_scores_full, dict):
        return None
    average = domain_scores_full.get("average")
    if not isinstance(average, (list, tuple)) or len(average) <= int(index):
        return None
    return float(average[int(index)])


def _build_ewok_payload(
    shared_ewok_module: Any,
    *,
    metrics_by_method_mean: dict[str, Any],
    per_item_records: list[dict[str, Any]],
    batch_size: int,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    babylm_mean = metrics_by_method_mean.get(shared_ewok_module.BABYLM_COMPLETION_CHOICE)
    context_mean = metrics_by_method_mean.get(shared_ewok_module.EWOK_CONTEXT_SENSITIVITY)
    pmi_method = getattr(shared_ewok_module, "PMI_COMPLETION_CHOICE", "pmi_completion_choice")
    pmi_mean = metrics_by_method_mean.get(pmi_method)
    babylm_full_mean = babylm_mean.get("domain_scores_full") if isinstance(babylm_mean, dict) else None
    context_full_mean = context_mean.get("domain_scores_full") if isinstance(context_mean, dict) else None
    pmi_full_mean = pmi_mean.get("domain_scores_full") if isinstance(pmi_mean, dict) else None

    summary = {
        "babylm_completion_choice_full_mean": babylm_full_mean,
        "ewok_context_sensitivity_full_mean": context_full_mean,
        "pmi_completion_choice_full_mean": pmi_full_mean,
        "babylm_completion_choice_full_mean_clean_average": _ewok_full_mean_clean_average(
            babylm_full_mean
        ),
        "ewok_context_sensitivity_full_mean_clean_average": _ewok_full_mean_clean_average(
            context_full_mean
        ),
        "pmi_completion_choice_full_mean_clean_average": _ewok_full_mean_clean_average(
            pmi_full_mean
        ),
        "babylm_completion_choice_official_mean_average": _ewok_official_average(babylm_mean),
        "ewok_context_sensitivity_official_mean_average": _ewok_official_average(context_mean),
        "pmi_completion_choice_official_mean_average": _ewok_official_average(pmi_mean),
        "pmi_completion_choice_symmetric_mean_average": _ewok_full_mean_average_member(
            pmi_full_mean,
            1,
        ),
        "num_items": int(len(shared_ewok_module.ewok_df)),
        "elapsed_seconds": float(elapsed_seconds),
    }

    payload = {
        "ewok_source": str(shared_ewok_module.SRC),
        "batch_size": int(batch_size),
        "elapsed_seconds": float(elapsed_seconds),
        "mean": {
            "metrics_by_method": metrics_by_method_mean,
            "num_items": int(len(shared_ewok_module.ewok_df)),
        },
        "summary": summary,
    }
    return payload, summary


def _evaluate_with_oom_retries(
    shared_ewok_module: Any,
    *,
    model: Any,
    tokenizer: Any,
    start_batch_size: int,
    status_callback: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    batch_size = max(1, int(start_batch_size))
    attempts: list[dict[str, Any]] = []
    while True:
        started = time.time()
        try:
            if status_callback is not None:
                status_callback(f"batch_size={int(batch_size)}")
            metrics_by_method_mean, per_item_records = shared_ewok_module.evaluate(
                model,
                tokenizer,
                batch_size=batch_size,
                return_per_item=True,
                score_reduction="mean",
                return_all_methods=True,
                show_progress=True,
            )
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "status": "completed",
                    "elapsed_seconds": float(time.time() - started),
                }
            )
            return metrics_by_method_mean, per_item_records, batch_size, attempts
        except RuntimeError as exc:
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "status": "failed",
                    "elapsed_seconds": float(time.time() - started),
                    "error": str(exc),
                }
            )
            if batch_size <= 1 or not _is_oom_error(exc):
                raise
            _release_memory()
            batch_size = max(1, batch_size // 2)


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _summarize_model_run(
    spec: ModelSpec,
    *,
    status: str,
    load_strategy: str | None,
    num_gpus_used: int | None,
    quantization: str | None,
    batch_size_final: int | None,
    elapsed_seconds: float | None,
    error: str | None,
    ewok_summary: dict[str, Any] | None,
    reused_existing: bool = False,
) -> dict[str, Any]:
    ewok_summary = ewok_summary or {}
    babylm_full_mean = ewok_summary.get("babylm_completion_choice_full_mean")
    context_full_mean = ewok_summary.get("ewok_context_sensitivity_full_mean")
    pmi_full_mean = ewok_summary.get("pmi_completion_choice_full_mean")
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "param_count_b": float(spec.param_count_b),
        "status": status,
        "load_strategy": load_strategy,
        "num_gpus_used": num_gpus_used,
        "quantization": quantization,
        "babylm_completion_choice_full_mean": babylm_full_mean,
        "ewok_context_sensitivity_full_mean": context_full_mean,
        "pmi_completion_choice_full_mean": pmi_full_mean,
        "ewok_babylm_completion_full_mean_avg": _ewok_full_mean_average(babylm_full_mean),
        "ewok_context_sensitivity_full_mean_avg": _ewok_full_mean_average(context_full_mean),
        "ewok_pmi_completion_full_mean_avg": _ewok_full_mean_average(pmi_full_mean),
        "ewok_babylm_completion_full_mean_clean_avg": ewok_summary.get(
            "babylm_completion_choice_full_mean_clean_average"
        ),
        "ewok_context_sensitivity_full_mean_clean_avg": ewok_summary.get(
            "ewok_context_sensitivity_full_mean_clean_average"
        ),
        "ewok_pmi_completion_full_mean_clean_avg": ewok_summary.get(
            "pmi_completion_choice_full_mean_clean_average"
        ),
        "ewok_babylm_completion_official_mean_avg": ewok_summary.get(
            "babylm_completion_choice_official_mean_average"
        ),
        "ewok_context_sensitivity_official_mean_avg": ewok_summary.get(
            "ewok_context_sensitivity_official_mean_average"
        ),
        "ewok_pmi_completion_official_mean_avg": ewok_summary.get(
            "pmi_completion_choice_official_mean_average"
        ),
        "ewok_pmi_completion_symmetric_mean_avg": ewok_summary.get(
            "pmi_completion_choice_symmetric_mean_average"
        ),
        "batch_size_final": batch_size_final,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
        "reused_existing": bool(reused_existing),
    }


def _existing_summary_row(spec: ModelSpec, output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    summary = _load_json(summary_path, None)
    if not isinstance(summary, dict):
        return None
    if summary.get("status") != "completed":
        return None
    row = _summarize_model_run(
        spec,
        status=str(summary.get("status")),
        load_strategy=_optional_str(summary.get("load_strategy")),
        num_gpus_used=summary.get("num_gpus_used"),
        quantization=_optional_str(summary.get("quantization")),
        batch_size_final=summary.get("batch_size_final"),
        elapsed_seconds=summary.get("elapsed_seconds"),
        error=_optional_str(summary.get("error")),
        ewok_summary={
            "babylm_completion_choice_full_mean": summary.get("babylm_completion_choice_full_mean")
            or (
                {"average": summary.get("ewok_babylm_completion_full_mean_avg")}
                if summary.get("ewok_babylm_completion_full_mean_avg") is not None
                else None
            ),
            "ewok_context_sensitivity_full_mean": summary.get("ewok_context_sensitivity_full_mean")
            or (
                {"average": summary.get("ewok_context_sensitivity_full_mean_avg")}
                if summary.get("ewok_context_sensitivity_full_mean_avg") is not None
                else None
            ),
            "pmi_completion_choice_full_mean": summary.get("pmi_completion_choice_full_mean")
            or (
                {"average": summary.get("ewok_pmi_completion_full_mean_avg")}
                if summary.get("ewok_pmi_completion_full_mean_avg") is not None
                else None
            ),
            "babylm_completion_choice_official_mean_average": summary.get(
                "ewok_babylm_completion_official_mean_avg"
            ),
            "ewok_context_sensitivity_official_mean_average": summary.get(
                "ewok_context_sensitivity_official_mean_avg"
            ),
            "pmi_completion_choice_full_mean_clean_average": summary.get(
                "ewok_pmi_completion_full_mean_clean_avg"
            ),
            "pmi_completion_choice_official_mean_average": summary.get(
                "ewok_pmi_completion_official_mean_avg"
            ),
            "pmi_completion_choice_symmetric_mean_average": summary.get(
                "ewok_pmi_completion_symmetric_mean_avg"
            ),
        },
        reused_existing=True,
    )
    return row


def run_model(
    spec: ModelSpec,
    *,
    output_root: str | os.PathLike[str] = DEFAULT_OUTPUT_ROOT,
    downloads_root: str | os.PathLike[str] = DEFAULT_DOWNLOADS_ROOT,
    dtype_name: str = "auto",
    hf_token: str | None = None,
    disable_xet: bool = DEFAULT_DISABLE_XET,
    download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    download_only: bool = False,
    shared_ewok_module: Any | None = None,
    hardware: dict[str, Any] | None = None,
    progress_callback: Optional[Callable[[str, str | None], None]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_root).expanduser().resolve() / spec.model_slug
    download_root = Path(downloads_root).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    ewok_metrics_path = output_dir / "ewok_metrics.json"
    ewok_items_path = output_dir / "ewok_items.jsonl"
    model_log_path = output_dir / DEFAULT_MODEL_LOG_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_callback = _compose_progress_callbacks(
        progress_callback,
        _make_phase_logger(model_log_path, model_slug=spec.model_slug),
    )

    hardware_info = hardware or detect_hardware()
    dtype = _auto_dtype(dtype_name, hardware_info)
    bnb_available = _bitsandbytes_available()
    load_attempts = build_load_attempts(
        spec,
        hardware_info,
        dtype,
        bitsandbytes_available=bnb_available,
    )

    manifest: dict[str, Any] = {
        "model": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "param_count_b": float(spec.param_count_b),
            "trust_remote_code": bool(spec.trust_remote_code),
            "tokenizer_id": spec.tokenizer_id,
            "model_slug": spec.model_slug,
        },
        "started_at": _utc_now_iso(),
        "status": "running",
        "paths": {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
            "ewok_metrics_path": str(ewok_metrics_path),
            "ewok_items_path": str(ewok_items_path),
            "run_log_path": str(model_log_path),
            "downloads_root": str(download_root),
        },
        "hardware": hardware_info,
        "dtype": str(dtype),
        "bitsandbytes_available": bool(bnb_available),
        "download_settings": {
            "disable_xet": bool(disable_xet),
            "hf_hub_disable_xet": "1" if disable_xet else "0",
            "max_retries": int(download_retries),
        },
        "download_only": bool(download_only),
        "download": {},
        "load_attempts": [],
        "evaluation": {},
        "cleanup": {
            "attempted": False,
            "removed_download_dir": False,
            "error": None,
        },
    }
    _atomic_write_json(manifest_path, manifest)

    started_at = time.time()
    cleanup_path: Path | None = download_root / spec.model_slug
    model = None
    tokenizer = None
    status = "failed"
    top_error: str | None = None
    top_traceback: str | None = None
    ewok_summary: dict[str, Any] | None = None
    load_strategy: str | None = None
    quantization: str | None = None
    num_gpus_used: int | None = None
    batch_size_final: int | None = None

    try:
        _notify_progress(progress_callback, "start", spec.model_id)
        _notify_progress(
            progress_callback,
            "download",
            f"{spec.model_id} xet={'disabled' if disable_xet else 'enabled'}",
        )
        download_started = time.time()
        asset_paths = _download_assets(
            spec,
            download_root=download_root,
            hf_token=hf_token,
            disable_xet=disable_xet,
            max_retries=download_retries,
        )
        cleanup_path = Path(asset_paths["download_dir"])
        manifest["download"] = {
            "status": "completed",
            "elapsed_seconds": float(time.time() - download_started),
            "disable_xet": bool(disable_xet),
            "download_dir": str(asset_paths["download_dir"]),
            "model_dir": str(asset_paths["model_dir"]),
            "tokenizer_dir": str(asset_paths["tokenizer_dir"]),
            "tokenizer_source": asset_paths["tokenizer_source"],
            "tokenizer_mode": asset_paths["tokenizer_mode"],
            "weight_family": asset_paths.get("weight_family"),
            "ignore_patterns": asset_paths.get("ignore_patterns"),
        }
        _atomic_write_json(manifest_path, manifest)
        _notify_progress(
            progress_callback,
            "downloaded",
            f"{float(time.time() - download_started):.1f}s",
        )
    except Exception as exc:
        status = "failed_download"
        top_error = str(exc)
        top_traceback = traceback.format_exc()
        manifest["status"] = status
        manifest["download"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": top_traceback,
        }
        _atomic_write_json(manifest_path, manifest)
        _notify_progress(progress_callback, status, str(exc))
    else:
        if download_only:
            status = "downloaded"
            top_error = None
            ewok_summary = {}
            _notify_progress(progress_callback, status, "download only; skipping evaluation")
        elif not load_attempts:
            status = "failed_capacity"
            top_error = "No viable load strategy was available for the detected hardware."
            _notify_progress(progress_callback, status, top_error)
        else:
            shared_ewok = shared_ewok_module or _load_shared_ewok_module()
            for attempt_idx, attempt in enumerate(load_attempts, start=1):
                attempt_record: dict[str, Any] = {
                    "load_strategy": attempt["load_strategy"],
                    "quantization": attempt.get("quantization"),
                    "device_mode": attempt["device_mode"],
                    "start_batch_size": int(attempt["start_batch_size"]),
                    "started_at": _utc_now_iso(),
                }
                attempt_started = time.time()
                try:
                    _notify_progress(
                        progress_callback,
                        "load",
                        f"attempt {attempt_idx}/{len(load_attempts)} {attempt['load_strategy']}",
                    )
                    model, tokenizer = _load_model_and_tokenizer_for_attempt(
                        spec,
                        asset_paths=asset_paths,
                        attempt=attempt,
                        dtype=dtype,
                        hardware=hardware_info,
                    )
                    attempt_record["status"] = "model_loaded"
                    attempt_record["load_elapsed_seconds"] = float(time.time() - attempt_started)
                    load_strategy = str(attempt["load_strategy"])
                    quantization = attempt.get("quantization")
                    num_gpus_used = _count_model_gpus(model)
                    attempt_record["model_loader"] = getattr(model, "_hf_ewok_loader_name", None)
                    attempt_record["attn_implementation"] = getattr(
                        model,
                        "_hf_ewok_attn_implementation",
                        None,
                    )

                    eval_started = time.time()
                    _notify_progress(
                        progress_callback,
                        "evaluate",
                        f"{load_strategy} batch_size={int(attempt['start_batch_size'])}",
                    )
                    metrics_by_method_mean, per_item_records, batch_size_final, eval_attempts = _evaluate_with_oom_retries(
                        shared_ewok,
                        model=model,
                        tokenizer=tokenizer,
                        start_batch_size=int(attempt["start_batch_size"]),
                        status_callback=lambda detail: _notify_progress(
                            progress_callback,
                            "evaluate",
                            f"{load_strategy} {detail}",
                        ),
                    )
                    ewok_payload, ewok_summary = _build_ewok_payload(
                        shared_ewok,
                        metrics_by_method_mean=metrics_by_method_mean,
                        per_item_records=per_item_records,
                        batch_size=batch_size_final,
                        elapsed_seconds=float(time.time() - eval_started),
                    )
                    _atomic_write_json(ewok_metrics_path, ewok_payload)
                    _write_jsonl(ewok_items_path, per_item_records)

                    attempt_record["status"] = "completed"
                    attempt_record["num_gpus_used"] = num_gpus_used
                    attempt_record["evaluation_attempts"] = eval_attempts
                    attempt_record["evaluation_elapsed_seconds"] = float(time.time() - eval_started)
                    manifest["load_attempts"].append(attempt_record)
                    status = "completed"
                    top_error = None
                    top_traceback = None
                    _notify_progress(
                        progress_callback,
                        status,
                        f"{load_strategy} batch_size={int(batch_size_final)}",
                    )
                    break
                except Exception as exc:
                    attempt_record["status"] = "failed"
                    attempt_record["error"] = str(exc)
                    attempt_record["traceback"] = traceback.format_exc()
                    attempt_record["elapsed_seconds"] = float(time.time() - attempt_started)
                    manifest["load_attempts"].append(attempt_record)
                    top_error = str(exc)
                    top_traceback = attempt_record["traceback"]
                    _release_memory()
                    model = None
                    tokenizer = None
                    _notify_progress(
                        progress_callback,
                        "retry" if _is_capacity_error(exc) else "failed_load",
                        f"{attempt['load_strategy']}: {str(exc)}",
                    )
                    if not _is_capacity_error(exc):
                        status = "failed_load" if attempt_record["status"] == "failed" else "failed"
                        break
            else:
                status = "failed_capacity" if top_error else "failed_capacity"

        if status != "completed" and status not in {"failed_download", "failed_load"}:
            if top_error is None and not load_attempts:
                top_error = "No viable load strategy was available for the detected hardware."
            status = "failed_capacity" if _optional_str(top_error) else status

    finally:
        _notify_progress(progress_callback, "cleanup", None)
        model = None
        tokenizer = None
        _release_memory()

        cleanup_error = None
        removed = False
        if cleanup_path is not None and not download_only:
            try:
                _remove_tree(cleanup_path)
                removed = not cleanup_path.exists()
            except Exception as exc:
                cleanup_error = str(exc)

        elapsed_seconds = float(time.time() - started_at)
        summary = _summarize_model_run(
            spec,
            status=status,
            load_strategy=load_strategy,
            num_gpus_used=num_gpus_used,
            quantization=quantization,
            batch_size_final=batch_size_final,
            elapsed_seconds=elapsed_seconds,
            error=top_error,
            ewok_summary=ewok_summary,
        )
        summary.update(
            {
                "started_at": manifest["started_at"],
                "finished_at": _utc_now_iso(),
            }
        )
        _atomic_write_json(summary_path, summary)

        manifest["status"] = status
        manifest["finished_at"] = summary["finished_at"]
        manifest["error"] = top_error
        if top_traceback is not None:
            manifest["traceback"] = top_traceback
        manifest["evaluation"].update(
            {
                "batch_size_final": batch_size_final,
                "load_strategy": load_strategy,
                "num_gpus_used": num_gpus_used,
                "quantization": quantization,
                "summary": ewok_summary,
            }
        )
        manifest["cleanup"] = {
            "attempted": cleanup_path is not None,
            "removed_download_dir": bool(removed),
            "error": cleanup_error,
        }
        _atomic_write_json(manifest_path, manifest)
        _notify_progress(progress_callback, status, summary.get("error"))

    return summary


def _write_queue_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(output_root / "queue_results.jsonl", rows)
    _write_csv(output_root / "queue_summary.csv", rows, QUEUE_SUMMARY_FIELDS)


def run_queue(
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    output_root: str | os.PathLike[str] = DEFAULT_OUTPUT_ROOT,
    downloads_root: str | os.PathLike[str] = DEFAULT_DOWNLOADS_ROOT,
    dtype_name: str = "auto",
    force: bool = False,
    hf_token: str | None = None,
    disable_xet: bool = DEFAULT_DISABLE_XET,
    download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    download_only: bool = False,
) -> list[dict[str, Any]]:
    specs = load_queue_config(config_path)
    if not specs:
        raise ValueError(f"No models were found in queue config: {config_path}")

    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    queue_log_path = resolved_output_root / DEFAULT_QUEUE_LOG_NAME
    rows: list[dict[str, Any]] = []
    hardware = detect_hardware()
    queue_logger = _make_phase_logger(queue_log_path)
    queue_logger(
        "queue_start",
        (
            f"config={Path(config_path).expanduser().resolve()} "
            f"downloads_root={Path(downloads_root).expanduser().resolve()} "
            f"dtype={dtype_name} force={bool(force)} "
            f"disable_xet={bool(disable_xet)} models={len(specs)}"
        ),
    )

    queue_bar = _build_queue_bar(len(specs))
    try:
        for spec in specs:
            model_output_dir = resolved_output_root / spec.model_slug
            if not force:
                existing = _existing_summary_row(spec, model_output_dir)
                if existing is not None:
                    rows.append(existing)
                    _write_queue_outputs(resolved_output_root, rows)
                    queue_logger("reuse", f"model={spec.model_slug} status=completed")
                    _set_queue_progress(queue_bar, model_slug=spec.model_slug, phase="reuse", detail="completed")
                    if queue_bar is not None:
                        queue_bar.update(1)
                    continue

            row = run_model(
                spec,
                output_root=resolved_output_root,
                downloads_root=downloads_root,
                dtype_name=dtype_name,
                hf_token=hf_token,
                disable_xet=disable_xet,
                download_retries=download_retries,
                download_only=download_only,
                hardware=hardware,
                progress_callback=_compose_progress_callbacks(
                    lambda phase, detail, model_slug=spec.model_slug: _set_queue_progress(
                        queue_bar,
                        model_slug=model_slug,
                        phase=phase,
                        detail=detail,
                    ),
                    _make_phase_logger(queue_log_path, model_slug=spec.model_slug),
                ),
            )
            rows.append(row)
            _write_queue_outputs(resolved_output_root, rows)
            _set_queue_progress(queue_bar, model_slug=spec.model_slug, phase=row["status"], detail=None)
            if queue_bar is not None:
                queue_bar.update(1)
    finally:
        if queue_bar is not None:
            queue_bar.close()
        completed = sum(1 for row in rows if row.get("status") == "completed")
        failed = sum(1 for row in rows if row.get("status") != "completed")
        queue_logger("queue_end", f"completed={completed} failed={failed} rows={len(rows)}")

    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially download Hugging Face causal LMs, evaluate fast EWoK "
            "with Moonshot mean scoring, write artifacts, and remove the local snapshot."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML queue config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for per-model result artifacts.",
    )
    parser.add_argument(
        "--downloads-root",
        type=str,
        default=str(DEFAULT_DOWNLOADS_ROOT),
        help="Root directory for temporary local model snapshots.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Requested model dtype.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if summary.json already records a completed run.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Defaults to the logged-in CLI token when omitted.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download model files (with verification) and skip evaluation.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=DEFAULT_DOWNLOAD_RETRIES,
        help="Number of download retry attempts for large models (default: 3).",
    )
    xet_group = parser.add_mutually_exclusive_group()
    xet_group.add_argument(
        "--disable-xet",
        dest="disable_xet",
        action="store_true",
        help="Disable the hf_xet/Xet download backend and force regular HTTP downloads.",
    )
    xet_group.add_argument(
        "--enable-xet",
        dest="disable_xet",
        action="store_false",
        help="Allow Hugging Face to use the hf_xet/Xet download backend.",
    )
    parser.set_defaults(disable_xet=DEFAULT_DISABLE_XET)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    rows = run_queue(
        config_path=args.config,
        output_root=args.output_root,
        downloads_root=args.downloads_root,
        dtype_name=args.dtype,
        force=bool(args.force),
        hf_token=args.hf_token,
        disable_xet=bool(args.disable_xet),
        download_retries=int(args.download_retries),
        download_only=bool(args.download_only),
    )
    completed = sum(1 for row in rows if row.get("status") == "completed")
    failed = sum(1 for row in rows if row.get("status") != "completed")
    print(
        f"Processed {len(rows)} models: {completed} completed, {failed} non-completed. "
        f"Queue summary: {Path(args.output_root).expanduser().resolve() / 'queue_summary.csv'}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
