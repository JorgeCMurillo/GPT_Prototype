# train_gpt2_finewebedu_bin.py
import os, random, argparse, json, math, inspect, subprocess, sys, re
from datetime import datetime
from contextlib import nullcontext
import gc

import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from tqdm import tqdm
import time

import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM

from torch.optim import AdamW

from transformers import (
    AutoTokenizer, GPT2Config,
)

from accelerate.utils import DataLoaderConfiguration
from accelerate import Accelerator

from evaluation.ewok import evaluate, ewok_df as EWOK_DF

from shard_loader import make_dataloader
from evaluation.runner import (
    build_ewok_row_category_lookup,
    run_ewok_eval_step,
    run_final_ewok_eval_main_process,
    run_hellaswag_eval_step,
    run_parallel_validation,
)
from training_utils.resume_trim import (
    derive_resume_skip_microsteps,
    fast_forward_iterator_data_only,
    trim_run_logs_after_step,
    validate_replay_compatibility,
)

try:
    from evaluation import hellaswag as hellaswag_eval
except Exception:
    hellaswag_eval = None


# -----------------------------
# JSON helpers

def to_jsonable(x):
    """Convert tensors / numpy / scalars inside dicts to JSON-safe Python types."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.ndim > 0 else x.item()
    return x


def save_metrics(metrics_list, out_path):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(to_jsonable(metrics_list), f, indent=2)
    os.replace(tmp_path, out_path)  # atomic write


def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")


def atomic_write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Repro + env

def set_all_seeds(seed_value: int):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)


# -----------------------------
# Norms + LR

def get_current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def get_llmc_lr(
    step: int,
    learning_rate: float,
    warmup_iters: int,
    num_iterations: int,
    learning_rate_decay_frac: float,
) -> float:
    min_lr = learning_rate * learning_rate_decay_frac
    if warmup_iters > 0 and step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if step > num_iterations:
        return min_lr
    if num_iterations <= warmup_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (num_iterations - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def build_llmc_style_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    device: torch.device,
):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"
    optimizer_kwargs = {"fused": use_fused} if fused_available else {}

    optimizer = AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
        **optimizer_kwargs,
    )
    return optimizer, len(decay_params), len(nodecay_params), use_fused


@torch.no_grad()
def global_param_norm_l2(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p is None:
            continue
        v = p.detach()
        tot += float(v.float().pow(2).sum().item())
    return float(tot ** 0.5)


@torch.no_grad()
def global_grad_norm_l2(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        tot += float(g.float().pow(2).sum().item())
    return float(tot ** 0.5)


def _flatten_meta(meta_obj):
    """Return a list[dict] from meta which might be dict, list of dict, etc."""
    out = []
    if meta_obj is None:
        return out
    if isinstance(meta_obj, dict):
        return [meta_obj]
    if isinstance(meta_obj, (list, tuple)):
        for x in meta_obj:
            out.extend(_flatten_meta(x))
    return out


def _to_scalar_int(x) -> int:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(x.shape)}")
        return int(x.item())
    return int(x)


def _resolve_ref_loss_file(ref_loss_dir: str, shard_path: str):
    base = os.path.basename(str(shard_path))
    if base.endswith(".bin"):
        base = base[:-4]
    for suffix, dtype in (("f16", np.float16), ("f32", np.float32)):
        p = os.path.join(ref_loss_dir, f"{base}.ref_loss.{suffix}.bin")
        if os.path.exists(p):
            return p, dtype
    raise FileNotFoundError(
        f"No ref-loss file found for shard '{shard_path}' in '{ref_loss_dir}'. "
        "Expected one of: *.ref_loss.f16.bin or *.ref_loss.f32.bin"
    )


def _list_split_shards(data_dir: str, split: str):
    prefix = f"{split}_"
    shards = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.startswith(prefix) and name.endswith(".bin")
    ]
    if not shards:
        raise FileNotFoundError(
            f"No shards found for split='{split}' in '{data_dir}' "
            f"(expected files like {split}_*.bin)."
        )
    return shards


def _count_uint16_tokens(path: str) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"uint16 shard has odd byte size (not divisible by 2): {path}")
    return nbytes // 2


def _count_ref_loss_tokens(path: str, ref_dtype) -> int:
    itemsize = int(np.dtype(ref_dtype).itemsize)
    nbytes = os.path.getsize(path)
    if nbytes % itemsize != 0:
        raise ValueError(
            f"Ref-loss file byte size not divisible by dtype itemsize={itemsize}: {path}"
        )
    return nbytes // itemsize


def _ref_loss_meta_path(ref_loss_path: str) -> str:
    if not ref_loss_path.endswith(".bin"):
        raise ValueError(f"Unexpected ref-loss filename (expected .bin): {ref_loss_path}")
    return ref_loss_path[:-4] + ".meta.json"


def _validate_rho_ref_loss_alignment(
    data_dir: str,
    ref_loss_dir: str,
    seq_len: int,
    micro_batch_size: int,
    split: str = "train",
    max_reported_errors: int = 12,
) -> int:
    """
    Fail-fast validation for rho reference losses before training starts.

    Checks for each split shard:
    - matching ref-loss file exists
    - ref-loss token count matches source shard token count
    - metadata file exists and has matching seq_len / batch_size / stride / block
    """
    shards = _list_split_shards(data_dir, split)
    expected_seq_len = int(seq_len)
    expected_bs = int(micro_batch_size)
    expected_stride = expected_seq_len * expected_bs
    expected_block = expected_stride + 1

    errors = []

    def _expect_meta_int(meta: dict, key: str, expected: int, shard_name: str):
        if key not in meta:
            errors.append(
                f"{shard_name}: meta missing key '{key}' (expected {expected})."
            )
            return
        try:
            got = int(meta.get(key))
        except Exception:
            errors.append(
                f"{shard_name}: meta key '{key}' is non-integer ({meta.get(key)!r}); expected {expected}."
            )
            return
        if got != int(expected):
            errors.append(
                f"{shard_name}: meta key '{key}'={got} but expected {expected}."
            )

    for shard_path in shards:
        shard_name = os.path.basename(shard_path)

        try:
            shard_tokens = _count_uint16_tokens(shard_path)
        except Exception as exc:
            errors.append(f"{shard_name}: cannot read shard token count: {exc}")
            continue

        try:
            ref_path, ref_dtype = _resolve_ref_loss_file(ref_loss_dir, shard_path)
        except Exception as exc:
            errors.append(f"{shard_name}: {exc}")
            continue

        try:
            ref_tokens = _count_ref_loss_tokens(ref_path, ref_dtype)
        except Exception as exc:
            errors.append(f"{shard_name}: cannot read ref-loss token count: {exc}")
            continue
        if ref_tokens != shard_tokens:
            errors.append(
                f"{shard_name}: ref token count mismatch: shard={shard_tokens}, ref={ref_tokens} ({ref_path})."
            )

        meta_path = _ref_loss_meta_path(ref_path)
        if not os.path.exists(meta_path):
            errors.append(
                f"{shard_name}: missing ref-loss metadata file: {meta_path}"
            )
            continue

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as exc:
            errors.append(f"{shard_name}: failed to parse meta JSON {meta_path}: {exc}")
            continue

        _expect_meta_int(meta, "seq_len", expected_seq_len, shard_name)
        _expect_meta_int(meta, "batch_size", expected_bs, shard_name)
        _expect_meta_int(meta, "stride_tokens", expected_stride, shard_name)
        _expect_meta_int(meta, "block_tokens", expected_block, shard_name)

        if "source_num_tokens" in meta:
            try:
                source_tokens = int(meta.get("source_num_tokens"))
                if source_tokens != shard_tokens:
                    errors.append(
                        f"{shard_name}: meta source_num_tokens={source_tokens} but shard has {shard_tokens}."
                    )
            except Exception:
                errors.append(
                    f"{shard_name}: meta source_num_tokens is non-integer ({meta.get('source_num_tokens')!r})."
                )

        source_base = meta.get("source_shard_basename")
        if source_base is not None and str(source_base) != shard_name:
            errors.append(
                f"{shard_name}: meta source_shard_basename={source_base!r} does not match shard name."
            )

    if errors:
        shown = errors[:max_reported_errors]
        lines = "\n".join(f"  - {msg}" for msg in shown)
        hidden = max(0, len(errors) - len(shown))
        if hidden > 0:
            lines += f"\n  - ... and {hidden} more validation errors."
        raise ValueError(
            "Rho ref-loss preflight validation failed before training.\n"
            f"Checked split='{split}' in data_dir='{data_dir}' against ref_loss_dir='{ref_loss_dir}'.\n"
            f"Expected seq_len={expected_seq_len}, micro_batch_size={expected_bs}.\n"
            "Common fix: recompute ref-loss shards with matching --seq_len and --batch_size.\n"
            f"Details:\n{lines}"
        )

    return len(shards)


def _checkpoint_step_from_dirname(path: str):
    name = os.path.basename(os.path.normpath(path))
    m = re.match(r"^ckpt_[^/]*_step(\d+)$", name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_resume_paths(resume_from_run: str):
    """
    Accept either:
    - a run directory containing ckpt_*_stepXXXXXXX folders
    - a specific checkpoint directory ckpt_*_stepXXXXXXX

    Returns (run_dir, ckpt_dir, step_hint).
    """
    raw = os.path.abspath(str(resume_from_run))
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"--resume_from_run path does not exist or is not a directory: {raw}")

    step_direct = _checkpoint_step_from_dirname(raw)
    if step_direct is not None:
        run_dir = os.path.dirname(raw)
        return run_dir, raw, int(step_direct)

    cands = []
    for name in sorted(os.listdir(raw)):
        ckpt_dir = os.path.join(raw, name)
        if not os.path.isdir(ckpt_dir):
            continue
        step = _checkpoint_step_from_dirname(ckpt_dir)
        if step is None:
            continue
        cands.append((int(step), float(os.path.getmtime(ckpt_dir)), ckpt_dir))

    if not cands:
        raise FileNotFoundError(
            f"No checkpoint directories found under '{raw}'. "
            "Expected folders named like ckpt_<tag>_step0001234."
        )

    cands.sort(key=lambda x: (x[0], x[1]))
    step, _mtime, ckpt_dir = cands[-1]
    return raw, ckpt_dir, int(step)


def _validate_ckpt_model_config_alignment(
    ckpt_dir: str,
    seq_len: int,
    vocab_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Checkpoint missing config.json: {cfg_path}")

    try:
        cfg = load_json(cfg_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint config: {cfg_path}: {exc}") from exc

    mismatches = []

    def _check(keys, expected: int, label: str):
        got = None
        for k in keys:
            if k in cfg:
                got = int(cfg[k])
                break
        if got is None:
            return
        if int(got) != int(expected):
            mismatches.append(f"{label}: ckpt={got}, requested={int(expected)}")

    _check(["vocab_size"], vocab_size, "vocab_size")
    _check(["n_embd"], n_embd, "n_embd")
    _check(["n_head"], n_head, "n_head")
    _check(["n_layer"], n_layer, "n_layer")
    # GPT2 config usually stores both n_positions and n_ctx; either must match seq_len.
    _check(["n_positions", "n_ctx"], seq_len, "seq_len")

    if mismatches:
        raise ValueError(
            "Checkpoint architecture does not match requested training args.\n"
            f"checkpoint: {ckpt_dir}\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


def _load_existing_step_metrics(metrics_path: str):
    if not os.path.exists(metrics_path):
        return []
    try:
        data = load_json(metrics_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse existing step_metrics file '{metrics_path}': {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected list in step_metrics file: {metrics_path}")
    return data


def _load_trainer_state(ckpt_dir: str):
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return {}
    try:
        data = load_json(state_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse trainer state '{state_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in trainer state file: {state_path}")
    return data


def _save_trainer_state(ckpt_dir: str, state: dict):
    atomic_write_json(os.path.join(ckpt_dir, "trainer_state.json"), state)


def _load_ref_loss_batch(
    ref_loss_dir: str,
    shard_meta: dict,
    expected_bt: int,
    ref_cache: dict,
):
    shard_path = shard_meta.get("shard_path", None)
    if shard_path is None:
        raise KeyError("Batch meta missing shard_path; cannot map to ref-loss file.")
    start = _to_scalar_int(shard_meta.get("start"))
    end = _to_scalar_int(shard_meta.get("end"))
    if (end - start) != (expected_bt + 1):
        raise ValueError(
            f"Unexpected block span from meta: end-start={end-start}, expected {expected_bt+1}. "
            "Reference-loss alignment requires matching seq_len and micro_batch_size."
        )

    cache_key = str(shard_path)
    if cache_key not in ref_cache:
        ref_path, ref_dtype = _resolve_ref_loss_file(ref_loss_dir, cache_key)
        ref_cache[cache_key] = {
            "path": ref_path,
            "dtype": ref_dtype,
            "mm": np.memmap(ref_path, dtype=ref_dtype, mode="r"),
        }
    mm = ref_cache[cache_key]["mm"]

    lo = start + 1
    hi = end
    arr = np.asarray(mm[lo:hi], dtype=np.float32)
    if arr.size != expected_bt:
        raise ValueError(
            f"Ref-loss slice size mismatch for shard '{shard_path}': got {arr.size}, expected {expected_bt}."
        )
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError(
            f"Ref-loss slice has no finite values for shard '{shard_path}' [{lo}:{hi}]. "
            "Check precompute alignment (same seq_len and micro_batch_size)."
        )
    # Keep non-finite values out of mask selection by making them very small score candidates.
    arr = np.where(valid, arr, np.float32(1e9)).astype(np.float32, copy=False)
    return arr, valid


def _resolve_mixed_precision(requested: str) -> str:
    if requested != "bf16":
        return requested
    if not torch.cuda.is_available():
        return "no"
    if not torch.cuda.is_bf16_supported():
        return "fp16"
    return "bf16"


def release_eval_memory():
    """Best-effort cleanup before heavy eval runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# Main

def main(
    seed: int,
    micro_batch_size: int,
    total_batch_tokens: int,
    max_train_steps: int,
    data_dir: str,
    experiments_dir: str = "experiments",
    seq_len: int = 1024,
    vocab_size: int = 50257,
    n_embd: int = 768,
    n_head: int = 12,
    n_layer: int = 12,
    num_workers: int = 0,
    shuffle_blocks: bool = True,
    grad_clip: float = 1.0,
    learning_rate: float = 6e-4,
    warmup_iters: int = 700,
    learning_rate_decay_frac: float = 0.0,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eval_every: int = 2000,
    hellaswag_every: int = 500,
    hellaswag_batch_size: int = 8,
    hellaswag_max_examples=1024,
    hellaswag_dataset: str = "hellaswag",
    hellaswag_dataset_config: str = None,
    hellaswag_split: str = "validation",
    hellaswag_local_files_only: bool = False,
    ewok_every: int = 250,
    ewok_batch_size: int = 8,
    skip_final_ewok: bool = False,
    save_every: int = 2000,
    exposure_every: int = 50,
    push_to_hub: bool = False,
    mixed_precision: str = "bf16",
    rho_ref_loss_dir: str = "",
    rho_keep_frac: float = 1.0,
    rho_warmup_steps: int = 0,
    rho_mode: str = "delta",
    rho_ref_loss_cap: float = 0.0,
    init_from_ckpt: str = "",
    resume_from_run: str = "",
) -> None:
    set_all_seeds(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    invocation_args = dict(locals())

    rho_enabled = bool(rho_ref_loss_dir)
    init_from_ckpt = str(init_from_ckpt).strip()
    resume_from_run = str(resume_from_run).strip()
    if init_from_ckpt and resume_from_run:
        raise ValueError("Use only one of --init_from_ckpt or --resume_from_run, not both.")
    if init_from_ckpt and not os.path.isdir(init_from_ckpt):
        raise FileNotFoundError(f"--init_from_ckpt not found: {init_from_ckpt}")

    if rho_enabled:
        if not os.path.isdir(rho_ref_loss_dir):
            raise FileNotFoundError(f"--rho_ref_loss_dir not found: {rho_ref_loss_dir}")
        if not (0.0 < float(rho_keep_frac) <= 1.0):
            raise ValueError(f"--rho_keep_frac must be in (0,1], got {rho_keep_frac}")
        if rho_mode not in {"delta", "ref_only"}:
            raise ValueError(f"--rho_mode must be one of ['delta','ref_only'], got {rho_mode}")
        if rho_warmup_steps < 0:
            raise ValueError(f"--rho_warmup_steps must be >=0, got {rho_warmup_steps}")
        if rho_ref_loss_cap < 0:
            raise ValueError(f"--rho_ref_loss_cap must be >=0, got {rho_ref_loss_cap}")

    # [FIX 1] Initialize Accelerator FIRST so we know the real world_size
    effective_mixed_precision = _resolve_mixed_precision(mixed_precision)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False, split_batches=False)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision=effective_mixed_precision,
        # We will set gradient_accumulation_steps manually below after calculation
    )
    device = accelerator.device
    world_size = accelerator.num_processes

    if rho_enabled:
        checked_shards = _validate_rho_ref_loss_alignment(
            data_dir=data_dir,
            ref_loss_dir=rho_ref_loss_dir,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            split="train",
        )
        if accelerator.is_local_main_process:
            print(
                f"rho preflight validation passed: checked {checked_shards} train shards "
                f"(seq_len={seq_len}, micro_batch_size={micro_batch_size})."
            )

    # [FIX 1] Calculate grad_accum_steps using the authoritative world_size
    tokens_per_microstep_global = world_size * micro_batch_size * seq_len
    if tokens_per_microstep_global <= 0:
        raise ValueError("Bad micro_batch_size/seq_len/world_size.")
    
    grad_accum_steps = max(1, math.ceil(total_batch_tokens / tokens_per_microstep_global))
    
    # Update accelerator with the calculated steps
    accelerator.gradient_accumulation_steps = grad_accum_steps

    effective_total_tokens = grad_accum_steps * tokens_per_microstep_global
    effective_global_batch_seqs = grad_accum_steps * micro_batch_size * world_size
    per_gpu_tokens_per_opt_step = grad_accum_steps * micro_batch_size * seq_len

    resume_mode = bool(resume_from_run)
    resume_run_dir = ""
    resume_ckpt_dir = ""
    resume_step_hint = 0
    resume_state = {}
    if resume_mode:
        resume_run_dir, resume_ckpt_dir, resume_step_hint = _resolve_resume_paths(resume_from_run)
        resume_state = _load_trainer_state(resume_ckpt_dir)
        out_dir = resume_run_dir
    else:
        run_name = (
            f"babygpt_fineweb_bin_mbs{micro_batch_size}_T{seq_len}_"
            f"d{n_embd}_h{n_head}_L{n_layer}_"
            f"tok{total_batch_tokens}_efftok{effective_total_tokens}_"
            f"ws{world_size}_gas{grad_accum_steps}_seed{seed}_"
            f"steps{max_train_steps}"
        )
        if rho_enabled:
            run_name += (
                f"_rho{rho_mode}_k{int(round(float(rho_keep_frac) * 1000.0)):04d}_"
                f"wu{int(rho_warmup_steps)}"
            )
        out_dir = os.path.join(experiments_dir, run_name)

    model_init_ckpt_dir = ""
    if resume_mode:
        model_init_ckpt_dir = resume_ckpt_dir
    elif init_from_ckpt:
        model_init_ckpt_dir = os.path.abspath(init_from_ckpt)

    if model_init_ckpt_dir:
        _validate_ckpt_model_config_alignment(
            ckpt_dir=model_init_ckpt_dir,
            seq_len=seq_len,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Logs
    exposures_dir = os.path.join(out_dir, "exposures")
    if exposure_every > 0 and accelerator.is_main_process:
        os.makedirs(exposures_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    rank_id = accelerator.process_index
    exposure_path = os.path.join(exposures_dir, f"exposures_rank{rank_id:04d}.jsonl")
    scalars_path = os.path.join(out_dir, "scalars.jsonl")
    ewok_items_path = os.path.join(out_dir, "ewok_items.jsonl")
    hellaswag_metrics_path = os.path.join(out_dir, "hellaswag_metrics.jsonl")
    metrics_path = os.path.join(out_dir, "step_metrics.json")
    run_config_path = os.path.join(out_dir, "run_config.json")

    if accelerator.is_main_process:
        if resume_mode:
            append_jsonl(
                os.path.join(out_dir, "resume_history.jsonl"),
                {
                    "timestamp": datetime.now().isoformat(),
                    "resume_from_run": resume_run_dir,
                    "resume_ckpt_dir": resume_ckpt_dir,
                    "resume_step_hint": int(resume_step_hint),
                    "invocation_args": invocation_args,
                },
            )
        elif not os.path.exists(run_config_path):
            atomic_write_json(
                run_config_path,
                {
                    "created_at": datetime.now().isoformat(),
                    "script": os.path.abspath(__file__),
                    "args": invocation_args,
                },
            )

    min_lr = learning_rate * learning_rate_decay_frac
    lr_step0 = get_llmc_lr(
        step=0,
        learning_rate=learning_rate,
        warmup_iters=warmup_iters,
        num_iterations=max_train_steps,
        learning_rate_decay_frac=learning_rate_decay_frac,
    )
    lr_warmup_end = get_llmc_lr(
        step=max(warmup_iters - 1, 0),
        learning_rate=learning_rate,
        warmup_iters=warmup_iters,
        num_iterations=max_train_steps,
        learning_rate_decay_frac=learning_rate_decay_frac,
    )
    lr_final = get_llmc_lr(
        step=max_train_steps,
        learning_rate=learning_rate,
        warmup_iters=warmup_iters,
        num_iterations=max_train_steps,
        learning_rate_decay_frac=learning_rate_decay_frac,
    )

    if accelerator.is_local_main_process:
        if resume_mode:
            print(f"resume_from_run             = {resume_run_dir}")
            print(f"resume_ckpt_dir             = {resume_ckpt_dir}")
            print(f"resume_step_hint            = {resume_step_hint}")
        elif model_init_ckpt_dir:
            print(f"init_from_ckpt              = {model_init_ckpt_dir}")
        print("---- Batch math ----")
        print("# Global token/accounting settings used to derive optimizer-step batch size.")
        print(f"world_size (processes)      = {world_size}")
        print(f"mixed_precision requested   = {mixed_precision}")
        print(f"mixed_precision effective   = {effective_mixed_precision}")
        print(f"micro_batch_size (per GPU)  = {micro_batch_size}")
        print(f"seq_len                     = {seq_len}")
        print(f"requested total_batch_tokens= {total_batch_tokens}")
        print(f"tokens per microstep global = {tokens_per_microstep_global}")
        print(f"grad_accum_steps            = {grad_accum_steps}")
        print(f"effective total tokens/opt  = {effective_total_tokens}")
        print(f"effective global batch seqs = {effective_global_batch_seqs}")
        print(f"per-GPU tokens/opt step     = {per_gpu_tokens_per_opt_step}")
        print("---- LR schedule ----")
        print("# Schedule: linear warmup to peak LR, then cosine decay to min LR.")
        print(f"learning_rate (peak/max)    = {learning_rate}")
        print(f"warmup_iters                = {warmup_iters}")
        print(f"learning_rate_decay_frac    = {learning_rate_decay_frac}")
        print(f"min_lr (peak * decay_frac)  = {min_lr}")
        print(f"lr@step0                    = {lr_step0}")
        print(f"lr@warmup_end               = {lr_warmup_end}")
        print(f"lr@final_step               = {lr_final}")
        print("--------------------")
        if rho_enabled:
            print("---- Rho masking ----")
            print(f"rho_ref_loss_dir            = {rho_ref_loss_dir}")
            print(f"rho_mode                    = {rho_mode}")
            print(f"rho_keep_frac               = {rho_keep_frac}")
            print(f"rho_warmup_steps            = {rho_warmup_steps}")
            print(f"rho_ref_loss_cap            = {rho_ref_loss_cap}")
            print("---------------------")
        print(f"out_dir = {out_dir}")

    # Tokenizer
    tokenizer_source = "gpt2"
    if model_init_ckpt_dir:
        ckpt_tok_cfg = os.path.join(model_init_ckpt_dir, "tokenizer_config.json")
        if os.path.exists(ckpt_tok_cfg):
            tokenizer_source = model_init_ckpt_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if accelerator.is_local_main_process:
        print(f"tokenizer source             = {tokenizer_source}")
    tokenizer.pad_token = tokenizer.eos_token
    EOS_ID = tokenizer.pad_token_id  # 50256
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    if vocab_size < tokenizer_vocab_size:
        raise ValueError(f"vocab_size={vocab_size} must be >= tokenizer vocab ({tokenizer_vocab_size}).")
    if vocab_size > tokenizer_vocab_size and accelerator.is_local_main_process:
        print(
            f"[warn] vocab_size={vocab_size} > tokenizer vocab={tokenizer_vocab_size}; "
            "training loss will use tokenizer vocab only (llm.c-style)."
        )
    if n_embd <= 0:
        raise ValueError(f"n_embd must be > 0, got {n_embd}")
    if n_head <= 0:
        raise ValueError(f"n_head must be > 0, got {n_head}")
    if n_layer <= 0:
        raise ValueError(f"n_layer must be > 0, got {n_layer}")
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd must be divisible by n_head, got n_embd={n_embd}, n_head={n_head}")
    loss_vocab_size = tokenizer_vocab_size
    can_probe_generate = (vocab_size == tokenizer_vocab_size)

    # Dataloaders
    # Metadata is only needed for rho masking and/or exposure logging.
    need_train_meta = bool(rho_enabled or (exposure_every > 0))
    if accelerator.is_local_main_process:
        print(f"train_loader metadata enabled = {need_train_meta}")

    if need_train_meta:
        try:
            train_loader = make_dataloader(
                data_dir=data_dir,
                split="train",
                batch_size=micro_batch_size,
                seq_len=seq_len,
                shuffle_blocks=shuffle_blocks,
                seed=seed,
                num_workers=num_workers,
                max_blocks=None,
                shard_by_rank=True,
                return_meta=True,
            )
        except TypeError:
            if rho_enabled:
                raise RuntimeError(
                    "shard_loader.make_dataloader does not support return_meta=True, "
                    "but rho masking needs per-batch shard/start/end metadata."
                )
            if accelerator.is_local_main_process:
                print(
                    "[warn] shard_loader.make_dataloader does not accept return_meta=True yet; "
                    "exposure meta will be empty."
                )
            train_loader = make_dataloader(
                data_dir=data_dir,
                split="train",
                batch_size=micro_batch_size,
                seq_len=seq_len,
                shuffle_blocks=shuffle_blocks,
                seed=seed,
                num_workers=num_workers,
                max_blocks=None,
                shard_by_rank=True,
            )
    else:
        train_loader = make_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=micro_batch_size,
            seq_len=seq_len,
            shuffle_blocks=shuffle_blocks,
            seed=seed,
            num_workers=num_workers,
            max_blocks=None,
            shard_by_rank=True,
        )

    val_loader = make_dataloader(
        data_dir=data_dir,
        split="val",
        batch_size=micro_batch_size,
        seq_len=seq_len,
        shuffle_blocks=False,
        seed=seed,
        num_workers=max(0, min(num_workers, 2)),
        max_blocks=None,
        # Keep validation prefix identical across ranks for comparable val-loss tracking.
        shard_by_rank=False,
    )

    # Model
    if model_init_ckpt_dir:
        model = AutoModelForCausalLM.from_pretrained(
            model_init_ckpt_dir,
            attn_implementation="sdpa",
        )
    else:
        config = GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=EOS_ID,
            eos_token_id=EOS_ID,
            n_ctx=seq_len,
            n_positions=seq_len,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            summary_first_dropout=0.0,
        )
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")

    if accelerator.is_local_main_process:
        if model_init_ckpt_dir:
            print(f"model init source            = {model_init_ckpt_dir}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (llm.c-style defaults + param grouping)
    optimizer, n_decay, n_nodecay, use_fused = build_llmc_style_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)

    if accelerator.is_local_main_process:
        print(f"Optimizer decayed tensors: {n_decay}, non-decayed tensors: {n_nodecay}")
        print(f"Using fused AdamW: {use_fused}")

    model, optimizer = accelerator.prepare(model, optimizer)

    resume_opt_step = 0
    resume_tokens_seen_local_total = 0
    resume_skip_microsteps = 0
    resume_trim_summary = None
    replay_check = None
    if resume_mode:
        def _state_int(name: str, fallback: int):
            if name not in resume_state:
                return int(fallback)
            return int(resume_state.get(name))

        resume_opt_step = _state_int("opt_step", resume_step_hint)
        resume_tokens_seen_local_total = _state_int("tokens_seen_local_total", 0)
        current_replay_cfg = {
            "seed": int(seed),
            "micro_batch_size": int(micro_batch_size),
            "seq_len": int(seq_len),
            "grad_accum_steps": int(grad_accum_steps),
            "shuffle_blocks": bool(shuffle_blocks),
            "num_workers": int(num_workers),
            "world_size": int(world_size),
            "data_dir": os.path.abspath(data_dir),
        }
        replay_check = validate_replay_compatibility(
            current_cfg=current_replay_cfg,
            trainer_state=resume_state,
            strict=True,
            fail_on_missing=False,
        )
        if accelerator.is_local_main_process:
            for msg in replay_check.get("warnings", []):
                print(f"[resume][warn] {msg}")

        if resume_opt_step < 0:
            raise ValueError(f"Invalid opt_step in trainer state: {resume_opt_step}")
        if resume_opt_step >= max_train_steps:
            raise ValueError(
                f"Checkpoint opt_step={resume_opt_step} is already >= max_train_steps={max_train_steps}."
            )
        resume_skip_microsteps = derive_resume_skip_microsteps(
            trainer_state=resume_state,
            resume_opt_step=resume_opt_step,
            grad_accum_steps=grad_accum_steps,
        )

        opt_state_path = os.path.join(resume_ckpt_dir, "optimizer.pt")
        if not os.path.exists(opt_state_path):
            raise FileNotFoundError(
                f"Resume requested but optimizer state is missing: {opt_state_path}\n"
                "Expected checkpoints saved by this script (which include optimizer.pt). "
                "If you only want weight initialization, use --init_from_ckpt instead."
            )
        accelerator.wait_for_everyone()
        opt_state = torch.load(opt_state_path, map_location="cpu")
        optimizer.load_state_dict(opt_state)
        del opt_state
        optimizer.zero_grad(set_to_none=True)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            print(f"Loaded optimizer state from {opt_state_path}")
            print(
                f"Resuming at optimizer step {resume_opt_step} "
                f"(tokens_seen_local_total={resume_tokens_seen_local_total}, "
                f"micro_steps_seen={resume_skip_microsteps})"
            )
        if accelerator.is_main_process:
            resume_trim_summary = trim_run_logs_after_step(
                run_dir=out_dir,
                max_step=resume_opt_step,
                include_exposure_logs=True,
                create_backup=True,
            )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and resume_trim_summary is not None:
            print(
                f"Resume preflight trim complete: removed "
                f"{int(resume_trim_summary.get('total_removed', 0))} records "
                f"with step > {resume_opt_step}."
            )

    # Plot buffers
    train_loss_history = []
    val_loss_history = []

    # EWoK/step metrics JSON (list)
    step_metrics = _load_existing_step_metrics(metrics_path) if resume_mode else []
    if accelerator.is_local_main_process and resume_mode:
        print(f"Loaded {len(step_metrics)} existing step_metrics entries from {metrics_path}")
    hellaswag_ds = None
    hellaswag_max_seq_len = None
    hellaswag_disabled = False
    ewok_category_columns = ("TargetDiff", "ContextDiff", "ContextType")
    ewok_row_category_lookup = build_ewok_row_category_lookup(EWOK_DF, ewok_category_columns)

    def save_plot():
        if not accelerator.is_main_process:
            return
        if plt is None:
            return
        if not train_loss_history and not val_loss_history:
            return
        plt.figure(figsize=(10, 6))
        if train_loss_history:
            tx, ty = zip(*train_loss_history)
            plt.plot(tx, ty, label="Train Loss", alpha=0.3)
        if val_loss_history:
            vx, vy = zip(*val_loss_history)
            plt.plot(vx, vy, label="Val Loss", linewidth=2, color="red", marker="o")
        plt.xlabel("Optimizer steps")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot to {plot_path}")

    def save_checkpoint(tag: str, step: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(out_dir, f"ckpt_{tag}_step{step:07d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            _save_trainer_state(
                ckpt_dir,
                {
                    "timestamp": datetime.now().isoformat(),
                    "opt_step": int(step),
                    "tokens_seen_local_total": int(tokens_seen_local_total),
                    "micro_batch_size": int(micro_batch_size),
                    "seq_len": int(seq_len),
                    "grad_accum_steps": int(grad_accum_steps),
                    "micro_steps_seen": int(step) * int(grad_accum_steps),
                    "effective_total_tokens": int(effective_total_tokens),
                    "rho_enabled": bool(rho_enabled),
                    "rho_mode": (rho_mode if rho_enabled else None),
                    "seed": int(seed),
                    "shuffle_blocks": bool(shuffle_blocks),
                    "num_workers": int(num_workers),
                    "world_size": int(world_size),
                    "data_dir": os.path.abspath(data_dir),
                },
            )
            print(f"Saved checkpoint to {ckpt_dir}/")
        accelerator.wait_for_everyone()

    # Training loop
    model.train()
    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    total_loss_sum = 0.0
    total_tokens = 0
    win_loss_sum = 0.0
    win_tokens = 0

    opt_step = int(resume_opt_step)
    micro_steps_total = max_train_steps * grad_accum_steps
    start_micro_step = int(resume_skip_microsteps) if resume_mode else 0
    if start_micro_step < 0:
        raise ValueError(f"Invalid start_micro_step={start_micro_step} (must be >=0).")
    if start_micro_step > int(micro_steps_total):
        raise ValueError(
            f"Resume micro_steps_seen={start_micro_step} exceeds total micro-steps "
            f"for this run ({micro_steps_total})."
        )

    train_iter = iter(train_loader)
    resume_fast_forward_stats = None
    if resume_mode and start_micro_step > 0:
        resume_fast_forward_stats = fast_forward_iterator_data_only(
            iterator=train_iter,
            skip_count=start_micro_step,
            report_every_s=5.0,
            is_main=accelerator.is_local_main_process,
        )
        if int(resume_fast_forward_stats.get("actual_skipped", 0)) != int(start_micro_step):
            raise RuntimeError(
                f"Resume dataloader fast-forward skipped "
                f"{resume_fast_forward_stats.get('actual_skipped')} micro-steps, "
                f"expected {start_micro_step}."
            )
        if bool(resume_fast_forward_stats.get("exhausted", False)):
            raise RuntimeError(
                "Resume dataloader fast-forward exhausted iterator early. "
                "Replay state cannot be reconstructed safely."
            )

    pbar = tqdm(
        train_iter,
        total=micro_steps_total,
        initial=start_micro_step,
        desc=f"Train (micro={micro_steps_total}, opt={max_train_steps})",
        disable=not accelerator.is_local_main_process,
    )

    # Throughput + exposure counters
    REPORT_EVERY_S = 5.0
    tokens_seen_local_recent = int(resume_tokens_seen_local_total)
    last_report_tokens_local = int(resume_tokens_seen_local_total)
    last_report_t = time.perf_counter()

    tokens_seen_local_total = int(resume_tokens_seen_local_total)  # cumulative on this rank
    if accelerator.is_local_main_process and resume_mode:
        if resume_fast_forward_stats is not None:
            print(
                f"[resume] dataloader replay aligned by skipping "
                f"{int(resume_fast_forward_stats.get('actual_skipped', 0))} micro-steps."
            )
        else:
            print("[resume] dataloader replay aligned with zero micro-step skip.")

    # SDPA backend preference (Flash -> Efficient -> Math)
    sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    VAL_BATCH_LIMIT = 50
    LOG_EVERY = 200  # optimizer steps

    # Buffer meta per optimizer step (i.e., across grad_accum microsteps)
    micro_meta_buf = []
    ref_loss_cache = {}

    # Keep last step scalars around so ewok records can include them
    last_lr = None
    last_grad_norm = None
    last_param_norm = None
    last_train_loss = None
    last_train_loss_micro = None
    last_rho_keep_frac = None
    last_rho_kept_tokens = None
    last_rho_ref_loss_mean = None

    # Track mean train loss across all microsteps in each optimizer step.
    opt_step_loss_sum = 0.0
    opt_step_loss_count = 0
    opt_step_rho_keep_frac_sum = 0.0
    opt_step_rho_keep_frac_count = 0
    opt_step_rho_kept_tokens_sum = 0
    opt_step_rho_ref_loss_sum = 0.0
    opt_step_rho_ref_loss_count = 0

    for micro_step, batch in enumerate(pbar, start=start_micro_step):
        if opt_step >= max_train_steps:
            break

        # Support both (x,y) and (x,y,meta)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            input_ids, labels, meta = batch
        else:
            input_ids, labels = batch
            meta = None

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # meta is CPU-side; keep it lightweight
        if meta is not None:
            micro_meta_buf.append(meta)

        # throughput update
        ntok = int(input_ids.numel())
        tokens_seen_local_recent += ntok
        tokens_seen_local_total += ntok

        now = time.perf_counter()
        if accelerator.is_local_main_process and (now - last_report_t) >= REPORT_EVERY_S:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = now - last_report_t
            delta = tokens_seen_local_recent - last_report_tokens_local
            tps_local = delta / max(dt, 1e-9)
            tps_global = tps_local * accelerator.num_processes
            pbar.set_postfix_str(f"tok/s≈{tps_global:,.0f}")
            last_report_t = now
            last_report_tokens_local = tokens_seen_local_recent

        with accelerator.accumulate(model):
            with autocast_ctx():
                with sdpa_kernel(sdpa_backends):
                    logits = model(input_ids=input_ids).logits  # no attention_mask

                # Token-wise NLL over (B*T) positions.
                token_loss = F.cross_entropy(
                    logits[..., :loss_vocab_size].reshape(-1, loss_vocab_size),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)

                rho_keep_frac_batch = None
                rho_kept_tokens_batch = None
                rho_ref_loss_mean_batch = None
                if rho_enabled and opt_step >= rho_warmup_steps:
                    if not isinstance(meta, dict):
                        raise RuntimeError(
                            "Rho masking requires dict metadata per batch. "
                            "Run with a shard_loader that returns meta."
                        )
                    ref_arr, ref_valid_np = _load_ref_loss_batch(
                        ref_loss_dir=rho_ref_loss_dir,
                        shard_meta=meta,
                        expected_bt=int(labels.numel()),
                        ref_cache=ref_loss_cache,
                    )
                    ref_loss = torch.from_numpy(ref_arr).to(device=device, non_blocking=True).reshape_as(labels)
                    ref_valid_mask = torch.from_numpy(ref_valid_np).to(device=device, non_blocking=True).reshape_as(labels)

                    score = (
                        token_loss.detach().float() - ref_loss
                        if rho_mode == "delta"
                        else -ref_loss
                    )

                    candidate_mask = ref_valid_mask
                    if rho_ref_loss_cap > 0:
                        candidate_mask = candidate_mask & (ref_loss <= float(rho_ref_loss_cap))

                    candidate_count = int(candidate_mask.sum().item())
                    if candidate_count <= 0:
                        # If cap removed everything, fall back to valid ref positions.
                        candidate_mask = ref_valid_mask
                        candidate_count = int(candidate_mask.sum().item())

                    if candidate_count <= 0:
                        keep_mask = torch.ones_like(token_loss, dtype=torch.bool)
                    elif rho_keep_frac >= 1.0:
                        keep_mask = candidate_mask
                    else:
                        k = max(1, int(math.ceil(float(rho_keep_frac) * candidate_count)))
                        cand_scores = score[candidate_mask]
                        if k >= int(cand_scores.numel()):
                            keep_mask = candidate_mask
                        else:
                            thresh = torch.topk(cand_scores, k, sorted=False).values.min()
                            keep_mask = candidate_mask & (score >= thresh)

                    mask_f = keep_mask.to(token_loss.dtype)
                    loss_raw = (token_loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)

                    rho_kept_tokens_batch = int(mask_f.sum().detach().item())
                    rho_keep_frac_batch = float(rho_kept_tokens_batch / max(1, token_loss.numel()))
                    rho_ref_loss_mean_batch = float(ref_loss[ref_valid_mask].mean().detach().item())
                else:
                    loss_raw = token_loss.mean()
                    if rho_enabled:
                        # During warmup, keep all tokens.
                        rho_kept_tokens_batch = int(token_loss.numel())
                        rho_keep_frac_batch = 1.0

                loss_raw_item = float(loss_raw.detach().item())
                opt_step_loss_sum += loss_raw_item
                opt_step_loss_count += 1
                if rho_keep_frac_batch is not None:
                    opt_step_rho_keep_frac_sum += float(rho_keep_frac_batch)
                    opt_step_rho_keep_frac_count += 1
                if rho_kept_tokens_batch is not None:
                    opt_step_rho_kept_tokens_sum += int(rho_kept_tokens_batch)
                if rho_ref_loss_mean_batch is not None:
                    opt_step_rho_ref_loss_sum += float(rho_ref_loss_mean_batch)
                    opt_step_rho_ref_loss_count += 1

                # Accelerate handles grad-accum loss scaling inside accelerator.backward().
                loss = loss_raw

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                step_lr = get_llmc_lr(
                    step=opt_step,
                    learning_rate=learning_rate,
                    warmup_iters=warmup_iters,
                    num_iterations=max_train_steps,
                    learning_rate_decay_frac=learning_rate_decay_frac,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = step_lr

                # norms + lr before update
                lr_before = get_current_lr(optimizer)

                if grad_clip > 0:
                    grad_norm_preclip = float(
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    )
                    grad_norm_postclip = global_grad_norm_l2(model)
                else:
                    grad_norm_preclip = global_grad_norm_l2(model)
                    grad_norm_postclip = None

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                lr_after = get_current_lr(optimizer)

                # param norm after update (main only to reduce overhead)
                param_norm = None
                if accelerator.is_main_process:
                    param_norm = global_param_norm_l2(accelerator.unwrap_model(model))

                # stash last-step scalars
                last_lr = lr_after
                last_grad_norm = grad_norm_preclip
                last_param_norm = param_norm
                last_train_loss = opt_step_loss_sum / max(1, opt_step_loss_count)
                last_train_loss_micro = loss_raw_item
                last_rho_keep_frac = (
                    opt_step_rho_keep_frac_sum / max(1, opt_step_rho_keep_frac_count)
                    if opt_step_rho_keep_frac_count > 0 else None
                )
                last_rho_kept_tokens = (
                    int(round(opt_step_rho_kept_tokens_sum / max(1, opt_step_rho_keep_frac_count)))
                    if opt_step_rho_keep_frac_count > 0 else None
                )
                last_rho_ref_loss_mean = (
                    opt_step_rho_ref_loss_sum / max(1, opt_step_rho_ref_loss_count)
                    if opt_step_rho_ref_loss_count > 0 else None
                )
                opt_step_loss_sum = 0.0
                opt_step_loss_count = 0
                opt_step_rho_keep_frac_sum = 0.0
                opt_step_rho_keep_frac_count = 0
                opt_step_rho_kept_tokens_sum = 0
                opt_step_rho_ref_loss_sum = 0.0
                opt_step_rho_ref_loss_count = 0

                # Approx global tokens seen so far (assumes each rank runs same # microbatches)
                tokens_seen_global_approx = int(tokens_seen_local_total * accelerator.num_processes)

                # Step-level scalars record (main only)
                if accelerator.is_main_process:
                    append_jsonl(scalars_path, {
                        "type": "scalars",
                        "step": opt_step,
                        "timestamp": datetime.now().isoformat(),
                        "lr_before_step": lr_before,
                        "lr_after_step": lr_after,
                        "train_loss_raw_last": last_train_loss_micro,
                        "train_loss_opt_step_mean": last_train_loss,
                        "grad_norm_l2_preclip": grad_norm_preclip,
                        "grad_norm_l2_postclip": grad_norm_postclip,
                        "param_norm_l2": param_norm,
                        "tokens_seen_global_approx": tokens_seen_global_approx,
                        "tokens_per_opt_step_global": int(effective_total_tokens),
                        "world_size": accelerator.num_processes,
                        "grad_accum_steps": int(grad_accum_steps),
                        "micro_batch_size": int(micro_batch_size),
                        "seq_len": int(seq_len),
                        "rho_enabled": bool(rho_enabled),
                        "rho_mode": (rho_mode if rho_enabled else None),
                        "rho_keep_frac_target": (float(rho_keep_frac) if rho_enabled else None),
                        "rho_keep_frac_mean": (float(last_rho_keep_frac) if last_rho_keep_frac is not None else None),
                        "rho_kept_tokens_mean": (int(last_rho_kept_tokens) if last_rho_kept_tokens is not None else None),
                        "rho_ref_loss_mean": (float(last_rho_ref_loss_mean) if last_rho_ref_loss_mean is not None else None),
                        "rho_warmup_steps": (int(rho_warmup_steps) if rho_enabled else None),
                        "rho_ref_loss_cap": (float(rho_ref_loss_cap) if rho_enabled else None),
                    })

        # stats (no masks, no padding)
        bs_tokens = int(labels.numel())
        if rho_enabled and (rho_kept_tokens_batch is not None):
            bs_tokens = int(rho_kept_tokens_batch)
        loss_val = loss_raw_item
        total_loss_sum += loss_val * bs_tokens
        total_tokens += bs_tokens
        win_loss_sum += loss_val * bs_tokens
        win_tokens += bs_tokens

        # Exposure snapshot (per rank file) at optimizer-step boundary
        if accelerator.sync_gradients:
            if exposure_every > 0 and (opt_step % exposure_every == 0):
                flat = []
                for m in micro_meta_buf:
                    flat.extend(_flatten_meta(m))

                rec = {
                    "type": "exposure",
                    "step": opt_step,
                    "timestamp": datetime.now().isoformat(),
                    "rank": accelerator.process_index,
                    "world_size": accelerator.num_processes,
                    "tokens_seen_local_total": int(tokens_seen_local_total),
                    "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
                    "tokens_per_opt_step_global": int(effective_total_tokens),
                    "micro_batches": [
                        {
                            "shard_idx": d.get("shard_idx"),
                            "shard_path": d.get("shard_path"),
                            "block_idx": d.get("block_idx"),
                            "start": d.get("start"),
                            "end": d.get("end"),
                            "worker_id": d.get("worker_id"),
                            "num_workers": d.get("num_workers"),
                        }
                        for d in flat if isinstance(d, dict)
                    ],
                }
                try:
                    append_jsonl(exposure_path, rec)
                except OSError as e:
                    if accelerator.is_local_main_process:
                        print(f"[warn] could not write exposure log to {exposure_path}: {e}")

            # clear buffer every optimizer step
            micro_meta_buf = []

        # -------------------------------------------------------------
        # Rank0-ish logging (Lightweight, no barriers)
        # -------------------------------------------------------------
        if accelerator.sync_gradients and accelerator.is_local_main_process:
            step_train_loss = float(last_train_loss) if last_train_loss is not None else loss_val
            train_loss_history.append((opt_step - 1, step_train_loss))

            if opt_step % LOG_EVERY == 0:
                avg = win_loss_sum / max(1, win_tokens)
                print(f"[Opt {opt_step:07d}/{max_train_steps}] last={step_train_loss:.4f} avg_token={avg:.4f}")
                win_loss_sum = 0.0
                win_tokens = 0
            
            # Generation check (Rank 0 only)
            if can_probe_generate and opt_step % 199 == 0:
                prompts = [
                    "Sunlight filtered through the ancient oak’s twisting branches.",
                    "A curious cat perched on the windowsill.",
                    "In the dim control room, tiny lights blinked like a field of artificial stars.",
                    "John believes his keys are in his pocket, but they are actually on the table. He reaches into his pocket to find ",
                    "Alice loves spicy food, while Bob hates it. When the waiter brought the extra-hot curry, Alice smiled, but Bob ",
                    "The explorer doubted the old rope bridge was safe. Before stepping onto it, she carefully ",
                    "The dog desperately wanted the steak on the high counter, but it was too short to reach. To get the food, it started to ",
                    "Mark didn't mean to bump into the stranger. Feeling bad about the accident, he quickly turned around to say ",
                ]
                model.eval()
                with torch.no_grad():
                    txt = random.choice(prompts)
                    inp = tokenizer.encode(txt, return_tensors="pt").to(device)
                    out = accelerator.unwrap_model(model).generate(inp, max_length=100, do_sample=True)
                    gen_txt = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"Generated: {gen_txt}")
                model.train()

        # -------------------------------------------------------------
        # [FIXED] PARALLEL VALIDATION (Runs on ALL ranks)
        # -------------------------------------------------------------
        if accelerator.sync_gradients:
            do_eval = (eval_every > 0 and opt_step % eval_every == 0)

            if do_eval:
                current_val_loss = run_parallel_validation(
                    accelerator=accelerator,
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    sdpa_backends=sdpa_backends,
                    loss_vocab_size=loss_vocab_size,
                    step=opt_step,
                    val_batch_limit=VAL_BATCH_LIMIT,
                    append_jsonl_fn=append_jsonl,
                    scalars_path=scalars_path,
                )
                if accelerator.is_main_process:
                    val_loss_history.append((opt_step, current_val_loss))

        # -------------------------------------------------------------
        # EWoK eval + checkpointing
        # -------------------------------------------------------------
        if accelerator.sync_gradients and (opt_step > 0):
            do_save = (save_every > 0 and opt_step % save_every == 0)
            do_hellaswag = (hellaswag_every > 0 and opt_step % hellaswag_every == 0)
            do_ewok = (ewok_every > 0 and opt_step % ewok_every == 0)

            if do_hellaswag:
                hellaswag_disabled, hellaswag_ds, hellaswag_max_seq_len = run_hellaswag_eval_step(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    hellaswag_eval_module=hellaswag_eval,
                    hellaswag_disabled=hellaswag_disabled,
                    hellaswag_ds=hellaswag_ds,
                    hellaswag_max_seq_len=hellaswag_max_seq_len,
                    hellaswag_dataset=hellaswag_dataset,
                    hellaswag_dataset_config=hellaswag_dataset_config,
                    hellaswag_split=hellaswag_split,
                    hellaswag_local_files_only=hellaswag_local_files_only,
                    hellaswag_max_examples=hellaswag_max_examples,
                    hellaswag_batch_size=hellaswag_batch_size,
                    opt_step=opt_step,
                    last_train_loss=last_train_loss,
                    loss_val=loss_val,
                    last_lr=last_lr,
                    optimizer=optimizer,
                    tokens_seen_local_total=tokens_seen_local_total,
                    scalars_path=scalars_path,
                    hellaswag_metrics_path=hellaswag_metrics_path,
                    step_metrics=step_metrics,
                    metrics_path=metrics_path,
                    append_jsonl_fn=append_jsonl,
                    save_metrics_fn=save_metrics,
                    get_current_lr_fn=get_current_lr,
                    release_eval_memory_fn=release_eval_memory,
                )

            if do_ewok:
                run_ewok_eval_step(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    evaluate_fn=evaluate,
                    ewok_batch_size=ewok_batch_size,
                    opt_step=opt_step,
                    last_train_loss=last_train_loss,
                    loss_val=loss_val,
                    last_lr=last_lr,
                    last_grad_norm=last_grad_norm,
                    last_param_norm=last_param_norm,
                    optimizer=optimizer,
                    tokens_seen_local_total=tokens_seen_local_total,
                    ewok_items_path=ewok_items_path,
                    step_metrics=step_metrics,
                    metrics_path=metrics_path,
                    ewok_row_category_lookup=ewok_row_category_lookup,
                    ewok_category_columns=ewok_category_columns,
                    append_jsonl_fn=append_jsonl,
                    save_metrics_fn=save_metrics,
                    get_current_lr_fn=get_current_lr,
                    to_jsonable_fn=to_jsonable,
                    release_eval_memory_fn=release_eval_memory,
                )

            if do_save:
                save_plot()
                save_checkpoint("periodic", opt_step)

    # Finalize
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        avg_loss = total_loss_sum / max(1, total_tokens)
        print(f"⇨ Done. token-avg loss = {avg_loss:.4f}")
        save_plot()

    # Final EWoK (+ per-item) + final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not skip_final_ewok:
            run_final_ewok_eval_main_process(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                evaluate_fn=evaluate,
                ewok_batch_size=ewok_batch_size,
                opt_step=opt_step,
                optimizer=optimizer,
                tokens_seen_local_total=tokens_seen_local_total,
                ewok_items_path=ewok_items_path,
                step_metrics=step_metrics,
                metrics_path=metrics_path,
                ewok_row_category_lookup=ewok_row_category_lookup,
                ewok_category_columns=ewok_category_columns,
                append_jsonl_fn=append_jsonl,
                save_metrics_fn=save_metrics,
                get_current_lr_fn=get_current_lr,
                to_jsonable_fn=to_jsonable,
            )
        else:
            print("[info] skipping final EWoK evaluation (--skip_final_ewok)")

        # Auto-generate run-local analysis plots from step_metrics.json
        plot_script = os.path.join(os.path.dirname(__file__), "plot_step_metrics.py")
        plot_dir = os.path.join(out_dir, "plots_from_step_metrics")
        if os.path.exists(plot_script):
            try:
                cmd = [sys.executable, plot_script, "--metrics", metrics_path, "--output-dir", plot_dir]
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.returncode == 0:
                    print(f"Generated analysis plots in {plot_dir}")
                    if proc.stdout:
                        print(proc.stdout.strip())
                else:
                    print(f"[warn] plot_step_metrics.py exited with code {proc.returncode}; skipping auto-plots.")
                    if proc.stderr:
                        print(proc.stderr.strip())
            except Exception as exc:
                print(f"[warn] failed to run plot_step_metrics.py: {exc}")
        else:
            print(f"[warn] plot script not found at {plot_script}; skipping auto-plots.")

    save_checkpoint("final", opt_step)

    if push_to_hub and accelerator.is_main_process:
        tokenizer.push_to_hub(out_dir)
        accelerator.unwrap_model(model).push_to_hub(out_dir)
        print(f"Pushed model to hub repo: {out_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GPT-2 with memmapped uint16 .bin shards + token-budget accumulation (step-based) + exposure + ewok per-item"
    )

    parser.add_argument("--seed", type=int, default=42)
    # [FIX 3] Default micro_batch_size changed to 10
    parser.add_argument("--micro_batch_size", type=int, default=10, help="Per-process/GPU micro-batch size (sequences)")
    parser.add_argument("--total_batch_tokens", type=int, default=524288, help="Global tokens per optimizer step target")
    parser.add_argument("--max_train_steps", type=int, default=20000, help="Total optimizer steps to run")

    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Parent directory where run folders are created")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Model vocab size (GPT-2 tokenizer is 50257; >50257 allowed with CE on first 50257 logits)")
    parser.add_argument("--n_embd", type=int, default=768,
                        help="Transformer hidden size (GPT-2 small uses 768; medium uses 1024)")
    parser.add_argument("--n_head", type=int, default=12,
                        help="Attention heads (GPT-2 small uses 12; medium uses 16)")
    parser.add_argument("--n_layer", type=int, default=12,
                        help="Transformer depth (GPT-2 small uses 12 layers)")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train_*.bin, val_*.bin, meta.json")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--shuffle_blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable block-level shuffle (recommended).",
    )

    parser.add_argument("--learning_rate", type=float, default=6e-4,
                        help="Peak learning rate (llm.c default for d12 run)")
    parser.add_argument("--warmup_iters", type=int, default=700,
                        help="Warmup iterations for llm.c-style LR schedule")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=0.0,
                        help="Final LR fraction for llm.c-style cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW decay for matrix/embedding params (llm.c-style)")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip norm (<=0 disables)")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Accelerate mixed precision mode (bf16 preferred where supported)")

    parser.add_argument("--rho_ref_loss_dir", type=str, default="",
                        help="Directory containing precomputed *.ref_loss.f16/f32.bin files (enables rho masking)")
    parser.add_argument("--rho_keep_frac", type=float, default=1.0,
                        help="Fraction of candidate tokens kept for optimization when rho is enabled")
    parser.add_argument("--rho_warmup_steps", type=int, default=0,
                        help="Number of optimizer steps to run without rho masking")
    parser.add_argument("--rho_mode", type=str, default="delta", choices=["delta", "ref_only"],
                        help="Token selection score: delta=(student_loss-ref_loss), ref_only=(-ref_loss)")
    parser.add_argument("--rho_ref_loss_cap", type=float, default=0.0,
                        help="Optional cap on acceptable ref_loss before top-k selection (0 disables)")
    parser.add_argument("--init_from_ckpt", type=str, default="",
                        help="Initialize model weights from a checkpoint directory (weights-only start).")
    parser.add_argument("--resume_from_run", type=str, default="",
                        help="Resume training from an existing run dir (or a specific ckpt_*_stepXXXXXXX dir).")

    # Intentional CLI overrides vs main() defaults.
    parser.add_argument("--eval_every", type=int, default=200,
                        help="Quick val-loss eval every N optimizer steps (0 disables)")
    parser.add_argument("--hellaswag_every", type=int, default=500,
                        help="Run HellaSwag eval every N optimizer steps (0 disables)")
    parser.add_argument("--hellaswag_batch_size", type=int, default=8,
                        help="Batch size for HellaSwag evaluate_hellaswag()")
    parser.add_argument("--hellaswag_max_examples", type=int, default=2048,
                        help="Optional max examples for HellaSwag split")
    parser.add_argument("--hellaswag_dataset", type=str, default="hellaswag",
                        help="Dataset name/path passed to datasets.load_dataset for HellaSwag")
    parser.add_argument("--hellaswag_dataset_config", type=str, default=None,
                        help="Optional datasets config name for HellaSwag")
    parser.add_argument("--hellaswag_split", type=str, default="validation",
                        help="Dataset split used for HellaSwag eval")
    parser.add_argument("--hellaswag_local_files_only", action="store_true",
                        help="Load HellaSwag dataset from local cache/files only")
    parser.add_argument("--ewok_every", type=int, default=250,
                        help="Run EWoK eval every N optimizer steps (0 disables)")
    parser.add_argument("--ewok_batch_size", type=int, default=4,
                        help="Batch size inside EWoK evaluate()")
    parser.add_argument("--skip_final_ewok", action="store_true",
                        help="Skip final EWoK eval at end of training (useful for quick smoke tests)")
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save checkpoint every N optimizer steps (0 disables)")

    parser.add_argument("--exposure_every", type=int, default=100,
                        help="Log exposure meta every N optimizer steps (0 disables)")

    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
