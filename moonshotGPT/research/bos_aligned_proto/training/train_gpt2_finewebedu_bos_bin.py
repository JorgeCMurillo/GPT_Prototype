"""Train GPT-style models on BOS-aligned row-packed token shards.

This prototype entrypoint mirrors the main training loop while using the
BOS-row loader and the same evaluation hooks for architecture comparisons.
"""

from dataclasses import asdict
import os, random, json, math, inspect, subprocess, sys
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
from transformers import AutoModelForCausalLM

from torch.optim import AdamW

from transformers import (
    AutoTokenizer, GPT2Config,
)

from accelerate.utils import DataLoaderConfiguration
from accelerate import Accelerator

# Allow running from locations where the repo root is not already on sys.path.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROTO_ROOT = os.path.dirname(_THIS_DIR)
_RESEARCH_ROOT = os.path.dirname(_PROTO_ROOT)
_REPO_ROOT = os.path.dirname(_RESEARCH_ROOT)
PLOT_STEP_METRICS_SCRIPT = os.path.join(_REPO_ROOT, "plot_step_metrics.py")
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

try:
    from attention_compat import sdpa_kernel, SDPBackend
except ImportError:
    from moonshotGPT.attention_compat import sdpa_kernel, SDPBackend
try:
    from runtime_memory import format_memory_usage_postfix, reset_peak_memory_stats
except ImportError:
    from moonshotGPT.runtime_memory import format_memory_usage_postfix, reset_peak_memory_stats

try:
    from research.bos_aligned_proto.training.config import (
        DEFAULT_EXPERIMENTS_DIR,
        TrainConfig,
        parse_args,
    )
except ImportError:
    from config import DEFAULT_EXPERIMENTS_DIR, TrainConfig, parse_args

try:
    from research.bos_aligned_proto.evaluation.ewok import (
        BABYLM_COMPLETION_CHOICE,
        EWOK_CONTEXT_SENSITIVITY,
        evaluate,
        ewok_df as EWOK_DF,
    )
except ImportError:
    from evaluation.ewok import (
        BABYLM_COMPLETION_CHOICE,
        EWOK_CONTEXT_SENSITIVITY,
        evaluate,
        ewok_df as EWOK_DF,
    )

try:
    from research.bos_aligned_proto.pipeline.bos_row_loader import (
        make_bos_row_dataloader as make_dataloader,
    )
except ImportError:
    from ..pipeline.bos_row_loader import make_bos_row_dataloader as make_dataloader

try:
    from research.bos_aligned_proto.evaluation.ewok_category import (
        aggregate_eval_full_by_category as _aggregate_eval_full_by_category,
        build_ewok_row_category_lookup as _build_ewok_row_category_lookup,
        plot_ewok_category_subplots as _plot_ewok_category_subplots,
    )
except ImportError:
    from ..evaluation.ewok_category import (
        aggregate_eval_full_by_category as _aggregate_eval_full_by_category,
        build_ewok_row_category_lookup as _build_ewok_row_category_lookup,
        plot_ewok_category_subplots as _plot_ewok_category_subplots,
    )

try:
    from research.bos_aligned_proto.evaluation import hellaswag as hellaswag_eval
except Exception:
    try:
        from evaluation import hellaswag as hellaswag_eval
    except Exception:
        hellaswag_eval = None

try:
    from research.bos_aligned_proto.evaluation import core as core_eval
except Exception:
    try:
        from evaluation import core as core_eval
    except Exception:
        core_eval = None

try:
    from evaluation.runner import (
        run_core_eval_step,
        run_ewok_eval_step,
        run_final_ewok_eval_main_process,
        run_hellaswag_eval_step,
        run_parallel_validation,
    )
except Exception:
    run_core_eval_step = None
    run_ewok_eval_step = None
    run_final_ewok_eval_main_process = None
    run_hellaswag_eval_step = None
    run_parallel_validation = None


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
    supported_fused_devices = {"cuda", "xpu", "privateuseone"}
    param_devices = {p.device.type for p in param_dict.values()}
    params_already_on_supported_device = len(param_devices) == 1 and next(iter(param_devices)) in supported_fused_devices
    use_fused = fused_available and params_already_on_supported_device
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


def release_eval_memory():
    """Best-effort cleanup before heavy eval runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _unpack_ewok_per_item(result):
    """
    Backward/forward compatible unpack for ewok_eval.evaluate(return_per_item=True).
    Older API returns 3 values, newer API returns 4 (with margin stats).
    """
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected EWoK return type: {type(result)}")
    if len(result) == 3:
        eval_off, eval_full, per_item = result
        return eval_off, eval_full, per_item, None
    if len(result) == 4:
        eval_off, eval_full, per_item, margin_stats = result
        return eval_off, eval_full, per_item, margin_stats
    raise ValueError(f"Unexpected EWoK return tuple length: {len(result)}")


def _evaluate_ewok_all_methods(model, tokenizer, *, batch_size: int, score_reduction: str):
    try:
        result = evaluate(
            model,
            tokenizer,
            batch_size=batch_size,
            return_per_item=True,
            score_reduction=score_reduction,
            return_all_methods=True,
        )
    except TypeError:
        eval_off, eval_full, per_item, margin_stats = _unpack_ewok_per_item(
            evaluate(
                model,
                tokenizer,
                batch_size=batch_size,
                return_per_item=True,
                score_reduction=score_reduction,
            )
        )
        return {
            BABYLM_COMPLETION_CHOICE: {
                "domain_scores_official": eval_off,
                "domain_scores_full": eval_full,
                "domain_margin_stats": margin_stats,
            },
            EWOK_CONTEXT_SENSITIVITY: None,
        }, per_item

    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError(
            "Expected evaluate(return_all_methods=True, return_per_item=True) "
            f"to return (metrics_by_method, per_item), got: {type(result)}"
        )

    metrics_by_method, per_item = result
    if not isinstance(metrics_by_method, dict):
        raise TypeError(f"Unexpected metrics_by_method type: {type(metrics_by_method)}")
    if not isinstance(per_item, list):
        raise TypeError(f"Unexpected per_item type: {type(per_item)}")
    if BABYLM_COMPLETION_CHOICE not in metrics_by_method:
        raise KeyError(f"Missing {BABYLM_COMPLETION_CHOICE} in EWoK metrics.")
    return metrics_by_method, per_item


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _pair_to_scalar(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2 and _is_finite_number(value[0]) and _is_finite_number(value[1]):
        return 0.5 * (float(value[0]) + float(value[1]))
    if _is_finite_number(value):
        return float(value)
    return None


def _extract_full_average_scalar(full_payload):
    if not isinstance(full_payload, dict):
        return None

    avg = _pair_to_scalar(full_payload.get("average"))
    if avg is not None:
        return avg

    vals = []
    for domain, value in full_payload.items():
        if str(domain) == "average":
            continue
        y = _pair_to_scalar(value)
        if y is not None:
            vals.append(y)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _plot_ewok_full_mean_average(step_metrics, out_dir):
    """Plot EWOK full-mean average across optimizer steps from in-memory step_metrics."""
    if plt is None or not step_metrics:
        return

    by_step = {}
    for rec in step_metrics:
        step = rec.get("step")
        full_mean = rec.get("eval_full_mean")
        if not isinstance(step, int) or not isinstance(full_mean, dict):
            continue
        y = _extract_full_average_scalar(full_mean)
        if y is None:
            continue
        by_step[step] = float(y)

    if not by_step:
        return

    points = sorted(by_step.items(), key=lambda t: t[0])
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color="#2a6f97", label="full_mean_average")
    ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.1, label="random chance = 50%")
    ax.set_title("EWOK Full Mean Average Across Steps")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_path = os.path.join(out_dir, "ewok_full_mean_average_by_step.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _extract_margin_average_scalar(margin_payload, metric_key):
    if not isinstance(margin_payload, dict):
        return None

    avg = margin_payload.get("average")
    if isinstance(avg, dict) and _is_finite_number(avg.get(metric_key)):
        return float(avg[metric_key])

    vals = []
    for domain, stats in margin_payload.items():
        if str(domain) == "average" or not isinstance(stats, dict):
            continue
        if _is_finite_number(stats.get(metric_key)):
            vals.append(float(stats[metric_key]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _plot_ewok_margin_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean"):
    if plt is None or not step_metrics:
        return

    ewok_records = [
        r for r in step_metrics
        if isinstance(r, dict) and isinstance(r.get("step"), int) and isinstance(r.get(metric_key), dict)
    ]
    if not ewok_records:
        return

    reduction_suffix = metric_key.replace("eval_margin_stats_", "").strip("_") or "unknown"
    by_domain = {}
    for rec in ewok_records:
        margin_payload = rec.get(metric_key, {})
        for domain, stats in margin_payload.items():
            if str(domain) == "average" or not isinstance(stats, dict):
                continue
            y_signed = stats.get("mean_signed_m")
            y_abs = stats.get("mean_abs_m")
            if _is_finite_number(y_signed) and _is_finite_number(y_abs):
                by_domain.setdefault(str(domain), []).append(
                    (rec["step"], float(y_signed), float(y_abs))
                )

    if not by_domain:
        return

    domains = sorted(by_domain)
    ncols = 3
    nrows = max(1, int(math.ceil(len(domains) / ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, max(14, nrows * 3.5)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        pts = sorted(by_domain[domain], key=lambda t: t[0])
        xs = [x for x, _, _ in pts]
        ys_signed = [a for _, a, _ in pts]
        ys_abs = [b for _, _, b in pts]
        ax.plot(xs, ys_signed, marker="o", linewidth=1.6, markersize=3.5, color="#1f77b4", label="mean signed margin")
        ax.plot(
            xs,
            ys_abs,
            marker="s",
            linewidth=1.4,
            markersize=3.2,
            linestyle=(0, (4, 2)),
            color="#ff7f0e",
            label="mean abs margin",
        )
        ax.axhline(0.0, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="zero margin")
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Margin", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"EWOK Mean Margins by Domain ({reduction_suffix})", fontsize=14)
    out_path = os.path.join(out_dir, f"ewok_margin_{reduction_suffix}_domains_4x3.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ewok_margin_average_all_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean"):
    if plt is None or not step_metrics:
        return

    reduction_suffix = metric_key.replace("eval_margin_stats_", "").strip("_") or "unknown"
    signed = []
    abs_margin = []
    for rec in step_metrics:
        step = rec.get("step")
        margin_payload = rec.get(metric_key)
        if not isinstance(step, int) or not isinstance(margin_payload, dict):
            continue
        y_signed = _extract_margin_average_scalar(margin_payload, "mean_signed_m")
        y_abs = _extract_margin_average_scalar(margin_payload, "mean_abs_m")
        if y_signed is not None:
            signed.append((step, float(y_signed)))
        if y_abs is not None:
            abs_margin.append((step, float(y_abs)))

    if not signed and not abs_margin:
        return

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    if signed:
        xs = [x for x, _ in signed]
        ys = [y for _, y in signed]
        ax.plot(xs, ys, linewidth=1.9, marker="o", markersize=3.5, color="#1f77b4", label="mean signed margin")

    if abs_margin:
        xs = [x for x, _ in abs_margin]
        ys = [y for _, y in abs_margin]
        ax.plot(
            xs,
            ys,
            linewidth=1.6,
            marker="s",
            markersize=3.4,
            linestyle=(0, (4, 2)),
            color="#ff7f0e",
            alpha=0.45,
            label="mean abs margin (reference)",
        )

    ax.axhline(0.0, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="zero margin")
    ax.set_title(f"EWOK Mean Margins Across Domains ({reduction_suffix})")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.25)
    ax.legend()

    reduction_label = (
        r"$s(C,T)=\sum_t \log P_{\theta}(t\mid C)$"
        if reduction_suffix == "sum"
        else r"$s(C,T)=\frac{1}{|T|}\sum_t \log P_{\theta}(t\mid C)$"
    )
    expl = (
        r"$m_1=s(C_1,T_1)-s(C_1,T_2),\ m_2=s(C_2,T_2)-s(C_2,T_1),\ m=\frac{1}{2}(m_1+m_2)$"
        "\n"
        r"$\mu_d=\mathbb{E}_i[m_i],\ \mathrm{plotted}=\frac{1}{D}\sum_d \mu_d$"
        "\n"
        + reduction_label
        + "; "
        + r"$\mathrm{abs\ ref}=\frac{1}{D}\sum_d \mathbb{E}_i[|m_i|]$"
        + "\n"
        + "Intuition: signed > 0 favors the correct direction; near 0 with high abs can indicate strong but inconsistent or biased discrimination."
    )
    ax.text(
        0.015,
        0.015,
        expl,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.75},
    )

    out_path = os.path.join(out_dir, f"ewok_margin_{reduction_suffix}_average_all_domains.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _refresh_ewok_analysis_plots(step_metrics, out_dir, include_sum_plots=False):
    _plot_ewok_full_mean_average(step_metrics, out_dir)
    _plot_ewok_category_subplots(step_metrics, out_dir, metric_key="eval_by_category_full_mean")
    _plot_ewok_margin_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean")
    _plot_ewok_margin_average_all_domains(step_metrics, out_dir, metric_key="eval_margin_stats_mean")
    if include_sum_plots:
        _plot_ewok_margin_domains(step_metrics, out_dir, metric_key="eval_margin_stats_sum")
        _plot_ewok_margin_average_all_domains(step_metrics, out_dir, metric_key="eval_margin_stats_sum")

def _has_row_shards(path: str) -> bool:
    """Best-effort check for expected BOS row-packed dataset layout."""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    has_meta = "meta.json" in entries
    has_train_bin = any(name.startswith("train_") and name.endswith(".bin") for name in entries)
    return has_meta and has_train_bin


def _resolve_row_data_dir(path: str) -> str:
    """Resolve BOS row-packed datasets from common repo-relative locations."""
    requested = os.path.expanduser(path)
    for candidate in (
        requested,
        os.path.join(_PROTO_ROOT, requested),
        os.path.join(_REPO_ROOT, requested),
        os.path.join(_REPO_ROOT, "data", requested),
    ):
        if _has_row_shards(candidate):
            return os.path.abspath(candidate)
    return requested


# -----------------------------
# Main

def main(cfg: TrainConfig) -> None:
    seed = cfg.seed
    micro_batch_size = cfg.micro_batch_size
    total_batch_tokens = cfg.total_batch_tokens
    max_train_steps = cfg.max_train_steps
    data_dir = cfg.data_dir
    experiments_dir = cfg.experiments_dir
    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size
    n_embd = cfg.n_embd
    n_head = cfg.n_head
    n_layer = cfg.n_layer
    num_workers = cfg.num_workers
    shuffle_blocks = cfg.shuffle_blocks
    grad_clip = cfg.grad_clip
    learning_rate = cfg.learning_rate
    warmup_iters = cfg.warmup_iters
    learning_rate_decay_frac = cfg.learning_rate_decay_frac
    weight_decay = cfg.weight_decay
    beta1 = cfg.beta1
    beta2 = cfg.beta2
    eval_every = cfg.eval_every
    hellaswag_every = cfg.hellaswag_every
    hellaswag_batch_size = cfg.hellaswag_batch_size
    hellaswag_max_examples = cfg.hellaswag_max_examples
    hellaswag_dataset = cfg.hellaswag_dataset
    hellaswag_dataset_config = cfg.hellaswag_dataset_config
    hellaswag_split = cfg.hellaswag_split
    hellaswag_local_files_only = cfg.hellaswag_local_files_only
    core_every = cfg.core_every
    core_max_per_task = cfg.core_max_per_task
    core_bundle_dir = cfg.core_bundle_dir
    core_local_files_only = cfg.core_local_files_only
    ewok_every = cfg.ewok_every
    ewok_batch_size = cfg.ewok_batch_size
    save_every = cfg.save_every
    exposure_every = cfg.exposure_every
    push_to_hub = cfg.push_to_hub
    skip_final_ewok = cfg.skip_final_ewok
    include_ewok_sum_plots = cfg.include_ewok_sum_plots

    set_all_seeds(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Resolve relative BOS datasets against the prototype root, repo root,
    # and a future top-level repo data/ directory.
    data_dir = _resolve_row_data_dir(data_dir)

    # [FIX 1] Initialize Accelerator FIRST so we know the real world_size
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False, split_batches=False)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        # We will set gradient_accumulation_steps manually below after calculation
    )
    device = accelerator.device
    world_size = accelerator.num_processes

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

    run_name = (
        f"babygpt_fineweb_bosrow_mbs{micro_batch_size}_T{seq_len}_"
        f"d{n_embd}_h{n_head}_L{n_layer}_"
        f"tok{total_batch_tokens}_efftok{effective_total_tokens}_"
        f"ws{world_size}_gas{grad_accum_steps}_seed{seed}_"
        f"steps{max_train_steps}"
    )
    out_dir = os.path.join(experiments_dir, run_name)
    analysis_plot_dir = os.path.join(out_dir, "plots_from_step_metrics")

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(analysis_plot_dir, exist_ok=True)
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
    core_metrics_path = os.path.join(out_dir, "core_metrics.jsonl")
    metrics_path = os.path.join(out_dir, "step_metrics.json")
    run_config_path = os.path.join(out_dir, "run_config.json")

    if accelerator.is_main_process and not os.path.exists(run_config_path):
        tmp_path = run_config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "script": os.path.abspath(__file__),
                    "config": to_jsonable(asdict(cfg)),
                },
                handle,
                indent=2,
            )
        os.replace(tmp_path, run_config_path)

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
        print("---- Batch math ----")
        print("# Global token/accounting settings used to derive optimizer-step batch size.")
        print(f"world_size (processes)      = {world_size}")
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
        print(f"data_dir                    = {data_dir}")
        print(f"out_dir = {out_dir}")
        print("---- Eval settings ----")
        print("# Interval settings are in optimizer steps; 0 disables that recurring action.")
        print(f"eval_every (val loss)       = {eval_every}")
        print(f"hellaswag_every             = {hellaswag_every}")
        print(f"hellaswag_batch_size        = {hellaswag_batch_size}")
        print(f"hellaswag_max_examples      = {hellaswag_max_examples}")
        print(f"hellaswag_dataset           = {hellaswag_dataset}")
        print(f"hellaswag_dataset_config    = {hellaswag_dataset_config}")
        print(f"hellaswag_split             = {hellaswag_split}")
        print(f"hellaswag_local_files_only  = {hellaswag_local_files_only}")
        print(f"core_every                  = {core_every}")
        print(f"core_max_per_task           = {core_max_per_task}")
        print(f"core_bundle_dir             = {core_bundle_dir or '<default>'}")
        print(f"core_local_files_only       = {core_local_files_only}")
        print(f"ewok_every                  = {ewok_every}")
        print(f"ewok_batch_size             = {ewok_batch_size}")
        print(f"save_every                  = {save_every}")
        print(f"exposure_every              = {exposure_every}")
        print(f"skip_final_ewok             = {skip_final_ewok}")
        print(f"include_ewok_sum_plots      = {include_ewok_sum_plots}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
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
        if accelerator.is_local_main_process:
            print("[warn] bos_row_loader.make_bos_row_dataloader does not accept return_meta=True yet; exposure meta will be empty.")
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
    # KV cache only helps incremental decoding; pretraining recomputes full sequences.
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.config.use_cache = False

    if accelerator.is_local_main_process:
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
    reset_peak_memory_stats(device)

    # Plot buffers
    train_loss_history = []
    val_loss_history = []

    # EWoK/step metrics JSON (list)
    step_metrics = []
    hellaswag_ds = None
    hellaswag_max_seq_len = None
    hellaswag_disabled = False
    core_disabled = False
    ewok_category_columns = ("TargetDiff", "ContextDiff", "ContextType")
    ewok_row_category_lookup = _build_ewok_row_category_lookup(EWOK_DF, ewok_category_columns)

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
            print(f"Saved checkpoint to {ckpt_dir}/")

    # Training loop
    model.train()
    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    total_loss_sum = 0.0
    total_tokens = 0
    win_loss_sum = 0.0
    win_tokens = 0

    opt_step = 0
    micro_steps_total = max_train_steps * grad_accum_steps

    pbar = tqdm(
        train_loader,
        total=micro_steps_total,
        desc=f"Train (micro={micro_steps_total}, opt={max_train_steps})",
        disable=not accelerator.is_local_main_process,
    )

    # Throughput + exposure counters
    REPORT_EVERY_S = 5.0
    tokens_seen_local_recent = 0
    last_report_tokens_local = 0
    last_report_t = time.perf_counter()

    tokens_seen_local_total = 0  # cumulative on this rank

    # SDPA backend preference (Flash -> Efficient -> Math)
    sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    VAL_BATCH_LIMIT = 50
    LOG_EVERY = 200  # optimizer steps

    # Buffer meta per optimizer step (i.e., across grad_accum microsteps)
    micro_meta_buf = []

    # Keep last step scalars around so ewok records can include them
    last_lr = None
    last_grad_norm = None
    last_param_norm = None
    last_train_loss = None
    last_train_loss_micro = None

    # Track mean train loss across all microsteps in each optimizer step.
    opt_step_loss_sum = 0.0
    opt_step_loss_count = 0

    for micro_step, batch in enumerate(pbar):
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
            postfix = f"tok/s≈{tps_global:,.0f}"
            mem_postfix = format_memory_usage_postfix(device)
            if mem_postfix:
                postfix = f"{postfix} {mem_postfix}"
            pbar.set_postfix_str(postfix)
            last_report_t = now
            last_report_tokens_local = tokens_seen_local_recent

        with accelerator.accumulate(model):
            with autocast_ctx():
                with sdpa_kernel(sdpa_backends):
                    logits = model(input_ids=input_ids).logits  # no attention_mask

                # mean NLL per token over (B*T) positions
                loss_raw = F.cross_entropy(
                    logits[..., :loss_vocab_size].reshape(-1, loss_vocab_size),
                    labels.reshape(-1),
                    reduction="mean",
                )
                loss_raw_item = float(loss_raw.detach().item())
                opt_step_loss_sum += loss_raw_item
                opt_step_loss_count += 1

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
                opt_step_loss_sum = 0.0
                opt_step_loss_count = 0

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
                    })

        # stats (no masks, no padding)
        bs_tokens = int(labels.numel())
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
            do_core = (core_every > 0 and opt_step % core_every == 0)
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

            if do_core:
                core_disabled = run_core_eval_step(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    core_eval_module=core_eval,
                    core_disabled=core_disabled,
                    core_bundle_dir=core_bundle_dir,
                    core_local_files_only=core_local_files_only,
                    core_max_per_task=core_max_per_task,
                    opt_step=opt_step,
                    last_train_loss=last_train_loss,
                    loss_val=loss_val,
                    last_lr=last_lr,
                    optimizer=optimizer,
                    tokens_seen_local_total=tokens_seen_local_total,
                    scalars_path=scalars_path,
                    core_metrics_path=core_metrics_path,
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
                if accelerator.is_main_process:
                    _refresh_ewok_analysis_plots(
                        step_metrics,
                        analysis_plot_dir,
                        include_sum_plots=include_ewok_sum_plots,
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

    # Final CORE / EWoK (+ per-item) + final checkpoint
    accelerator.wait_for_everyone()
    if core_every > 0:
        core_disabled = run_core_eval_step(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            device=device,
            core_eval_module=core_eval,
            core_disabled=core_disabled,
            core_bundle_dir=core_bundle_dir,
            core_local_files_only=core_local_files_only,
            core_max_per_task=core_max_per_task,
            opt_step=opt_step,
            last_train_loss=last_train_loss,
            loss_val=float(last_train_loss) if last_train_loss is not None else 0.0,
            last_lr=last_lr,
            optimizer=optimizer,
            tokens_seen_local_total=tokens_seen_local_total,
            scalars_path=scalars_path,
            core_metrics_path=core_metrics_path,
            step_metrics=step_metrics,
            metrics_path=metrics_path,
            append_jsonl_fn=append_jsonl,
            save_metrics_fn=save_metrics,
            get_current_lr_fn=get_current_lr,
            release_eval_memory_fn=release_eval_memory,
            final=True,
        )

    if accelerator.is_main_process and not skip_final_ewok:
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
        _refresh_ewok_analysis_plots(
            step_metrics,
            analysis_plot_dir,
            include_sum_plots=include_ewok_sum_plots,
        )
    elif accelerator.is_main_process and skip_final_ewok:
        print("[info] skipping final EWoK evaluation (--skip_final_ewok)")

    if accelerator.is_main_process:
        _refresh_ewok_analysis_plots(
            step_metrics,
            analysis_plot_dir,
            include_sum_plots=include_ewok_sum_plots,
        )

        # Auto-generate run-local analysis plots from step_metrics.json
        plot_script = PLOT_STEP_METRICS_SCRIPT
        plot_dir = analysis_plot_dir
        if os.path.exists(plot_script) and os.path.exists(metrics_path):
            try:
                cmd = [sys.executable, plot_script, "--metrics", metrics_path, "--output-dir", plot_dir]
                if include_ewok_sum_plots:
                    cmd.append("--include-ewok-sum-plots")
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
            if not os.path.exists(plot_script):
                print(f"[warn] plot script not found at {plot_script}; skipping auto-plots.")
            else:
                print(f"[warn] metrics file not found at {metrics_path}; skipping auto-plots.")

    save_checkpoint("final", opt_step)

    if push_to_hub and accelerator.is_main_process:
        tokenizer.push_to_hub(out_dir)
        accelerator.unwrap_model(model).push_to_hub(out_dir)
        print(f"Pushed model to hub repo: {out_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
