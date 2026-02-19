# train_gpt2_finewebedu_bin_step_exposure_fixed.py
import os, random, argparse, json, math, inspect, subprocess, sys
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

try:
    from ewok_eval import evaluate, ewok_df as EWOK_DF
except ImportError:
    from ewok_eval import evaluate
    EWOK_DF = None

from shard_loader import make_dataloader

try:
    import hellaswag_eval
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
        return eval_off, eval_full, per_item
    if len(result) == 4:
        eval_off, eval_full, per_item, _margin_stats = result
        return eval_off, eval_full, per_item
    raise ValueError(f"Unexpected EWoK return tuple length: {len(result)}")


def _canonicalize_category_value(value) -> str:
    if value is None:
        return "<NA>"
    if isinstance(value, str):
        return value.strip().replace("_", " ")
    try:
        if np.isnan(value):
            return "<NA>"
    except Exception:
        pass
    return str(value)


def _build_ewok_row_category_lookup(
    ewok_df,
    category_columns=("TargetDiff", "ContextDiff", "ContextType"),
):
    lookup = {}
    if ewok_df is None:
        return lookup
    for row_idx, row in ewok_df.iterrows():
        row_key = int(row_idx)
        lookup[row_key] = {
            col: _canonicalize_category_value(row.get(col))
            for col in category_columns
        }
    return lookup


def _aggregate_eval_full_by_category(
    per_item_records,
    row_category_lookup,
    category_columns=("TargetDiff", "ContextDiff", "ContextType"),
):
    counters = {col: {} for col in category_columns}

    for rec in per_item_records:
        row_idx = rec.get("row_index")
        if not isinstance(row_idx, int):
            continue
        row_meta = row_category_lookup.get(row_idx)
        if not isinstance(row_meta, dict):
            continue

        off_ok = 1 if bool(rec.get("correct_official", False)) else 0
        sym_ok = 1 if bool(rec.get("correct_symmetric", False)) else 0

        for col in category_columns:
            cat = row_meta.get(col, "<NA>")
            bucket = counters[col].setdefault(cat, {"off_ok": 0, "sym_ok": 0, "n": 0})
            bucket["off_ok"] += off_ok
            bucket["sym_ok"] += sym_ok
            bucket["n"] += 1

    out = {}
    for col in category_columns:
        col_map = {}
        acc1_vals = []
        acc2_vals = []
        for cat in sorted(counters[col].keys()):
            b = counters[col][cat]
            n = int(b["n"])
            if n <= 0:
                continue
            acc1 = float(b["off_ok"] / n)
            acc2 = float(b["sym_ok"] / n)
            col_map[str(cat)] = (acc1, acc2)
            acc1_vals.append(acc1)
            acc2_vals.append(acc2)
        if acc1_vals:
            col_map["average"] = (float(np.mean(acc1_vals)), float(np.mean(acc2_vals)))
        out[col] = col_map

    return out


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
    ewok_every: int = 5000,
    ewok_batch_size: int = 8,
    save_every: int = 2000,
    exposure_every: int = 50,
    push_to_hub: bool = False,
) -> None:
    set_all_seeds(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        f"babygpt_fineweb_bin_mbs{micro_batch_size}_T{seq_len}_"
        f"L{n_layer}_"
        f"tok{total_batch_tokens}_efftok{effective_total_tokens}_"
        f"ws{world_size}_gas{grad_accum_steps}_seed{seed}_"
        f"steps{max_train_steps}"
    )
    out_dir = os.path.join(experiments_dir, run_name)

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
        print(f"out_dir = {out_dir}")

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
    if n_layer <= 0:
        raise ValueError(f"n_layer must be > 0, got {n_layer}")
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
            print("[warn] shard_loader.make_dataloader does not accept return_meta=True yet; exposure meta will be empty.")
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
        n_embd=768,
        n_head=12,
        n_layer=n_layer,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")

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

    # Plot buffers
    train_loss_history = []
    val_loss_history = []

    # EWoK/step metrics JSON (list)
    step_metrics = []
    hellaswag_ds = None
    hellaswag_max_seq_len = None
    hellaswag_disabled = False
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
            pbar.set_postfix_str(f"tok/s≈{tps_global:,.0f}")
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
                # [BARRIER] Ensure synchronization before eval
                accelerator.wait_for_everyone()

                # Reset to start so each eval uses the same validation prefix.
                val_iter = iter(val_loader)
                model.eval()
                
                # Each rank computes local sums
                local_val_loss_sum = 0.0
                local_val_tokens_sum = 0
                
                # Only print on local main process
                if accelerator.is_local_main_process:
                    print('Validation')
                    
                with torch.no_grad():
                    # Run on ALL ranks
                    for _ in range(VAL_BATCH_LIMIT):
                        try:
                            v_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader) # Restart
                            v_batch = next(val_iter)

                        if isinstance(v_batch, (list, tuple)) and len(v_batch) == 3:
                            v_ids, v_labels, _ = v_batch
                        else:
                            v_ids, v_labels = v_batch
                        
                        v_ids = v_ids.to(device, non_blocking=True)
                        v_labels = v_labels.to(device, non_blocking=True)

                        with autocast_ctx():
                            with sdpa_kernel(sdpa_backends):
                                v_logits = model(input_ids=v_ids).logits

                        v_loss_mean = F.cross_entropy(
                            v_logits[..., :loss_vocab_size].reshape(-1, loss_vocab_size),
                            v_labels.reshape(-1),
                            reduction="mean",
                        )

                        vtoks = int(v_labels.numel())
                        local_val_loss_sum += float(v_loss_mean.item()) * vtoks
                        local_val_tokens_sum += vtoks

                # Aggregate results from all ranks
                # Create tensors on device for reduction
                tr_loss = torch.tensor(local_val_loss_sum, device=device)
                tr_tokens = torch.tensor(local_val_tokens_sum, device=device)
                
                # Sum across all ranks
                global_loss_sum = accelerator.reduce(tr_loss, reduction="sum")
                global_tokens_sum = accelerator.reduce(tr_tokens, reduction="sum")
                
                # Compute average
                current_val_loss = global_loss_sum.item() / max(1, global_tokens_sum.item())
                
                if accelerator.is_local_main_process:
                    print(f" --> Val loss (first {VAL_BATCH_LIMIT} val batches) @ step {opt_step}: {current_val_loss:.4f}")
                if accelerator.is_main_process:
                    val_loss_history.append((opt_step, current_val_loss))

                    # Log to jsonl
                    append_jsonl(scalars_path, {
                        "type": "val_loss",
                        "step": opt_step,
                        "timestamp": datetime.now().isoformat(),
                        "val_loss": float(current_val_loss),
                        "val_batches": int(VAL_BATCH_LIMIT),
                    })
                    
                model.train()
                
                # [BARRIER] Ensure everyone is done before moving on
                accelerator.wait_for_everyone()

        # -------------------------------------------------------------
        # EWoK eval + checkpointing
        # -------------------------------------------------------------
        if accelerator.sync_gradients and (opt_step > 0):
            do_save = (save_every > 0 and opt_step % save_every == 0)
            do_hellaswag = (hellaswag_every > 0 and opt_step % hellaswag_every == 0)
            do_ewok = (ewok_every > 0 and opt_step % ewok_every == 0)

            if do_hellaswag:
                # [BARRIER 1] Sync entry so everyone stops here before one rank evaluates
                accelerator.wait_for_everyone()
                model.eval()

                release_eval_memory()

                if accelerator.is_main_process:
                    if hellaswag_disabled:
                        pass
                    elif hellaswag_eval is None:
                        print("[warn] hellaswag_eval import failed; skipping HellaSwag eval.")
                        hellaswag_disabled = True
                    else:
                        if hellaswag_ds is None:
                            print(f"Loading HellaSwag dataset: {hellaswag_dataset} ({hellaswag_split})")
                            try:
                                hellaswag_ds = hellaswag_eval.load_dataset_compat(
                                    dataset_name=hellaswag_dataset,
                                    dataset_config=hellaswag_dataset_config,
                                    split=hellaswag_split,
                                    local_files_only=hellaswag_local_files_only,
                                )
                                if hellaswag_max_examples is not None:
                                    hellaswag_ds = hellaswag_ds.select(
                                        range(min(int(hellaswag_max_examples), len(hellaswag_ds)))
                                    )
                            except Exception as exc:
                                print(f"[warn] failed to load HellaSwag dataset: {exc}")
                                hellaswag_ds = None
                                hellaswag_disabled = True

                        if hellaswag_ds is not None:
                            print("starting HellaSwag evaluation on main process")
                            hs_model = accelerator.unwrap_model(model)
                            hs_model.eval()
                            hellaswag_max_seq_len = hellaswag_eval.infer_max_seq_len(hs_model, tokenizer)
                            with torch.no_grad():
                                hs_metrics = hellaswag_eval.evaluate_hellaswag(
                                    model=hs_model,
                                    tokenizer=tokenizer,
                                    dataset=hellaswag_ds,
                                    batch_size=hellaswag_batch_size,
                                    device=device,
                                    max_seq_len=hellaswag_max_seq_len,
                                )

                            hs_record = {
                                "step": opt_step,
                                "timestamp": datetime.now().isoformat(),
                                "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
                                "lr": float(last_lr) if last_lr is not None else get_current_lr(optimizer),
                                "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
                                "hellaswag": {
                                    "dataset": hellaswag_dataset,
                                    "dataset_config": hellaswag_dataset_config,
                                    "split": hellaswag_split,
                                    "num_examples": int(hs_metrics["num_examples"]),
                                    "accuracy": float(hs_metrics["accuracy"]),
                                    "accuracy_norm": float(hs_metrics["accuracy_norm"]),
                                    "batch_size": int(hellaswag_batch_size),
                                    "max_seq_len": int(hellaswag_max_seq_len),
                                },
                            }
                            step_metrics.append(hs_record)
                            save_metrics(step_metrics, metrics_path)
                            hs_scalar = {
                                "type": "hellaswag",
                                "step": opt_step,
                                "timestamp": datetime.now().isoformat(),
                                "num_examples": int(hs_metrics["num_examples"]),
                                "accuracy": float(hs_metrics["accuracy"]),
                                "accuracy_norm": float(hs_metrics["accuracy_norm"]),
                                "batch_size": int(hellaswag_batch_size),
                                "max_seq_len": int(hellaswag_max_seq_len),
                                "dataset": hellaswag_dataset,
                                "dataset_config": hellaswag_dataset_config,
                                "split": hellaswag_split,
                            }
                            append_jsonl(scalars_path, hs_scalar)
                            append_jsonl(hellaswag_metrics_path, hs_scalar)
                            print(
                                f"HellaSwag @ step {opt_step}: "
                                f"acc={hs_metrics['accuracy']:.4f}, "
                                f"acc_norm={hs_metrics['accuracy_norm']:.4f}"
                            )

                # [BARRIER 2] Forces other ranks to wait for rank 0 to finish eval
                accelerator.wait_for_everyone()
                model.train()

            if do_ewok:
                # [BARRIER 1] Sync entry so everyone stops here before one rank evaluates
                accelerator.wait_for_everyone()
                model.eval()
                # print('managed to pass barrier, starting ewok eval on main process')
                # Memory cleanup to prevent OOM during eval
                release_eval_memory()

                if accelerator.is_main_process:
                    print('starting EWoK evaluation on main process')
                    with torch.no_grad():
                        eval_off_sum, eval_full_sum, per_item_sum = _unpack_ewok_per_item(
                            evaluate(
                                accelerator.unwrap_model(model),
                                tokenizer,
                                batch_size=ewok_batch_size,
                                return_per_item=True,
                                score_reduction="sum",
                            )
                        )
                        eval_off_mean, eval_full_mean, per_item_mean = _unpack_ewok_per_item(
                            evaluate(
                                accelerator.unwrap_model(model),
                                tokenizer,
                                batch_size=ewok_batch_size,
                                return_per_item=True,
                                score_reduction="mean",
                            )
                        )
                        eval_by_category_full_sum = _aggregate_eval_full_by_category(
                            per_item_sum,
                            ewok_row_category_lookup,
                            ewok_category_columns,
                        )
                        eval_by_category_full_mean = _aggregate_eval_full_by_category(
                            per_item_mean,
                            ewok_row_category_lookup,
                            ewok_category_columns,
                        )

                    # per-item jsonl
                    for r in per_item_sum:
                        rr = dict(r)
                        rr.update({
                            "type": "ewok_item",
                            "step": opt_step,
                            "timestamp": datetime.now().isoformat(),
                        })
                        append_jsonl(ewok_items_path, rr)
                    for r in per_item_mean:
                        rr = dict(r)
                        rr.update({
                            "type": "ewok_item_mean",
                            "step": opt_step,
                            "timestamp": datetime.now().isoformat(),
                        })
                        append_jsonl(ewok_items_path, rr)

                    record = {
                        "step": opt_step,
                        "timestamp": datetime.now().isoformat(),
                        "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
                        "lr": float(last_lr) if last_lr is not None else get_current_lr(optimizer),
                        "grad_norm_l2": float(last_grad_norm) if last_grad_norm is not None else None,
                        "param_norm_l2": float(last_param_norm) if last_param_norm is not None else None,
                        "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
                        # Keep these keys as backward-compatible aliases for sum-reduction.
                        "eval_official": to_jsonable(eval_off_sum),
                        "eval_full": to_jsonable(eval_full_sum),
                        "eval_official_sum": to_jsonable(eval_off_sum),
                        "eval_full_sum": to_jsonable(eval_full_sum),
                        "eval_official_mean": to_jsonable(eval_off_mean),
                        "eval_full_mean": to_jsonable(eval_full_mean),
                        "eval_by_category_full_sum": to_jsonable(eval_by_category_full_sum),
                        "eval_by_category_full_mean": to_jsonable(eval_by_category_full_mean),
                    }
                    step_metrics.append(record)
                    save_metrics(step_metrics, metrics_path)
                    print(f"Saved metrics to {metrics_path}")
                    print(f"Appended per-item EWoK to {ewok_items_path}")
                
                # [BARRIER 2] Forces Ranks 1-5 to wait for Rank 0 to finish eval
                accelerator.wait_for_everyone()
                model.train()

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
        model.eval()
        accelerator.unwrap_model(model).eval()
        with torch.no_grad():
            eval_off_sum, eval_full_sum, per_item_sum = _unpack_ewok_per_item(
                evaluate(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    batch_size=ewok_batch_size,
                    return_per_item=True,
                    score_reduction="sum",
                )
            )
            eval_off_mean, eval_full_mean, per_item_mean = _unpack_ewok_per_item(
                evaluate(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    batch_size=ewok_batch_size,
                    return_per_item=True,
                    score_reduction="mean",
                )
            )
            eval_by_category_full_sum = _aggregate_eval_full_by_category(
                per_item_sum,
                ewok_row_category_lookup,
                ewok_category_columns,
            )
            eval_by_category_full_mean = _aggregate_eval_full_by_category(
                per_item_mean,
                ewok_row_category_lookup,
                ewok_category_columns,
            )

        for r in per_item_sum:
            rr = dict(r)
            rr.update({
                "type": "ewok_item_final",
                "step": opt_step,
                "timestamp": datetime.now().isoformat(),
            })
            append_jsonl(ewok_items_path, rr)
        for r in per_item_mean:
            rr = dict(r)
            rr.update({
                "type": "ewok_item_final_mean",
                "step": opt_step,
                "timestamp": datetime.now().isoformat(),
            })
            append_jsonl(ewok_items_path, rr)

        record = {
            "step": opt_step,
            "timestamp": datetime.now().isoformat(),
            "final": True,
            "lr": get_current_lr(optimizer),
            "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
            # Keep these keys as backward-compatible aliases for sum-reduction.
            "eval_official": to_jsonable(eval_off_sum),
            "eval_full": to_jsonable(eval_full_sum),
            "eval_official_sum": to_jsonable(eval_off_sum),
            "eval_full_sum": to_jsonable(eval_full_sum),
            "eval_official_mean": to_jsonable(eval_off_mean),
            "eval_full_mean": to_jsonable(eval_full_mean),
            "eval_by_category_full_sum": to_jsonable(eval_by_category_full_sum),
            "eval_by_category_full_mean": to_jsonable(eval_by_category_full_mean),
        }
        step_metrics.append(record)
        save_metrics(step_metrics, metrics_path)
        print(f"Saved final metrics to {metrics_path}")
        print(f"Appended final per-item EWoK to {ewok_items_path}")

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
    parser.add_argument("--ewok_every", type=int, default=1000,
                        help="Run EWoK eval every N optimizer steps (0 disables)")
    parser.add_argument("--ewok_batch_size", type=int, default=4,
                        help="Batch size inside EWoK evaluate()")
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save checkpoint every N optimizer steps (0 disables)")

    parser.add_argument("--exposure_every", type=int, default=100,
                        help="Log exposure meta every N optimizer steps (0 disables)")

    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
