import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _log(msg: str, rank: int = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]"
    if rank is not None:
        prefix += f"[rank{int(rank):02d}]"
    print(f"{prefix} {msg}", flush=True)


@contextmanager
def _stage(name: str, enabled: bool, rank: int = None):
    t0 = time.time()
    if enabled:
        _log(f"START: {name}", rank=rank)
    try:
        yield
    finally:
        if enabled:
            _log(f"DONE:  {name} ({time.time() - t0:.1f}s)", rank=rank)


def atomic_write_json(path: str, payload: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _list_shards(data_dir: str, split: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(f"{split}_") and name.endswith(".bin"):
            out.append(os.path.join(data_dir, name))
    return out


def _dtype_suffix(dtype_name: str) -> str:
    if dtype_name == "float16":
        return "f16"
    if dtype_name == "float32":
        return "f32"
    raise ValueError(f"Unsupported out dtype: {dtype_name}")


def _dtype_np(dtype_name: str):
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"Unsupported out dtype: {dtype_name}")


def _resolve_mixed_precision(requested: str) -> str:
    if requested != "bf16":
        return requested
    if not torch.cuda.is_available():
        return "no"
    if not torch.cuda.is_bf16_supported():
        return "fp16"
    return "bf16"


def _model_dtype_for_mp(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def _ref_loss_paths(out_dir: str, shard_path: str, out_dtype: str) -> Tuple[str, str]:
    base = os.path.basename(shard_path)
    if not base.endswith(".bin"):
        raise ValueError(f"Expected .bin shard, got {shard_path}")
    stem = base[:-4]
    suffix = _dtype_suffix(out_dtype)
    loss_path = os.path.join(out_dir, f"{stem}.ref_loss.{suffix}.bin")
    meta_path = os.path.join(out_dir, f"{stem}.ref_loss.{suffix}.meta.json")
    return loss_path, meta_path


def _count_tokens(path: str) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte-size must be divisible by 2 for uint16: {path}")
    return nbytes // 2


def _estimate_blocks_for_shard(path: str, seq_len: int, batch_size: int) -> int:
    n_tokens = _count_tokens(path)
    stride_tokens = batch_size * seq_len
    block_tokens = stride_tokens + 1
    if n_tokens < block_tokens:
        return 0
    return int(1 + (n_tokens - block_tokens) // stride_tokens)


def _has_complete_ref_loss(shard_path: str, out_dir: str, out_dtype: str) -> bool:
    """Return True only if both the loss bin and metadata json already exist."""
    loss_path, meta_path = _ref_loss_paths(out_dir, shard_path, out_dtype)
    return os.path.exists(loss_path) and os.path.exists(meta_path)


def _partition_pending_shards(
    shards: List[str], out_dir: str, out_dtype: str, overwrite: bool
) -> Tuple[List[str], List[str]]:
    """
    Split shards into:
    - pending: needs compute in this run
    - skipped_existing: already fully computed

    Why this exists:
    On resume runs, assigning all shards first and then skipping in-loop creates
    heavy rank imbalance. Filtering to pending shards before assignment keeps the
    multi-GPU workload balanced and reduces barrier timeouts.
    """
    if overwrite:
        return list(shards), []

    pending = []
    skipped_existing = []
    for shard_path in shards:
        if _has_complete_ref_loss(shard_path, out_dir, out_dtype):
            skipped_existing.append(shard_path)
        else:
            pending.append(shard_path)
    return pending, skipped_existing


def _assign_shards_greedy_by_blocks(
    pending_shards: List[str], block_counts: Dict[str, int], world_size: int
) -> Tuple[List[List[str]], List[int]]:
    """
    Greedy block-balanced assignment (Longest-Processing-Time first).

    Algorithm:
    1) Sort pending shards by descending estimated blocks.
    2) Repeatedly assign next shard to the currently least-loaded rank.

    Why greedy:
    It is simple, deterministic, and gives a much better balance than naive
    round-robin when shard sizes differ or when only a tail subset remains.
    """
    per_rank_shards: List[List[str]] = [[] for _ in range(world_size)]
    per_rank_blocks: List[int] = [0 for _ in range(world_size)]

    ordered = sorted(
        pending_shards,
        key=lambda p: (-int(block_counts.get(p, 0)), os.path.basename(p)),
    )
    for shard_path in ordered:
        rank = min(range(world_size), key=lambda r: (per_rank_blocks[r], r))
        per_rank_shards[rank].append(shard_path)
        per_rank_blocks[rank] += int(block_counts.get(shard_path, 0))

    return per_rank_shards, per_rank_blocks


def _gather_rank_results(accelerator: Accelerator, rank_results: List[Dict]) -> List[List[Dict]]:
    """
    Gather per-rank result lists into a list-of-lists with one entry per rank.

    We prefer torch.distributed.all_gather_object for explicit behavior. Fallback
    to accelerate.gather_object for environments where distributed is unavailable.
    """
    world = int(accelerator.num_processes)
    if world <= 1:
        return [rank_results]

    if dist.is_available() and dist.is_initialized():
        gathered: List[List[Dict]] = [None for _ in range(world)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, rank_results)
        return [x if isinstance(x, list) else [] for x in gathered]

    if hasattr(accelerator, "gather_object"):
        gathered = accelerator.gather_object(rank_results)
        if isinstance(gathered, list) and gathered and isinstance(gathered[0], list):
            return gathered
        if isinstance(gathered, list) and gathered and isinstance(gathered[0], dict):
            return [gathered]
        return [rank_results] if accelerator.is_main_process else []

    return [rank_results] if accelerator.is_main_process else []


@torch.no_grad()
def _run_startup_probe(model, tokenizer, accelerator: Accelerator, probe_tokens: int = 8) -> None:
    n = max(2, int(probe_tokens))
    vocab = int(tokenizer.vocab_size)
    x = torch.randint(0, vocab, (1, n), device=accelerator.device, dtype=torch.long)
    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
    with autocast_ctx():
        y = model(input_ids=x).logits
    if y.ndim != 3 or int(y.shape[0]) != 1 or int(y.shape[1]) != n:
        raise RuntimeError(f"Startup probe produced unexpected logits shape: {tuple(y.shape)}")


@torch.no_grad()
def compute_one_shard(
    shard_path: str,
    out_dir: str,
    split: str,
    seq_len: int,
    batch_size: int,
    out_dtype: str,
    model,
    tokenizer,
    accelerator: Accelerator,
    overwrite: bool,
    progress: bool,
    requested_mixed_precision: str,
    ref_model_name: str,
    heartbeat_seconds: int = 0,
    heartbeat_enabled: bool = False,
) -> Dict:
    np_dtype = _dtype_np(out_dtype)
    loss_path, meta_path = _ref_loss_paths(out_dir, shard_path, out_dtype)

    if os.path.exists(loss_path) and os.path.exists(meta_path) and not overwrite:
        return {
            "shard_path": shard_path,
            "loss_path": loss_path,
            "meta_path": meta_path,
            "status": "skipped_existing",
        }

    os.makedirs(out_dir, exist_ok=True)

    n_tokens = _count_tokens(shard_path)
    mm = np.memmap(shard_path, dtype=np.uint16, mode="r")

    stride_tokens = batch_size * seq_len
    block_tokens = stride_tokens + 1
    n_blocks = max(0, 1 + (n_tokens - block_tokens) // stride_tokens) if n_tokens >= block_tokens else 0

    tmp_loss_path = loss_path + ".tmp"
    out_mm = np.memmap(tmp_loss_path, dtype=np_dtype, mode="w+", shape=(n_tokens,))
    out_mm[:] = np.nan

    loss_vocab_size = int(tokenizer.vocab_size)

    pbar = tqdm(
        total=n_blocks,
        desc=f"rank{accelerator.process_index} {os.path.basename(shard_path)}",
        disable=not progress,
        dynamic_ncols=True,
        unit="block",
    )

    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    tic = time.time()
    last_heartbeat = tic
    for bidx in range(n_blocks):
        start = bidx * stride_tokens
        end = start + block_tokens

        chunk = mm[start:end]
        chunk_t = torch.from_numpy(np.asarray(chunk, dtype=np.int64))

        x = chunk_t[:-1].reshape(batch_size, seq_len).to(accelerator.device, non_blocking=True)
        y = chunk_t[1:].reshape(batch_size, seq_len).to(accelerator.device, non_blocking=True)

        with autocast_ctx():
            logits = model(input_ids=x).logits

        token_loss = F.cross_entropy(
            logits[..., :loss_vocab_size].reshape(-1, loss_vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape(batch_size, seq_len)

        out_mm[start + 1:end] = token_loss.detach().float().cpu().numpy().reshape(-1).astype(np_dtype, copy=False)
        pbar.update(1)

        if heartbeat_enabled and heartbeat_seconds > 0:
            now = time.time()
            if (now - last_heartbeat) >= heartbeat_seconds:
                done = bidx + 1
                elapsed = now - tic
                pct = 100.0 * done / max(1, n_blocks)
                rate = done / max(elapsed, 1e-9)
                rem = (n_blocks - done) / max(rate, 1e-9)
                _log(
                    f"HEARTBEAT split={split} shard={os.path.basename(shard_path)} "
                    f"blocks={done}/{n_blocks} ({pct:.1f}%) "
                    f"elapsed={elapsed/60.0:.1f}m eta={rem/60.0:.1f}m",
                    rank=accelerator.process_index,
                )
                last_heartbeat = now

    pbar.close()
    out_mm.flush()
    os.replace(tmp_loss_path, loss_path)

    covered_tokens = int(n_blocks * stride_tokens)
    elapsed = float(time.time() - tic)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source_shard_path": shard_path,
        "source_shard_basename": os.path.basename(shard_path),
        "source_num_tokens": int(n_tokens),
        "ref_loss_path": loss_path,
        "ref_loss_dtype": out_dtype,
        "ref_model": ref_model_name,
        "tokenizer": tokenizer.name_or_path,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "stride_tokens": int(stride_tokens),
        "block_tokens": int(block_tokens),
        "num_blocks": int(n_blocks),
        "covered_target_tokens": int(covered_tokens),
        "unfilled_tokens": int(n_tokens - covered_tokens),
        "coverage_fraction": float(covered_tokens / max(1, n_tokens)),
        "token_index_0_is_nan": True,
        "mixed_precision_requested": requested_mixed_precision,
        "mixed_precision_effective": accelerator.mixed_precision,
        "elapsed_seconds": elapsed,
        "throughput_target_tokens_per_sec": float(covered_tokens / max(elapsed, 1e-9)),
        "notes": "Loss is aligned to raw token indices; positions [start+1:end] are filled per training-style block.",
    }
    atomic_write_json(meta_path, meta)

    return {
        "shard_path": shard_path,
        "loss_path": loss_path,
        "meta_path": meta_path,
        "status": "ok",
        "num_blocks": n_blocks,
        "covered_target_tokens": covered_tokens,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute per-token reference-model loss aligned to fineweb token shards")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with train_*.bin / val_*.bin shards")
    p.add_argument("--out_dir", type=str, required=True, help="Directory for *.ref_loss.*.bin outputs")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "all"], help="Which shard split(s) to process")
    p.add_argument("--seq_len", type=int, default=1024, help="Sequence length T used by training")
    p.add_argument("--batch_size", type=int, default=10, help="Micro-batch size B used by training")
    p.add_argument("--ref_model", type=str, default="openai-community/gpt2-medium", help="HF reference model id")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer id/path")
    p.add_argument("--out_dtype", type=str, default="float16", choices=["float16", "float32"], help="Output storage dtype")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode")
    p.add_argument("--startup_diagnostics", action=argparse.BooleanOptionalAction, default=True,
                   help="Print timestamped stage logs for startup and per-shard progress")
    p.add_argument("--startup_probe_tokens", type=int, default=8,
                   help="Dummy token length for startup forward-pass probe (0 disables probe)")
    p.add_argument("--heartbeat_seconds", type=int, default=180,
                   help="Emit periodic heartbeat log during shard compute (0 disables)")
    p.add_argument("--dist_timeout_minutes", type=int, default=180,
                   help="Distributed collectives timeout in minutes (increase for long tail rank skew)")
    p.add_argument("--max_shards", type=int, default=0, help="If >0, cap shards per split (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing ref-loss outputs")
    args = p.parse_args()

    if args.seq_len <= 0:
        raise ValueError("--seq_len must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat_seconds must be >= 0")
    if args.dist_timeout_minutes <= 0:
        raise ValueError("--dist_timeout_minutes must be > 0")

    requested_mp = args.mixed_precision
    effective_mp = _resolve_mixed_precision(requested_mp)

    dist_timeout = timedelta(minutes=int(args.dist_timeout_minutes))
    pg_kwargs = InitProcessGroupKwargs(timeout=dist_timeout)
    with _stage("Accelerator initialization", args.startup_diagnostics):
        accelerator = Accelerator(mixed_precision=effective_mp, kwargs_handlers=[pg_kwargs])
    rank = accelerator.process_index

    if args.startup_diagnostics:
        _log(
            f"process online: world_size={accelerator.num_processes}, "
            f"local_process_index={accelerator.local_process_index}, device={accelerator.device}",
            rank=rank,
        )
        if torch.cuda.is_available():
            try:
                dev_idx = torch.cuda.current_device()
                _log(f"cuda device: index={dev_idx}, name={torch.cuda.get_device_name(dev_idx)}", rank=rank)
            except Exception as exc:
                _log(f"cuda device query failed: {exc}", rank=rank)

    if accelerator.is_local_main_process:
        _log("---- reference loss precompute ----")
        _log(f"data_dir                 = {args.data_dir}")
        _log(f"out_dir                  = {args.out_dir}")
        _log(f"split                    = {args.split}")
        _log(f"seq_len                  = {args.seq_len}")
        _log(f"batch_size               = {args.batch_size}")
        _log(f"ref_model                = {args.ref_model}")
        _log(f"tokenizer                = {args.tokenizer}")
        _log(f"out_dtype                = {args.out_dtype}")
        _log(f"mixed_precision requested= {requested_mp}")
        _log(f"mixed_precision effective= {effective_mp}")
        _log(f"dist_timeout_minutes     = {args.dist_timeout_minutes}")
        _log(f"world_size               = {accelerator.num_processes}")
        _log("-----------------------------------")

    torch_dtype = _model_dtype_for_mp(effective_mp)
    with _stage("Load reference model", args.startup_diagnostics, rank=rank):
        model = AutoModelForCausalLM.from_pretrained(args.ref_model, torch_dtype=torch_dtype)
    with _stage("Load tokenizer", args.startup_diagnostics, rank=rank):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    with _stage("Accelerator prepare(model)", args.startup_diagnostics, rank=rank):
        model = accelerator.prepare(model)
    if int(args.startup_probe_tokens) > 0:
        with _stage("Startup forward-pass probe", args.startup_diagnostics, rank=rank):
            _run_startup_probe(model=model, tokenizer=tokenizer, accelerator=accelerator, probe_tokens=args.startup_probe_tokens)

    splits = ["train", "val"] if args.split == "all" else [args.split]

    all_results: List[Dict] = []
    split_stats: List[Dict] = []

    for split in splits:
        with _stage(f"Discover shards for split={split}", args.startup_diagnostics, rank=rank):
            shards = _list_shards(args.data_dir, split)
        if args.max_shards > 0:
            shards = shards[: args.max_shards]

        if not shards:
            if accelerator.is_main_process:
                print(f"[warn] no shards found for split={split} in {args.data_dir}")
            continue

        rank = accelerator.process_index
        world = accelerator.num_processes

        with _stage(
            f"Partition pending shards for split={split}",
            args.startup_diagnostics and accelerator.is_local_main_process,
            rank=rank,
        ):
            pending_shards, skipped_existing_shards = _partition_pending_shards(
                shards=shards,
                out_dir=args.out_dir,
                out_dtype=args.out_dtype,
                overwrite=args.overwrite,
            )
        skipped_existing_count = len(skipped_existing_shards)

        if pending_shards:
            with _stage(
                f"Estimate block counts for pending split={split}",
                args.startup_diagnostics and accelerator.is_local_main_process,
                rank=rank,
            ):
                pending_block_counts = {
                    p: _estimate_blocks_for_shard(p, args.seq_len, args.batch_size)
                    for p in pending_shards
                }
            with _stage(
                f"Greedy block-balanced assignment for split={split}",
                args.startup_diagnostics and accelerator.is_local_main_process,
                rank=rank,
            ):
                per_rank_shards, per_rank_blocks = _assign_shards_greedy_by_blocks(
                    pending_shards=pending_shards,
                    block_counts=pending_block_counts,
                    world_size=world,
                )
        else:
            pending_block_counts = {}
            per_rank_shards = [[] for _ in range(world)]
            per_rank_blocks = [0 for _ in range(world)]

        rank_shards = per_rank_shards[rank]
        rank_block_counts = {p: int(pending_block_counts.get(p, 0)) for p in rank_shards}
        total_rank_blocks = int(per_rank_blocks[rank])

        rank_pbar = tqdm(
            total=total_rank_blocks,
            desc=f"rank{rank} {split} total",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            unit="block",
        )

        if accelerator.is_local_main_process:
            print(
                f"split={split}: total_shards={len(shards)} "
                f"pending={len(pending_shards)} "
                f"skipped_existing={skipped_existing_count} "
                f"rank{rank}_assigned={len(rank_shards)} "
                f"rank{rank}_blocks={total_rank_blocks} "
                f"(strategy=greedy_block_balanced_pending)"
            )
            assignment_summary = ", ".join(
                f"r{rr}:{len(per_rank_shards[rr])} shards/{int(per_rank_blocks[rr])} blocks"
                for rr in range(world)
            )
            print(f"split={split}: assignment={assignment_summary}")

        rank_results = []
        for sidx, shard_path in enumerate(rank_shards, start=1):
            shard_blocks = int(rank_block_counts.get(shard_path, 0))
            if accelerator.is_local_main_process:
                rank_pbar.set_postfix_str(os.path.basename(shard_path))
            if args.startup_diagnostics and accelerator.is_local_main_process:
                _log(
                    f"split={split} shard {sidx}/{len(rank_shards)} start: "
                    f"{os.path.basename(shard_path)} (blocks={shard_blocks})",
                    rank=rank,
                )
            result = compute_one_shard(
                shard_path=shard_path,
                out_dir=args.out_dir,
                split=split,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                out_dtype=args.out_dtype,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                overwrite=args.overwrite,
                progress=False,
                requested_mixed_precision=requested_mp,
                ref_model_name=args.ref_model,
                heartbeat_seconds=int(args.heartbeat_seconds),
                heartbeat_enabled=bool(args.startup_diagnostics and accelerator.is_local_main_process),
            )
            rank_pbar.update(shard_blocks)
            rank_results.append(result)
            if args.startup_diagnostics and accelerator.is_local_main_process:
                _log(
                    f"split={split} shard {sidx}/{len(rank_shards)} done: "
                    f"{os.path.basename(shard_path)} status={result.get('status')} "
                    f"elapsed={result.get('elapsed_seconds', 'n/a')}",
                    rank=rank,
                )

        rank_pbar.close()
        accelerator.wait_for_everyone()

        gathered = _gather_rank_results(accelerator, rank_results)
        if accelerator.is_main_process:
            flat = []
            for rr in gathered:
                if isinstance(rr, list):
                    flat.extend(rr)
            done = sum(1 for r in flat if r.get("status") == "ok")
            skipped_runtime = sum(1 for r in flat if r.get("status") == "skipped_existing")
            skipped = skipped_existing_count + skipped_runtime
            print(
                f"split={split}: completed={done}, skipped_existing={skipped}, "
                f"total_rank_results={len(flat)}"
            )
            split_stats.append(
                {
                    "split": split,
                    "total_shards": int(len(shards)),
                    "pending_shards": int(len(pending_shards)),
                    "skipped_existing_before_run": int(skipped_existing_count),
                    "assignment_strategy": "greedy_block_balanced_pending",
                    "assigned_blocks_by_rank": [int(x) for x in per_rank_blocks],
                    "assigned_shards_by_rank": [int(len(x)) for x in per_rank_shards],
                    "completed_shards": int(done),
                    "skipped_existing_runtime": int(skipped_runtime),
                    "skipped_existing_total": int(skipped),
                }
            )
            all_results.extend(flat)

    if accelerator.is_main_process:
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": args.data_dir,
            "out_dir": args.out_dir,
            "split": args.split,
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "ref_model": args.ref_model,
            "tokenizer": args.tokenizer,
            "out_dtype": args.out_dtype,
            "mixed_precision_requested": requested_mp,
            "mixed_precision_effective": effective_mp,
            "dist_timeout_minutes": int(args.dist_timeout_minutes),
            "world_size": int(accelerator.num_processes),
            "split_stats": split_stats,
            "results": all_results,
        }
        atomic_write_json(os.path.join(args.out_dir, "ref_loss_run_summary.json"), summary)
        print(f"Wrote run summary to {os.path.join(args.out_dir, 'ref_loss_run_summary.json')}")

    # Explicit distributed teardown avoids noisy warnings at process exit.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
