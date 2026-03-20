"""Shared training-time evaluation helpers for validation and benchmarks.

These functions run distributed-safe validation plus EWOK, HellaSwag, BLiMP,
and CORE evaluations, then merge the resulting metrics into run logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
try:
    from attention_compat import sdpa_kernel
except ImportError:
    from moonshotGPT.attention_compat import sdpa_kernel

from .ewok import BABYLM_COMPLETION_CHOICE, EWOK_CONTEXT_SENSITIVITY


def run_parallel_validation(
    *,
    accelerator,
    model,
    val_loader,
    device,
    autocast_ctx,
    sdpa_backends,
    loss_vocab_size: int,
    step: int,
    val_batch_limit: int,
    append_jsonl_fn,
    scalars_path: str,
) -> float:
    """Run validation prefix on all ranks and return global token-weighted loss."""
    # Barrier before evaluation so every rank enters together.
    accelerator.wait_for_everyone()

    val_iter = iter(val_loader)
    model.eval()
    local_val_loss_sum = 0.0
    local_val_tokens_sum = 0

    if accelerator.is_local_main_process:
        print("Validation")

    with torch.no_grad():
        for _ in range(int(val_batch_limit)):
            try:
                v_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
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

    tr_loss = torch.tensor(local_val_loss_sum, device=device)
    tr_tokens = torch.tensor(local_val_tokens_sum, device=device)
    global_loss_sum = accelerator.reduce(tr_loss, reduction="sum")
    global_tokens_sum = accelerator.reduce(tr_tokens, reduction="sum")
    current_val_loss = global_loss_sum.item() / max(1, global_tokens_sum.item())

    if accelerator.is_local_main_process:
        print(
            f" --> Val loss (first {int(val_batch_limit)} val batches) "
            f"@ step {int(step)}: {current_val_loss:.4f}"
        )
    if accelerator.is_main_process:
        append_jsonl_fn(
            scalars_path,
            {
                "type": "val_loss",
                "step": int(step),
                "timestamp": datetime.now().isoformat(),
                "val_loss": float(current_val_loss),
                "val_batches": int(val_batch_limit),
            },
        )

    model.train()
    # Barrier after eval so no rank exits early into training.
    accelerator.wait_for_everyone()
    return float(current_val_loss)


def run_hellaswag_eval_step(
    *,
    accelerator,
    model,
    tokenizer,
    device,
    hellaswag_eval_module,
    hellaswag_disabled: bool,
    hellaswag_ds,
    hellaswag_max_seq_len,
    hellaswag_dataset: str,
    hellaswag_dataset_config,
    hellaswag_split: str,
    hellaswag_local_files_only: bool,
    hellaswag_max_examples,
    hellaswag_batch_size: int,
    opt_step: int,
    last_train_loss,
    loss_val: float,
    last_lr,
    optimizer,
    tokens_seen_local_total: int,
    scalars_path: str,
    hellaswag_metrics_path: str,
    step_metrics: list,
    metrics_path: str,
    append_jsonl_fn,
    save_metrics_fn,
    get_current_lr_fn,
    release_eval_memory_fn,
) -> Tuple[bool, Any, Any]:
    """Run HellaSwag eval on main process and synchronize all ranks."""
    accelerator.wait_for_everyone()
    model.eval()
    release_eval_memory_fn()

    if accelerator.is_main_process:
        if hellaswag_disabled:
            pass
        elif hellaswag_eval_module is None:
            print("[warn] hellaswag_eval import failed; skipping HellaSwag eval.")
            hellaswag_disabled = True
        else:
            if hellaswag_ds is None:
                print(f"Loading HellaSwag dataset: {hellaswag_dataset} ({hellaswag_split})")
                try:
                    hellaswag_ds = hellaswag_eval_module.load_dataset_compat(
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
                hellaswag_max_seq_len = hellaswag_eval_module.infer_max_seq_len(hs_model, tokenizer)
                with torch.no_grad():
                    hs_metrics = hellaswag_eval_module.evaluate_hellaswag(
                        model=hs_model,
                        tokenizer=tokenizer,
                        dataset=hellaswag_ds,
                        batch_size=hellaswag_batch_size,
                        device=device,
                        max_seq_len=hellaswag_max_seq_len,
                    )

                hs_record = {
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
                    "lr": float(last_lr) if last_lr is not None else get_current_lr_fn(optimizer),
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
                save_metrics_fn(step_metrics, metrics_path)
                hs_scalar = {
                    "type": "hellaswag",
                    "step": int(opt_step),
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
                append_jsonl_fn(scalars_path, hs_scalar)
                append_jsonl_fn(hellaswag_metrics_path, hs_scalar)
                print(
                    f"HellaSwag @ step {int(opt_step)}: "
                    f"acc={hs_metrics['accuracy']:.4f}, "
                    f"acc_norm={hs_metrics['accuracy_norm']:.4f}"
                )

    accelerator.wait_for_everyone()
    model.train()
    return bool(hellaswag_disabled), hellaswag_ds, hellaswag_max_seq_len


def run_core_eval_step(
    *,
    accelerator,
    model,
    tokenizer,
    device,
    core_eval_module,
    core_disabled: bool,
    core_bundle_dir: str,
    core_local_files_only: bool,
    core_max_per_task: int,
    opt_step: int,
    last_train_loss,
    loss_val: float,
    last_lr,
    optimizer,
    tokens_seen_local_total: int,
    scalars_path: str,
    core_metrics_path: str,
    step_metrics: list,
    metrics_path: str,
    append_jsonl_fn,
    save_metrics_fn,
    get_current_lr_fn,
    release_eval_memory_fn,
    final: bool = False,
) -> bool:
    """Run CORE eval on main process and synchronize all ranks."""
    accelerator.wait_for_everyone()
    model.eval()
    release_eval_memory_fn()

    if accelerator.is_main_process:
        if core_disabled:
            pass
        elif core_eval_module is None:
            print("[warn] core_eval import failed; skipping CORE eval.")
            core_disabled = True
        else:
            try:
                print("starting CORE evaluation on main process")
                core_model = accelerator.unwrap_model(model)
                core_model.eval()
                with torch.no_grad():
                    core_results = core_eval_module.evaluate_core(
                        model=core_model,
                        tokenizer=tokenizer,
                        device=device,
                        max_per_task=core_max_per_task,
                        bundle_dir=(core_bundle_dir or None),
                        local_files_only=core_local_files_only,
                        distributed=False,
                    )

                core_record = {
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
                    "lr": float(last_lr) if last_lr is not None else get_current_lr_fn(optimizer),
                    "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
                    "core": {
                        "num_tasks": int(core_results["num_tasks"]),
                        "max_per_task": int(core_results["max_per_task"]),
                        "bundle_source": str(core_results.get("bundle_source", core_results["bundle_dir"])),
                        "bundle_dir": str(core_results["bundle_dir"]),
                        "results": dict(core_results["results"]),
                        "centered_results": dict(core_results["centered_results"]),
                        "core_metric": float(core_results["core_metric"]),
                    },
                }
                if final:
                    core_record["final"] = True
                step_metrics.append(core_record)
                save_metrics_fn(step_metrics, metrics_path)

                core_scalar = {
                    "type": "core",
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "num_tasks": int(core_results["num_tasks"]),
                    "max_per_task": int(core_results["max_per_task"]),
                    "bundle_source": str(core_results.get("bundle_source", core_results["bundle_dir"])),
                    "bundle_dir": str(core_results["bundle_dir"]),
                    "core_metric": float(core_results["core_metric"]),
                }
                if final:
                    core_scalar["final"] = True

                core_metrics_record = {
                    **core_scalar,
                    "results": dict(core_results["results"]),
                    "centered_results": dict(core_results["centered_results"]),
                    "examples_per_task": dict(core_results["examples_per_task"]),
                }

                append_jsonl_fn(scalars_path, core_scalar)
                append_jsonl_fn(core_metrics_path, core_metrics_record)
                print(
                    f"CORE @ step {int(opt_step)}: "
                    f"core_metric={core_results['core_metric']:.4f}, "
                    f"tasks={int(core_results['num_tasks'])}"
                )
            except Exception as exc:
                print(f"[warn] failed to run CORE eval: {exc}")
                core_disabled = True

    accelerator.wait_for_everyone()
    model.train()
    return bool(core_disabled)


def run_blimp_eval_step(
    *,
    accelerator,
    model,
    tokenizer,
    device,
    blimp_eval_module,
    blimp_disabled: bool,
    blimp_records,
    blimp_source_path,
    blimp_data_dir: str,
    blimp_max_examples_per_subset: int,
    blimp_batch_size: int,
    opt_step: int,
    last_train_loss,
    loss_val: float,
    last_lr,
    optimizer,
    tokens_seen_local_total: int,
    scalars_path: str,
    blimp_items_path: str,
    blimp_metrics_path: str,
    step_metrics: list,
    metrics_path: str,
    append_jsonl_fn,
    save_metrics_fn,
    get_current_lr_fn,
    to_jsonable_fn,
    release_eval_memory_fn,
) -> Tuple[bool, Any, Any]:
    """Run BLiMP-fast eval on main process and synchronize all ranks."""
    accelerator.wait_for_everyone()
    model.eval()
    release_eval_memory_fn()

    if accelerator.is_main_process:
        if blimp_disabled:
            pass
        elif blimp_eval_module is None:
            print("[warn] blimp_eval import failed; skipping BLiMP eval.")
            blimp_disabled = True
        else:
            if blimp_records is None:
                print("Loading BLiMP dataset")
                try:
                    blimp_records, blimp_source_path = blimp_eval_module.load_blimp_records(
                        data_dir=(blimp_data_dir or None),
                        max_examples_per_subset=blimp_max_examples_per_subset,
                    )
                except Exception as exc:
                    print(f"[warn] failed to load BLiMP dataset: {exc}")
                    blimp_records = None
                    blimp_source_path = None
                    blimp_disabled = True

            if blimp_records is not None:
                print("starting BLiMP evaluation on main process")
                blimp_model = accelerator.unwrap_model(model)
                blimp_model.eval()
                with torch.no_grad():
                    blimp_sum = blimp_eval_module.evaluate(
                        blimp_model,
                        tokenizer,
                        batch_size=blimp_batch_size,
                        return_per_item=True,
                        score_reduction="sum",
                        records=blimp_records,
                        source_path=blimp_source_path,
                        device=device,
                    )
                    blimp_mean = blimp_eval_module.evaluate(
                        blimp_model,
                        tokenizer,
                        batch_size=blimp_batch_size,
                        return_per_item=True,
                        score_reduction="mean",
                        records=blimp_records,
                        source_path=blimp_source_path,
                        device=device,
                    )

                per_item_sum = list(blimp_sum.pop("per_item", []))
                per_item_mean = list(blimp_mean.pop("per_item", []))

                for rec in per_item_sum:
                    rr = dict(rec)
                    rr.update(
                        {
                            "type": "blimp_item",
                            "step": int(opt_step),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    append_jsonl_fn(blimp_items_path, rr)
                for rec in per_item_mean:
                    rr = dict(rec)
                    rr.update(
                        {
                            "type": "blimp_item_mean",
                            "step": int(opt_step),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    append_jsonl_fn(blimp_items_path, rr)

                record = {
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
                    "lr": float(last_lr) if last_lr is not None else get_current_lr_fn(optimizer),
                    "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
                    "blimp_sum": to_jsonable_fn(blimp_sum),
                    "blimp_mean": to_jsonable_fn(blimp_mean),
                }
                step_metrics.append(record)
                save_metrics_fn(step_metrics, metrics_path)

                sum_scalar = {
                    "type": "blimp",
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "score_reduction": "sum",
                    "num_examples": int(blimp_sum["num_examples"]),
                    "num_subsets": int(blimp_sum["num_subsets"]),
                    "accuracy": float(blimp_sum["accuracy"]),
                    "accuracy_macro_uid": float(blimp_sum["accuracy_macro_uid"]),
                    "tie_rate": float(blimp_sum["tie_rate"]),
                    "batch_size": int(blimp_batch_size),
                    "source": str(blimp_sum["source"]),
                }
                mean_scalar = {
                    "type": "blimp",
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                    "score_reduction": "mean",
                    "num_examples": int(blimp_mean["num_examples"]),
                    "num_subsets": int(blimp_mean["num_subsets"]),
                    "accuracy": float(blimp_mean["accuracy"]),
                    "accuracy_macro_uid": float(blimp_mean["accuracy_macro_uid"]),
                    "tie_rate": float(blimp_mean["tie_rate"]),
                    "batch_size": int(blimp_batch_size),
                    "source": str(blimp_mean["source"]),
                }
                append_jsonl_fn(scalars_path, sum_scalar)
                append_jsonl_fn(scalars_path, mean_scalar)
                append_jsonl_fn(blimp_metrics_path, sum_scalar)
                append_jsonl_fn(blimp_metrics_path, mean_scalar)
                print(
                    f"BLiMP @ step {int(opt_step)}: "
                    f"sum_acc={blimp_sum['accuracy']:.4f}, "
                    f"sum_uid_macro={blimp_sum['accuracy_macro_uid']:.4f}, "
                    f"mean_acc={blimp_mean['accuracy']:.4f}, "
                    f"mean_uid_macro={blimp_mean['accuracy_macro_uid']:.4f}"
                )

    accelerator.wait_for_everyone()
    model.train()
    return bool(blimp_disabled), blimp_records, blimp_source_path


def _unpack_ewok_per_item(result):
    """Backward/forward compatible unpack for ewok_eval.evaluate(return_per_item=True)."""
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected EWoK return type: {type(result)}")
    if len(result) == 3:
        eval_off, eval_full, per_item = result
        return eval_off, eval_full, per_item, None
    if len(result) == 4:
        eval_off, eval_full, per_item, margin_stats = result
        return eval_off, eval_full, per_item, margin_stats
    raise ValueError(f"Unexpected EWoK return tuple length: {len(result)}")


def _evaluate_ewok_all_methods(
    evaluate_fn,
    model,
    tokenizer,
    *,
    batch_size: int,
    score_reduction: str,
):
    try:
        result = evaluate_fn(
            model,
            tokenizer,
            batch_size=batch_size,
            return_per_item=True,
            score_reduction=score_reduction,
            return_all_methods=True,
        )
    except TypeError:
        eval_off, eval_full, per_item, margin_stats = _unpack_ewok_per_item(
            evaluate_fn(
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


def build_ewok_row_category_lookup(
    ewok_df,
    category_columns=("TargetDiff", "ContextDiff", "ContextType"),
):
    """Build row-index -> category metadata lookup for EWoK category aggregations."""
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


def run_ewok_eval_step(
    *,
    accelerator,
    model,
    tokenizer,
    evaluate_fn,
    ewok_batch_size: int,
    opt_step: int,
    last_train_loss,
    loss_val: float,
    last_lr,
    last_grad_norm,
    last_param_norm,
    optimizer,
    tokens_seen_local_total: int,
    ewok_items_path: str,
    step_metrics: list,
    metrics_path: str,
    ewok_row_category_lookup,
    ewok_category_columns,
    append_jsonl_fn,
    save_metrics_fn,
    get_current_lr_fn,
    to_jsonable_fn,
    release_eval_memory_fn,
) -> None:
    """Run periodic EWoK eval on main process and synchronize all ranks."""
    accelerator.wait_for_everyone()
    model.eval()
    release_eval_memory_fn()

    if accelerator.is_main_process:
        print("starting EWoK evaluation on main process")
        with torch.no_grad():
            metrics_by_method_sum, per_item_sum = _evaluate_ewok_all_methods(
                evaluate_fn,
                accelerator.unwrap_model(model),
                tokenizer,
                batch_size=ewok_batch_size,
                score_reduction="sum",
            )
            metrics_by_method_mean, per_item_mean = _evaluate_ewok_all_methods(
                evaluate_fn,
                accelerator.unwrap_model(model),
                tokenizer,
                batch_size=ewok_batch_size,
                score_reduction="mean",
            )
            babylm_sum = metrics_by_method_sum[BABYLM_COMPLETION_CHOICE]
            babylm_mean = metrics_by_method_mean[BABYLM_COMPLETION_CHOICE]
            context_sum = metrics_by_method_sum.get(EWOK_CONTEXT_SENSITIVITY)
            context_mean = metrics_by_method_mean.get(EWOK_CONTEXT_SENSITIVITY)
            eval_off_sum = babylm_sum["domain_scores_official"]
            eval_full_sum = babylm_sum["domain_scores_full"]
            eval_margin_stats_sum = babylm_sum.get("domain_margin_stats")
            eval_off_mean = babylm_mean["domain_scores_official"]
            eval_full_mean = babylm_mean["domain_scores_full"]
            eval_margin_stats_mean = babylm_mean.get("domain_margin_stats")
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
            rr.update(
                {
                    "type": "ewok_item",
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            append_jsonl_fn(ewok_items_path, rr)
        for r in per_item_mean:
            rr = dict(r)
            rr.update(
                {
                    "type": "ewok_item_mean",
                    "step": int(opt_step),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            append_jsonl_fn(ewok_items_path, rr)

        record = {
            "step": int(opt_step),
            "timestamp": datetime.now().isoformat(),
            "train_loss_last": float(last_train_loss) if last_train_loss is not None else float(loss_val),
            "lr": float(last_lr) if last_lr is not None else get_current_lr_fn(optimizer),
            "grad_norm_l2": float(last_grad_norm) if last_grad_norm is not None else None,
            "param_norm_l2": float(last_param_norm) if last_param_norm is not None else None,
            "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
            # Keep these keys as backward-compatible aliases for sum-reduction.
            "eval_official": to_jsonable_fn(eval_off_sum),
            "eval_full": to_jsonable_fn(eval_full_sum),
            "eval_official_sum": to_jsonable_fn(eval_off_sum),
            "eval_full_sum": to_jsonable_fn(eval_full_sum),
            "eval_official_mean": to_jsonable_fn(eval_off_mean),
            "eval_full_mean": to_jsonable_fn(eval_full_mean),
            # Margin stats include mean_signed_m / mean_abs_m per domain and global average.
            "eval_margin_stats": to_jsonable_fn(eval_margin_stats_sum),
            "eval_margin_stats_sum": to_jsonable_fn(eval_margin_stats_sum),
            "eval_margin_stats_mean": to_jsonable_fn(eval_margin_stats_mean),
            "eval_by_category_full_sum": to_jsonable_fn(eval_by_category_full_sum),
            "eval_by_category_full_mean": to_jsonable_fn(eval_by_category_full_mean),
            "eval_babylm_completion_choice_official": to_jsonable_fn(eval_off_sum),
            "eval_babylm_completion_choice_full": to_jsonable_fn(eval_full_sum),
            "eval_babylm_completion_choice_official_sum": to_jsonable_fn(eval_off_sum),
            "eval_babylm_completion_choice_full_sum": to_jsonable_fn(eval_full_sum),
            "eval_babylm_completion_choice_official_mean": to_jsonable_fn(eval_off_mean),
            "eval_babylm_completion_choice_full_mean": to_jsonable_fn(eval_full_mean),
            "eval_babylm_completion_choice_margin_stats": to_jsonable_fn(eval_margin_stats_sum),
            "eval_babylm_completion_choice_margin_stats_sum": to_jsonable_fn(eval_margin_stats_sum),
            "eval_babylm_completion_choice_margin_stats_mean": to_jsonable_fn(eval_margin_stats_mean),
            "eval_babylm_completion_choice_by_category_full_sum": to_jsonable_fn(eval_by_category_full_sum),
            "eval_babylm_completion_choice_by_category_full_mean": to_jsonable_fn(eval_by_category_full_mean),
        }
        if context_sum is not None:
            record.update(
                {
                    "eval_context_sensitivity_official": to_jsonable_fn(
                        context_sum["domain_scores_official"]
                    ),
                    "eval_context_sensitivity_full": to_jsonable_fn(
                        context_sum["domain_scores_full"]
                    ),
                    "eval_context_sensitivity_official_sum": to_jsonable_fn(
                        context_sum["domain_scores_official"]
                    ),
                    "eval_context_sensitivity_full_sum": to_jsonable_fn(
                        context_sum["domain_scores_full"]
                    ),
                    "eval_context_sensitivity_margin_stats": to_jsonable_fn(
                        context_sum.get("domain_margin_stats")
                    ),
                    "eval_context_sensitivity_margin_stats_sum": to_jsonable_fn(
                        context_sum.get("domain_margin_stats")
                    ),
                    "eval_ewok_paper_context_sensitivity_official": to_jsonable_fn(
                        context_sum["domain_scores_official"]
                    ),
                    "eval_ewok_paper_context_sensitivity_full": to_jsonable_fn(
                        context_sum["domain_scores_full"]
                    ),
                    "eval_ewok_paper_context_sensitivity_official_sum": to_jsonable_fn(
                        context_sum["domain_scores_official"]
                    ),
                    "eval_ewok_paper_context_sensitivity_full_sum": to_jsonable_fn(
                        context_sum["domain_scores_full"]
                    ),
                    "eval_ewok_paper_context_sensitivity_margin_stats": to_jsonable_fn(
                        context_sum.get("domain_margin_stats")
                    ),
                    "eval_ewok_paper_context_sensitivity_margin_stats_sum": to_jsonable_fn(
                        context_sum.get("domain_margin_stats")
                    ),
                }
            )
        if context_mean is not None:
            record.update(
                {
                    "eval_context_sensitivity_official_mean": to_jsonable_fn(
                        context_mean["domain_scores_official"]
                    ),
                    "eval_context_sensitivity_full_mean": to_jsonable_fn(
                        context_mean["domain_scores_full"]
                    ),
                    "eval_context_sensitivity_margin_stats_mean": to_jsonable_fn(
                        context_mean.get("domain_margin_stats")
                    ),
                    "eval_ewok_paper_context_sensitivity_official_mean": to_jsonable_fn(
                        context_mean["domain_scores_official"]
                    ),
                    "eval_ewok_paper_context_sensitivity_full_mean": to_jsonable_fn(
                        context_mean["domain_scores_full"]
                    ),
                    "eval_ewok_paper_context_sensitivity_margin_stats_mean": to_jsonable_fn(
                        context_mean.get("domain_margin_stats")
                    ),
                }
            )
        step_metrics.append(record)
        save_metrics_fn(step_metrics, metrics_path)
        print(f"Saved metrics to {metrics_path}")
        print(f"Appended per-item EWoK to {ewok_items_path}")

    accelerator.wait_for_everyone()
    model.train()


def run_final_ewok_eval_main_process(
    *,
    accelerator,
    model,
    tokenizer,
    evaluate_fn,
    ewok_batch_size: int,
    opt_step: int,
    optimizer,
    tokens_seen_local_total: int,
    ewok_items_path: str,
    step_metrics: list,
    metrics_path: str,
    ewok_row_category_lookup,
    ewok_category_columns,
    append_jsonl_fn,
    save_metrics_fn,
    get_current_lr_fn,
    to_jsonable_fn,
) -> None:
    """Run final EWoK eval and logging on main process only."""
    if not accelerator.is_main_process:
        return

    model.eval()
    accelerator.unwrap_model(model).eval()
    with torch.no_grad():
        metrics_by_method_sum, per_item_sum = _evaluate_ewok_all_methods(
            evaluate_fn,
            accelerator.unwrap_model(model),
            tokenizer,
            batch_size=ewok_batch_size,
            score_reduction="sum",
        )
        metrics_by_method_mean, per_item_mean = _evaluate_ewok_all_methods(
            evaluate_fn,
            accelerator.unwrap_model(model),
            tokenizer,
            batch_size=ewok_batch_size,
            score_reduction="mean",
        )
        babylm_sum = metrics_by_method_sum[BABYLM_COMPLETION_CHOICE]
        babylm_mean = metrics_by_method_mean[BABYLM_COMPLETION_CHOICE]
        context_sum = metrics_by_method_sum.get(EWOK_CONTEXT_SENSITIVITY)
        context_mean = metrics_by_method_mean.get(EWOK_CONTEXT_SENSITIVITY)
        eval_off_sum = babylm_sum["domain_scores_official"]
        eval_full_sum = babylm_sum["domain_scores_full"]
        eval_margin_stats_sum = babylm_sum.get("domain_margin_stats")
        eval_off_mean = babylm_mean["domain_scores_official"]
        eval_full_mean = babylm_mean["domain_scores_full"]
        eval_margin_stats_mean = babylm_mean.get("domain_margin_stats")
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
        rr.update(
            {
                "type": "ewok_item_final",
                "step": int(opt_step),
                "timestamp": datetime.now().isoformat(),
            }
        )
        append_jsonl_fn(ewok_items_path, rr)
    for r in per_item_mean:
        rr = dict(r)
        rr.update(
            {
                "type": "ewok_item_final_mean",
                "step": int(opt_step),
                "timestamp": datetime.now().isoformat(),
            }
        )
        append_jsonl_fn(ewok_items_path, rr)

    record = {
        "step": int(opt_step),
        "timestamp": datetime.now().isoformat(),
        "final": True,
        "lr": get_current_lr_fn(optimizer),
        "tokens_seen_global_approx": int(tokens_seen_local_total * accelerator.num_processes),
        # Keep these keys as backward-compatible aliases for sum-reduction.
        "eval_official": to_jsonable_fn(eval_off_sum),
        "eval_full": to_jsonable_fn(eval_full_sum),
        "eval_official_sum": to_jsonable_fn(eval_off_sum),
        "eval_full_sum": to_jsonable_fn(eval_full_sum),
        "eval_official_mean": to_jsonable_fn(eval_off_mean),
        "eval_full_mean": to_jsonable_fn(eval_full_mean),
        # Margin stats include mean_signed_m / mean_abs_m per domain and global average.
        "eval_margin_stats": to_jsonable_fn(eval_margin_stats_sum),
        "eval_margin_stats_sum": to_jsonable_fn(eval_margin_stats_sum),
        "eval_margin_stats_mean": to_jsonable_fn(eval_margin_stats_mean),
        "eval_by_category_full_sum": to_jsonable_fn(eval_by_category_full_sum),
        "eval_by_category_full_mean": to_jsonable_fn(eval_by_category_full_mean),
        "eval_babylm_completion_choice_official": to_jsonable_fn(eval_off_sum),
        "eval_babylm_completion_choice_full": to_jsonable_fn(eval_full_sum),
        "eval_babylm_completion_choice_official_sum": to_jsonable_fn(eval_off_sum),
        "eval_babylm_completion_choice_full_sum": to_jsonable_fn(eval_full_sum),
        "eval_babylm_completion_choice_official_mean": to_jsonable_fn(eval_off_mean),
        "eval_babylm_completion_choice_full_mean": to_jsonable_fn(eval_full_mean),
        "eval_babylm_completion_choice_margin_stats": to_jsonable_fn(eval_margin_stats_sum),
        "eval_babylm_completion_choice_margin_stats_sum": to_jsonable_fn(eval_margin_stats_sum),
        "eval_babylm_completion_choice_margin_stats_mean": to_jsonable_fn(eval_margin_stats_mean),
        "eval_babylm_completion_choice_by_category_full_sum": to_jsonable_fn(eval_by_category_full_sum),
        "eval_babylm_completion_choice_by_category_full_mean": to_jsonable_fn(eval_by_category_full_mean),
    }
    if context_sum is not None:
        record.update(
            {
                "eval_context_sensitivity_official": to_jsonable_fn(
                    context_sum["domain_scores_official"]
                ),
                "eval_context_sensitivity_full": to_jsonable_fn(
                    context_sum["domain_scores_full"]
                ),
                "eval_context_sensitivity_official_sum": to_jsonable_fn(
                    context_sum["domain_scores_official"]
                ),
                "eval_context_sensitivity_full_sum": to_jsonable_fn(
                    context_sum["domain_scores_full"]
                ),
                "eval_context_sensitivity_margin_stats": to_jsonable_fn(
                    context_sum.get("domain_margin_stats")
                ),
                "eval_context_sensitivity_margin_stats_sum": to_jsonable_fn(
                    context_sum.get("domain_margin_stats")
                ),
                "eval_ewok_paper_context_sensitivity_official": to_jsonable_fn(
                    context_sum["domain_scores_official"]
                ),
                "eval_ewok_paper_context_sensitivity_full": to_jsonable_fn(
                    context_sum["domain_scores_full"]
                ),
                "eval_ewok_paper_context_sensitivity_official_sum": to_jsonable_fn(
                    context_sum["domain_scores_official"]
                ),
                "eval_ewok_paper_context_sensitivity_full_sum": to_jsonable_fn(
                    context_sum["domain_scores_full"]
                ),
                "eval_ewok_paper_context_sensitivity_margin_stats": to_jsonable_fn(
                    context_sum.get("domain_margin_stats")
                ),
                "eval_ewok_paper_context_sensitivity_margin_stats_sum": to_jsonable_fn(
                    context_sum.get("domain_margin_stats")
                ),
            }
        )
    if context_mean is not None:
        record.update(
            {
                "eval_context_sensitivity_official_mean": to_jsonable_fn(
                    context_mean["domain_scores_official"]
                ),
                "eval_context_sensitivity_full_mean": to_jsonable_fn(
                    context_mean["domain_scores_full"]
                ),
                "eval_context_sensitivity_margin_stats_mean": to_jsonable_fn(
                    context_mean.get("domain_margin_stats")
                ),
                "eval_ewok_paper_context_sensitivity_official_mean": to_jsonable_fn(
                    context_mean["domain_scores_official"]
                ),
                "eval_ewok_paper_context_sensitivity_full_mean": to_jsonable_fn(
                    context_mean["domain_scores_full"]
                ),
                "eval_ewok_paper_context_sensitivity_margin_stats_mean": to_jsonable_fn(
                    context_mean.get("domain_margin_stats")
                ),
            }
        )
    step_metrics.append(record)
    save_metrics_fn(step_metrics, metrics_path)
    print(f"Saved final metrics to {metrics_path}")
    print(f"Appended final per-item EWoK to {ewok_items_path}")
