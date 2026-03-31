"""Shared helpers for TrackStar continued-pretraining ablations."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from ..common.export import write_json, write_jsonl
from ..common.checkpoints import build_model_from_checkpoint, load_tokenizer_from_checkpoint

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
    from research.bos_aligned_proto.evaluation.ewok_category import build_ewok_row_category_lookup
except ImportError:
    from ....evaluation.ewok_category import build_ewok_row_category_lookup


REPO_ROOT = Path(__file__).resolve().parents[5]
TRAINER_MODULE = "research.bos_aligned_proto.training.trainer"
DEFAULT_METRIC_NAME = "babylm_completion_choice_margin_combined"
DEFAULT_GROUP_BY = "average"
DEFAULT_REDUCTION = "mean"
DEFAULT_MICRO_BATCH_SIZE = 4
DEFAULT_TOTAL_BATCH_TOKENS = 32 * 1024
DEFAULT_NUM_EPOCHS = 3
DEFAULT_NUM_PROCESSES = 1
DEFAULT_EWOK_BATCH_SIZE = 4
DEFAULT_LRS = (4e-5,)
DEFAULT_SEEDS = (42,)
RECOMMENDED_LR_SWEEP = (1e-5, 2e-5, 4e-5, 8e-5)
SUPPORTED_GROUP_BYS = ("average", "domain", "ContextDiff", "TargetDiff", "ContextType")
SUPPORTED_REDUCTIONS = ("mean", "sum")
GROUPING_COLUMNS = ("TargetDiff", "ContextDiff", "ContextType")
EWOK_ITEM_TYPES = {
    "ewok_item": ("sum", False),
    "ewok_item_final": ("sum", True),
    "ewok_item_mean": ("mean", False),
    "ewok_item_final_mean": ("mean", True),
    "baseline_ewok_item": ("sum", False),
    "baseline_ewok_item_mean": ("mean", False),
}
ROW_CATEGORY_LOOKUP = build_ewok_row_category_lookup(EWOK_DF, GROUPING_COLUMNS)


@dataclass(frozen=True)
class CheckpointModelConfig:
    seq_len: int
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int


@dataclass(frozen=True)
class TrainingBudget:
    seq_len: int
    num_rows: int
    micro_batch_size: int
    total_batch_tokens: int
    num_processes: int
    grad_accum_steps: int
    effective_global_batch_seqs: int
    steps_per_epoch: int
    max_train_steps: int


@dataclass(frozen=True)
class AblationRunSpec:
    arm: str
    learning_rate: float
    seed: int
    data_dir: Path
    experiments_dir: Path
    expected_run_name: str
    expected_run_dir: Path
    budget: TrainingBudget
    warmup_iters: int
    ewok_batch_size: int
    command: tuple[str, ...]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned.strip("_") or "unknown"


def format_lr_tag(value: float) -> str:
    return _safe_name(f"{float(value):.0e}")


def parse_float_list(raw: str | Sequence[float] | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(float(value) for value in default)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    if not values:
        return tuple(float(value) for value in default)
    return tuple(float(value) for value in values)


def parse_int_list(raw: str | Sequence[int] | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(value) for value in default)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    if not values:
        return tuple(int(value) for value in default)
    return tuple(int(value) for value in values)


def load_checkpoint_model_config(checkpoint_dir: str | Path) -> CheckpointModelConfig:
    payload = _load_json(Path(checkpoint_dir) / "config.json")
    seq_len = payload.get("n_positions", payload.get("n_ctx"))
    if seq_len is None:
        raise ValueError(f"Checkpoint {checkpoint_dir} is missing n_positions/n_ctx in config.json")
    return CheckpointModelConfig(
        seq_len=int(seq_len),
        vocab_size=int(payload["vocab_size"]),
        n_embd=int(payload["n_embd"]),
        n_head=int(payload["n_head"]),
        n_layer=int(payload["n_layer"]),
    )


def _list_train_shards(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("train_*.bin"))


def _infer_num_rows(meta: dict[str, Any], train_shards: Sequence[Path]) -> int:
    if "num_rows" in meta:
        return int(meta["num_rows"])
    row_tokens = int(meta["row_tokens"])
    total_tokens = 0
    for shard in train_shards:
        total_tokens += int(shard.stat().st_size // 2)
    if total_tokens % row_tokens != 0:
        raise ValueError(
            f"Could not infer num_rows cleanly for {train_shards[0].parent if train_shards else '<empty>'}: "
            f"total_tokens={total_tokens}, row_tokens={row_tokens}"
        )
    return total_tokens // row_tokens


def compute_training_budget(
    *,
    seq_len: int,
    num_rows: int,
    micro_batch_size: int,
    total_batch_tokens: int,
    num_processes: int,
    num_epochs: int,
) -> TrainingBudget:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if num_rows <= 0:
        raise ValueError("num_rows must be > 0")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be > 0")
    if total_batch_tokens <= 0:
        raise ValueError("total_batch_tokens must be > 0")
    if num_processes <= 0:
        raise ValueError("num_processes must be > 0")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")

    tokens_per_microstep_global = int(num_processes) * int(micro_batch_size) * int(seq_len)
    grad_accum_steps = max(1, math.ceil(int(total_batch_tokens) / tokens_per_microstep_global))
    effective_global_batch_seqs = grad_accum_steps * int(micro_batch_size) * int(num_processes)
    steps_per_epoch = max(1, math.ceil(int(num_rows) / effective_global_batch_seqs))
    max_train_steps = int(num_epochs) * steps_per_epoch
    return TrainingBudget(
        seq_len=int(seq_len),
        num_rows=int(num_rows),
        micro_batch_size=int(micro_batch_size),
        total_batch_tokens=int(total_batch_tokens),
        num_processes=int(num_processes),
        grad_accum_steps=int(grad_accum_steps),
        effective_global_batch_seqs=int(effective_global_batch_seqs),
        steps_per_epoch=int(steps_per_epoch),
        max_train_steps=int(max_train_steps),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == str(src.resolve()):
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_bos_row_training_view(source_dataset_dir: str | Path, output_dir: str | Path) -> Path:
    source_dir = Path(source_dataset_dir).expanduser().resolve()
    view_dir = Path(output_dir).expanduser().resolve()
    view_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(_load_json(source_dir / "meta.json"))
    train_shards = _list_train_shards(source_dir)
    if not train_shards:
        raise FileNotFoundError(f"No train shards found under {source_dir}")

    for shard in train_shards:
        _symlink_or_copy(shard, view_dir / shard.name)

    rows_path = source_dir / "rows.jsonl"
    if rows_path.exists():
        _symlink_or_copy(rows_path, view_dir / rows_path.name)

    val_path = view_dir / "val_000000.bin"
    with val_path.open("wb") as out_handle:
        for shard in train_shards:
            with shard.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle)

    meta["cpt_ablation"] = {
        "source_dataset_dir": str(source_dir),
        "synthetic_val_split": "concatenate_train_shards",
    }
    write_json(view_dir / "meta.json", meta)
    return view_dir


def resolve_matched_pool_datasets(matched_pool_dir: str | Path) -> dict[str, Path]:
    root = Path(matched_pool_dir).expanduser().resolve()
    treated = root / "treated_dataset"
    control = root / "control_dataset"
    if not treated.is_dir():
        raise FileNotFoundError(f"Missing treated dataset under {root}")
    if not control.is_dir():
        raise FileNotFoundError(f"Missing control dataset under {root}")
    return {"treated": treated, "control": control}


def build_expected_bos_run_name(
    *,
    loader_kind: str,
    micro_batch_size: int,
    seq_len: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    total_batch_tokens: int,
    effective_total_tokens: int,
    num_processes: int,
    grad_accum_steps: int,
    seed: int,
    max_train_steps: int,
) -> str:
    if loader_kind == "stream":
        loader_tag = "stream"
    elif loader_kind == "bos_row":
        loader_tag = "bosrow"
    else:
        loader_tag = "bospackedindex"
    return (
        f"babygpt_fineweb_{loader_tag}_mbs{micro_batch_size}_T{seq_len}_"
        f"d{n_embd}_h{n_head}_L{n_layer}_"
        f"tok{total_batch_tokens}_efftok{effective_total_tokens}_"
        f"ws{num_processes}_gas{grad_accum_steps}_seed{seed}_"
        f"steps{max_train_steps}"
    )


def build_bos_trainer_command(
    *,
    base_ckpt: str | Path,
    data_dir: str | Path,
    experiments_dir: str | Path,
    model_config: CheckpointModelConfig,
    budget: TrainingBudget,
    learning_rate: float,
    warmup_iters: int,
    seed: int,
    ewok_every: int,
    ewok_batch_size: int,
    num_processes: int,
    num_workers: int = 0,
) -> list[str]:
    if int(num_processes) <= 1:
        prefix = [sys.executable, "-m", TRAINER_MODULE]
    else:
        prefix = [
            "accelerate",
            "launch",
            "--num_processes",
            str(int(num_processes)),
            "-m",
            TRAINER_MODULE,
        ]

    return prefix + [
        "--loader_kind",
        "bos_row",
        "--data_dir",
        str(Path(data_dir).expanduser().resolve()),
        "--experiments_dir",
        str(Path(experiments_dir).expanduser().resolve()),
        "--init_from_ckpt",
        str(Path(base_ckpt).expanduser().resolve()),
        "--seed",
        str(int(seed)),
        "--micro_batch_size",
        str(int(budget.micro_batch_size)),
        "--total_batch_tokens",
        str(int(budget.total_batch_tokens)),
        "--max_train_steps",
        str(int(budget.max_train_steps)),
        "--seq_len",
        str(int(model_config.seq_len)),
        "--vocab_size",
        str(int(model_config.vocab_size)),
        "--n_embd",
        str(int(model_config.n_embd)),
        "--n_head",
        str(int(model_config.n_head)),
        "--n_layer",
        str(int(model_config.n_layer)),
        "--num_workers",
        str(int(num_workers)),
        "--learning_rate",
        f"{float(learning_rate):.12g}",
        "--warmup_iters",
        str(int(warmup_iters)),
        "--learning_rate_decay_frac",
        "0.0",
        "--eval_every",
        "0",
        "--hellaswag_every",
        "0",
        "--core_every",
        "0",
        "--ewok_every",
        str(int(ewok_every)),
        "--ewok_batch_size",
        str(int(ewok_batch_size)),
        "--save_every",
        "0",
        "--exposure_every",
        "0",
    ]


def locate_trainer_run_dir(experiments_dir: str | Path, expected_run_name: str) -> Path | None:
    experiments_path = Path(experiments_dir).expanduser().resolve()
    expected = experiments_path / expected_run_name
    if expected.is_dir():
        return expected

    candidates = [child for child in experiments_path.iterdir() if child.is_dir()] if experiments_path.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    return None


def _unpack_ewok_per_item(result: Any) -> tuple[dict, dict, list[dict], dict | None]:
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected EWoK return type: {type(result)}")
    if len(result) == 3:
        eval_off, eval_full, per_item = result
        return eval_off, eval_full, per_item, None
    if len(result) == 4:
        eval_off, eval_full, per_item, margin_stats = result
        return eval_off, eval_full, per_item, margin_stats
    raise ValueError(f"Unexpected EWoK return tuple length: {len(result)}")


def evaluate_ewok_all_methods(
    model,
    tokenizer,
    *,
    batch_size: int,
    score_reduction: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        raise ValueError(f"Unexpected EWoK result from evaluate(..., return_all_methods=True): {type(result)}")
    metrics_by_method, per_item = result
    if not isinstance(metrics_by_method, dict):
        raise TypeError(f"Unexpected metrics_by_method type: {type(metrics_by_method)}")
    if not isinstance(per_item, list):
        raise TypeError(f"Unexpected per_item type: {type(per_item)}")
    return metrics_by_method, per_item


def aggregate_margin_records(
    records: Iterable[dict[str, Any]],
    *,
    group_by: str,
    metric_name: str = DEFAULT_METRIC_NAME,
    row_category_lookup: dict[int, dict[str, str]] | None = None,
) -> dict[str, float]:
    if group_by not in SUPPORTED_GROUP_BYS:
        raise ValueError(f"Unsupported group_by={group_by!r}; expected one of {SUPPORTED_GROUP_BYS}")
    lookup = row_category_lookup or ROW_CATEGORY_LOOKUP
    buckets: dict[str, list[float]] = {}
    for record in records:
        if metric_name not in record:
            continue
        try:
            value = float(record[metric_name])
        except Exception:
            continue
        if group_by == "average":
            group_name = "average"
        elif group_by == "domain":
            group_name = str(record.get("domain", "<NA>"))
        else:
            row_idx = record.get("row_index")
            if not isinstance(row_idx, int):
                continue
            group_name = lookup.get(int(row_idx), {}).get(group_by, "<NA>")
        buckets.setdefault(str(group_name), []).append(value)
    return {
        str(group_name): float(np.mean(values))
        for group_name, values in sorted(buckets.items())
        if values
    }


def evaluate_checkpoint_baseline(
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    ewok_batch_size: int,
    metric_name: str = DEFAULT_METRIC_NAME,
    device: str | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "baseline_ewok_items.jsonl"
    summary_path = out_dir / "baseline_summary.json"
    if items_path.exists() and summary_path.exists():
        return {"items_path": items_path, "summary_path": summary_path}

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint_dir, device=resolved_device)
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)

    timestamp = datetime.now().isoformat()
    _, per_item_sum = evaluate_ewok_all_methods(
        model,
        tokenizer,
        batch_size=int(ewok_batch_size),
        score_reduction="sum",
    )
    _, per_item_mean = evaluate_ewok_all_methods(
        model,
        tokenizer,
        batch_size=int(ewok_batch_size),
        score_reduction="mean",
    )

    baseline_rows: list[dict[str, Any]] = []
    for record in per_item_sum:
        baseline_rows.append(
            {
                **record,
                "type": "baseline_ewok_item",
                "step": 0,
                "timestamp": timestamp,
            }
        )
    for record in per_item_mean:
        baseline_rows.append(
            {
                **record,
                "type": "baseline_ewok_item_mean",
                "step": 0,
                "timestamp": timestamp,
            }
        )
    write_jsonl(items_path, baseline_rows)

    summary = {
        "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
        "metric_name": str(metric_name),
        "ewok_batch_size": int(ewok_batch_size),
        "device": str(resolved_device),
        "category_columns": list(GROUPING_COLUMNS),
        "reductions": {
            "sum": {
                group_by: aggregate_margin_records(per_item_sum, group_by=group_by, metric_name=metric_name)
                for group_by in SUPPORTED_GROUP_BYS
            },
            "mean": {
                group_by: aggregate_margin_records(per_item_mean, group_by=group_by, metric_name=metric_name)
                for group_by in SUPPORTED_GROUP_BYS
            },
        },
    }
    write_json(summary_path, summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"items_path": items_path, "summary_path": summary_path}


def _is_ewok_step_metric(record: dict[str, Any]) -> bool:
    return bool(
        "eval_official" in record
        or "eval_official_sum" in record
        or "eval_official_mean" in record
        or "eval_babylm_completion_choice_official" in record
        or "eval_babylm_completion_choice_official_sum" in record
        or "eval_babylm_completion_choice_official_mean" in record
    )


def _dedupe_ewok_step_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict) or not _is_ewok_step_metric(record):
            continue
        step = record.get("step")
        if not isinstance(step, int):
            continue
        grouped.setdefault(int(step), []).append(record)

    deduped: list[dict[str, Any]] = []
    for step, group in grouped.items():
        preferred = [record for record in group if not bool(record.get("final", False))]
        candidates = preferred if preferred else group
        candidates = sorted(candidates, key=lambda row: str(row.get("timestamp", "")), reverse=True)
        deduped.append(candidates[0])
    deduped.sort(key=lambda row: int(row["step"]))
    return deduped


def load_deduped_ewok_items_frame(path: str | Path) -> pd.DataFrame:
    rows = []
    for row in _load_jsonl(path):
        record_type = str(row.get("type", ""))
        mapped = EWOK_ITEM_TYPES.get(record_type)
        if mapped is None:
            continue
        reduction, is_final = mapped
        step = row.get("step")
        row_index = row.get("row_index")
        if not isinstance(step, int) or not isinstance(row_index, int):
            continue
        rows.append(
            {
                **row,
                "reduction": reduction,
                "is_final_eval": bool(is_final),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows)
    frame = frame.sort_values(
        ["reduction", "step", "row_index", "is_final_eval", "timestamp"],
        ascending=[True, True, True, True, False],
    )
    frame = frame.drop_duplicates(subset=["reduction", "step", "row_index"], keep="first").reset_index(drop=True)
    return frame


def _group_label_order(group_by: str, available: Iterable[str]) -> list[str]:
    available_set = {str(value) for value in available}
    if group_by == "average":
        return ["average"] if "average" in available_set else sorted(available_set)
    if group_by == "domain":
        domain_order = [str(value) for value in EWOK_DF["Domain"].unique().tolist()]
        ordered = [value for value in domain_order if value in available_set]
        tail = sorted(available_set - set(ordered))
        return ordered + tail
    if group_by in GROUPING_COLUMNS:
        values = []
        for _, row in EWOK_DF.iterrows():
            value = ROW_CATEGORY_LOOKUP.get(int(_), {}).get(group_by, "<NA>")
            if value not in values:
                values.append(value)
        ordered = [value for value in values if value in available_set]
        tail = sorted(available_set - set(ordered))
        return ordered + tail
    return sorted(available_set)


def build_curves_for_run(
    *,
    run_record: dict[str, Any],
    baseline_summary: dict[str, Any],
    metric_name: str = DEFAULT_METRIC_NAME,
) -> list[dict[str, Any]]:
    step_metrics = _load_json(run_record["step_metrics_path"])
    ewok_steps = _dedupe_ewok_step_metrics(step_metrics)
    step_to_final = {int(record["step"]): bool(record.get("final", False)) for record in ewok_steps}
    step_values = {int(record["step"]) for record in ewok_steps}

    items_frame = load_deduped_ewok_items_frame(run_record["ewok_items_path"])
    if items_frame.empty:
        return []

    if step_values:
        items_frame = items_frame.loc[items_frame["step"].isin(step_values)].copy()

    curves: list[dict[str, Any]] = []
    for reduction in SUPPORTED_REDUCTIONS:
        reduction_frame = items_frame.loc[items_frame["reduction"] == reduction].copy()
        if reduction_frame.empty:
            continue
        for group_by in SUPPORTED_GROUP_BYS:
            if group_by == "average":
                reduction_frame["group_name"] = "average"
            elif group_by == "domain":
                reduction_frame["group_name"] = reduction_frame["domain"].astype(str)
            else:
                reduction_frame["group_name"] = reduction_frame["row_index"].map(
                    lambda idx: ROW_CATEGORY_LOOKUP.get(int(idx), {}).get(group_by, "<NA>")
                )
            grouped = (
                reduction_frame.groupby(["step", "group_name"], as_index=False)[metric_name]
                .mean()
                .rename(columns={metric_name: "value"})
            )
            for row in grouped.itertuples(index=False):
                baseline_value = (
                    baseline_summary.get("reductions", {})
                    .get(reduction, {})
                    .get(group_by, {})
                    .get(str(row.group_name))
                )
                curves.append(
                    {
                        "arm": str(run_record["arm"]),
                        "lr": float(run_record["learning_rate"]),
                        "seed": int(run_record["seed"]),
                        "run_dir": str(run_record["run_dir"]),
                        "step": int(row.step),
                        "epoch": float(int(row.step) / max(1, int(run_record["steps_per_epoch"]))),
                        "final": bool(step_to_final.get(int(row.step), False)),
                        "metric_name": metric_name,
                        "reduction": reduction,
                        "group_by": group_by,
                        "group_name": str(row.group_name),
                        "value": float(row.value),
                        "baseline_value": (None if baseline_value is None else float(baseline_value)),
                        "delta_from_baseline": (
                            None if baseline_value is None else float(row.value) - float(baseline_value)
                        ),
                    }
                )
    return curves


def build_ablation_summary(curves: Sequence[dict[str, Any]], *, metric_name: str = DEFAULT_METRIC_NAME) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(curves)
    if frame.empty:
        return {
            "metric_name": metric_name,
            "default_view": {
                "group_by": DEFAULT_GROUP_BY,
                "group_name": "average",
                "reduction": DEFAULT_REDUCTION,
            },
            "final_records": [],
            "seed_summary": [],
        }

    frame = frame.sort_values(["lr", "seed", "arm", "reduction", "group_by", "group_name", "step"])
    final_frame = (
        frame.groupby(["lr", "seed", "arm", "reduction", "group_by", "group_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    records: list[dict[str, Any]] = []
    grouped = final_frame.groupby(["lr", "seed", "reduction", "group_by", "group_name"], as_index=False)
    for _, row_frame in grouped:
        treated_rows = row_frame.loc[row_frame["arm"] == "treated"]
        control_rows = row_frame.loc[row_frame["arm"] == "control"]
        if treated_rows.empty or control_rows.empty:
            continue
        treated_value = float(treated_rows["value"].iloc[-1])
        control_value = float(control_rows["value"].iloc[-1])
        treated_delta = treated_rows["delta_from_baseline"].iloc[-1]
        control_delta = control_rows["delta_from_baseline"].iloc[-1]
        records.append(
            {
                "lr": float(row_frame["lr"].iloc[0]),
                "seed": int(row_frame["seed"].iloc[0]),
                "reduction": str(row_frame["reduction"].iloc[0]),
                "group_by": str(row_frame["group_by"].iloc[0]),
                "group_name": str(row_frame["group_name"].iloc[0]),
                "treated_value": treated_value,
                "control_value": control_value,
                "treated_delta_from_baseline": (None if pd.isna(treated_delta) else float(treated_delta)),
                "control_delta_from_baseline": (None if pd.isna(control_delta) else float(control_delta)),
                "treated_minus_control": treated_value - control_value,
            }
        )
    summary_frame = pd.DataFrame.from_records(records)
    seed_summary: list[dict[str, Any]] = []
    if not summary_frame.empty:
        grouped = summary_frame.groupby(["lr", "reduction", "group_by", "group_name"], as_index=False)
        for group in grouped:
            (_, row_frame) = group
            seed_summary.append(
                {
                    "lr": float(row_frame["lr"].iloc[0]),
                    "reduction": str(row_frame["reduction"].iloc[0]),
                    "group_by": str(row_frame["group_by"].iloc[0]),
                    "group_name": str(row_frame["group_name"].iloc[0]),
                    "num_seeds": int(len(row_frame)),
                    "treated_minus_control_mean": float(row_frame["treated_minus_control"].mean()),
                    "treated_minus_control_std": (
                        float(row_frame["treated_minus_control"].std(ddof=0))
                        if len(row_frame) > 1
                        else 0.0
                    ),
                    "treated_value_mean": float(row_frame["treated_value"].mean()),
                    "control_value_mean": float(row_frame["control_value"].mean()),
                }
            )
    return {
        "metric_name": metric_name,
        "default_view": {
            "group_by": DEFAULT_GROUP_BY,
            "group_name": "average",
            "reduction": DEFAULT_REDUCTION,
        },
        "final_records": records,
        "seed_summary": seed_summary,
    }


def write_ablation_outputs(
    *,
    output_dir: str | Path,
    run_records: Sequence[dict[str, Any]],
    curves: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    runs_path = root / "ablation_runs.json"
    curves_path = root / "ablation_curves.jsonl"
    summary_path = root / "ablation_summary.json"
    write_json(runs_path, list(run_records))
    write_jsonl(curves_path, list(curves))
    write_json(summary_path, summary)
    return {
        "runs_path": runs_path,
        "curves_path": curves_path,
        "summary_path": summary_path,
    }


def launch_training_run(spec: AblationRunSpec, *, dry_run: bool = False) -> dict[str, Any]:
    existing_run_dir = locate_trainer_run_dir(spec.experiments_dir, spec.expected_run_name)
    if existing_run_dir is not None:
        step_metrics_path = existing_run_dir / "step_metrics.json"
        ewok_items_path = existing_run_dir / "ewok_items.jsonl"
        if step_metrics_path.exists() and ewok_items_path.exists():
            return {
                "arm": spec.arm,
                "learning_rate": float(spec.learning_rate),
                "seed": int(spec.seed),
                "data_dir": str(spec.data_dir),
                "experiments_dir": str(spec.experiments_dir),
                "expected_run_name": spec.expected_run_name,
                "run_dir": str(existing_run_dir),
                "step_metrics_path": str(step_metrics_path),
                "ewok_items_path": str(ewok_items_path),
                "steps_per_epoch": int(spec.budget.steps_per_epoch),
                "max_train_steps": int(spec.budget.max_train_steps),
                "status": "reused",
                "command": list(spec.command),
            }

    if dry_run:
        return {
            "arm": spec.arm,
            "learning_rate": float(spec.learning_rate),
            "seed": int(spec.seed),
            "data_dir": str(spec.data_dir),
            "experiments_dir": str(spec.experiments_dir),
            "expected_run_name": spec.expected_run_name,
            "run_dir": None,
            "step_metrics_path": None,
            "ewok_items_path": None,
            "steps_per_epoch": int(spec.budget.steps_per_epoch),
            "max_train_steps": int(spec.budget.max_train_steps),
            "status": "planned",
            "command": list(spec.command),
        }

    spec.experiments_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(list(spec.command), cwd=str(REPO_ROOT), check=True)
    run_dir = locate_trainer_run_dir(spec.experiments_dir, spec.expected_run_name)
    if run_dir is None:
        raise FileNotFoundError(
            f"Training completed but the ablation runner could not resolve a child run directory under {spec.experiments_dir}"
        )
    return {
        "arm": spec.arm,
        "learning_rate": float(spec.learning_rate),
        "seed": int(spec.seed),
        "data_dir": str(spec.data_dir),
        "experiments_dir": str(spec.experiments_dir),
        "expected_run_name": spec.expected_run_name,
        "run_dir": str(run_dir),
        "step_metrics_path": str(run_dir / "step_metrics.json"),
        "ewok_items_path": str(run_dir / "ewok_items.jsonl"),
        "steps_per_epoch": int(spec.budget.steps_per_epoch),
        "max_train_steps": int(spec.budget.max_train_steps),
        "status": "completed",
        "command": list(spec.command),
    }


def build_ablation_run_specs(
    *,
    base_ckpt: str | Path,
    matched_pool_dir: str | Path,
    output_dir: str | Path,
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    micro_batch_size: int,
    total_batch_tokens: int,
    num_epochs: int,
    num_processes: int,
    ewok_batch_size: int,
    num_workers: int = 0,
    warmup_iters: int | None = None,
) -> tuple[dict[str, Path], list[AblationRunSpec]]:
    output_root = Path(output_dir).expanduser().resolve()
    datasets = resolve_matched_pool_datasets(matched_pool_dir)
    views_root = output_root / "data_views"
    training_views = {
        arm: prepare_bos_row_training_view(dataset_dir, views_root / arm)
        for arm, dataset_dir in datasets.items()
    }

    model_config = load_checkpoint_model_config(base_ckpt)
    specs: list[AblationRunSpec] = []
    for arm, data_dir in training_views.items():
        meta = _load_json(data_dir / "meta.json")
        train_shards = _list_train_shards(data_dir)
        num_rows = _infer_num_rows(meta, train_shards)
        budget = compute_training_budget(
            seq_len=int(meta.get("seq_len", model_config.seq_len)),
            num_rows=num_rows,
            micro_batch_size=micro_batch_size,
            total_batch_tokens=total_batch_tokens,
            num_processes=num_processes,
            num_epochs=num_epochs,
        )
        if num_rows < int(micro_batch_size):
            raise ValueError(
                f"{arm} dataset under {data_dir} only has {num_rows} rows, which is smaller than micro_batch_size={micro_batch_size}"
            )
        effective_total_tokens = (
            budget.grad_accum_steps * budget.micro_batch_size * budget.seq_len * budget.num_processes
        )
        resolved_warmup_iters = (
            int(warmup_iters)
            if warmup_iters is not None
            else max(1, int(round(0.05 * budget.max_train_steps)))
        )
        for learning_rate in learning_rates:
            lr_tag = format_lr_tag(float(learning_rate))
            for seed in seeds:
                experiments_dir = output_root / "child_runs" / arm / f"lr_{lr_tag}" / f"seed_{int(seed)}" / "experiments"
                expected_run_name = build_expected_bos_run_name(
                    loader_kind="bos_row",
                    micro_batch_size=budget.micro_batch_size,
                    seq_len=model_config.seq_len,
                    n_embd=model_config.n_embd,
                    n_head=model_config.n_head,
                    n_layer=model_config.n_layer,
                    total_batch_tokens=budget.total_batch_tokens,
                    effective_total_tokens=effective_total_tokens,
                    num_processes=num_processes,
                    grad_accum_steps=budget.grad_accum_steps,
                    seed=int(seed),
                    max_train_steps=budget.max_train_steps,
                )
                command = build_bos_trainer_command(
                    base_ckpt=base_ckpt,
                    data_dir=data_dir,
                    experiments_dir=experiments_dir,
                    model_config=model_config,
                    budget=budget,
                    learning_rate=float(learning_rate),
                    warmup_iters=resolved_warmup_iters,
                    seed=int(seed),
                    ewok_every=budget.steps_per_epoch,
                    ewok_batch_size=ewok_batch_size,
                    num_processes=num_processes,
                    num_workers=num_workers,
                )
                specs.append(
                    AblationRunSpec(
                        arm=arm,
                        learning_rate=float(learning_rate),
                        seed=int(seed),
                        data_dir=data_dir,
                        experiments_dir=experiments_dir,
                        expected_run_name=expected_run_name,
                        expected_run_dir=experiments_dir / expected_run_name,
                        budget=budget,
                        warmup_iters=resolved_warmup_iters,
                        ewok_batch_size=int(ewok_batch_size),
                        command=tuple(command),
                    )
                )
    return training_views, specs


def run_ablation_aggregation(
    *,
    output_dir: str | Path,
    run_records: Sequence[dict[str, Any]],
    baseline_summary_path: str | Path,
    metric_name: str = DEFAULT_METRIC_NAME,
) -> dict[str, Path]:
    baseline_summary = _load_json(baseline_summary_path)
    curves: list[dict[str, Any]] = []
    for run_record in run_records:
        step_metrics_path = run_record.get("step_metrics_path")
        ewok_items_path = run_record.get("ewok_items_path")
        if not step_metrics_path or not ewok_items_path:
            continue
        if not Path(step_metrics_path).exists() or not Path(ewok_items_path).exists():
            continue
        curves.extend(build_curves_for_run(run_record=run_record, baseline_summary=baseline_summary, metric_name=metric_name))
    summary = build_ablation_summary(curves, metric_name=metric_name)
    return write_ablation_outputs(output_dir=output_dir, run_records=run_records, curves=curves, summary=summary)


def spec_to_record(spec: AblationRunSpec) -> dict[str, Any]:
    return {
        "arm": spec.arm,
        "learning_rate": float(spec.learning_rate),
        "seed": int(spec.seed),
        "data_dir": str(spec.data_dir),
        "experiments_dir": str(spec.experiments_dir),
        "expected_run_name": spec.expected_run_name,
        "expected_run_dir": str(spec.expected_run_dir),
        "budget": asdict(spec.budget),
        "warmup_iters": int(spec.warmup_iters),
        "ewok_batch_size": int(spec.ewok_batch_size),
        "command": list(spec.command),
    }


__all__ = [
    "AblationRunSpec",
    "DEFAULT_GROUP_BY",
    "DEFAULT_EWOK_BATCH_SIZE",
    "DEFAULT_LRS",
    "DEFAULT_METRIC_NAME",
    "DEFAULT_MICRO_BATCH_SIZE",
    "DEFAULT_NUM_EPOCHS",
    "DEFAULT_NUM_PROCESSES",
    "DEFAULT_REDUCTION",
    "DEFAULT_SEEDS",
    "DEFAULT_TOTAL_BATCH_TOKENS",
    "GROUPING_COLUMNS",
    "RECOMMENDED_LR_SWEEP",
    "SUPPORTED_GROUP_BYS",
    "SUPPORTED_REDUCTIONS",
    "build_ablation_run_specs",
    "build_ablation_summary",
    "compute_training_budget",
    "evaluate_checkpoint_baseline",
    "format_lr_tag",
    "launch_training_run",
    "load_deduped_ewok_items_frame",
    "parse_float_list",
    "parse_int_list",
    "prepare_bos_row_training_view",
    "resolve_matched_pool_datasets",
    "run_ablation_aggregation",
    "spec_to_record",
]
