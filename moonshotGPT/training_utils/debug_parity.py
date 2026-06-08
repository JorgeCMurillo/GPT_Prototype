"""Shared debug helpers for training-parity and tiny-overfit runs.

These utilities are intentionally training-loop friendly:
- cache a small fixed set of local batches from an existing dataloader
- replay those cached batches forever for tiny-overfit checks
- write per-microstep JSONL traces with optimizer-step boundary metadata
- summarize a run and optionally diff it against another run's trace
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


FLOAT_DIFF_ATOL = 1e-7
FLOAT_DIFF_RTOL = 1e-5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def append_jsonl(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path_obj)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path_obj)


def _clone_debug_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {k: _clone_debug_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_debug_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_debug_value(v) for v in value)
    return value


def clone_batch_to_cpu(batch: Any) -> Any:
    """Deep-clone a dataloader batch so replayed overfit batches stay immutable."""
    return _clone_debug_value(batch)


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _tensor_sha1(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha1()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def split_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1], None
    raise TypeError(f"Unsupported batch structure for debug replay: {type(batch)!r}")


def summarize_batch(batch: Any) -> dict[str, Any]:
    input_ids, labels, meta = split_batch(batch)
    label_ignore_count = int((labels == -100).sum().item()) if torch.is_tensor(labels) else 0
    summary = {
        "input_shape": list(input_ids.shape),
        "label_shape": list(labels.shape),
        "tokens_per_batch": int(input_ids.numel()),
        "label_ignore_count": label_ignore_count,
        "input_ids_sha1": _tensor_sha1(input_ids),
        "labels_sha1": _tensor_sha1(labels),
        "meta_present": meta is not None,
        "meta_sha1": (_sha1_bytes(_stable_json_bytes(meta)) if meta is not None else None),
    }
    return summary


def cache_debug_batches(train_loader: Iterable[Any], num_batches: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")
    iterator = iter(train_loader)
    cached_batches: list[Any] = []
    batch_summaries: list[dict[str, Any]] = []
    for batch_idx in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Requested {num_batches} cached batches, but the train loader yielded only {batch_idx}."
            ) from exc
        cloned = clone_batch_to_cpu(batch)
        cached_batches.append(cloned)
        batch_summary = summarize_batch(cloned)
        batch_summary["cached_batch_index"] = int(batch_idx)
        batch_summaries.append(batch_summary)
    return cached_batches, batch_summaries


def replay_cached_batches(cached_batches: Sequence[Any]) -> Iterator[Any]:
    if not cached_batches:
        raise ValueError("cached_batches must be non-empty")
    while True:
        for batch in cached_batches:
            yield clone_batch_to_cpu(batch)


def replay_cached_batches_with_index(cached_batches: Sequence[Any]) -> Iterator[tuple[int, Any]]:
    if not cached_batches:
        raise ValueError("cached_batches must be non-empty")
    while True:
        for batch_idx, batch in enumerate(cached_batches):
            yield int(batch_idx), clone_batch_to_cpu(batch)


def yield_batches_with_index(train_loader: Iterable[Any]) -> Iterator[tuple[int | None, Any]]:
    for batch in train_loader:
        yield None, batch


@dataclass(frozen=True)
class DebugParityConfig:
    enabled: bool = False
    overfit_batches: int = 0
    output_dir: str = ""
    compare_to: str = ""
    compute_update_norm: bool = False
    disable_fused_adamw: bool = False

    @property
    def mode(self) -> str:
        return "tiny_overfit" if int(self.overfit_batches) > 0 else "short_parity"


@dataclass(frozen=True)
class DebugParityPaths:
    root_dir: str
    jsonl_path: str
    manifest_path: str
    summary_json_path: str
    summary_txt_path: str
    diff_json_path: str
    diff_txt_path: str


def resolve_debug_paths(
    *,
    out_dir: str,
    output_dir_override: str,
    rank: int,
) -> DebugParityPaths:
    root_dir = os.path.abspath(output_dir_override) if output_dir_override else os.path.join(out_dir, "debug_training")
    return DebugParityPaths(
        root_dir=root_dir,
        jsonl_path=os.path.join(root_dir, f"train_debug_rank{int(rank):04d}.jsonl"),
        manifest_path=os.path.join(root_dir, "manifest.json"),
        summary_json_path=os.path.join(root_dir, "summary_rank0000.json"),
        summary_txt_path=os.path.join(root_dir, "summary_rank0000.txt"),
        diff_json_path=os.path.join(root_dir, "diff_report.json"),
        diff_txt_path=os.path.join(root_dir, "diff_report.txt"),
    )


def write_debug_manifest(
    *,
    paths: DebugParityPaths,
    config: DebugParityConfig,
    manifest: dict[str, Any],
) -> None:
    payload = {
        "created_at": _utc_now_iso(),
        "mode": config.mode,
        "config": asdict(config),
        **manifest,
    }
    atomic_write_json(paths.manifest_path, payload)


@dataclass(frozen=True)
class StepDebugRecord:
    event: str
    timestamp: str
    script_name: str
    mode: str
    rank: int
    world_size: int
    loader_kind: str
    optimizer_step_before: int
    optimizer_step_after: int
    optimizer_step_target: int
    optimizer_step_occurred: bool
    scheduler_step_occurred: bool
    lr_schedule_applied: bool
    global_micro_step: int
    micro_step_in_optimizer_step: int
    grad_accum_steps: int
    grad_accum_count: int
    train_loss_raw: float
    train_loss_opt_step_running_mean: float
    lr_before_step: float | None
    lr_after_step: float | None
    lr_current: float
    grad_norm_l2_preclip: float | None
    grad_norm_l2_postclip: float | None
    param_norm_l2: float | None
    update_norm_l2: float | None
    optimizer_use_fused: bool
    tokens_this_microstep: int
    tokens_seen_local_total: int
    tokens_seen_global_approx: int
    tokens_per_opt_step_global: int
    fixed_batch_mode: bool
    fixed_batch_index: int | None
    cached_batch_count: int
    input_shape: list[int]
    label_shape: list[int]
    label_ignore_count: int
    input_ids_sha1: str
    labels_sha1: str
    meta_present: bool
    meta_sha1: str | None


def write_step_debug_record(path: str, record: StepDebugRecord) -> None:
    append_jsonl(path, asdict(record))


def maybe_snapshot_parameters(model: torch.nn.Module, enabled: bool) -> list[torch.Tensor] | None:
    if not enabled:
        return None
    snapshot: list[torch.Tensor] = []
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            snapshot.append(param.detach().clone())
    return snapshot


def compute_update_norm_l2(model: torch.nn.Module, snapshot: Sequence[torch.Tensor] | None) -> float | None:
    if snapshot is None:
        return None
    total_sq = 0.0
    snap_idx = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            before = snapshot[snap_idx]
            snap_idx += 1
            diff = param.detach().float() - before.detach().float()
            total_sq += float(diff.pow(2).sum().item())
    return float(math.sqrt(total_sq))


def _load_jsonl_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Debug parity log not found: {path}")
    records: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _microstep_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == "train_microstep"]


def _float_differs(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left != right
    if math.isnan(left) or math.isnan(right):
        return not (math.isnan(left) and math.isnan(right))
    return not math.isclose(float(left), float(right), rel_tol=FLOAT_DIFF_RTOL, abs_tol=FLOAT_DIFF_ATOL)


def summarize_debug_run(jsonl_path: str) -> dict[str, Any]:
    records = _microstep_records(_load_jsonl_records(jsonl_path))
    if not records:
        raise RuntimeError(f"No train_microstep records found in {jsonl_path}")

    optimizer_step_records = [record for record in records if bool(record.get("optimizer_step_occurred", False))]
    first_loss = float(records[0]["train_loss_raw"])
    last_loss = float(records[-1]["train_loss_raw"])
    min_loss = min(float(record["train_loss_raw"]) for record in records)
    max_loss = max(float(record["train_loss_raw"]) for record in records)

    cadence_ok = all(
        bool(record.get("optimizer_step_occurred", False))
        == (int(record["micro_step_in_optimizer_step"]) == int(record["grad_accum_steps"]))
        for record in records
    )
    lr_boundary_only = all(
        bool(record.get("lr_schedule_applied", False)) == bool(record.get("optimizer_step_occurred", False))
        for record in records
    )
    scheduler_never_called = not any(bool(record.get("scheduler_step_occurred", False)) for record in records)
    fixed_batch_mode = any(bool(record.get("fixed_batch_mode", False)) for record in records)
    cached_batch_count = max(int(record.get("cached_batch_count", 0)) for record in records)
    fused_flags = sorted({bool(record.get("optimizer_use_fused", False)) for record in records})

    summary = {
        "created_at": _utc_now_iso(),
        "jsonl_path": os.path.abspath(jsonl_path),
        "script_name": records[0].get("script_name"),
        "mode": records[0].get("mode"),
        "loader_kind": records[0].get("loader_kind"),
        "rank": int(records[0].get("rank", 0)),
        "world_size": int(records[0].get("world_size", 1)),
        "num_microsteps": len(records),
        "num_optimizer_steps": len(optimizer_step_records),
        "grad_accum_steps": int(records[0]["grad_accum_steps"]),
        "first_train_loss_raw": first_loss,
        "last_train_loss_raw": last_loss,
        "min_train_loss_raw": min_loss,
        "max_train_loss_raw": max_loss,
        "loss_delta_raw": last_loss - first_loss,
        "optimizer_step_loss_first": (
            float(optimizer_step_records[0]["train_loss_opt_step_running_mean"]) if optimizer_step_records else None
        ),
        "optimizer_step_loss_last": (
            float(optimizer_step_records[-1]["train_loss_opt_step_running_mean"]) if optimizer_step_records else None
        ),
        "tokens_seen_local_total_last": int(records[-1]["tokens_seen_local_total"]),
        "tokens_seen_global_approx_last": int(records[-1]["tokens_seen_global_approx"]),
        "tokens_per_opt_step_global": int(records[-1]["tokens_per_opt_step_global"]),
        "optimizer_step_cadence_matches_grad_accum": cadence_ok,
        "lr_schedule_applied_only_on_optimizer_steps": lr_boundary_only,
        "scheduler_step_called_anywhere": not scheduler_never_called,
        "fixed_batch_mode": fixed_batch_mode,
        "cached_batch_count": cached_batch_count,
        "optimizer_use_fused_values": fused_flags,
        "first_input_ids_sha1": records[0]["input_ids_sha1"],
        "first_labels_sha1": records[0]["labels_sha1"],
    }
    return summary


def summary_to_text(summary: dict[str, Any]) -> str:
    lines = [
        f"script: {summary.get('script_name')}",
        f"mode: {summary.get('mode')}",
        f"loader_kind: {summary.get('loader_kind')}",
        f"rank/world_size: {summary.get('rank')}/{summary.get('world_size')}",
        f"microsteps: {summary.get('num_microsteps')}",
        f"optimizer_steps: {summary.get('num_optimizer_steps')}",
        f"grad_accum_steps: {summary.get('grad_accum_steps')}",
        f"loss raw: {summary.get('first_train_loss_raw'):.6f} -> {summary.get('last_train_loss_raw'):.6f}"
        f" (delta {summary.get('loss_delta_raw'):+.6f})",
        f"optimizer-step mean loss: {summary.get('optimizer_step_loss_first')} -> {summary.get('optimizer_step_loss_last')}",
        f"tokens_seen_global_approx_last: {summary.get('tokens_seen_global_approx_last')}",
        f"optimizer_step cadence matches grad_accum: {summary.get('optimizer_step_cadence_matches_grad_accum')}",
        f"lr schedule only at optimizer boundary: {summary.get('lr_schedule_applied_only_on_optimizer_steps')}",
        f"scheduler.step() called anywhere: {summary.get('scheduler_step_called_anywhere')}",
        f"fixed_batch_mode: {summary.get('fixed_batch_mode')} (cached_batch_count={summary.get('cached_batch_count')})",
        f"optimizer_use_fused values: {summary.get('optimizer_use_fused_values')}",
    ]
    return "\n".join(lines) + "\n"


def compare_debug_runs(current_jsonl_path: str, reference_jsonl_path: str) -> dict[str, Any]:
    current_records = _microstep_records(_load_jsonl_records(current_jsonl_path))
    reference_records = _microstep_records(_load_jsonl_records(reference_jsonl_path))

    exact_fields = (
        "optimizer_step_occurred",
        "scheduler_step_occurred",
        "lr_schedule_applied",
        "optimizer_step_before",
        "optimizer_step_after",
        "optimizer_step_target",
        "global_micro_step",
        "micro_step_in_optimizer_step",
        "grad_accum_steps",
        "grad_accum_count",
        "tokens_this_microstep",
        "tokens_seen_local_total",
        "tokens_seen_global_approx",
        "tokens_per_opt_step_global",
        "fixed_batch_mode",
        "fixed_batch_index",
        "cached_batch_count",
        "input_shape",
        "label_shape",
        "label_ignore_count",
        "input_ids_sha1",
        "labels_sha1",
        "optimizer_use_fused",
    )
    float_fields = (
        "train_loss_raw",
        "train_loss_opt_step_running_mean",
        "lr_before_step",
        "lr_after_step",
        "lr_current",
        "grad_norm_l2_preclip",
        "grad_norm_l2_postclip",
        "param_norm_l2",
        "update_norm_l2",
    )

    limit = min(len(current_records), len(reference_records))
    first_divergence: dict[str, Any] | None = None
    for record_idx in range(limit):
        current = current_records[record_idx]
        reference = reference_records[record_idx]
        mismatches: list[dict[str, Any]] = []
        for field in exact_fields:
            if current.get(field) != reference.get(field):
                mismatches.append(
                    {
                        "field": field,
                        "kind": "exact",
                        "current": current.get(field),
                        "reference": reference.get(field),
                    }
                )
        for field in float_fields:
            if _float_differs(current.get(field), reference.get(field)):
                mismatches.append(
                    {
                        "field": field,
                        "kind": "float",
                        "current": current.get(field),
                        "reference": reference.get(field),
                        "delta": (
                            None
                            if current.get(field) is None or reference.get(field) is None
                            else float(current.get(field) - reference.get(field))
                        ),
                    }
                )
        if mismatches:
            first_divergence = {
                "record_index": int(record_idx),
                "optimizer_step_target": int(current.get("optimizer_step_target", 0)),
                "micro_step_in_optimizer_step": int(current.get("micro_step_in_optimizer_step", 0)),
                "current_record": current,
                "reference_record": reference,
                "mismatches": mismatches,
            }
            break

    status = "identical"
    if first_divergence is not None:
        status = "diverged"
    elif len(current_records) != len(reference_records):
        status = "different_length"

    result = {
        "created_at": _utc_now_iso(),
        "status": status,
        "current_jsonl_path": os.path.abspath(current_jsonl_path),
        "reference_jsonl_path": os.path.abspath(reference_jsonl_path),
        "current_num_microsteps": len(current_records),
        "reference_num_microsteps": len(reference_records),
        "first_divergence": first_divergence,
    }
    if status == "different_length":
        trailing_record = current_records[limit] if len(current_records) > limit else reference_records[limit]
        result["first_divergence"] = {
            "record_index": int(limit),
            "optimizer_step_target": int(trailing_record.get("optimizer_step_target", 0)),
            "micro_step_in_optimizer_step": int(trailing_record.get("micro_step_in_optimizer_step", 0)),
            "reason": "different_length",
        }
    return result


def diff_to_text(diff: dict[str, Any]) -> str:
    lines = [
        f"status: {diff.get('status')}",
        f"current_jsonl_path: {diff.get('current_jsonl_path')}",
        f"reference_jsonl_path: {diff.get('reference_jsonl_path')}",
        f"current_num_microsteps: {diff.get('current_num_microsteps')}",
        f"reference_num_microsteps: {diff.get('reference_num_microsteps')}",
    ]
    first_divergence = diff.get("first_divergence")
    if first_divergence:
        lines.append(
            "first_divergence: "
            f"record_index={first_divergence.get('record_index')}, "
            f"optimizer_step_target={first_divergence.get('optimizer_step_target')}, "
            f"micro_step_in_optimizer_step={first_divergence.get('micro_step_in_optimizer_step')}"
        )
        mismatches = first_divergence.get("mismatches", [])
        for mismatch in mismatches[:8]:
            lines.append(
                f"  {mismatch.get('field')}: current={mismatch.get('current')} "
                f"reference={mismatch.get('reference')}"
            )
        if first_divergence.get("reason"):
            lines.append(f"  reason: {first_divergence.get('reason')}")
    return "\n".join(lines) + "\n"


def finalize_debug_outputs(
    *,
    paths: DebugParityPaths,
    compare_to: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summary = summarize_debug_run(paths.jsonl_path)
    atomic_write_json(paths.summary_json_path, summary)
    atomic_write_text(paths.summary_txt_path, summary_to_text(summary))

    diff = None
    if compare_to:
        diff = compare_debug_runs(paths.jsonl_path, compare_to)
        atomic_write_json(paths.diff_json_path, diff)
        atomic_write_text(paths.diff_txt_path, diff_to_text(diff))
    return summary, diff
