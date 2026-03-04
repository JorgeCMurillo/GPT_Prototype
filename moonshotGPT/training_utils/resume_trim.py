"""Resume-safe log trimming and data-stream replay helpers.

This module serves two related resume preflight needs:
1) Trim run logs (JSON/JSONL) so records above checkpoint step are removed.
2) Rebuild dataloader replay state by fast-forwarding an iterator data-only.

Design notes:
- Trimming is backup-first (single backup file per target path).
- Rewrites are atomic (write temp file, then os.replace).
- JSONL parsing is permissive: invalid lines are preserved, never dropped.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from typing import Any, Dict, Iterator, List, Tuple


def _parse_step(value: Any):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _atomic_write_text(path: str, text: str):
    # Atomic rewrite flow: write side file first, then replace destination.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_json(path: str, payload: Any):
    # Atomic rewrite flow: write side file first, then replace destination.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _maybe_backup(path: str, backup_suffix: str):
    # Backup semantics: create once and keep stable across repeated trims.
    # We never overwrite an existing backup to avoid losing the original state.
    backup_path = path + backup_suffix
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
    return backup_path


def trim_records_by_step(records: List[Any], max_step: int, step_key: str = "step") -> Tuple[List[Any], int]:
    """Return filtered records where dict items with step>max_step are removed."""
    kept = []
    removed = 0
    max_step = int(max_step)
    for rec in records:
        if not isinstance(rec, dict):
            kept.append(rec)
            continue
        step = _parse_step(rec.get(step_key))
        if step is not None and step > max_step:
            removed += 1
            continue
        kept.append(rec)
    return kept, removed


def trim_json_file_by_step(
    path: str,
    max_step: int,
    step_key: str = "step",
    create_backup: bool = True,
    backup_suffix: str = ".resume_pretrim.bak",
) -> Dict[str, Any]:
    """Trim a JSON list file in place, preserving items with step<=max_step."""
    out = {
        "path": path,
        "exists": os.path.exists(path),
        "type": "json",
        "removed": 0,
        "kept": 0,
        "rewritten": False,
        "backup_path": None,
    }
    if not out["exists"]:
        return out

    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(payload)}")

    kept, removed = trim_records_by_step(payload, max_step=max_step, step_key=step_key)
    out["removed"] = int(removed)
    out["kept"] = int(len(kept))
    if removed > 0:
        if create_backup:
            out["backup_path"] = _maybe_backup(path, backup_suffix=backup_suffix)
        # Atomic rewrite prevents partially-written files if interrupted.
        _atomic_write_json(path, kept)
        out["rewritten"] = True
    return out


def trim_jsonl_file_by_step(
    path: str,
    max_step: int,
    step_key: str = "step",
    create_backup: bool = True,
    backup_suffix: str = ".resume_pretrim.bak",
) -> Dict[str, Any]:
    """Trim a JSONL file by step while preserving invalid/non-JSON lines."""
    out = {
        "path": path,
        "exists": os.path.exists(path),
        "type": "jsonl",
        "removed": 0,
        "kept": 0,
        "invalid_lines_kept": 0,
        "rewritten": False,
        "backup_path": None,
    }
    if not out["exists"]:
        return out

    with open(path, "r") as f:
        raw_lines = f.readlines()

    kept_lines = []
    removed = 0
    invalid = 0
    max_step = int(max_step)

    for line in raw_lines:
        candidate = line.rstrip("\n")
        if candidate.strip() == "":
            kept_lines.append(line)
            continue
        try:
            rec = json.loads(candidate)
        except Exception:
            # Keep malformed lines verbatim so the log remains audit-complete.
            kept_lines.append(line)
            invalid += 1
            continue
        if isinstance(rec, dict):
            step = _parse_step(rec.get(step_key))
            if step is not None and step > max_step:
                removed += 1
                continue
        kept_lines.append(line if line.endswith("\n") else (line + "\n"))

    out["removed"] = int(removed)
    out["kept"] = int(len(kept_lines))
    out["invalid_lines_kept"] = int(invalid)
    if removed > 0:
        if create_backup:
            out["backup_path"] = _maybe_backup(path, backup_suffix=backup_suffix)
        # Atomic rewrite prevents partially-written files if interrupted.
        _atomic_write_text(path, "".join(kept_lines))
        out["rewritten"] = True
    return out


def trim_run_logs_after_step(
    run_dir: str,
    max_step: int,
    include_exposure_logs: bool = False,
    create_backup: bool = True,
    backup_suffix: str = ".resume_pretrim.bak",
) -> Dict[str, Any]:
    """Trim run-level metrics logs to a checkpoint step.

    Files handled:
    - step_metrics.json (list JSON)
    - scalars.jsonl
    - hellaswag_metrics.jsonl
    - ewok_items.jsonl
    - exposures/exposures_rank*.jsonl (optional)
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    results = {}
    total_removed = 0

    step_metrics_path = os.path.join(run_dir, "step_metrics.json")
    results["step_metrics"] = trim_json_file_by_step(
        step_metrics_path,
        max_step=max_step,
        step_key="step",
        create_backup=create_backup,
        backup_suffix=backup_suffix,
    )
    total_removed += int(results["step_metrics"]["removed"])

    jsonl_names = [
        "scalars.jsonl",
        "hellaswag_metrics.jsonl",
        "ewok_items.jsonl",
    ]
    for name in jsonl_names:
        p = os.path.join(run_dir, name)
        res = trim_jsonl_file_by_step(
            p,
            max_step=max_step,
            step_key="step",
            create_backup=create_backup,
            backup_suffix=backup_suffix,
        )
        results[name] = res
        total_removed += int(res["removed"])

    if include_exposure_logs:
        exposure_pattern = os.path.join(run_dir, "exposures", "exposures_rank*.jsonl")
        exposure_paths = sorted(glob.glob(exposure_pattern))
        exposure_results = []
        for p in exposure_paths:
            res = trim_jsonl_file_by_step(
                p,
                max_step=max_step,
                step_key="step",
                create_backup=create_backup,
                backup_suffix=backup_suffix,
            )
            exposure_results.append(res)
            total_removed += int(res["removed"])
        results["exposures"] = exposure_results

    return {
        "run_dir": run_dir,
        "max_step": int(max_step),
        "total_removed": int(total_removed),
        "files": results,
    }


def derive_resume_skip_microsteps(
    trainer_state: Dict[str, Any],
    resume_opt_step: int,
    grad_accum_steps: int,
) -> int:
    """Derive number of micro-steps to consume before resume.

    Preference order:
    1) `trainer_state["micro_steps_seen"]` when available
    2) `resume_opt_step * grad_accum_steps` fallback for old checkpoints
    """
    if int(grad_accum_steps) <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")

    explicit = trainer_state.get("micro_steps_seen")
    if explicit is not None:
        try:
            explicit_i = int(explicit)
        except Exception as exc:
            raise ValueError(
                f"Invalid trainer_state['micro_steps_seen']={explicit!r}; expected integer."
            ) from exc
        if explicit_i < 0:
            raise ValueError(f"micro_steps_seen must be >= 0, got {explicit_i}")
        return explicit_i

    fallback = int(resume_opt_step) * int(grad_accum_steps)
    if fallback < 0:
        raise ValueError(
            f"Derived negative fallback micro-step count: "
            f"opt_step={resume_opt_step}, grad_accum_steps={grad_accum_steps}"
        )
    return fallback


def validate_replay_compatibility(
    current_cfg: Dict[str, Any],
    trainer_state: Dict[str, Any],
    strict: bool = True,
    fail_on_missing: bool = False,
) -> Dict[str, Any]:
    """Validate whether runtime settings can safely replay the checkpoint stream.

    Comparison is key-by-key from `current_cfg`. Missing checkpoint keys are
    expected for older checkpoints; they are warnings by default.
    """

    def _norm(v: Any):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return int(v) if float(v).is_integer() else float(v)
        if v is None:
            return None
        return str(v)

    mismatches: List[str] = []
    missing: List[str] = []
    checked: List[str] = []

    for key, expected in current_cfg.items():
        checked.append(key)
        if key not in trainer_state:
            missing.append(key)
            continue
        got = trainer_state.get(key)
        if _norm(got) != _norm(expected):
            mismatches.append(f"{key}: checkpoint={got!r}, current={expected!r}")

    if strict and mismatches:
        raise ValueError(
            "Replay compatibility check failed due to mismatched fields:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )
    if fail_on_missing and missing:
        raise ValueError(
            "Replay compatibility check failed: checkpoint missing fields:\n"
            + "\n".join(f"  - {k}" for k in missing)
        )

    warnings = []
    if missing:
        warnings.append(
            "Checkpoint trainer_state is missing replay fields (likely older checkpoint): "
            + ", ".join(sorted(missing))
        )

    return {
        "ok": (len(mismatches) == 0) and (not fail_on_missing or len(missing) == 0),
        "strict": bool(strict),
        "fail_on_missing": bool(fail_on_missing),
        "checked_keys": checked,
        "mismatches": mismatches,
        "missing": missing,
        "warnings": warnings,
    }


def fast_forward_iterator_data_only(
    iterator: Iterator[Any],
    skip_count: int,
    report_every_s: float = 5.0,
    is_main: bool = False,
) -> Dict[str, Any]:
    """Consume `skip_count` entries from an iterator without model compute.

    Intended use: dataloader replay during resume so stream position matches
    checkpointed progress before training loop starts.
    """
    skip_count = int(skip_count)
    if skip_count < 0:
        raise ValueError(f"skip_count must be >= 0, got {skip_count}")

    t0 = time.perf_counter()
    t_last_report = t0
    skipped = 0
    exhausted = False

    if is_main and skip_count > 0:
        print(f"[resume] fast-forwarding dataloader by {skip_count} micro-steps (data-only).")

    while skipped < skip_count:
        try:
            next(iterator)
            skipped += 1
        except StopIteration:
            exhausted = True
            break

        now = time.perf_counter()
        if (
            is_main
            and report_every_s > 0
            and (now - t_last_report) >= float(report_every_s)
        ):
            rate = skipped / max(now - t0, 1e-9)
            print(
                f"[resume] fast-forward progress: {skipped}/{skip_count} "
                f"micro-steps ({rate:,.1f} it/s)"
            )
            t_last_report = now

    elapsed_s = time.perf_counter() - t0
    stats = {
        "requested_skip_count": int(skip_count),
        "actual_skipped": int(skipped),
        "exhausted": bool(exhausted),
        "elapsed_s": float(elapsed_s),
        "iters_per_s": float(skipped / max(elapsed_s, 1e-9)),
    }
    if is_main and skip_count > 0:
        print(
            f"[resume] fast-forward complete: skipped={stats['actual_skipped']}/"
            f"{stats['requested_skip_count']} in {stats['elapsed_s']:.2f}s."
        )
    return stats
