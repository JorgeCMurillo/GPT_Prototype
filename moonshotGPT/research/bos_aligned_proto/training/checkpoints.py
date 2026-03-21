"""Checkpoint and resume helpers for the research trainer."""

from __future__ import annotations

import os
import re

try:
    from research.bos_aligned_proto.training.reporting import atomic_write_json, load_json
except ImportError:
    from reporting import atomic_write_json, load_json


def checkpoint_step_from_dirname(path: str):
    name = os.path.basename(os.path.normpath(path))
    match = re.match(r"^ckpt_[^/]*_step(\d+)$", name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def resolve_resume_paths(resume_from_run: str):
    raw = os.path.abspath(str(resume_from_run))
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"--resume_from_run path does not exist or is not a directory: {raw}")

    step_direct = checkpoint_step_from_dirname(raw)
    if step_direct is not None:
        run_dir = os.path.dirname(raw)
        return run_dir, raw, int(step_direct)

    candidates = []
    for name in sorted(os.listdir(raw)):
        ckpt_dir = os.path.join(raw, name)
        if not os.path.isdir(ckpt_dir):
            continue
        step = checkpoint_step_from_dirname(ckpt_dir)
        if step is None:
            continue
        candidates.append((int(step), float(os.path.getmtime(ckpt_dir)), ckpt_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directories found under '{raw}'. "
            "Expected folders named like ckpt_<tag>_step0001234."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _mtime, ckpt_dir = candidates[-1]
    return raw, ckpt_dir, int(step)


def validate_ckpt_model_config_alignment(
    ckpt_dir: str,
    seq_len: int,
    vocab_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
) -> None:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Checkpoint missing config.json: {cfg_path}")

    try:
        cfg = load_json(cfg_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint config: {cfg_path}: {exc}") from exc

    mismatches = []

    def check(keys, expected: int, label: str) -> None:
        got = None
        for key in keys:
            if key in cfg:
                got = int(cfg[key])
                break
        if got is None:
            return
        if int(got) != int(expected):
            mismatches.append(f"{label}: ckpt={got}, requested={int(expected)}")

    check(["vocab_size"], vocab_size, "vocab_size")
    check(["n_embd"], n_embd, "n_embd")
    check(["n_head"], n_head, "n_head")
    check(["n_layer"], n_layer, "n_layer")
    check(["n_positions", "n_ctx"], seq_len, "seq_len")

    if mismatches:
        raise ValueError(
            "Checkpoint architecture does not match requested training args.\n"
            f"checkpoint: {ckpt_dir}\n"
            + "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        )


def load_existing_step_metrics(metrics_path: str):
    if not os.path.exists(metrics_path):
        return []
    try:
        data = load_json(metrics_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse existing step_metrics file '{metrics_path}': {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected list in step_metrics file: {metrics_path}")
    return data


def load_trainer_state(ckpt_dir: str):
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


def save_trainer_state(ckpt_dir: str, state: dict) -> None:
    atomic_write_json(os.path.join(ckpt_dir, "trainer_state.json"), state)
