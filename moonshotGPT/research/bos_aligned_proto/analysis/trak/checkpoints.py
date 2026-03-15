"""Checkpoint discovery and loading helpers for BOS research runs.

This module normalizes the run directories produced by the BOS-aligned
prototype into a consistent set of checkpoint references. The rest of the TRAK
package uses these helpers to discover periodic or final checkpoints and to
load the matching model, tokenizer, or raw state dict when attribution starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer


_CHECKPOINT_RE = re.compile(r"^ckpt_(?P<kind>periodic|final)_step(?P<step>\d+)$")


@dataclass(frozen=True)
class CheckpointRef:
    step: int
    path: Path
    kind: str


def discover_checkpoints(run_dir: str | Path) -> list[CheckpointRef]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    by_step: dict[int, CheckpointRef] = {}
    for child in sorted(run_path.iterdir()):
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match is None:
            continue
        ref = CheckpointRef(
            step=int(match.group("step")),
            path=child,
            kind=str(match.group("kind")),
        )
        current = by_step.get(ref.step)
        if current is None or (current.kind != "final" and ref.kind == "final"):
            by_step[ref.step] = ref
    return [by_step[step] for step in sorted(by_step)]


def select_checkpoints(
    checkpoints: list[CheckpointRef],
    requested_steps: tuple[int, ...] = (),
) -> list[CheckpointRef]:
    if not requested_steps:
        return checkpoints

    by_step = {ref.step: ref for ref in checkpoints}
    missing = [step for step in requested_steps if step not in by_step]
    if missing:
        raise ValueError(f"Requested checkpoint steps not found: {missing}")
    return [by_step[step] for step in requested_steps]


def load_checkpoint_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"

    if safetensors_path.exists():
        return load_safetensors_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint type at {bin_path}: {type(state)!r}")
        return state
    raise FileNotFoundError(
        f"No supported checkpoint weights found in {checkpoint_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def build_model_from_checkpoint(checkpoint_dir: str | Path, device: str = "cpu"):
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()
    return model


def load_tokenizer_from_checkpoint(checkpoint_dir: str | Path):
    return AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)


__all__ = [
    "CheckpointRef",
    "build_model_from_checkpoint",
    "discover_checkpoints",
    "load_checkpoint_state_dict",
    "load_tokenizer_from_checkpoint",
    "select_checkpoints",
]
