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
    """Stable handle for one discovered checkpoint directory.

    The outer attribution pipeline only needs three pieces of identity:
    the integer training step, the filesystem path, and whether the folder
    came from a periodic or final save. Keeping that state in a tiny immutable
    dataclass makes it easy to sort, select, cache against, and serialize.
    """

    step: int
    path: Path
    kind: str


def discover_checkpoints(run_dir: str | Path) -> list[CheckpointRef]:
    """Scan a run directory and return one canonical checkpoint per step.

    Runs may contain both `ckpt_periodic_stepXXXXXXX` and
    `ckpt_final_stepXXXXXXX` folders for the same step. Attribution only wants
    one concrete checkpoint path per step, so this helper prefers the `final`
    directory when both exist and otherwise keeps the periodic one.
    """

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
    """Filter discovered checkpoints to an explicit ordered step subset.

    The CLI exposes `--checkpoint_steps` as raw integers. This helper keeps the
    user-specified order, validates that each requested step was discovered,
    and falls back to the full discovered list when no explicit subset is
    requested.
    """

    if not requested_steps:
        return checkpoints

    by_step = {ref.step: ref for ref in checkpoints}
    missing = [step for step in requested_steps if step not in by_step]
    if missing:
        raise ValueError(f"Requested checkpoint steps not found: {missing}")
    return [by_step[step] for step in requested_steps]


def align_state_dict_to_model(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module | None,
) -> dict[str, torch.Tensor]:
    """Fill in missing tied-weight aliases before a strict model load.

    This exists because some Hugging Face checkpoints, including the GPT-2
    style checkpoints we use in BOS attribution, save only one side of a tied
    parameter pair. A common example is:

    - `transformer.wte.weight` present in the checkpoint
    - `lm_head.weight` absent from the checkpoint

    even though the instantiated model exposes both keys in its state dict.
    Bergson and TRAK both reload checkpoints into an already-constructed model
    with `strict=True`, so without this normalization we fail on an apparently
    "missing" key even though the tensor is semantically shared.

    The implementation groups model parameters by data pointer, which lets us
    detect alias sets that share storage in the live model. If the checkpoint
    contains one alias from that set and another alias is missing, we clone the
    present tensor into the missing key so that strict loading succeeds.
    """

    if model is None:
        return state_dict

    model_state = model.state_dict()
    pointer_to_keys: dict[int, list[str]] = {}
    for key, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        pointer_to_keys.setdefault(int(value.data_ptr()), []).append(str(key))

    aligned = dict(state_dict)
    for tied_keys in pointer_to_keys.values():
        if len(tied_keys) <= 1:
            continue
        present_keys = [key for key in tied_keys if key in aligned]
        missing_keys = [key for key in tied_keys if key not in aligned]
        if not present_keys or not missing_keys:
            continue

        for missing_key in missing_keys:
            expected = model_state.get(missing_key)
            if not isinstance(expected, torch.Tensor):
                continue
            # We only synthesize the alias when the saved tensor already looks
            # like a shape/dtype match for the model slot we are about to fill.
            source_key = next(
                (
                    candidate
                    for candidate in present_keys
                    if tuple(aligned[candidate].shape) == tuple(expected.shape)
                    and aligned[candidate].dtype == expected.dtype
                ),
                None,
            )
            if source_key is None:
                continue
            aligned[missing_key] = aligned[source_key].clone()
    return aligned


def load_checkpoint_state_dict(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Load raw checkpoint weights from disk and normalize them for the model.

    `AutoModelForCausalLM.from_pretrained(...)` is convenient when we want to
    build a model from scratch, but the attribution backends reuse a live model
    instance and swap checkpoint weights into it repeatedly. That makes raw
    state-dict loading the common path here.

    If `model` is supplied, we immediately run `align_state_dict_to_model(...)`
    so strict loading can tolerate omitted tied-weight aliases such as the
    GPT-2 `lm_head.weight` case.
    """

    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"

    if safetensors_path.exists():
        state = load_safetensors_file(str(safetensors_path), device="cpu")
        return align_state_dict_to_model(state, model)
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint type at {bin_path}: {type(state)!r}")
        return align_state_dict_to_model(state, model)
    raise FileNotFoundError(
        f"No supported checkpoint weights found in {checkpoint_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def build_model_from_checkpoint(checkpoint_dir: str | Path, device: str = "cpu"):
    """Instantiate the causal LM that matches a checkpoint directory."""

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()
    return model


def load_tokenizer_from_checkpoint(checkpoint_dir: str | Path):
    """Load the tokenizer stored alongside a checkpoint directory."""

    return AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)


__all__ = [
    "CheckpointRef",
    "align_state_dict_to_model",
    "build_model_from_checkpoint",
    "discover_checkpoints",
    "load_checkpoint_state_dict",
    "load_tokenizer_from_checkpoint",
    "select_checkpoints",
]
