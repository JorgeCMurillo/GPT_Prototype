"""Model/tokenizer loading for EWoK linear probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_ref: str
    tokenizer_source: str
    device: torch.device
    dtype: object


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device


def resolve_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_arg]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_arg!r}") from exc


def load_model_and_tokenizer(
    model_ref: str,
    *,
    device_arg: str = "auto",
    dtype_arg: str = "auto",
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> LoadedModel:
    device = resolve_device(device_arg)
    dtype = resolve_dtype(dtype_arg, device)

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only or Path(model_ref).expanduser().exists(),
        use_fast=True,
    )
    tokenizer_source = str(model_ref)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if getattr(model, "get_input_embeddings", None) is not None:
        embed = model.get_input_embeddings()
        if embed is not None and embed.num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_ref=str(model_ref),
        tokenizer_source=tokenizer_source,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "LoadedModel",
    "load_model_and_tokenizer",
    "resolve_device",
    "resolve_dtype",
]
