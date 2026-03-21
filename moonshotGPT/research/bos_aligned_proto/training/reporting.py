"""Small JSON and metrics helpers shared by the research trainer."""

from __future__ import annotations

import json
import os

import numpy as np
import torch


def to_jsonable(value):
    """Convert tensors / numpy / scalars inside dicts to JSON-safe Python types."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim > 0 else value.item()
    return value


def save_metrics(metrics_list, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics_list), handle, indent=2)
    os.replace(tmp_path, out_path)


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record)) + "\n")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str, payload) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)
    os.replace(tmp_path, path)
