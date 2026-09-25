"""Sequential Hugging Face EWoK evaluation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DOWNLOADS_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "ModelSpec",
    "build_load_attempts",
    "detect_hardware",
    "load_queue_config",
    "run_model",
    "run_queue_models",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".run_queue", __name__)
    if name == "run_queue_models":
        return getattr(module, "run_queue")
    return getattr(module, name)
