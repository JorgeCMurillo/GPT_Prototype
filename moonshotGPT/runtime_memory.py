"""Helpers for reporting process RAM and CUDA memory in training progress bars."""

from __future__ import annotations

import os

import torch


_GIB = float(1024 ** 3)


def _format_gib(num_bytes: int) -> str:
    return f"{float(num_bytes) / _GIB:.1f}GiB"


def get_process_rss_bytes() -> int | None:
    """Return current process RSS in bytes when available."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.readline().split()
        if len(parts) >= 2:
            resident_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return resident_pages * page_size
    except Exception:
        return None
    return None


def reset_peak_memory_stats(device: torch.device | None) -> None:
    """Reset CUDA peak memory tracking when the active device supports it."""
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def format_memory_usage_postfix(device: torch.device | None) -> str:
    """Return a short tqdm-friendly memory usage string."""
    parts: list[str] = []

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            parts.append(f"vram={_format_gib(allocated)}/{_format_gib(reserved)}")
            parts.append(f"peak={_format_gib(peak_reserved)}")
        except Exception:
            pass

    rss_bytes = get_process_rss_bytes()
    if rss_bytes is not None:
        parts.append(f"ram={_format_gib(rss_bytes)}")

    return " ".join(parts)
