"""Compatibility helpers for SDPA backend selection across PyTorch versions."""

from __future__ import annotations

from contextlib import contextmanager

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:
    from torch.backends.cuda import sdp_kernel as _legacy_sdp_kernel
    from torch.backends.cuda import SDPBackend

    @contextmanager
    def sdpa_kernel(backends):
        """Backport the newer list-based SDPA API to older torch releases."""
        if isinstance(backends, SDPBackend):
            backend_set = {backends}
        else:
            backend_set = set(backends)

        with _legacy_sdp_kernel(
            enable_flash=SDPBackend.FLASH_ATTENTION in backend_set,
            enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION in backend_set,
            enable_math=SDPBackend.MATH in backend_set,
        ):
            yield

__all__ = ["sdpa_kernel", "SDPBackend"]
