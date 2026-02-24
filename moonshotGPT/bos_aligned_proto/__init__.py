"""BOS-aligned prototype pipeline for moonshotGPT."""

from .bos_row_loader import make_bos_row_dataloader

__all__ = ["make_bos_row_dataloader"]
