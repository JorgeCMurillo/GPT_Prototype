"""Pipeline code for BOS-aligned dataset preparation and loading."""

from .bos_row_loader import make_bos_row_dataloader

__all__ = ["make_bos_row_dataloader"]
