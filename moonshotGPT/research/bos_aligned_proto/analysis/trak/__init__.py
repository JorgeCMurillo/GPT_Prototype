"""Public entrypoints for the BOS-row TRAK analysis package.

This module keeps the external surface of the package intentionally small. It
re-exports the validated configuration object and the top-level runner so other
code can launch the attribution pipeline without importing the internal helper
modules directly.
"""

from .config import TRAKConfig, parse_args
from .run_trak import run

__all__ = ["TRAKConfig", "parse_args", "run"]
