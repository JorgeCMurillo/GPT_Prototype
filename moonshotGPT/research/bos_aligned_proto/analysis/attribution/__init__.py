"""BOS-aligned attribution backends and runners.

This package keeps runner exports lazy so `python -m ...run_trak` and
`python -m ...run_trackstar` do not pre-import their own modules through the
package `__init__`, which would trigger a `runpy` warning.
"""

from __future__ import annotations


__all__ = ["run_trackstar", "run_trak"]


def __getattr__(name: str):
    if name == "run_trak":
        from .run_trak import run

        return run
    if name == "run_trackstar":
        from .run_trackstar import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
