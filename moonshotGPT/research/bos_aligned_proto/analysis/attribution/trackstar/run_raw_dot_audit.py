"""CLI entrypoint for the raw-gradient TrackStar audit."""

from __future__ import annotations

from .raw_dot_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
