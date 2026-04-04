"""CLI entrypoint for the TrackStar score-path audit."""

from __future__ import annotations

from .score_path_audit import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
