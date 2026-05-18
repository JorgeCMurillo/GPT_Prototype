"""CLI entrypoint for the TrackStar projection-geometry audit."""

from __future__ import annotations

from .projection_geometry_audit import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
