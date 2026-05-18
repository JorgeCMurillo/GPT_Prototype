"""CLI entrypoint for the TrackStar Adam-projection diagnostic."""

from __future__ import annotations

from .adam_projection_diagnostic import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
