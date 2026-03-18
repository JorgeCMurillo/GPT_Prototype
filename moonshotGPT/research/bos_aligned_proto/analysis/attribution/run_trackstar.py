"""TrackStar/Bergson entrypoint for BOS-row attribution runs."""

from __future__ import annotations

from .run_trak import run as run_attribution
from .trackstar.config import TrackstarConfig, parse_args


def run(config: TrackstarConfig) -> dict:
    return run_attribution(config)


def main(argv: list[str] | None = None) -> dict:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    main()
