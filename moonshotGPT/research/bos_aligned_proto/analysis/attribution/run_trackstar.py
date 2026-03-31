"""TrackStar/Bergson entrypoint for BOS-row attribution runs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trackstar.config import TrackstarConfig


def _startup_status(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [attribution][trackstar] {message}", flush=True)


def _startup_progress_bar():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=2, desc="TrackStar startup", unit="stage", leave=False)


def run(config: "TrackstarConfig") -> dict:
    from .run_trak import run as run_attribution

    return run_attribution(config)


def main(argv: list[str] | None = None) -> dict:
    progress = _startup_progress_bar()
    try:
        if progress is None:
            _startup_status("bootstrapping TrackStar CLI")
        else:
            progress.set_postfix_str("parsing args")

        from .trackstar.config import parse_args

        config = parse_args(argv)

        if progress is None:
            _startup_status("parsed CLI; loading shared attribution runner")
        else:
            progress.update(1)
            progress.set_postfix_str("loading runner")

        from .run_trak import run as run_attribution

        if progress is None:
            _startup_status("shared runner loaded; starting attribution pipeline")
        else:
            progress.update(1)
            progress.set_postfix_str("starting run")
        return run_attribution(config)
    finally:
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    main()
