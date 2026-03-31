"""Helpers for recovering training-example semantics from finished runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .training_examples import PACKED_INDEX_FORMAT, CandidateKind


@dataclass(frozen=True)
class TrainingExampleSpec:
    """Minimal facts needed to reconstruct the model's training examples."""

    candidate_kind: CandidateKind
    seq_len: int


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_for_data_dir(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _candidate_kind_from_meta(meta: dict[str, Any]) -> CandidateKind:
    if str(meta.get("format", "")) == PACKED_INDEX_FORMAT or "row_tokens" in meta:
        return "bos_packed_row"
    return "stream_window"


def _iter_checkpoint_dirs(run_dir: Path) -> Iterable[Path]:
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("ckpt_"):
            yield child


def _extract_seq_len_from_checkpoint(checkpoint_dir: Path) -> int | None:
    trainer_state = _read_json_if_exists(checkpoint_dir / "trainer_state.json") or {}
    for key in ("seq_len", "n_ctx", "n_positions"):
        value = trainer_state.get(key)
        if value is not None:
            return int(value)

    config = _read_json_if_exists(checkpoint_dir / "config.json") or {}
    for key in ("n_ctx", "n_positions"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return None


def resolve_training_example_spec(
    *,
    run_dir: str | Path,
    data_dir: str | Path,
) -> TrainingExampleSpec:
    """Recover whether a run trained on BOS rows or stream windows.

    The attribution CLI only receives a finished run directory plus the data
    directory to analyze. Stream artifacts do not store `seq_len` inside the
    raw token-shard metadata, so we recover it from the run config or
    checkpoints and pair it with the candidate kind inferred from the data/run.
    """

    run_path = Path(run_dir).expanduser().resolve()
    data_path = Path(data_dir).expanduser().resolve()
    meta = _meta_for_data_dir(data_path)

    run_config = _read_json_if_exists(run_path / "run_config.json") or {}
    run_args = run_config.get("args", {}) if isinstance(run_config, dict) else {}
    loader_kind = run_args.get("loader_kind")
    script_name = Path(str(run_config.get("script", ""))).name if isinstance(run_config, dict) else ""

    if loader_kind == "bos_packed_index":
        candidate_kind: CandidateKind = "bos_packed_row"
    elif loader_kind == "stream":
        candidate_kind = "stream_window"
    elif script_name == "train_gpt2_finewebedu_bin.py":
        # Older stream-training entrypoint predates the unified loader_kind arg.
        candidate_kind = "stream_window"
    else:
        candidate_kind = _candidate_kind_from_meta(meta)

    seq_len = run_args.get("seq_len")
    if seq_len is None and candidate_kind == "bos_packed_row":
        seq_len = meta.get("seq_len")
    if seq_len is None:
        for checkpoint_dir in _iter_checkpoint_dirs(run_path):
            seq_len = _extract_seq_len_from_checkpoint(checkpoint_dir)
            if seq_len is not None:
                break
    if seq_len is None:
        raise ValueError(
            f"Could not resolve seq_len for run {run_path}. "
            "Stream-window attribution needs seq_len to reconstruct exact training examples."
        )

    return TrainingExampleSpec(
        candidate_kind=candidate_kind,
        seq_len=int(seq_len),
    )


__all__ = [
    "TrainingExampleSpec",
    "resolve_training_example_spec",
]
