"""Notebook-friendly helpers for inspecting attribution run outputs.

This module is intentionally designed for interactive analysis rather than for
the main attribution pipeline. The goal is to make the exported CSV/JSONL
artifacts readable for someone who did not write the backend and may not know
what each file means yet.

The helpers here answer a few practical notebook questions:

- What files exist in an attribution output directory?
- Which checkpoint steps were exported?
- Which targets were hardest or easiest under the EWoK objective?
- Which BOS rows recur across many targets?
- Which rows dominate within a particular domain?
- How do I inspect one target or one candidate row without manually parsing
  JSONL files?
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from research.bos_aligned_proto.pipeline.bos_packed_index import (
        PACKED_INDEX_FORMAT,
        PackedIndexView,
    )
except ImportError:
    from ....pipeline.bos_packed_index import PACKED_INDEX_FORMAT, PackedIndexView


_STEP_RE = re.compile(r"step(?P<step>\d{8})")
_TOP_ROWS_RE = re.compile(r"^top_rows_step(?P<step>\d{8})\.csv$")
_BOTTOM_ROWS_RE = re.compile(r"^bottom_rows_step(?P<step>\d{8})\.csv$")
_ROW_SUMMARY_RE = re.compile(r"^row_summary_step(?P<step>\d{8})\.csv$")
_DOMAIN_SUMMARY_RE = re.compile(r"^domain_summary_step(?P<step>\d{8})\.csv$")
_TARGET_DIAGNOSTICS_RE = re.compile(r"^target_diagnostics_step(?P<step>\d{8})\.jsonl$")
_DENSE_SCORES_RE = re.compile(r"^dense_scores_step(?P<step>\d{8})\.npy$")

_ARTIFACT_HELP = {
    "config.json": "Resolved CLI configuration used for the run.",
    "run_summary.json": "Run-level provenance and output summary.",
    "checkpoint_manifest.json": "Which checkpoint directories were selected.",
    "target_items.jsonl": "One JSON row per EWoK item with context/target text and metadata.",
    "checkpoint_compare.csv": "Cross-checkpoint comparison table. Empty for one-checkpoint runs.",
    "top_rows": "Top-ranked candidate BOS rows for each EWoK target item.",
    "bottom_rows": "Lowest-ranked candidate BOS rows for each EWoK target item when bottom-k export is enabled.",
    "row_summary": "Candidate rows aggregated over all target items at one checkpoint.",
    "domain_summary": "Candidate rows aggregated separately within each EWoK domain.",
    "target_diagnostics": "Per-target scalar diagnostics such as margins and softplus loss.",
    "dense_scores": "Full dense target-by-candidate score matrix, only present when requested.",
}


@dataclass(frozen=True)
class StepArtifacts:
    """All checkpoint-local export paths for one attribution step."""

    step: int
    top_rows_path: Path | None = None
    bottom_rows_path: Path | None = None
    row_summary_path: Path | None = None
    domain_summary_path: Path | None = None
    target_diagnostics_path: Path | None = None
    dense_scores_path: Path | None = None


@dataclass(frozen=True)
class AttributionRunArtifacts:
    """A parsed attribution output directory plus convenient loaders."""

    output_dir: Path
    config: dict[str, Any]
    run_summary: dict[str, Any]
    checkpoint_manifest: tuple[dict[str, Any], ...]
    target_items_path: Path | None
    checkpoint_compare_path: Path | None
    steps: dict[int, StepArtifacts]

    @property
    def available_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self.steps))

    @property
    def default_step(self) -> int:
        if self.available_steps:
            return int(self.available_steps[0])
        raise ValueError(f"No step-local artifacts were discovered under {self.output_dir}")

    @property
    def data_dir(self) -> Path | None:
        value = self.config.get("data_dir")
        if not value:
            return None
        return Path(value).expanduser().resolve()

    @property
    def tokenizer_checkpoint_path(self) -> Path | None:
        if self.checkpoint_manifest:
            first = self.checkpoint_manifest[0]
            path_value = first.get("path")
            if path_value:
                return Path(path_value).expanduser().resolve()
        run_dir = self.config.get("run_dir")
        if not run_dir:
            return None
        run_path = Path(run_dir).expanduser().resolve()
        candidates = sorted(run_path.glob("ckpt_*_step*"))
        if candidates:
            return candidates[0]
        return None

    def step_artifacts(self, step: int | None = None) -> StepArtifacts:
        target_step = self.default_step if step is None else int(step)
        if target_step not in self.steps:
            raise KeyError(
                f"Step {target_step} is not available. Known steps: {list(self.available_steps)!r}"
            )
        return self.steps[target_step]

    def load_top_rows(self, step: int | None = None) -> pd.DataFrame:
        path = self.step_artifacts(step).top_rows_path
        return _read_csv(path)

    def load_bottom_rows(self, step: int | None = None) -> pd.DataFrame:
        path = self.step_artifacts(step).bottom_rows_path
        return _read_csv(path)

    def load_row_summary(self, step: int | None = None) -> pd.DataFrame:
        path = self.step_artifacts(step).row_summary_path
        return _read_csv(path)

    def load_domain_summary(self, step: int | None = None) -> pd.DataFrame:
        path = self.step_artifacts(step).domain_summary_path
        return _read_csv(path)

    def load_target_diagnostics(self, step: int | None = None) -> pd.DataFrame:
        path = self.step_artifacts(step).target_diagnostics_path
        return _read_jsonl_frame(path)

    def load_target_items(self) -> pd.DataFrame:
        return _augment_target_items_with_ewok_metadata(_read_jsonl_frame(self.target_items_path))

    def load_checkpoint_compare(self) -> pd.DataFrame:
        return _read_csv(self.checkpoint_compare_path)

    def load_dense_scores(self, step: int | None = None) -> np.ndarray | None:
        path = self.step_artifacts(step).dense_scores_path
        if path is None or not path.exists():
            return None
        return np.load(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_jsonl_frame(path: Path | None) -> pd.DataFrame:
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def _infer_ewok_variant(target_items: pd.DataFrame) -> str:
    if target_items.empty or "target_id" not in target_items.columns:
        return "fast"
    sample = str(target_items["target_id"].iloc[0])
    if sample.startswith("ewok-full:"):
        return "full"
    return "fast"


@lru_cache(maxsize=2)
def _load_ewok_lookup(variant: str) -> pd.DataFrame:
    from evaluation.ewok_data import load_ewok_df

    df, _ = load_ewok_df(variant)
    lookup = df.reset_index().rename(columns={"index": "row_index"}).copy()
    lookup["concept_a"] = lookup["ConceptA"].astype(str)
    lookup["concept_b"] = lookup["ConceptB"].astype(str)
    return lookup[["row_index", "concept_a", "concept_b"]]


def _augment_target_items_with_ewok_metadata(target_items: pd.DataFrame) -> pd.DataFrame:
    if target_items.empty or "row_index" not in target_items.columns:
        return target_items.copy()

    if {"concept_a", "concept_b"}.issubset(target_items.columns):
        augmented = target_items.copy()
        if "concept_pair" not in augmented.columns:
            augmented["concept_pair"] = [
                f"{left} <-> {right}" if pd.notna(left) and pd.notna(right) else ""
                for left, right in zip(augmented["concept_a"], augmented["concept_b"])
            ]
        return augmented

    variant = _infer_ewok_variant(target_items)
    try:
        lookup = _load_ewok_lookup(variant)
    except Exception:
        return target_items.copy()

    augmented = target_items.merge(lookup, on="row_index", how="left")
    if "concept_pair" not in augmented.columns and {"concept_a", "concept_b"}.issubset(augmented.columns):
        augmented["concept_pair"] = [
            f"{left} <-> {right}" if pd.notna(left) and pd.notna(right) else ""
            for left, right in zip(augmented["concept_a"], augmented["concept_b"])
        ]
    return augmented


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    # Some export paths intentionally exist for schema consistency even when a
    # one-checkpoint run has nothing meaningful to write yet. In that case the
    # file may contain only a trailing newline, which pandas treats as an empty
    # CSV and raises EmptyDataError for. Notebook analysis should interpret that
    # as "no rows available" rather than failing.
    if not path.read_text(encoding="utf-8").strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _require_matplotlib():
    """Import matplotlib only for plotting helpers.

    The notebook should still be able to load CSV/JSONL artifacts and answer
    table-oriented questions in environments that do not have plotting
    dependencies installed. We only raise a plotting-specific error when the
    user actually calls a plot helper.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting attribution outputs. "
            "Install matplotlib in the active environment to use the plot_* helpers."
        ) from exc
    return plt


@lru_cache(maxsize=8)
def _load_row_pack_meta(data_dir: str) -> dict[str, Any]:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_local_tokenizer(checkpoint_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, local_files_only=True)


@lru_cache(maxsize=256)
def _memmap_shard(shard_path: str) -> np.memmap:
    return np.memmap(Path(shard_path), dtype=np.uint16, mode="r")


def _clean_decoded_row_text(text: str, *, bos_token: str | None, eos_token: str | None) -> str:
    cleaned = text
    if bos_token:
        cleaned = cleaned.replace(bos_token, "\n[BOS]\n")
    if eos_token and eos_token != bos_token:
        cleaned = cleaned.replace(eos_token, "\n[EOS]\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.strip()
    if cleaned.startswith("[BOS]"):
        cleaned = cleaned.removeprefix("[BOS]").strip()
    return cleaned


def _truncate_text(text: str, *, max_chars: int | None = None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    clipped = text[: max(0, int(max_chars) - 3)].rstrip()
    return clipped + "..."


def _compact_target_pair(context: str, target: str, *, context_label: str, target_label: str) -> str:
    return f"{context_label}: {context}\n{target_label}: {target}"


def _pair_contrast_text(left: str, right: str, *, max_parts: int = 3) -> str:
    left_tokens = str(left).split()
    right_tokens = str(right).split()
    matcher = SequenceMatcher(a=left_tokens, b=right_tokens)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left_part = " ".join(left_tokens[i1:i2]).strip()
        right_part = " ".join(right_tokens[j1:j2]).strip()
        if not left_part and not right_part:
            continue
        if not left_part:
            parts.append(f"+ {right_part}")
        elif not right_part:
            parts.append(f"- {left_part}")
        else:
            parts.append(f"{left_part} <-> {right_part}")
        if len(parts) >= int(max_parts):
            break
    return " | ".join(parts)


def _concept_hint_from_pairs(
    *,
    context1: str,
    context2: str,
    target1: str,
    target2: str,
) -> str:
    target_contrast = _pair_contrast_text(target1, target2)
    context_contrast = _pair_contrast_text(context1, context2)
    if target_contrast and context_contrast:
        return f"target: {target_contrast}; context: {context_contrast}"
    if target_contrast:
        return f"target: {target_contrast}"
    if context_contrast:
        return f"context: {context_contrast}"
    return ""


def _decode_exported_row_text(
    *,
    shard_path: str,
    local_row_idx: int | None,
    data_dir: Path,
    checkpoint_path: Path,
    candidate_id: int | None = None,
    token_offset_start: int | None = None,
    token_offset_end: int | None = None,
    max_chars: int | None = 500,
) -> str:
    meta = _load_row_pack_meta(str(data_dir))
    if meta.get("format") == PACKED_INDEX_FORMAT:
        view = PackedIndexView(data_dir)
        split_name = Path(str(shard_path)).name.split("_", 1)[0]
        if candidate_id is not None:
            tokens = view.reconstruct_row(split_name, int(candidate_id)).astype(np.int64, copy=False)
        else:
            if local_row_idx is None:
                return ""
            shard_candidates = view.virtual_shards(split_name)
            shard_match = next((shard for shard in shard_candidates if shard.shard_path == str(shard_path)), None)
            if shard_match is None:
                raise KeyError(
                    f"Virtual shard path {shard_path!r} was not found in packed-index manifest for split {split_name!r}."
                )
            global_row_id = int(shard_match.row_start + int(local_row_idx))
            tokens = view.reconstruct_row(split_name, global_row_id).astype(np.int64, copy=False)
    else:
        mm = _memmap_shard(str(Path(shard_path).expanduser().resolve()))
        if token_offset_start is not None and token_offset_end is not None:
            tokens = np.asarray(mm[int(token_offset_start) : int(token_offset_end)], dtype=np.int64)
        else:
            row_tokens = meta.get("row_tokens")
            if row_tokens is None or local_row_idx is None:
                return ""
            start = int(local_row_idx) * int(row_tokens)
            tokens = np.asarray(mm[start : start + int(row_tokens)], dtype=np.int64)
    tokenizer = _load_local_tokenizer(str(checkpoint_path))
    text = tokenizer.decode(tokens.tolist(), clean_up_tokenization_spaces=False)
    text = _clean_decoded_row_text(text, bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token)
    return _truncate_text(text, max_chars=max_chars)


def attach_row_text(
    frame: pd.DataFrame,
    run: AttributionRunArtifacts,
    *,
    text_column: str = "row_text",
    max_chars: int | None = 500,
) -> pd.DataFrame:
    """Attach decoded BOS row text to a frame with shard/local-row metadata."""

    if frame.empty:
        return frame.copy()
    if text_column in frame.columns:
        return frame.copy()
    if "shard_path" not in frame.columns:
        return frame.copy()
    data_dir = run.data_dir
    checkpoint_path = run.tokenizer_checkpoint_path
    if data_dir is None or checkpoint_path is None:
        return frame.copy()

    annotated = frame.copy()
    texts: list[str] = []
    for record in annotated.to_dict(orient="records"):
        local_row_idx = record.get("local_row_idx")
        texts.append(
            _decode_exported_row_text(
                shard_path=str(record["shard_path"]),
                local_row_idx=None if pd.isna(local_row_idx) else int(local_row_idx),
                data_dir=data_dir,
                checkpoint_path=checkpoint_path,
                candidate_id=(
                    None
                    if "candidate_id" not in record or pd.isna(record["candidate_id"])
                    else int(record["candidate_id"])
                ),
                token_offset_start=(
                    None
                    if "token_offset_start" not in record or pd.isna(record["token_offset_start"])
                    else int(record["token_offset_start"])
                ),
                token_offset_end=(
                    None
                    if "token_offset_end" not in record or pd.isna(record["token_offset_end"])
                    else int(record["token_offset_end"])
                ),
                max_chars=max_chars,
            )
        )
    annotated[text_column] = texts
    return annotated


def prepare_target_text_frame(target_items: pd.DataFrame, target_id: str) -> pd.DataFrame:
    """Return a compact, text-first view of one EWoK item."""

    if target_items.empty or "target_id" not in target_items.columns:
        return pd.DataFrame()
    frame = target_items.loc[target_items["target_id"] == target_id].copy()
    if frame.empty:
        return frame
    frame["pair_1"] = [
        _compact_target_pair(context, target, context_label="C1", target_label="T1")
        for context, target in zip(frame["context1"], frame["target1"])
    ]
    frame["pair_2"] = [
        _compact_target_pair(context, target, context_label="C2", target_label="T2")
        for context, target in zip(frame["context2"], frame["target2"])
    ]
    if {"concept_a", "concept_b"}.issubset(frame.columns):
        frame["concept_pair"] = [
            f"{left} <-> {right}" if pd.notna(left) and pd.notna(right) else ""
            for left, right in zip(frame["concept_a"], frame["concept_b"])
        ]
    frame["target_contrast"] = [
        _pair_contrast_text(left, right)
        for left, right in zip(frame["target1"], frame["target2"])
    ]
    frame["context_contrast"] = [
        _pair_contrast_text(left, right)
        for left, right in zip(frame["context1"], frame["context2"])
    ]
    frame["concept_hint"] = [
        _concept_hint_from_pairs(context1=context1, context2=context2, target1=target1, target2=target2)
        for context1, context2, target1, target2 in zip(
            frame["context1"], frame["context2"], frame["target1"], frame["target2"]
        )
    ]
    if "concept_pair" not in frame.columns and {"concept_a", "concept_b"}.issubset(frame.columns):
        frame["concept_pair"] = [
            f"{left} <-> {right}" if pd.notna(left) and pd.notna(right) else ""
            for left, right in zip(frame["concept_a"], frame["concept_b"])
        ]
    frame["concept"] = frame.get("concept_pair", pd.Series([""] * len(frame)))
    preferred = ["pair_1", "pair_2", "context_type", "context_diff", "concept"]
    available = [column for column in preferred if column in frame.columns]
    return frame[available].reset_index(drop=True)


def _target_index_from_items(target_items: pd.DataFrame, target_id: str) -> int | None:
    if target_items.empty or "target_id" not in target_items.columns:
        return None
    matches = target_items.index[target_items["target_id"] == target_id].tolist()
    if not matches:
        return None
    return int(matches[0])


def _target_negative_rows_from_dense(
    *,
    run: AttributionRunArtifacts,
    row_summary: pd.DataFrame,
    target_items: pd.DataFrame,
    target_id: str,
    step: int | None,
    bottom_n: int,
    max_chars: int | None,
) -> pd.DataFrame:
    dense_scores = run.load_dense_scores(step)
    target_idx = _target_index_from_items(target_items.reset_index(drop=True), target_id)
    if dense_scores is None or target_idx is None or row_summary.empty:
        return pd.DataFrame()

    candidate_info = row_summary.reset_index(drop=True)
    if dense_scores.shape[1] != len(candidate_info):
        return pd.DataFrame()

    row_scores = np.asarray(dense_scores[target_idx], dtype=np.float64)
    bottom_indices = np.argsort(row_scores)[: int(bottom_n)]
    frame = candidate_info.loc[bottom_indices, ["row_id", "shard_path", "local_row_idx"]].copy()
    frame["score"] = row_scores[bottom_indices]
    frame["influence_direction"] = "negative"
    frame = frame.sort_values("score", ascending=True).reset_index(drop=True)
    return attach_row_text(frame, run, max_chars=max_chars)


def _upsert_step_artifact(steps: dict[int, StepArtifacts], step: int, **kwargs) -> None:
    existing = steps.get(int(step), StepArtifacts(step=int(step)))
    steps[int(step)] = StepArtifacts(
        step=int(step),
        top_rows_path=kwargs.get("top_rows_path", existing.top_rows_path),
        bottom_rows_path=kwargs.get("bottom_rows_path", existing.bottom_rows_path),
        row_summary_path=kwargs.get("row_summary_path", existing.row_summary_path),
        domain_summary_path=kwargs.get("domain_summary_path", existing.domain_summary_path),
        target_diagnostics_path=kwargs.get("target_diagnostics_path", existing.target_diagnostics_path),
        dense_scores_path=kwargs.get("dense_scores_path", existing.dense_scores_path),
    )


def load_attribution_run(output_dir: str | Path) -> AttributionRunArtifacts:
    """Load an attribution output directory and discover its exported artifacts."""

    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Attribution output directory not found: {root}")

    steps: dict[int, StepArtifacts] = {}
    for child in root.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if (match := _TOP_ROWS_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), top_rows_path=child)
        elif (match := _BOTTOM_ROWS_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), bottom_rows_path=child)
        elif (match := _ROW_SUMMARY_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), row_summary_path=child)
        elif (match := _DOMAIN_SUMMARY_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), domain_summary_path=child)
        elif (match := _TARGET_DIAGNOSTICS_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), target_diagnostics_path=child)
        elif (match := _DENSE_SCORES_RE.match(name)) is not None:
            _upsert_step_artifact(steps, int(match.group("step")), dense_scores_path=child)

    return AttributionRunArtifacts(
        output_dir=root,
        config=_read_json(root / "config.json"),
        run_summary=_read_json(root / "run_summary.json"),
        checkpoint_manifest=tuple(_read_json(root / "checkpoint_manifest.json") or []),
        target_items_path=(root / "target_items.jsonl") if (root / "target_items.jsonl").exists() else None,
        checkpoint_compare_path=(root / "checkpoint_compare.csv") if (root / "checkpoint_compare.csv").exists() else None,
        steps=steps,
    )


def artifact_inventory(run: AttributionRunArtifacts) -> pd.DataFrame:
    """Return a human-readable inventory of files inside one output directory."""

    rows: list[dict[str, Any]] = []
    for path in sorted(run.output_dir.iterdir()):
        if path.name == "cache":
            continue
        kind = path.name
        if path.name.startswith("top_rows_step"):
            kind = "top_rows"
        elif path.name.startswith("bottom_rows_step"):
            kind = "bottom_rows"
        elif path.name.startswith("row_summary_step"):
            kind = "row_summary"
        elif path.name.startswith("domain_summary_step"):
            kind = "domain_summary"
        elif path.name.startswith("target_diagnostics_step"):
            kind = "target_diagnostics"
        elif path.name.startswith("dense_scores_step"):
            kind = "dense_scores"
        rows.append(
            {
                "artifact": path.name,
                "kind": kind,
                "description": _ARTIFACT_HELP.get(kind, ""),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_overview(run: AttributionRunArtifacts) -> pd.DataFrame:
    """Summarize the run at a glance for notebook display."""

    cfg = run.config or run.run_summary.get("config", {})
    row = {
        "backend": cfg.get("backend"),
        "exp_name": cfg.get("exp_name"),
        "output_dir": str(run.output_dir),
        "available_steps": list(run.available_steps),
        "ewok_score_view": cfg.get("ewok_score_view"),
        "score_reduction": cfg.get("score_reduction"),
        "candidate_strategy": cfg.get("candidate_strategy"),
        "max_candidate_rows": cfg.get("max_candidate_rows"),
        "topk": cfg.get("topk"),
        "max_targets": cfg.get("max_targets"),
        "use_fast_jl": cfg.get("use_fast_jl"),
        "proj_dim": cfg.get("proj_dim"),
        "device": cfg.get("device"),
        "distributed": cfg.get("distributed"),
    }
    return pd.DataFrame([row])


def step_overview(run: AttributionRunArtifacts, step: int | None = None) -> pd.DataFrame:
    """Summarize one checkpoint step using the exported analysis artifacts."""

    chosen_step = run.default_step if step is None else int(step)
    top_rows = run.load_top_rows(chosen_step)
    row_summary = run.load_row_summary(chosen_step)
    domain_summary = run.load_domain_summary(chosen_step)
    diagnostics = run.load_target_diagnostics(chosen_step)

    row = {
        "step": chosen_step,
        "target_count": int(diagnostics["target_id"].nunique()) if "target_id" in diagnostics else 0,
        "candidate_count": int(len(row_summary)),
        "domains": int(diagnostics["domain"].nunique()) if "domain" in diagnostics else 0,
        "top_rows_entries": int(len(top_rows)),
        "rows_in_domain_summary": int(len(domain_summary)),
        "mean_softplus_loss": float(diagnostics["softplus_loss"].mean()) if "softplus_loss" in diagnostics else np.nan,
        "median_softplus_loss": float(diagnostics["softplus_loss"].median()) if "softplus_loss" in diagnostics else np.nan,
        "mean_abs_candidate_score": float(row_summary["mean_abs_score"].mean()) if "mean_abs_score" in row_summary else np.nan,
        "max_abs_candidate_score": float(row_summary["max_abs_score"].max()) if "max_abs_score" in row_summary else np.nan,
    }
    return pd.DataFrame([row])


def merge_target_metadata(frame: pd.DataFrame, target_items: pd.DataFrame) -> pd.DataFrame:
    """Attach target text fields to a diagnostics or top-rows frame."""

    if frame.empty or target_items.empty or "target_id" not in frame or "target_id" not in target_items:
        return frame.copy()
    target_columns = [
        "target_id",
        "domain",
        "concept_a",
        "concept_b",
        "concept_pair",
        "context1",
        "context2",
        "target1",
        "target2",
        "context_type",
        "context_diff",
        "target_diff",
    ]
    available_columns = [column for column in target_columns if column in target_items.columns]
    merged = frame.merge(
        target_items[available_columns].drop_duplicates("target_id"),
        on="target_id",
        how="left",
        suffixes=("", "_item"),
    )
    if "context1" in merged.columns and "target1" in merged.columns:
        merged["pair_1"] = [
            _compact_target_pair(context, target, context_label="C1", target_label="T1")
            for context, target in zip(merged["context1"], merged["target1"])
        ]
    if "context2" in merged.columns and "target2" in merged.columns:
        merged["pair_2"] = [
            _compact_target_pair(context, target, context_label="C2", target_label="T2")
            for context, target in zip(merged["context2"], merged["target2"])
        ]
    if {"context1", "context2", "target1", "target2"}.issubset(merged.columns):
        merged["target_contrast"] = [
            _pair_contrast_text(left, right)
            for left, right in zip(merged["target1"], merged["target2"])
        ]
        merged["context_contrast"] = [
            _pair_contrast_text(left, right)
            for left, right in zip(merged["context1"], merged["context2"])
        ]
        merged["concept_hint"] = [
            _concept_hint_from_pairs(context1=context1, context2=context2, target1=target1, target2=target2)
            for context1, context2, target1, target2 in zip(
                merged["context1"], merged["context2"], merged["target1"], merged["target2"]
            )
        ]
    return merged


def top_targets_by_loss(
    diagnostics: pd.DataFrame,
    *,
    top_n: int = 15,
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank EWoK items by softplus loss.

    High loss means the model is struggling more on that item under the chosen
    score view and reduction.
    """

    if diagnostics.empty:
        return diagnostics.copy()
    columns = [
        "target_id",
        "domain",
        "softplus_loss",
        "combined_margin",
        "margin_1",
        "margin_2",
        "score",
    ]
    available = [column for column in columns if column in diagnostics.columns]
    return diagnostics.sort_values("softplus_loss", ascending=ascending)[available].head(int(top_n)).reset_index(drop=True)


def most_recurrent_rows(top_rows: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Find candidate rows that appear in many per-target top-k lists."""

    if top_rows.empty:
        return top_rows.copy()
    grouped = (
        top_rows.groupby(["row_id", "shard_path", "local_row_idx"], as_index=False)
        .agg(
            target_hits=("target_id", "nunique"),
            domain_hits=("domain", "nunique"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            mean_score=("score", "mean"),
            mean_abs_score=("score", lambda x: float(np.abs(x).mean())),
        )
        .sort_values(["target_hits", "best_rank", "mean_abs_score"], ascending=[False, True, False])
    )
    return grouped.head(int(top_n)).reset_index(drop=True)


def top_candidate_rows(
    row_summary: pd.DataFrame,
    *,
    top_n: int = 20,
    by: str = "mean_abs_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the most influential rows under one row-level summary statistic."""

    if row_summary.empty:
        return row_summary.copy()
    columns = [
        "row_id",
        "mean_score",
        "mean_abs_score",
        "positive_score_sum",
        "negative_score_sum",
        "max_abs_score",
        "target_count",
        "shard_path",
        "local_row_idx",
    ]
    available = [column for column in columns if column in row_summary.columns]
    return row_summary.sort_values(by, ascending=ascending)[available].head(int(top_n)).reset_index(drop=True)


def top_domain_rows(
    domain_summary: pd.DataFrame,
    *,
    domain: str | None = None,
    top_n: int = 15,
    by: str = "mean_abs_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the top rows within one domain or across all domains."""

    if domain_summary.empty:
        return domain_summary.copy()
    frame = domain_summary.copy()
    if domain is not None:
        normalized = domain if domain.startswith("domain:") else f"domain:{domain}"
        frame = frame.loc[frame["group"] == normalized].copy()
    columns = [
        "group",
        "row_id",
        "mean_score",
        "mean_abs_score",
        "positive_score_sum",
        "negative_score_sum",
        "max_abs_score",
        "target_count",
        "shard_path",
        "local_row_idx",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.sort_values(by, ascending=ascending)[available].head(int(top_n)).reset_index(drop=True)


def target_report(
    run: AttributionRunArtifacts,
    top_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    target_items: pd.DataFrame,
    target_id: str,
    *,
    step: int | None = None,
    row_summary: pd.DataFrame | None = None,
    top_n_rows: int = 20,
    bottom_n_rows: int = 20,
    row_text_chars: int | None = 500,
) -> dict[str, pd.DataFrame | str]:
    """Collect a text-first report for one target item.

    The report is intentionally notebook-oriented. It tries to answer the
    practical human questions first:

    - What are `C1/T1` and `C2/T2` in plain text?
    - What was the item's softplus loss and margins?
    - Which BOS rows push this item most positively?
    - Which rows push it most negatively, if dense scores were exported?
    """

    item = prepare_target_text_frame(target_items, target_id)
    diag = diagnostics.loc[diagnostics["target_id"] == target_id].copy() if not diagnostics.empty else pd.DataFrame()
    positive_rows = (
        top_rows.loc[top_rows["target_id"] == target_id]
        .sort_values(["score", "rank"], ascending=[False, True])
        .head(int(top_n_rows))
        .copy()
    )
    positive_rows["influence_direction"] = "positive"
    positive_rows = attach_row_text(positive_rows, run, max_chars=row_text_chars)
    positive_rows["row_idx"] = positive_rows["row_id"]
    positive_rows = positive_rows[
        [column for column in ("softplus_loss", "influence_direction", "row_idx", "row_text") if column in positive_rows.columns]
    ]

    negative_rows = run.load_bottom_rows(step)
    if not negative_rows.empty:
        negative_rows = (
            negative_rows.loc[negative_rows["target_id"] == target_id]
            .sort_values(["score", "rank"], ascending=[True, True])
            .head(int(bottom_n_rows))
            .copy()
        )
        negative_rows["influence_direction"] = "negative"
        negative_rows = attach_row_text(negative_rows, run, max_chars=row_text_chars)
        negative_rows["row_idx"] = negative_rows["row_id"]
        negative_rows = negative_rows[
            [column for column in ("softplus_loss", "influence_direction", "row_idx", "row_text") if column in negative_rows.columns]
        ]

    note = ""
    if negative_rows.empty and row_summary is not None:
        negative_rows = _target_negative_rows_from_dense(
            run=run,
            row_summary=row_summary,
            target_items=target_items,
            target_id=target_id,
            step=step,
            bottom_n=int(bottom_n_rows),
            max_chars=row_text_chars,
        )
    if negative_rows.empty:
        note = (
            "Per-target negative rows are only available when the run exported "
            "`bottom_rows_stepXXXXXXXX.csv` via `--bottomk` or "
            "`dense_scores_stepXXXXXXXX.npy` via `--write_dense_scores`."
        )
    return {
        "target_item": item.reset_index(drop=True),
        "diagnostics": diag.reset_index(drop=True),
        "top_positive_rows": positive_rows.reset_index(drop=True),
        "top_negative_rows": negative_rows.reset_index(drop=True),
        "negative_rows_note": note,
    }


def row_report(
    run: AttributionRunArtifacts,
    row_id: int,
    top_rows: pd.DataFrame,
    row_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    *,
    top_n_targets: int = 15,
) -> dict[str, pd.DataFrame]:
    """Collect the most useful tables for inspecting one candidate row."""

    row_value = int(row_id)
    overall = row_summary.loc[row_summary["row_id"] == row_value].copy() if not row_summary.empty else pd.DataFrame()
    by_domain = domain_summary.loc[domain_summary["row_id"] == row_value].copy() if not domain_summary.empty else pd.DataFrame()
    appearances = (
        top_rows.loc[top_rows["row_id"] == row_value]
        .sort_values(["rank", "score"], ascending=[True, False])
        .head(int(top_n_targets))
        .copy()
        if not top_rows.empty
        else pd.DataFrame()
    )
    overall = attach_row_text(overall, run)
    by_domain = attach_row_text(by_domain, run)
    appearances = merge_target_metadata(appearances, run.load_target_items())
    return {
        "overall_summary": overall.reset_index(drop=True),
        "domain_summary": by_domain.reset_index(drop=True),
        "top_target_appearances": appearances.reset_index(drop=True),
    }


def plot_target_loss_distribution(
    diagnostics: pd.DataFrame,
    *,
    ax=None,
    bins: int = 25,
):
    """Plot the distribution of EWoK softplus losses."""

    if diagnostics.empty or "softplus_loss" not in diagnostics:
        raise ValueError("Expected a diagnostics frame with a softplus_loss column")
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diagnostics["softplus_loss"], bins=int(bins), color="#4C78A8", alpha=0.85)
    ax.set_title("EWoK Softplus Loss Distribution")
    ax.set_xlabel("softplus_loss")
    ax.set_ylabel("target_count")
    return ax


def plot_row_score_distribution(
    row_summary: pd.DataFrame,
    *,
    column: str = "mean_abs_score",
    ax=None,
    bins: int = 30,
):
    """Plot the distribution of one row-summary score column."""

    if row_summary.empty or column not in row_summary:
        raise ValueError(f"Expected row_summary to contain {column!r}")
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(row_summary[column], bins=int(bins), color="#F58518", alpha=0.85)
    ax.set_title(f"Candidate Row Distribution: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("row_count")
    return ax


def plot_domain_heatmap(
    domain_summary: pd.DataFrame,
    row_summary: pd.DataFrame,
    *,
    top_n_rows: int = 12,
    score_column: str = "mean_abs_score",
    ax=None,
) -> tuple[Any, pd.DataFrame]:
    """Plot a domain-by-row heatmap for the most influential overall rows."""

    if domain_summary.empty or row_summary.empty:
        raise ValueError("Expected both row_summary and domain_summary to be non-empty")
    plt = _require_matplotlib()
    top_rows = top_candidate_rows(row_summary, top_n=int(top_n_rows), by=score_column)
    if top_rows.empty:
        raise ValueError("Could not determine top rows for the heatmap")

    subset = domain_summary.loc[domain_summary["row_id"].isin(top_rows["row_id"])].copy()
    subset["domain"] = subset["group"].str.removeprefix("domain:")
    pivot = (
        subset.pivot_table(index="row_id", columns="domain", values=score_column, fill_value=0.0)
        .reindex(index=top_rows["row_id"].tolist())
        .fillna(0.0)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 0.8 * len(pivot.columns)), max(4, 0.5 * len(pivot.index))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(f"Domain Specialization Heatmap ({score_column})")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(row_id) for row_id in pivot.index])
    ax.set_xlabel("domain")
    ax.set_ylabel("row_id")
    plt.colorbar(image, ax=ax, shrink=0.8)
    return ax, pivot


def plot_checkpoint_compare(checkpoint_compare: pd.DataFrame, *, ax=None):
    """Visualize shared-row overlap between adjacent checkpoints when available."""

    if checkpoint_compare.empty:
        raise ValueError("checkpoint_compare.csv is empty; compare runs need at least two checkpoints")
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(checkpoint_compare))
    if "shared_row_count" in checkpoint_compare:
        ax.plot(x, checkpoint_compare["shared_row_count"], marker="o", label="shared_row_count")
    if "topk_abs_overlap" in checkpoint_compare:
        ax.plot(x, checkpoint_compare["topk_abs_overlap"], marker="s", label="topk_abs_overlap")
    ax.set_title("Checkpoint-to-Checkpoint Overlap")
    ax.set_xlabel("adjacent checkpoint pair index")
    ax.legend()
    return ax


__all__ = [
    "AttributionRunArtifacts",
    "StepArtifacts",
    "attach_row_text",
    "artifact_inventory",
    "load_attribution_run",
    "merge_target_metadata",
    "most_recurrent_rows",
    "plot_checkpoint_compare",
    "plot_domain_heatmap",
    "plot_row_score_distribution",
    "plot_target_loss_distribution",
    "prepare_target_text_frame",
    "row_report",
    "run_overview",
    "step_overview",
    "target_report",
    "top_candidate_rows",
    "top_domain_rows",
    "top_targets_by_loss",
]
