"""Build BOS-aligned row-packed FineWeb-Edu shards for prototype training.

Pipeline overview:
1. A background prefetcher reads dataset examples into text batches.
2. The main thread tokenizes each batch and prepends BOS to every document.
3. A row packer fills fixed-width `(seq_len + 1)` rows using best-fit packing.
4. A shard sink writes rows into `train_*.bin` / `val_*.bin` shard files.

This script preserves the existing BOS-row packing semantics. The performance
changes are limited to overlapping dataset reads and shard flushes with the
main packing loop.

Resume support:
- The script records shard-boundary resume points under `out_dir/.resume_state/`.
- Only the active exact resume snapshot and one safety duplicate are retained to avoid unbounded disk growth.
- When resuming from shard `N`, that snapshot is kept until shard `N+1` is fully checkpointed.
- `--resume_from_shard N` restarts from the beginning of shard `N` when that snapshot exists.
- Exact BOS-row resume stores the remaining best-fit document buffer and the
  cumulative crop count, because output depends on packing state, not just the
  next source document index.

CLI hyperparameters / knobs:
- `--dataset`: Hugging Face dataset id to read from.
- `--config`: Dataset subset/config to stream, such as `sample-10BT` or
  `sample-100BT`.
- `--data_files`: Optional local dataset files. When set, this bypasses the HF
  config selection and reads local files instead.
- `--split`: Dataset split to use, typically `train`.
- `--text_field`: Field name containing raw text inside each dataset example.
- `--out_dir`: Directory where row-packed shard files and `meta.json` are
  written.
- `--tokenizer`: Tokenizer name or path. The script assumes GPT-2-style
  token ids that fit in `uint16`.
- `--batch_docs`: Number of raw documents tokenized per tokenizer call. Larger
  values usually improve throughput but increase peak RAM.
- `--seq_len`: Target training sequence length. Each packed row has
  `seq_len + 1` tokens so loaders can form `(x, y)` pairs.
- `--buffer_docs`: Number of tokenized documents kept in the best-fit packing
  buffer. Larger values usually improve packing quality and reduce cropping,
  but increase RAM and packing search work.
- `--shard_rows`: Number of rows per output shard. `0` means auto-size to
  roughly 100M output tokens per shard.
- `--val_shards`: Number of initial shard indices labeled as validation
  instead of training.
- `--max_docs`: Optional debug limit. If set above zero, stop after this many
  source documents.
- `--prefetch_batches`: Number of raw-text batches the background reader can
  queue ahead of tokenization. Higher values can hide dataset stalls but use
  more RAM.
- `--write_queue_shards`: Number of completed shard buffers that can wait for
  the background writer. `0` disables async writes; larger values trade RAM for
  more overlap with disk I/O.
- `--progress_metrics`: Comma-separated tqdm diagnostics to show. By default
  the bar only shows the most important run-health signals: RAM, free disk, and
  crop rate.
- `--progress_refresh_secs`: How often the selected diagnostics are refreshed.
  The token counter itself still advances continuously.
- `--resume_from_shard`: Restart from the beginning of a shard using saved
  packing state in `out_dir/.resume_state/`.
"""

import argparse
import json
import os
import queue
import re
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Callable, Iterator, Sequence

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


PROGRESS_METRIC_CHOICES = ("rss", "disk", "crop", "buf", "pre", "wr", "docs", "shard")
DEFAULT_PROGRESS_METRICS = ("rss", "disk", "crop")


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically replace `path` with `data` so interrupted runs do not leave torn files."""
    dirpath = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + "."
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=prefix, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_bin(path: str, arr: np.ndarray) -> None:
    """Atomically write a uint16 shard file to disk."""
    assert arr.dtype == np.uint16
    dirpath = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + "."
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=prefix, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        arr.tofile(f)
    os.replace(tmp, path)


def atomic_write_array(path: str, arr: np.ndarray) -> None:
    """Atomically write a raw NumPy array without imposing a specific dtype."""
    dirpath = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + "."
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=prefix, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        arr.tofile(f)
    os.replace(tmp, path)


def format_bytes(num_bytes: int | None) -> str:
    """Format byte counts compactly for tqdm postfix output."""
    if num_bytes is None:
        return "?"
    units = ("B", "K", "M", "G", "T", "P")
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if value >= 10 or unit_idx == 0:
        return f"{value:.0f}{units[unit_idx]}"
    return f"{value:.1f}{units[unit_idx]}"


def format_count(value: int) -> str:
    """Format large counters compactly for status displays."""
    thresholds = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    )
    for threshold, suffix in thresholds:
        if value >= threshold:
            scaled = value / float(threshold)
            if scaled >= 10:
                return f"{scaled:.0f}{suffix}"
            return f"{scaled:.1f}{suffix}"
    return str(value)


def get_process_rss_bytes() -> int | None:
    """Read current resident memory from `/proc/self/status` on Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def parse_progress_metrics(raw_value: str) -> tuple[str, ...]:
    """Normalize progress metric selection from a CLI string."""
    value = raw_value.strip().lower()
    if value in ("", "default"):
        return DEFAULT_PROGRESS_METRICS
    if value == "all":
        return PROGRESS_METRIC_CHOICES
    if value == "none":
        return ()

    metrics: list[str] = []
    for item in value.split(","):
        metric = item.strip().lower()
        if not metric:
            continue
        if metric not in PROGRESS_METRIC_CHOICES:
            choices = ", ".join(PROGRESS_METRIC_CHOICES)
            raise ValueError(
                f"Unknown progress metric '{metric}'. Expected one of: {choices}, or 'default', 'all', 'none'."
            )
        if metric not in metrics:
            metrics.append(metric)
    return tuple(metrics)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI separately so `main()` only coordinates the run."""
    parser = argparse.ArgumentParser(
        description=(
            "Pretokenize FineWeb/FineWeb-Edu into BOS-aligned row-packed uint16 .bin shards "
            "with nanochat-style best-fit + shortest-crop packing"
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu or HuggingFaceFW/fineweb",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sample-10BT",
        help="Dataset config/name, e.g. sample-10BT or sample-100BT",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="Optional data files for local datasets. If set, --config is ignored.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split (usually 'train')")
    parser.add_argument("--text_field", type=str, default="text", help="Field containing raw text")
    parser.add_argument("--out_dir", type=str, default="fineweb_edu_10B_bosrow", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    parser.add_argument(
        "--batch_docs",
        type=int,
        default=256,
        help="Docs per tokenizer batch; larger batches improve throughput if RAM allows",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Training sequence length. Packed rows always have seq_len + 1 tokens.",
    )
    parser.add_argument(
        "--buffer_docs",
        type=int,
        default=1000,
        help="How many tokenized docs to keep in the best-fit packing buffer",
    )
    parser.add_argument(
        "--shard_rows",
        type=int,
        default=0,
        help="Rows per shard. 0 means auto-size to about 100M tokens/shard.",
    )
    parser.add_argument("--val_shards", type=int, default=1, help="Number of initial shards to label as val")
    parser.add_argument("--max_docs", type=int, default=0, help="If >0, stop after this many docs (debug)")
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
        help="How many text batches to prefetch ahead of tokenization",
    )
    parser.add_argument(
        "--write_queue_shards",
        type=int,
        default=1,
        help="How many completed shard buffers may wait for the background writer; 0 disables async writes",
    )
    parser.add_argument(
        "--progress_metrics",
        type=str,
        default="default",
        help=(
            "Comma-separated tqdm diagnostics to show. Choices: "
            f"{', '.join(PROGRESS_METRIC_CHOICES)}, plus default/all/none."
        ),
    )
    parser.add_argument(
        "--progress_refresh_secs",
        type=float,
        default=3.0,
        help="How often to refresh tqdm diagnostics like RAM/disk/crop. Token progress still updates continuously.",
    )
    parser.add_argument(
        "--resume_from_shard",
        type=int,
        default=0,
        help=(
            "Restart from the beginning of this shard index using the exact state saved in out_dir/.resume_state. "
            "By default only the most recent snapshot is retained."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid settings before expensive setup starts."""
    if args.seq_len <= 0:
        raise ValueError("--seq_len must be > 0")
    if args.batch_docs <= 0:
        raise ValueError("--batch_docs must be > 0")
    if args.buffer_docs <= 0:
        raise ValueError("--buffer_docs must be > 0")
    if args.shard_rows < 0:
        raise ValueError("--shard_rows must be >= 0")
    if args.val_shards < 0:
        raise ValueError("--val_shards must be >= 0")
    if args.max_docs < 0:
        raise ValueError("--max_docs must be >= 0")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch_batches must be > 0")
    if args.write_queue_shards < 0:
        raise ValueError("--write_queue_shards must be >= 0")
    if args.progress_refresh_secs <= 0:
        raise ValueError("--progress_refresh_secs must be > 0")
    if args.resume_from_shard < 0:
        raise ValueError("--resume_from_shard must be >= 0")
    parse_progress_metrics(args.progress_metrics)


def load_tokenizer(tokenizer_name: str):
    """Load the tokenizer once and normalize BOS/EOS handling for GPT-2."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; GPT-2 should have one (50256).")
    bos_token_id = int(eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    return tokenizer, bos_token_id, vocab_size


def load_dataset_source(args: argparse.Namespace):
    """Load either a streaming HF dataset or a local debug dataset."""
    if args.data_files:
        # Local debug path: avoid streaming mode because some environments disallow
        # the shared-memory setup used by some dataset backends.
        return load_dataset(args.dataset, data_files=args.data_files, split=args.split, streaming=False)
    return load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)


def tokenize_batch(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Tokenize one text batch while leaving BOS insertion to the document builder."""
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return encoded["input_ids"]


def document_tokens_to_uint16(token_ids: Sequence[int], bos_token_id: int) -> np.ndarray:
    """Prepend BOS and validate that every token id fits in the uint16 output format."""
    ids = np.asarray(token_ids, dtype=np.int64)
    max_id = bos_token_id
    min_id = bos_token_id
    if ids.size:
        max_id = max(max_id, int(ids.max()))
        min_id = min(min_id, int(ids.min()))
    if max_id >= 2**16 or min_id < 0:
        raise ValueError("Token id out of uint16 range; expected GPT-2 token ids.")

    out = np.empty((ids.size + 1,), dtype=np.uint16)
    out[0] = bos_token_id
    if ids.size:
        out[1:] = ids.astype(np.uint16, copy=False)
    return out


@dataclass(frozen=True)
class TextBatch:
    """A batch of raw text documents ready for tokenization."""

    texts: list[str]


@dataclass(frozen=True)
class ShardWriteRequest:
    """A completed shard buffer that is ready to be persisted."""

    path: str
    data: np.ndarray


@dataclass
class PackingStats:
    """Run statistics collected during BOS-row packing."""

    docs_processed: int = 0
    tokens_cropped_total: int = 0


@dataclass(frozen=True)
class PackingStatusSnapshot:
    """Compact runtime status persisted for human inspection while packing runs."""

    phase: str
    current_shard_idx: int
    latest_completed_shard_idx: int | None
    resume_from_shard: int
    active_resume_state_shard_idx: int | None
    docs_processed: int
    source_docs_seen_total: int
    source_replay_target_docs: int
    source_replay_remaining_docs: int
    current_shard_fill_rows: int
    current_shard_fill_tokens: int
    tokens_written_total: int
    tokens_cropped_total: int
    doc_buffer_len: int
    buffer_docs_capacity: int
    explanation: str
    updated_utc: str

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "current_shard_idx": int(self.current_shard_idx),
            "latest_completed_shard_idx": (
                None if self.latest_completed_shard_idx is None else int(self.latest_completed_shard_idx)
            ),
            "resume_from_shard": int(self.resume_from_shard),
            "active_resume_state_shard_idx": (
                None if self.active_resume_state_shard_idx is None else int(self.active_resume_state_shard_idx)
            ),
            "docs_processed": int(self.docs_processed),
            "source_docs_seen_total": int(self.source_docs_seen_total),
            "source_replay_target_docs": int(self.source_replay_target_docs),
            "source_replay_remaining_docs": int(self.source_replay_remaining_docs),
            "current_shard_fill_rows": int(self.current_shard_fill_rows),
            "current_shard_fill_tokens": int(self.current_shard_fill_tokens),
            "tokens_written_total": int(self.tokens_written_total),
            "tokens_cropped_total": int(self.tokens_cropped_total),
            "doc_buffer_len": int(self.doc_buffer_len),
            "buffer_docs_capacity": int(self.buffer_docs_capacity),
            "explanation": self.explanation,
            "updated_utc": self.updated_utc,
        }


def explain_packing_status(snapshot: PackingStatusSnapshot) -> str:
    """Return a short human-readable explanation of the current packing phase."""
    if snapshot.phase == "resume_replay":
        target = format_count(snapshot.source_replay_target_docs)
        seen = format_count(snapshot.source_docs_seen_total)
        remaining = format_count(snapshot.source_replay_remaining_docs)
        return (
            f"Replaying source docs to resume shard {snapshot.current_shard_idx}: "
            f"{seen}/{target} docs replayed, {remaining} still to skip. "
            f"state_{snapshot.active_resume_state_shard_idx:06d} stays active until a newer shard checkpoint is saved."
            if snapshot.active_resume_state_shard_idx is not None
            else f"Replaying source docs to resume shard {snapshot.current_shard_idx}: "
            f"{seen}/{target} docs replayed, {remaining} still to skip."
        )
    if snapshot.phase == "packing":
        latest = (
            "none yet"
            if snapshot.latest_completed_shard_idx is None
            else str(snapshot.latest_completed_shard_idx)
        )
        return (
            f"Packing shard {snapshot.current_shard_idx}; latest completed shard is {latest}. "
            f"The active resume snapshot will be replaced at the next shard boundary."
            if snapshot.active_resume_state_shard_idx is not None
            else f"Packing shard {snapshot.current_shard_idx}; latest completed shard is {latest}."
        )
    if snapshot.phase == "completed":
        return "Packing finished successfully; no more shard work is pending."
    if snapshot.phase == "failed":
        return "Packing failed; inspect the log and the last active resume snapshot before restarting."
    return "Preparing packing state."


def write_packing_status(out_dir: str, snapshot: PackingStatusSnapshot) -> None:
    """Persist live packing status for users watching the output directory."""
    status_path = os.path.join(out_dir, "packing_status.json")
    atomic_write_bytes(
        status_path,
        json.dumps(snapshot.to_dict(), indent=2).encode("utf-8"),
    )


@dataclass(frozen=True)
class ResumeConfigSignature:
    """Output-shaping settings that must match for an exact resume."""

    dataset: str
    config: str | None
    data_files: str | None
    split: str
    text_field: str
    tokenizer: str
    seq_len: int
    row_tokens: int
    buffer_docs: int
    shard_rows: int
    val_shards: int

    @classmethod
    def from_runtime(
        cls,
        args: argparse.Namespace,
        row_tokens: int,
        shard_rows: int,
    ) -> "ResumeConfigSignature":
        return cls(
            dataset=args.dataset,
            config=args.config,
            data_files=args.data_files,
            split=args.split,
            text_field=args.text_field,
            tokenizer=args.tokenizer,
            seq_len=int(args.seq_len),
            row_tokens=int(row_tokens),
            buffer_docs=int(args.buffer_docs),
            shard_rows=int(shard_rows),
            val_shards=int(args.val_shards),
        )


@dataclass(frozen=True)
class TokenizedDocument:
    """One tokenized source document paired with the cumulative doc count."""

    docs_consumed: int
    tokens: np.ndarray


@dataclass(frozen=True)
class PackingResumeState:
    """Exact restart information for the beginning of a BOS-row shard."""

    next_shard_idx: int
    docs_consumed: int
    tokens_cropped_total: int
    buffered_docs: list[np.ndarray]


class DocBufferCodec:
    """Serialize and deserialize the buffered tokenized docs used by best-fit packing."""

    _LENGTHS_DTYPE = np.uint32

    @classmethod
    def write(cls, lengths_path: str, tokens_path: str, docs: Sequence[np.ndarray]) -> None:
        lengths = np.asarray([int(doc.size) for doc in docs], dtype=cls._LENGTHS_DTYPE)
        if docs:
            flat_tokens = np.concatenate([np.asarray(doc, dtype=np.uint16) for doc in docs])
        else:
            flat_tokens = np.empty((0,), dtype=np.uint16)
        atomic_write_array(lengths_path, lengths)
        atomic_write_array(tokens_path, flat_tokens)

    @classmethod
    def read(cls, lengths_path: str, tokens_path: str) -> list[np.ndarray]:
        lengths = cls._read_array(lengths_path, dtype=cls._LENGTHS_DTYPE)
        flat_tokens = cls._read_array(tokens_path, dtype=np.uint16)
        expected_total = int(lengths.astype(np.int64).sum())
        if flat_tokens.size != expected_total:
            raise ValueError(
                f"Buffered-doc token file expected {expected_total} uint16 values, found {flat_tokens.size}."
            )

        docs: list[np.ndarray] = []
        offset = 0
        for length in lengths.tolist():
            end = offset + int(length)
            docs.append(flat_tokens[offset:end].copy())
            offset = end
        return docs

    @staticmethod
    def _read_array(path: str, dtype) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing resume buffer file: {path}")
        return np.fromfile(path, dtype=dtype)


class PackingResumeStateStore:
    """Persist exact BOS-row shard-boundary state under `out_dir/.resume_state/`.

    Unlike flat-stream tokenization, BOS-row packing is stateful because future
    rows depend on the current best-fit document buffer and on cumulative crop
    accounting. Both must be restored exactly to resume without drift. To keep
    storage bounded, only the active snapshot and one safety duplicate are
    retained. After resuming from shard `N`, that snapshot pair is preserved
    until a newer shard-boundary snapshot replaces it.
    """

    _SHARD_RE = re.compile(r"^(train|val)_(\d{6})\.bin$")
    _STATE_JSON_RE = re.compile(r"^state_(\d{6})(?:\.backup)?\.json$")
    _STATE_LENGTHS_RE = re.compile(r"^state_(\d{6})(?:\.backup)?\.buffer_lengths\.bin$")
    _STATE_TOKENS_RE = re.compile(r"^state_(\d{6})(?:\.backup)?\.buffer_tokens\.bin$")

    def __init__(self, out_dir: str, signature: ResumeConfigSignature) -> None:
        self._out_dir = out_dir
        self._state_dir = os.path.join(out_dir, ".resume_state")
        self._signature = signature
        os.makedirs(self._state_dir, exist_ok=True)

    def save(self, state: PackingResumeState) -> None:
        primary_lengths_path = self._buffer_lengths_path(state.next_shard_idx)
        primary_tokens_path = self._buffer_tokens_path(state.next_shard_idx)
        backup_lengths_path = self._buffer_lengths_path(state.next_shard_idx, backup=True)
        backup_tokens_path = self._buffer_tokens_path(state.next_shard_idx, backup=True)
        DocBufferCodec.write(primary_lengths_path, primary_tokens_path, state.buffered_docs)
        DocBufferCodec.write(backup_lengths_path, backup_tokens_path, state.buffered_docs)

        payload_base = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "next_shard_idx": int(state.next_shard_idx),
            "docs_consumed": int(state.docs_consumed),
            "tokens_cropped_total": int(state.tokens_cropped_total),
            "buffered_doc_count": int(len(state.buffered_docs)),
            "dataset": self._signature.dataset,
            "config": self._signature.config,
            "data_files": self._signature.data_files,
            "split": self._signature.split,
            "text_field": self._signature.text_field,
            "tokenizer": self._signature.tokenizer,
            "seq_len": int(self._signature.seq_len),
            "row_tokens": int(self._signature.row_tokens),
            "buffer_docs": int(self._signature.buffer_docs),
            "shard_rows": int(self._signature.shard_rows),
            "val_shards": int(self._signature.val_shards),
        }
        primary_payload = {
            **payload_base,
            "buffer_lengths_path": os.path.basename(primary_lengths_path),
            "buffer_tokens_path": os.path.basename(primary_tokens_path),
        }
        atomic_write_bytes(
            self._state_path(state.next_shard_idx),
            json.dumps(primary_payload, indent=2).encode("utf-8"),
        )
        backup_payload = {
            **payload_base,
            "buffer_lengths_path": os.path.basename(backup_lengths_path),
            "buffer_tokens_path": os.path.basename(backup_tokens_path),
        }
        atomic_write_bytes(
            self._state_path(state.next_shard_idx, backup=True),
            json.dumps(backup_payload, indent=2).encode("utf-8"),
        )
        self._prune_all_but(state.next_shard_idx)

    def load(self, next_shard_idx: int) -> PackingResumeState:
        primary_state_path = self._state_path(next_shard_idx)
        backup_state_path = self._state_path(next_shard_idx, backup=True)
        last_error: BaseException | None = None

        for state_path, label in (
            (primary_state_path, "primary"),
            (backup_state_path, "backup"),
        ):
            if not os.path.exists(state_path):
                continue
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self._validate_payload(payload, expected_next_shard_idx=next_shard_idx)

                buffered_docs = DocBufferCodec.read(
                    os.path.join(self._state_dir, payload["buffer_lengths_path"]),
                    os.path.join(self._state_dir, payload["buffer_tokens_path"]),
                )
                expected_count = int(payload.get("buffered_doc_count", len(buffered_docs)))
                if len(buffered_docs) != expected_count:
                    raise ValueError(
                        f"Resume state expected {expected_count} buffered docs, found {len(buffered_docs)}."
                    )
                if label == "backup":
                    print(
                        f"[resume] primary snapshot for shard {next_shard_idx} was unavailable; "
                        "falling back to the safety duplicate."
                    )
                return PackingResumeState(
                    next_shard_idx=int(payload["next_shard_idx"]),
                    docs_consumed=int(payload["docs_consumed"]),
                    tokens_cropped_total=int(payload["tokens_cropped_total"]),
                    buffered_docs=buffered_docs,
                )
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"Failed to load resume state for shard {next_shard_idx} from either the active snapshot "
                "or its safety duplicate."
            ) from last_error
        raise FileNotFoundError(
            f"No resume state found for shard {next_shard_idx} at '{primary_state_path}' "
            f"or '{backup_state_path}'. Resume is only available for runs created with the resume-aware script."
        )

    def cleanup_from_shard(self, start_shard_idx: int, keep_state_shard_idx: int | None = None) -> None:
        """Remove outputs/state from `start_shard_idx` onward before rewriting them."""
        for name in os.listdir(self._out_dir):
            match = self._SHARD_RE.match(name)
            if not match:
                continue
            shard_idx = int(match.group(2))
            if shard_idx >= start_shard_idx:
                os.remove(os.path.join(self._out_dir, name))

        for name in os.listdir(self._state_dir):
            for pattern in (self._STATE_JSON_RE, self._STATE_LENGTHS_RE, self._STATE_TOKENS_RE):
                match = pattern.match(name)
                if match:
                    shard_idx = int(match.group(1))
                    if shard_idx >= start_shard_idx and shard_idx != keep_state_shard_idx:
                        os.remove(os.path.join(self._state_dir, name))
                    break

    def _validate_payload(self, payload: dict, expected_next_shard_idx: int) -> None:
        expected = {
            "dataset": self._signature.dataset,
            "config": self._signature.config,
            "data_files": self._signature.data_files,
            "split": self._signature.split,
            "text_field": self._signature.text_field,
            "tokenizer": self._signature.tokenizer,
            "seq_len": int(self._signature.seq_len),
            "row_tokens": int(self._signature.row_tokens),
            "buffer_docs": int(self._signature.buffer_docs),
            "shard_rows": int(self._signature.shard_rows),
            "val_shards": int(self._signature.val_shards),
        }
        for key, expected_value in expected.items():
            observed_value = payload.get(key)
            if observed_value != expected_value:
                raise ValueError(
                    f"Resume state mismatch for '{key}': expected {expected_value!r}, found {observed_value!r}."
                )
        observed_next_shard = int(payload.get("next_shard_idx", -1))
        if observed_next_shard != expected_next_shard_idx:
            raise ValueError(
                f"Resume state file expected next_shard_idx={expected_next_shard_idx}, "
                f"found {observed_next_shard}."
            )

    def _state_path(self, next_shard_idx: int, backup: bool = False) -> str:
        suffix = ".backup" if backup else ""
        return os.path.join(self._state_dir, f"state_{next_shard_idx:06d}{suffix}.json")

    def _buffer_lengths_path(self, next_shard_idx: int, backup: bool = False) -> str:
        suffix = ".backup" if backup else ""
        return os.path.join(self._state_dir, f"state_{next_shard_idx:06d}{suffix}.buffer_lengths.bin")

    def _buffer_tokens_path(self, next_shard_idx: int, backup: bool = False) -> str:
        suffix = ".backup" if backup else ""
        return os.path.join(self._state_dir, f"state_{next_shard_idx:06d}{suffix}.buffer_tokens.bin")

    def _prune_all_but(self, keep_next_shard_idx: int) -> None:
        for name in os.listdir(self._state_dir):
            matched_idx = None
            for pattern in (self._STATE_JSON_RE, self._STATE_LENGTHS_RE, self._STATE_TOKENS_RE):
                match = pattern.match(name)
                if match:
                    matched_idx = int(match.group(1))
                    break
            if matched_idx is None or matched_idx == keep_next_shard_idx:
                continue
            os.remove(os.path.join(self._state_dir, name))


class TextBatchPrefetcher:
    """Read examples in the background and emit ordered text batches.

    The dataset iterator itself stays on one thread. This reduces idle time when
    the main thread is busy tokenizing or row-packing.
    """

    _END = object()

    def __init__(
        self,
        dataset,
        text_field: str,
        batch_docs: int,
        max_docs: int,
        prefetch_batches: int,
        skip_docs: int = 0,
        progress_callback: Callable[[int, int, bool], None] | None = None,
        progress_report_interval_seconds: float = 5.0,
    ) -> None:
        self._dataset = dataset
        self._text_field = text_field
        self._batch_docs = batch_docs
        self._max_docs = max_docs
        self._skip_docs = skip_docs
        self._progress_callback = progress_callback
        self._progress_report_interval_seconds = progress_report_interval_seconds
        self._last_progress_report_time = 0.0
        self._queue: queue.Queue[object] = queue.Queue(maxsize=prefetch_batches)
        self._cancel_event = Event()
        self._thread = Thread(target=self._run, name="bos-row-prefetch", daemon=True)
        self._exception: BaseException | None = None
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def close(self, cancel: bool = False) -> None:
        if not self._started:
            return
        if cancel:
            self._cancel_event.set()
        self._thread.join()
        if self._exception is not None and not cancel:
            raise self._exception

    def set_progress_callback(
        self,
        callback: Callable[[int, int, bool], None] | None,
    ) -> None:
        """Attach or replace the progress callback before or during a run."""
        self._progress_callback = callback

    def __iter__(self) -> Iterator[TextBatch]:
        self.start()
        while True:
            item = self._queue.get()
            if item is self._END:
                break
            if not isinstance(item, TextBatch):
                raise TypeError(f"Unexpected prefetch item type: {type(item)!r}")
            yield item
        self.close(cancel=False)

    def _put(self, item: object) -> bool:
        while not self._cancel_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _emit_progress(self, docs_seen_total: int, replay_complete: bool, force: bool = False) -> None:
        if self._progress_callback is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_progress_report_time) < self._progress_report_interval_seconds:
            return
        self._last_progress_report_time = now
        self._progress_callback(docs_seen_total, self._skip_docs, replay_complete)

    def _run(self) -> None:
        try:
            batch_texts: list[str] = []
            docs_seen_total = 0
            self._emit_progress(docs_seen_total, replay_complete=(self._skip_docs == 0), force=True)

            for example in self._dataset:
                docs_seen_total += 1

                # Resume works by replaying the source stream up to the saved
                # document count. That is exact, but still linear in skipped docs.
                if docs_seen_total <= self._skip_docs:
                    self._emit_progress(docs_seen_total, replay_complete=False)
                    if self._max_docs > 0 and docs_seen_total >= self._max_docs:
                        break
                    continue

                text = example.get(self._text_field)
                if text is None:
                    raise KeyError(
                        f"Example missing text field '{self._text_field}'. Keys: {list(example.keys())}"
                    )

                batch_texts.append(text)

                reached_doc_limit = self._max_docs > 0 and docs_seen_total >= self._max_docs
                if len(batch_texts) >= self._batch_docs or reached_doc_limit:
                    if not self._put(TextBatch(texts=batch_texts)):
                        return
                    batch_texts = []
                    self._emit_progress(docs_seen_total, replay_complete=True)

                if reached_doc_limit:
                    break

            if batch_texts:
                self._put(TextBatch(texts=batch_texts))
                self._emit_progress(docs_seen_total, replay_complete=True)
        except BaseException as exc:
            self._exception = exc
        finally:
            self._emit_progress(docs_seen_total if "docs_seen_total" in locals() else 0, replay_complete=True, force=True)
            self._put(self._END)

    @property
    def pending_batches(self) -> int:
        """Approximate number of prefetched text batches waiting to be tokenized."""
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        """Configured maximum number of prefetched batches."""
        return self._queue.maxsize


class ShardWriter(ABC):
    """Small abstraction so row packing does not depend on sync vs async writes."""

    @abstractmethod
    def submit(self, path: str, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    def pending_writes(self) -> int:
        """Approximate number of completed shard buffers waiting for disk I/O."""
        return 0

    @property
    def queue_capacity(self) -> int:
        """Maximum number of queued background writes; zero means synchronous writes."""
        return 0


class SyncShardWriter(ShardWriter):
    """Write shards on the caller's thread."""

    def submit(self, path: str, data: np.ndarray) -> None:
        atomic_write_bin(path, data)

    def close(self) -> None:
        return


class AsyncShardWriter(ShardWriter):
    """Write completed shard buffers on a dedicated background thread."""

    _END = object()

    def __init__(self, max_pending_shards: int) -> None:
        if max_pending_shards <= 0:
            raise ValueError("max_pending_shards must be > 0 for AsyncShardWriter")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_pending_shards)
        self._thread = Thread(target=self._run, name="bos-row-shard-writer", daemon=True)
        self._exception: BaseException | None = None
        self._closed = False
        self._thread.start()

    def submit(self, path: str, data: np.ndarray) -> None:
        self._put(ShardWriteRequest(path=path, data=data))

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._put(self._END)
        self._thread.join()
        self._closed = True
        self._raise_if_failed()

    def _put(self, item: object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise self._exception

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._END:
                    return
                if not isinstance(item, ShardWriteRequest):
                    raise TypeError(f"Unexpected shard write item type: {type(item)!r}")
                atomic_write_bin(item.path, item.data)
        except BaseException as exc:
            self._exception = exc

    @property
    def pending_writes(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize


def build_shard_writer(write_queue_shards: int) -> ShardWriter:
    """Return the shard writer implementation selected by the CLI flags."""
    if write_queue_shards == 0:
        return SyncShardWriter()
    return AsyncShardWriter(max_pending_shards=write_queue_shards)


class BosRowShardSink:
    """Collect fixed-width rows into shard files and hand completed buffers to a writer."""

    def __init__(
        self,
        out_dir: str,
        row_tokens: int,
        shard_rows: int,
        val_shards: int,
        writer: ShardWriter,
        on_tokens_written: Callable[[int], None] | None = None,
        start_shard_idx: int = 0,
    ) -> None:
        self._out_dir = out_dir
        self._row_tokens = row_tokens
        self._shard_rows = shard_rows
        self._val_shards = val_shards
        self._writer = writer
        self._on_tokens_written = on_tokens_written

        self._token_buf = np.empty((shard_rows * row_tokens,), dtype=np.uint16)
        self._buf_rows = 0
        self._shard_idx = start_shard_idx
        self.total_rows_written = start_shard_idx * shard_rows
        self.total_tokens_written = self.total_rows_written * row_tokens
        self.run_rows_written = 0
        self.run_tokens_written = 0

    @property
    def num_shards_total(self) -> int:
        return self._shard_idx

    @property
    def current_shard_idx(self) -> int:
        return self._shard_idx

    @property
    def current_shard_fill_rows(self) -> int:
        return self._buf_rows

    @property
    def current_shard_fill_tokens(self) -> int:
        return self._buf_rows * self._row_tokens

    def set_on_tokens_written(self, callback: Callable[[int], None] | None) -> None:
        """Attach or replace the progress callback after construction."""
        self._on_tokens_written = callback

    def append_row(self, row: np.ndarray) -> int | None:
        """Append one fully packed row to the current shard.

        Returns the next shard index when appending this row completed a full
        shard flush. The packer uses that moment to persist exact resume state.
        """
        if row.dtype != np.uint16:
            raise TypeError(f"Expected row dtype uint16, got {row.dtype!r}")
        if row.size != self._row_tokens:
            raise ValueError(f"Expected row with {self._row_tokens} tokens, got {row.size}")

        start = self._buf_rows * self._row_tokens
        self._token_buf[start : start + self._row_tokens] = row
        self._buf_rows += 1

        if self._on_tokens_written is not None:
            self._on_tokens_written(self._row_tokens)

        if self._buf_rows == self._shard_rows:
            self._flush_full_shard()
            return self._shard_idx
        return None

    def finalize(self) -> None:
        """Flush the trailing partial shard, if any rows remain."""
        if self._buf_rows == 0:
            return

        path = self._shard_path(self._shard_idx)
        end = self._buf_rows * self._row_tokens
        self._writer.submit(path, self._token_buf[:end].copy())
        self.total_rows_written += self._buf_rows
        self.total_tokens_written += end
        self.run_rows_written += self._buf_rows
        self.run_tokens_written += end
        self._shard_idx += 1
        self._buf_rows = 0

    def _flush_full_shard(self) -> None:
        path = self._shard_path(self._shard_idx)
        full_buffer = self._token_buf

        # Swap buffers before submitting the completed shard so row packing can
        # continue while the previous shard is still being written.
        self._token_buf = np.empty((self._shard_rows * self._row_tokens,), dtype=np.uint16)
        self._buf_rows = 0

        self._writer.submit(path, full_buffer)
        shard_tokens = self._shard_rows * self._row_tokens
        self.total_rows_written += self._shard_rows
        self.total_tokens_written += shard_tokens
        self.run_rows_written += self._shard_rows
        self.run_tokens_written += shard_tokens
        self._shard_idx += 1

    def _shard_split(self, shard_idx: int) -> str:
        return "val" if shard_idx < self._val_shards else "train"

    def _shard_path(self, shard_idx: int) -> str:
        return os.path.join(self._out_dir, f"{self._shard_split(shard_idx)}_{shard_idx:06d}.bin")


class ProgressMonitor:
    """Own tqdm updates and live diagnostics without polluting packing logic.

    Useful runtime signals here are the ones that tell you whether the job is
    healthy and which stage is limiting throughput:
    - `rss`: current process RAM usage
    - `disk`: free space on the output filesystem
    - `buf`: how full the best-fit document buffer is
    - `pre`: prefetched text batches waiting for tokenization
    - `wr`: completed shard buffers waiting for disk writes
    - `docs`: source documents already consumed from the dataset
    - `crop`: current crop fraction, since packing quality matters for this format
    - `shard`: the shard index currently being filled
    """

    def __init__(
        self,
        progress,
        out_dir: str,
        buffer_docs: int,
        prefetcher: TextBatchPrefetcher,
        writer: ShardWriter,
        sink: BosRowShardSink,
        stats: PackingStats,
        metrics: Sequence[str],
        refresh_interval_seconds: float = 1.0,
        resume_from_shard: int = 0,
        active_resume_state_shard_idx: int | None = None,
    ) -> None:
        self._progress = progress
        self._out_dir = out_dir
        self._buffer_docs = buffer_docs
        self._prefetcher = prefetcher
        self._writer = writer
        self._sink = sink
        self._stats = stats
        self._metrics = tuple(metrics)
        self._refresh_interval_seconds = refresh_interval_seconds
        self._doc_buffer_len = 0
        self._last_refresh_time = 0.0
        self._resume_from_shard = int(resume_from_shard)
        self._active_resume_state_shard_idx = active_resume_state_shard_idx
        self._source_docs_seen_total = 0
        self._source_replay_target_docs = int(stats.docs_processed)
        self._source_replay_complete = self._resume_from_shard == 0
        self._phase = "starting"
        self._lock = Lock()

    def on_tokens_written(self, tokens_written: int) -> None:
        with self._lock:
            self._progress.update(tokens_written)
            self._refresh_if_needed_locked()

    def set_doc_buffer_len(self, doc_buffer_len: int) -> None:
        with self._lock:
            self._doc_buffer_len = doc_buffer_len
            self._refresh_if_needed_locked()

    def set_source_progress(self, docs_seen_total: int, replay_target_docs: int, replay_complete: bool) -> None:
        with self._lock:
            self._source_docs_seen_total = int(docs_seen_total)
            self._source_replay_target_docs = int(replay_target_docs)
            self._source_replay_complete = bool(replay_complete)
            self._refresh_if_needed_locked(redraw=True)

    def on_shard_checkpoint_saved(self, next_shard_idx: int) -> None:
        with self._lock:
            self._active_resume_state_shard_idx = int(next_shard_idx)
            self._refresh_if_needed_locked(force=True, redraw=True)

    def mark_completed(self) -> None:
        with self._lock:
            self._phase = "completed"
            self._refresh_if_needed_locked(force=True, redraw=True)

    def mark_failed(self) -> None:
        with self._lock:
            self._phase = "failed"
            self._refresh_if_needed_locked(force=True, redraw=True)

    def emit_startup_summary(self) -> None:
        with self._lock:
            snapshot = self._build_status_snapshot()
            self._refresh_if_needed_locked(force=True, redraw=True)
        print(f"[status] {snapshot.explanation}")
        print(f"[status] live status file: {self._out_dir}/packing_status.json")

    def refresh_if_needed(self, force: bool = False, redraw: bool = False) -> None:
        with self._lock:
            self._refresh_if_needed_locked(force=force, redraw=redraw)

    def _refresh_if_needed_locked(self, force: bool = False, redraw: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_refresh_time) < self._refresh_interval_seconds:
            return
        self._last_refresh_time = now
        snapshot = self._build_status_snapshot()
        write_packing_status(self._out_dir, snapshot)
        self._progress.set_postfix_str(self._build_postfix(snapshot), refresh=redraw)

    def close(self) -> None:
        with self._lock:
            self._refresh_if_needed_locked(force=True)
            self._progress.close()

    def _build_status_snapshot(self) -> PackingStatusSnapshot:
        if self._phase in ("completed", "failed"):
            phase = self._phase
        elif self._resume_from_shard > 0 and not self._source_replay_complete:
            phase = "resume_replay"
        else:
            phase = "packing"

        latest_completed_shard_idx = (
            int(self._sink.current_shard_idx - 1)
            if self._sink.current_shard_idx > 0
            else None
        )
        replay_remaining_docs = max(0, self._source_replay_target_docs - self._source_docs_seen_total)
        snapshot = PackingStatusSnapshot(
            phase=phase,
            current_shard_idx=int(self._sink.current_shard_idx),
            latest_completed_shard_idx=latest_completed_shard_idx,
            resume_from_shard=self._resume_from_shard,
            active_resume_state_shard_idx=self._active_resume_state_shard_idx,
            docs_processed=int(self._stats.docs_processed),
            source_docs_seen_total=int(self._source_docs_seen_total),
            source_replay_target_docs=int(self._source_replay_target_docs),
            source_replay_remaining_docs=int(replay_remaining_docs),
            current_shard_fill_rows=int(self._sink.current_shard_fill_rows),
            current_shard_fill_tokens=int(self._sink.current_shard_fill_tokens),
            tokens_written_total=int(self._sink.total_tokens_written),
            tokens_cropped_total=int(self._stats.tokens_cropped_total),
            doc_buffer_len=int(self._doc_buffer_len),
            buffer_docs_capacity=int(self._buffer_docs),
            explanation="",
            updated_utc=datetime.now(timezone.utc).isoformat(),
        )
        return PackingStatusSnapshot(
            **{
                **snapshot.to_dict(),
                "explanation": explain_packing_status(snapshot),
            }
        )

    def _build_postfix(self, snapshot: PackingStatusSnapshot) -> str:
        if not self._metrics:
            return f"phase={snapshot.phase}"

        accounted_tokens = self._sink.total_tokens_written + self._sink.current_shard_fill_tokens
        total_accounted = accounted_tokens + self._stats.tokens_cropped_total
        crop_fraction = (
            float(self._stats.tokens_cropped_total) / float(total_accounted)
            if total_accounted > 0
            else 0.0
        )

        parts: list[str] = []
        if snapshot.phase == "resume_replay":
            parts.append(
                f"replay={format_count(snapshot.source_docs_seen_total)}/{format_count(snapshot.source_replay_target_docs)}"
            )
            parts.append(f"remain={format_count(snapshot.source_replay_remaining_docs)}")
            parts.append(f"target={snapshot.current_shard_idx}")
        for metric in self._metrics:
            if metric == "shard":
                parts.append(f"shard={self._sink.current_shard_idx}")
            elif metric == "buf":
                parts.append(f"buf={self._doc_buffer_len}/{self._buffer_docs}")
            elif metric == "pre":
                parts.append(f"pre={self._prefetcher.pending_batches}/{self._prefetcher.queue_capacity}")
            elif metric == "wr":
                writer_cap = self._writer.queue_capacity
                writer_status = f"{self._writer.pending_writes}/{writer_cap}" if writer_cap > 0 else "sync"
                parts.append(f"wr={writer_status}")
            elif metric == "docs":
                parts.append(f"docs={format_count(self._stats.docs_processed)}")
            elif metric == "crop":
                parts.append(f"crop={100.0 * crop_fraction:.2f}%")
            elif metric == "rss":
                parts.append(f"rss={format_bytes(get_process_rss_bytes())}")
            elif metric == "disk":
                parts.append(f"disk={format_bytes(shutil.disk_usage(self._out_dir).free)} free")
        return " ".join(parts)


def iter_tokenized_documents(
    prefetcher: TextBatchPrefetcher,
    tokenizer,
    bos_token_id: int,
    docs_consumed_start: int,
) -> Iterator[TokenizedDocument]:
    """Yield tokenized docs paired with the exact cumulative source-doc count.

    Resume checkpoints are written at shard boundaries, which may occur in the
    middle of a tokenizer batch. Tracking the cumulative document count per
    document keeps those checkpoints exact.
    """
    docs_consumed = docs_consumed_start
    for batch in prefetcher:
        batch_input_ids = tokenize_batch(tokenizer, batch.texts)
        for token_ids in batch_input_ids:
            docs_consumed += 1
            yield TokenizedDocument(
                docs_consumed=docs_consumed,
                tokens=document_tokens_to_uint16(token_ids, bos_token_id),
            )


def find_largest_that_fits(doc_buffer: list[np.ndarray], remaining: int) -> int:
    """Pick the largest buffered document that still fits in the current row tail."""
    best_idx = -1
    best_len = 0
    for i, doc in enumerate(doc_buffer):
        n = int(doc.size)
        if n <= remaining and n > best_len:
            best_idx = i
            best_len = n
    return best_idx


def find_shortest(doc_buffer: list[np.ndarray]) -> int:
    """Fallback choice when no buffered document fits in the remaining row space."""
    shortest_idx = 0
    shortest_len = int(doc_buffer[0].size)
    for i in range(1, len(doc_buffer)):
        n = int(doc_buffer[i].size)
        if n < shortest_len:
            shortest_idx = i
            shortest_len = n
    return shortest_idx


def pack_rows(
    token_docs: Iterator[TokenizedDocument],
    row_tokens: int,
    buffer_docs: int,
    sink: BosRowShardSink,
    stats: PackingStats,
    initial_doc_buffer: Sequence[np.ndarray] | None = None,
    on_shard_completed: Callable[[PackingResumeState], None] | None = None,
    on_buffer_state_changed: Callable[[int], None] | None = None,
) -> None:
    """Pack tokenized docs into fixed-width rows with best-fit plus crop fallback.

    Semantics match the original implementation:
    - rows are filled from a candidate buffer of tokenized documents
    - if something fits, use the largest fitting document
    - otherwise crop the shortest buffered document to finish the row
    - drop the final incomplete row instead of padding it
    """

    doc_buffer = [np.asarray(doc, dtype=np.uint16).copy() for doc in initial_doc_buffer or ()]
    if len(doc_buffer) > buffer_docs:
        raise ValueError(
            f"Resume state restored {len(doc_buffer)} buffered docs, which exceeds --buffer_docs={buffer_docs}."
        )

    docs_exhausted = False
    docs_consumed = stats.docs_processed

    def emit_buffer_state() -> None:
        if on_buffer_state_changed is not None:
            on_buffer_state_changed(len(doc_buffer))

    def refill_buffer() -> None:
        nonlocal docs_consumed, docs_exhausted
        while len(doc_buffer) < buffer_docs and not docs_exhausted:
            try:
                doc = next(token_docs)
            except StopIteration:
                docs_exhausted = True
                break
            doc_buffer.append(doc.tokens)
            docs_consumed = doc.docs_consumed
            stats.docs_processed = docs_consumed
        emit_buffer_state()

    emit_buffer_state()
    refill_buffer()
    while True:
        if not doc_buffer:
            break

        row = np.empty((row_tokens,), dtype=np.uint16)
        pos = 0

        while pos < row_tokens:
            if not doc_buffer:
                refill_buffer()
                if not doc_buffer:
                    break  # Preserve the original behavior: drop the final partial row.

            remaining = row_tokens - pos
            best_idx = find_largest_that_fits(doc_buffer, remaining)

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                n = int(doc.size)
                row[pos : pos + n] = doc
                pos += n
            else:
                shortest_idx = find_shortest(doc_buffer)
                doc = doc_buffer.pop(shortest_idx)
                n = int(doc.size)
                row[pos : pos + remaining] = doc[:remaining]
                stats.tokens_cropped_total += n - remaining
                pos += remaining

            if len(doc_buffer) < buffer_docs:
                refill_buffer()

        if pos < row_tokens:
            break

        next_shard_idx = sink.append_row(row)
        if next_shard_idx is not None and on_shard_completed is not None:
            on_shard_completed(
                PackingResumeState(
                    next_shard_idx=next_shard_idx,
                    docs_consumed=docs_consumed,
                    tokens_cropped_total=stats.tokens_cropped_total,
                    buffered_docs=[doc.copy() for doc in doc_buffer],
                )
            )


def build_meta(
    args: argparse.Namespace,
    vocab_size: int,
    bos_token_id: int,
    row_tokens: int,
    shard_rows: int,
    elapsed_seconds: float,
    stats: PackingStats,
    sink: BosRowShardSink,
) -> dict:
    """Collect the run summary for debugging and downstream loaders."""
    total_output_tokens = stats.tokens_cropped_total + sink.total_tokens_written
    crop_fraction = (
        float(stats.tokens_cropped_total) / float(total_output_tokens)
        if total_output_tokens > 0
        else 0.0
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "data_files": args.data_files,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "use_fast": True,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "format": "bos_row_packed_bestfit",
        "seq_len": int(args.seq_len),
        "row_tokens": int(row_tokens),
        "packing_algo": "largest_fit_then_shortest_crop",
        "batch_docs": int(args.batch_docs),
        "buffer_docs": int(args.buffer_docs),
        "prefetch_batches": int(args.prefetch_batches),
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(bos_token_id),
        "bos_is_eos": True,
        "doc_format": "[BOS] + gpt2_bpe(text)",
        "shard_rows": int(shard_rows),
        "shard_tokens": int(shard_rows * row_tokens),
        "write_queue_shards": int(args.write_queue_shards),
        "progress_metrics": list(parse_progress_metrics(args.progress_metrics)),
        "progress_refresh_secs": float(args.progress_refresh_secs),
        "val_shards": int(args.val_shards),
        "resume_from_shard": int(args.resume_from_shard),
        "num_shards_total": int(sink.num_shards_total),
        "rows_written_total": int(sink.total_rows_written),
        "tokens_written_total": int(sink.total_tokens_written),
        "run_rows_written": int(sink.run_rows_written),
        "run_tokens_written": int(sink.run_tokens_written),
        "tokens_cropped_total": int(stats.tokens_cropped_total),
        "crop_fraction": float(crop_fraction),
        "docs_processed": int(stats.docs_processed),
        "elapsed_seconds": float(elapsed_seconds),
        "throughput_tokens_per_sec": float(sink.run_tokens_written / max(elapsed_seconds, 1e-9)),
        "notes": (
            "Rows are fixed length (seq_len+1), each row starts with BOS, and are packed with "
            "largest-fit then shortest-doc crop fallback."
        ),
    }


def resolve_resume_state(
    args: argparse.Namespace,
    resume_store: PackingResumeStateStore,
) -> PackingResumeState:
    """Load and validate the requested resume point, or return a fresh start state."""
    if args.resume_from_shard == 0:
        return PackingResumeState(
            next_shard_idx=0,
            docs_consumed=0,
            tokens_cropped_total=0,
            buffered_docs=[],
        )

    state = resume_store.load(args.resume_from_shard)
    resume_store.cleanup_from_shard(
        args.resume_from_shard,
        keep_state_shard_idx=args.resume_from_shard,
    )
    return state


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the full BOS-row pretokenization job and return the generated metadata."""
    os.makedirs(args.out_dir, exist_ok=True)
    progress_metrics = parse_progress_metrics(args.progress_metrics)

    tokenizer, bos_token_id, vocab_size = load_tokenizer(args.tokenizer)
    row_tokens = args.seq_len + 1
    auto_rows = max(1, 100_000_000 // row_tokens)
    shard_rows = auto_rows if args.shard_rows <= 0 else int(args.shard_rows)
    resume_store = PackingResumeStateStore(
        out_dir=args.out_dir,
        signature=ResumeConfigSignature.from_runtime(args, row_tokens=row_tokens, shard_rows=shard_rows),
    )
    resume_state = resolve_resume_state(args, resume_store)
    if args.max_docs > 0 and resume_state.docs_consumed > args.max_docs:
        raise ValueError(
            f"--max_docs={args.max_docs} is smaller than the resume point docs_consumed={resume_state.docs_consumed}."
        )

    dataset = load_dataset_source(args)

    writer = build_shard_writer(args.write_queue_shards)
    stats = PackingStats(
        docs_processed=resume_state.docs_consumed,
        tokens_cropped_total=resume_state.tokens_cropped_total,
    )
    progress = tqdm(unit="tok", desc="packing", dynamic_ncols=True)
    sink = BosRowShardSink(
        out_dir=args.out_dir,
        row_tokens=row_tokens,
        shard_rows=shard_rows,
        val_shards=args.val_shards,
        writer=writer,
        start_shard_idx=resume_state.next_shard_idx,
    )
    prefetcher = TextBatchPrefetcher(
        dataset=dataset,
        text_field=args.text_field,
        batch_docs=args.batch_docs,
        max_docs=args.max_docs,
        prefetch_batches=args.prefetch_batches,
        skip_docs=resume_state.docs_consumed,
        progress_callback=None,
        progress_report_interval_seconds=max(1.0, min(args.progress_refresh_secs, 5.0)),
    )
    monitor = ProgressMonitor(
        progress=progress,
        out_dir=args.out_dir,
        buffer_docs=args.buffer_docs,
        prefetcher=prefetcher,
        writer=writer,
        sink=sink,
        stats=stats,
        metrics=progress_metrics,
        refresh_interval_seconds=args.progress_refresh_secs,
        resume_from_shard=args.resume_from_shard,
        active_resume_state_shard_idx=(args.resume_from_shard if args.resume_from_shard > 0 else None),
    )
    sink.set_on_tokens_written(monitor.on_tokens_written)
    prefetcher.set_progress_callback(monitor.set_source_progress)
    monitor.emit_startup_summary()

    start_time = time.time()
    main_error: BaseException | None = None
    main_error_traceback = None

    try:
        token_docs = iter_tokenized_documents(
            prefetcher=prefetcher,
            tokenizer=tokenizer,
            bos_token_id=bos_token_id,
            docs_consumed_start=resume_state.docs_consumed,
        )

        def on_shard_completed(state: PackingResumeState) -> None:
            resume_store.save(state)
            monitor.on_shard_checkpoint_saved(state.next_shard_idx)

        pack_rows(
            token_docs=token_docs,
            row_tokens=row_tokens,
            buffer_docs=args.buffer_docs,
            sink=sink,
            stats=stats,
            initial_doc_buffer=resume_state.buffered_docs,
            on_shard_completed=on_shard_completed,
            on_buffer_state_changed=monitor.set_doc_buffer_len,
        )
        sink.finalize()
    except BaseException as exc:
        main_error = exc
        main_error_traceback = exc.__traceback__
    finally:
        prefetcher.close(cancel=main_error is not None)
        writer_error: BaseException | None = None
        try:
            writer.close()
        except BaseException as exc:
            writer_error = exc
        finally:
            if main_error is None and writer_error is None:
                monitor.mark_completed()
            else:
                monitor.mark_failed()
            monitor.close()
        if writer_error is not None and main_error is None:
            raise writer_error

    if main_error is not None:
        raise main_error.with_traceback(main_error_traceback)

    elapsed_seconds = time.time() - start_time
    meta = build_meta(
        args=args,
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        row_tokens=row_tokens,
        shard_rows=shard_rows,
        elapsed_seconds=elapsed_seconds,
        stats=stats,
        sink=sink,
    )
    meta_path = os.path.join(args.out_dir, "meta.json")
    atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))
    return meta


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    meta = run_pipeline(args)

    print("\nDone.")
    print(f"  out_dir: {args.out_dir}")
    print(f"  shards:  {meta['num_shards_total']} (val_shards={args.val_shards})")
    print(f"  rows:    {meta['rows_written_total']:,}")
    print(f"  tokens:  {meta['tokens_written_total']:,}")
    print(f"  cropped: {meta['tokens_cropped_total']:,} ({100.0 * meta['crop_fraction']:.2f}%)")
    print(f"  docs:    {meta['docs_processed']:,}")
    print(f"  speed:   {meta['throughput_tokens_per_sec']:.0f} tok/s")


if __name__ == "__main__":
    main()
