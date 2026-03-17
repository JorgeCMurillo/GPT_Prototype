"""Pretokenize FineWeb/FineWeb-Edu into raw GPT-2 token shards.

Pipeline overview:
1. A background prefetcher reads streamed dataset examples into text batches.
2. The main thread tokenizes each batch with the fast GPT-2 tokenizer.
3. A shard sink packs token IDs into fixed-size uint16 shard buffers.
4. A shard writer persists completed buffers, optionally on a background thread.

The output format stays compatible with the original serial version:
uint16 `train_*.bin` / `val_*.bin` shards plus a `meta.json`.

Resume support:
- The script records shard-boundary resume points under `out_dir/.resume_state/`.
- `--resume_from_shard N` restarts from the beginning of shard `N`.
- Resume works even if a document crossed the shard boundary, because the script
  stores the pending token suffix needed to reconstruct the exact stream.

CLI knobs:
- `--max_shards`: Optional output cap. If set above zero, stop after writing
  this many shard files total. The limit is enforced at shard boundaries so
  the existing resume snapshots remain exact.
"""

import argparse
import json
import os
import queue
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Callable, Iterator, Sequence

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically replace `path` with `data` so interrupted runs do not leave torn files."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_bin(path: str, arr: np.ndarray) -> None:
    """Atomically write a uint16 token shard to disk."""
    assert arr.dtype == np.uint16
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        arr.tofile(f)
    os.replace(tmp, path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI in one place so `main()` only orchestrates the run."""
    parser = argparse.ArgumentParser(
        description="Pretokenize FineWeb/FineWeb-Edu to raw uint16 .bin shards (GPT-2 tokenizer)"
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
    parser.add_argument("--split", type=str, default="train", help="Dataset split (usually 'train')")
    parser.add_argument("--text_field", type=str, default="text", help="Field containing raw text")
    parser.add_argument("--out_dir", type=str, default="fineweb_edu_10B", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    parser.add_argument(
        "--batch_docs",
        type=int,
        default=256,
        help="Docs per tokenizer batch; larger batches improve throughput if RAM allows",
    )
    parser.add_argument(
        "--shard_tokens",
        type=int,
        default=100_000_000,
        help="Tokens per shard (uint16). 100M tokens is about 200 MB on disk",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="If >0, stop after writing this many shard files total. 0 means no limit.",
    )
    parser.add_argument("--val_shards", type=int, default=1, help="Number of initial shards to label as 'val'")
    parser.add_argument("--max_docs", type=int, default=0, help="If >0, stop after this many docs (debug)")
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
        help="How many text batches to prefetch from the streaming dataset ahead of tokenization",
    )
    parser.add_argument(
        "--write_queue_shards",
        type=int,
        default=1,
        help="How many completed shard buffers may wait for the background writer; 0 disables async writes",
    )
    parser.add_argument(
        "--resume_from_shard",
        type=int,
        default=0,
        help="Restart from the beginning of this shard index using state saved in out_dir/.resume_state",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on impossible settings before expensive setup starts."""
    if args.batch_docs <= 0:
        raise ValueError("--batch_docs must be > 0")
    if args.shard_tokens <= 0:
        raise ValueError("--shard_tokens must be > 0")
    if args.max_shards < 0:
        raise ValueError("--max_shards must be >= 0")
    if args.val_shards < 0:
        raise ValueError("--val_shards must be >= 0")
    if args.max_docs < 0:
        raise ValueError("--max_docs must be >= 0")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch_batches must be > 0")
    if args.write_queue_shards < 0:
        raise ValueError("--write_queue_shards must be >= 0")
    if args.resume_from_shard < 0:
        raise ValueError("--resume_from_shard must be >= 0")


def load_tokenizer(tokenizer_name: str):
    """Load the tokenizer once and normalize the BOS/EOS handling used by GPT-2."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; GPT-2 should have one (50256).")
    bos_token_id = int(eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    return tokenizer, bos_token_id, vocab_size


def load_streaming_dataset(args: argparse.Namespace):
    """Return the iterable streaming dataset so raw text is not materialized in RAM."""
    return load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)


def tokenize_batch(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Tokenize one text batch while leaving BOS insertion to the shard packer."""
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
        raise ValueError("Token id out of uint16 range; are you sure this is GPT-2 tokenization?")

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


@dataclass(frozen=True)
class ResumeState:
    """Exact restart information for the beginning of a shard."""

    next_shard_idx: int
    docs_consumed: int
    pending_tokens: np.ndarray


@dataclass(frozen=True)
class ResumeConfigSignature:
    """Minimal run configuration that must match when resuming."""

    dataset: str
    config: str
    split: str
    text_field: str
    tokenizer: str
    shard_tokens: int
    val_shards: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ResumeConfigSignature":
        return cls(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            text_field=args.text_field,
            tokenizer=args.tokenizer,
            shard_tokens=int(args.shard_tokens),
            val_shards=int(args.val_shards),
        )


@dataclass(frozen=True)
class TokenizedDocument:
    """One tokenized document paired with the cumulative source-doc count."""

    docs_consumed: int
    tokens: np.ndarray


class ResumeStateStore:
    """Persist resume points so a rerun can restart from a shard boundary exactly.

    The tricky case is when a document spans shards. In that case the next shard
    starts with the still-unwritten suffix of the current document, so the store
    records both the dataset doc count to skip and the pending uint16 token tail.
    """

    _SHARD_RE = re.compile(r"^(train|val)_(\d{6})\.bin$")
    _STATE_JSON_RE = re.compile(r"^state_(\d{6})\.json$")
    _STATE_PENDING_RE = re.compile(r"^state_(\d{6})\.pending\.bin$")

    def __init__(self, out_dir: str, args: argparse.Namespace) -> None:
        self._out_dir = out_dir
        self._state_dir = os.path.join(out_dir, ".resume_state")
        self._signature = ResumeConfigSignature.from_args(args)
        os.makedirs(self._state_dir, exist_ok=True)

    def save_resume_state(self, state: ResumeState) -> None:
        if state.pending_tokens.dtype != np.uint16:
            raise TypeError(f"Expected pending_tokens dtype uint16, got {state.pending_tokens.dtype!r}")

        pending_path = self._pending_path(state.next_shard_idx)
        if state.pending_tokens.size > 0:
            atomic_write_bin(pending_path, state.pending_tokens)
        elif os.path.exists(pending_path):
            os.remove(pending_path)

        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "next_shard_idx": int(state.next_shard_idx),
            "docs_consumed": int(state.docs_consumed),
            "pending_tokens_path": os.path.basename(pending_path) if state.pending_tokens.size > 0 else None,
            "pending_tokens_count": int(state.pending_tokens.size),
            "dataset": self._signature.dataset,
            "config": self._signature.config,
            "split": self._signature.split,
            "text_field": self._signature.text_field,
            "tokenizer": self._signature.tokenizer,
            "shard_tokens": int(self._signature.shard_tokens),
            "val_shards": int(self._signature.val_shards),
        }
        state_path = self._state_path(state.next_shard_idx)
        atomic_write_bytes(state_path, json.dumps(payload, indent=2).encode("utf-8"))

    def load_resume_state(self, next_shard_idx: int) -> ResumeState:
        state_path = self._state_path(next_shard_idx)
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"No resume state found for shard {next_shard_idx} at '{state_path}'. "
                "Resume state is only available for runs created with the resume-aware script."
            )

        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._validate_payload(payload, next_shard_idx)

        pending_tokens_path = payload.get("pending_tokens_path")
        if pending_tokens_path:
            pending_path = os.path.join(self._state_dir, pending_tokens_path)
            if not os.path.exists(pending_path):
                raise FileNotFoundError(
                    f"Resume state for shard {next_shard_idx} is missing its pending token file: {pending_path}"
                )
            pending_tokens = np.fromfile(pending_path, dtype=np.uint16)
        else:
            pending_tokens = np.empty((0,), dtype=np.uint16)

        expected_pending = int(payload.get("pending_tokens_count", pending_tokens.size))
        if pending_tokens.size != expected_pending:
            raise ValueError(
                f"Resume state for shard {next_shard_idx} expected {expected_pending} pending tokens, "
                f"found {pending_tokens.size}."
            )

        return ResumeState(
            next_shard_idx=int(payload["next_shard_idx"]),
            docs_consumed=int(payload["docs_consumed"]),
            pending_tokens=pending_tokens,
        )

    def cleanup_from_shard(self, start_shard_idx: int) -> None:
        """Remove outputs/state from `start_shard_idx` onward before rewriting them."""
        for name in os.listdir(self._out_dir):
            match = self._SHARD_RE.match(name)
            if not match:
                continue
            shard_idx = int(match.group(2))
            if shard_idx >= start_shard_idx:
                os.remove(os.path.join(self._out_dir, name))

        for name in os.listdir(self._state_dir):
            for pattern in (self._STATE_JSON_RE, self._STATE_PENDING_RE):
                match = pattern.match(name)
                if match:
                    shard_idx = int(match.group(1))
                    if shard_idx >= start_shard_idx:
                        os.remove(os.path.join(self._state_dir, name))
                    break

    def _validate_payload(self, payload: dict, expected_next_shard_idx: int) -> None:
        expected = {
            "dataset": self._signature.dataset,
            "config": self._signature.config,
            "split": self._signature.split,
            "text_field": self._signature.text_field,
            "tokenizer": self._signature.tokenizer,
            "shard_tokens": int(self._signature.shard_tokens),
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

    def _state_path(self, next_shard_idx: int) -> str:
        return os.path.join(self._state_dir, f"state_{next_shard_idx:06d}.json")

    def _pending_path(self, next_shard_idx: int) -> str:
        return os.path.join(self._state_dir, f"state_{next_shard_idx:06d}.pending.bin")


class TextBatchPrefetcher:
    """Read the streaming dataset on a background thread and emit fixed-size text batches.

    The dataset iterator stays on a single thread. The consumer only sees ordered
    `TextBatch` objects, so tokenization behavior remains unchanged.
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
    ) -> None:
        self._dataset = dataset
        self._text_field = text_field
        self._batch_docs = batch_docs
        self._max_docs = max_docs
        self._skip_docs = skip_docs
        self._queue: queue.Queue[object] = queue.Queue(maxsize=prefetch_batches)
        self._cancel_event = Event()
        self._thread = Thread(target=self._run, name="fineweb-prefetch", daemon=True)
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

    def _run(self) -> None:
        try:
            batch_texts: list[str] = []
            docs_seen_total = 0

            for example in self._dataset:
                docs_seen_total += 1

                # Resume works by replaying the stream up to the saved doc count.
                # This is exact but still linear in the number of skipped docs.
                if docs_seen_total <= self._skip_docs:
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

                if reached_doc_limit:
                    break

            if batch_texts:
                self._put(TextBatch(texts=batch_texts))
        except BaseException as exc:
            self._exception = exc
        finally:
            self._put(self._END)


class ShardWriter(ABC):
    """Small abstraction so packing code does not depend on sync vs async writes."""

    @abstractmethod
    def submit(self, path: str, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SyncShardWriter(ShardWriter):
    """Reference implementation: write shards on the caller's thread."""

    def submit(self, path: str, data: np.ndarray) -> None:
        atomic_write_bin(path, data)

    def close(self) -> None:
        return


class AsyncShardWriter(ShardWriter):
    """Write completed shard buffers on a dedicated thread.

    The main thread still owns ordering. It only hands off already completed
    buffers, so the background writer can never reorder shard indices.
    """

    _END = object()

    def __init__(self, max_pending_shards: int) -> None:
        if max_pending_shards <= 0:
            raise ValueError("max_pending_shards must be > 0 for AsyncShardWriter")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_pending_shards)
        self._thread = Thread(target=self._run, name="fineweb-shard-writer", daemon=True)
        self._exception: BaseException | None = None
        self._closed = False
        self._thread.start()

    def submit(self, path: str, data: np.ndarray) -> None:
        request = ShardWriteRequest(path=path, data=data)
        self._put(request)

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
                request = item
                atomic_write_bin(request.path, request.data)
        except BaseException as exc:
            self._exception = exc


def build_shard_writer(write_queue_shards: int) -> ShardWriter:
    """Return the writer implementation chosen by the CLI flags."""
    if write_queue_shards == 0:
        return SyncShardWriter()
    return AsyncShardWriter(max_pending_shards=write_queue_shards)


def iter_tokenized_documents(
    prefetcher: TextBatchPrefetcher,
    tokenizer,
    bos_token_id: int,
    docs_consumed_start: int,
) -> Iterator[TokenizedDocument]:
    """Yield tokenized documents paired with the exact cumulative doc count.

    Resume boundaries depend on knowing how many source documents have been fully
    consumed at the moment a shard flush happens. Tracking that per document
    keeps the saved resume points exact even when a flush occurs mid-batch.
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


class TokenShardSink:
    """Pack per-document token ids into fixed-size raw shard files.

    This class has one job: maintain shard state and hand completed buffers to a
    writer. It does not know where text came from or how tokenization happened.
    """

    def __init__(
        self,
        out_dir: str,
        shard_tokens: int,
        max_shards: int | None,
        val_shards: int,
        writer: ShardWriter,
        on_tokens_written: Callable[[int], None] | None = None,
        on_shard_completed: Callable[[ResumeState], None] | None = None,
        start_shard_idx: int = 0,
    ) -> None:
        self._out_dir = out_dir
        self._shard_tokens = shard_tokens
        self._max_shards = max_shards
        self._val_shards = val_shards
        self._writer = writer
        self._on_tokens_written = on_tokens_written
        self._on_shard_completed = on_shard_completed

        self._buf = np.empty((shard_tokens,), dtype=np.uint16)
        self._buf_len = 0
        self._shard_idx = start_shard_idx
        self.total_tokens_written = start_shard_idx * shard_tokens
        self.run_tokens_written = 0

    @property
    def num_shards_total(self) -> int:
        return self._shard_idx

    def append_document(self, token_ids: Sequence[int], bos_token_id: int, docs_consumed: int) -> None:
        """Append one document to the token stream, inserting BOS exactly once."""
        tokens = document_tokens_to_uint16(token_ids, bos_token_id)
        self.append_raw_tokens(tokens, docs_consumed=docs_consumed)

    def append_raw_tokens(self, tokens: np.ndarray, docs_consumed: int) -> None:
        """Append already-materialized uint16 tokens without modifying them.

        This is used by resume logic to inject the saved suffix of a document
        that previously crossed a shard boundary.
        """
        if tokens.dtype != np.uint16:
            raise TypeError(f"Expected raw token dtype uint16, got {tokens.dtype!r}")
        self._append_tokens(tokens, docs_consumed=docs_consumed)

    def finalize(self) -> None:
        """Flush the final partial shard, if any, after all documents were processed."""
        if self._buf_len == 0:
            return
        if self._max_shards is not None and self._shard_idx >= self._max_shards:
            return

        path = self._shard_path(self._shard_idx)
        # Partial shards need a trimmed copy because only the populated prefix is valid.
        self._writer.submit(path, self._buf[: self._buf_len].copy())
        self.total_tokens_written += self._buf_len
        self.run_tokens_written += self._buf_len
        self._shard_idx += 1
        self._buf_len = 0

    def _append_tokens(self, tokens: np.ndarray, docs_consumed: int) -> None:
        if self._max_shards is not None and self._shard_idx >= self._max_shards:
            return

        offset = 0
        total = int(tokens.size)

        while offset < total:
            space = self._shard_tokens - self._buf_len
            take = min(space, total - offset)
            self._buf[self._buf_len : self._buf_len + take] = tokens[offset : offset + take]
            self._buf_len += take
            offset += take

            if self._on_tokens_written is not None:
                self._on_tokens_written(take)

            if self._buf_len == self._shard_tokens:
                pending_tokens = tokens[offset:].copy()
                self._flush_full_shard(docs_consumed=docs_consumed, pending_tokens=pending_tokens)
                if self._max_shards is not None and self._shard_idx >= self._max_shards:
                    return

    def _flush_full_shard(self, docs_consumed: int, pending_tokens: np.ndarray) -> None:
        path = self._shard_path(self._shard_idx)
        full_buffer = self._buf

        # Swap buffers before handing work to the writer. This is the key
        # low-risk overlap: tokenization can continue while the previous shard
        # is still being flushed to disk.
        self._buf = np.empty((self._shard_tokens,), dtype=np.uint16)
        self._buf_len = 0

        self._writer.submit(path, full_buffer)
        self.total_tokens_written += self._shard_tokens
        self.run_tokens_written += self._shard_tokens
        self._shard_idx += 1
        if self._on_shard_completed is not None:
            self._on_shard_completed(
                ResumeState(
                    next_shard_idx=self._shard_idx,
                    docs_consumed=docs_consumed,
                    pending_tokens=pending_tokens,
                )
            )

    def _shard_split(self, shard_idx: int) -> str:
        return "val" if shard_idx < self._val_shards else "train"

    def _shard_path(self, shard_idx: int) -> str:
        split = self._shard_split(shard_idx)
        return os.path.join(self._out_dir, f"{split}_{shard_idx:06d}.bin")


def build_meta(
    args: argparse.Namespace,
    vocab_size: int,
    eos_token_id: int,
    docs_processed: int,
    elapsed_seconds: float,
    sink: TokenShardSink,
) -> dict:
    """Collect the run summary that downstream scripts use for inspection/debugging."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "use_fast": True,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "bos_token_id": eos_token_id,
        "eos_token_id": eos_token_id,
        "bos_is_eos": True,
        "doc_format": "[BOS] + gpt2_bpe(text)",
        "shard_tokens": int(args.shard_tokens),
        "max_shards": (None if args.max_shards <= 0 else int(args.max_shards)),
        "val_shards": int(args.val_shards),
        "prefetch_batches": int(args.prefetch_batches),
        "write_queue_shards": int(args.write_queue_shards),
        "resume_from_shard": int(args.resume_from_shard),
        "num_shards_total": int(sink.num_shards_total),
        "tokens_written_total": int(sink.total_tokens_written),
        "run_tokens_written": int(sink.run_tokens_written),
        "docs_processed": int(docs_processed),
        "elapsed_seconds": float(elapsed_seconds),
        "throughput_tokens_per_sec": float(sink.run_tokens_written / max(elapsed_seconds, 1e-9)),
        "notes": "Raw .bin shards contain a single 1D stream of uint16 token IDs; each doc is prepended with BOS (=50256).",
    }


def resolve_resume_state(args: argparse.Namespace, resume_store: ResumeStateStore) -> ResumeState:
    """Load and validate the requested resume point, or return a fresh start state."""
    if args.resume_from_shard == 0:
        return ResumeState(
            next_shard_idx=0,
            docs_consumed=0,
            pending_tokens=np.empty((0,), dtype=np.uint16),
        )

    state = resume_store.load_resume_state(args.resume_from_shard)
    resume_store.cleanup_from_shard(args.resume_from_shard)
    return state


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the full pretokenization job and return the metadata for reporting."""
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer, bos_token_id, vocab_size = load_tokenizer(args.tokenizer)
    resume_store = ResumeStateStore(args.out_dir, args)
    resume_state = resolve_resume_state(args, resume_store)
    if args.max_docs > 0 and resume_state.docs_consumed > args.max_docs:
        raise ValueError(
            f"--max_docs={args.max_docs} is smaller than the resume point docs_consumed={resume_state.docs_consumed}."
        )
    if args.max_shards > 0 and resume_state.next_shard_idx > args.max_shards:
        raise ValueError(
            f"--max_shards={args.max_shards} is smaller than the resume point next_shard_idx={resume_state.next_shard_idx}."
        )
    dataset = load_streaming_dataset(args)
    prefetcher = TextBatchPrefetcher(
        dataset=dataset,
        text_field=args.text_field,
        batch_docs=args.batch_docs,
        max_docs=args.max_docs,
        prefetch_batches=args.prefetch_batches,
        skip_docs=resume_state.docs_consumed,
    )
    writer = build_shard_writer(args.write_queue_shards)
    progress = tqdm(unit="tok", desc="writing", dynamic_ncols=True)
    sink = TokenShardSink(
        out_dir=args.out_dir,
        shard_tokens=args.shard_tokens,
        max_shards=(None if args.max_shards <= 0 else int(args.max_shards)),
        val_shards=args.val_shards,
        writer=writer,
        on_tokens_written=progress.update,
        on_shard_completed=resume_store.save_resume_state,
        start_shard_idx=resume_state.next_shard_idx,
    )

    docs_processed = resume_state.docs_consumed
    start_time = time.time()
    main_error: BaseException | None = None
    main_error_traceback = None

    try:
        if resume_state.pending_tokens.size > 0:
            sink.append_raw_tokens(resume_state.pending_tokens, docs_consumed=resume_state.docs_consumed)

        shard_limit_reached = args.max_shards > 0 and sink.num_shards_total >= args.max_shards
        if not shard_limit_reached:
            for doc in iter_tokenized_documents(
                prefetcher=prefetcher,
                tokenizer=tokenizer,
                bos_token_id=bos_token_id,
                docs_consumed_start=resume_state.docs_consumed,
            ):
                docs_processed = doc.docs_consumed
                sink.append_raw_tokens(doc.tokens, docs_consumed=doc.docs_consumed)
                if args.max_shards > 0 and sink.num_shards_total >= args.max_shards:
                    break

        sink.finalize()
    except BaseException as exc:
        main_error = exc
        main_error_traceback = exc.__traceback__
    finally:
        progress.close()
        prefetcher.close(cancel=main_error is not None)
        try:
            writer.close()
        except BaseException:
            if main_error is None:
                raise

    if main_error is not None:
        raise main_error.with_traceback(main_error_traceback)

    elapsed_seconds = time.time() - start_time
    meta = build_meta(
        args=args,
        vocab_size=vocab_size,
        eos_token_id=bos_token_id,
        docs_processed=docs_processed,
        elapsed_seconds=elapsed_seconds,
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
    print(f"  tokens:  {meta['tokens_written_total']:,}")
    print(f"  docs:    {meta['docs_processed']:,}")
    print(f"  speed:   {meta['throughput_tokens_per_sec']:.0f} tok/s")


if __name__ == "__main__":
    main()
