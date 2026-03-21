"""Exact BOS packed-index artifacts, loaders, and builder helpers.

This module replaces duplicated BOS-row token shards with a compact index over
the existing pretokenized FineWebEdu stream. It preserves the exact
`bos_row_packed_bestfit` semantics:

- split logical documents on BOS boundaries in the raw token stream
- fill fixed-width rows with the largest buffered doc that fits
- otherwise crop the shortest buffered doc to finish the row
- drop the trailing partial row

The stored artifact is an index, not duplicated row tokens. Each packed row is
represented as a sequence of source-shard token slices that can be replayed on
demand by training and attribution code.
"""

from __future__ import annotations

import argparse
import bisect
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


PACKED_INDEX_FORMAT = "bos_row_packed_bestfit_index_v1"
PACKING_ALGO = "largest_fit_then_shortest_crop"
SEGMENT_DTYPE = np.dtype(
    [
        ("source_shard_idx", np.uint32),
        ("source_token_start", np.uint64),
        ("source_token_end", np.uint64),
    ],
    align=False,
)
DOC_LENGTHS_DTYPE = np.uint32
DOC_SEGMENT_COUNTS_DTYPE = np.uint32
ROW_POINTER_DTYPE = np.uint64
DEFAULT_BOS_TOKEN_ID = 50256
_SHARD_NAME_RE = re.compile(r"^(train|val)_(\d{6})\.bin$")
_STATE_JSON_RE = re.compile(r"^state_(\d{6})(?:\.backup)?\.json$")
_STATE_BIN_RE = re.compile(r"^state_(\d{6})(?:\.backup)?\..+\.bin$")


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def atomic_write_array(path: str, arr: np.ndarray) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        np.asarray(arr).tofile(handle)
    os.replace(tmp_path, path)


def _count_tokens(path: str | Path) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"Shard byte size is not divisible by 2: {path}")
    return nbytes // 2


def _natural_sort(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths)


def _split_work(total: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    base = total // num_workers
    rem = total % num_workers
    start = worker_id * base + min(worker_id, rem)
    end = start + base + (1 if worker_id < rem else 0)
    return start, end


def _get_dist_info() -> tuple[int, int]:
    rank = os.environ.get("RANK", None)
    world = os.environ.get("WORLD_SIZE", None)

    if rank is None:
        rank = os.environ.get("ACCELERATE_PROCESS_INDEX", None)
    if world is None:
        world = os.environ.get("ACCELERATE_NUM_PROCESSES", None)
    if world is None:
        world = os.environ.get("ACCELERATE_PROCESS_COUNT", None)

    try:
        resolved_rank = int(rank) if rank is not None else 0
    except ValueError:
        resolved_rank = 0
    try:
        resolved_world = int(world) if world is not None else 1
    except ValueError:
        resolved_world = 1

    return max(0, resolved_rank), max(1, resolved_world)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional_meta(data_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        return _load_json(meta_path)
    return {}


def _ordered_source_shards(data_dir: str | Path) -> tuple[Path, ...]:
    paths = [Path(path) for path in glob.glob(str(Path(data_dir) / "*.bin"))]
    parsed: list[tuple[int, Path]] = []
    for path in paths:
        match = _SHARD_NAME_RE.match(path.name)
        if match is None:
            continue
        parsed.append((int(match.group(2)), path))
    if not parsed:
        raise FileNotFoundError(f"No raw shard files found under {data_dir}")
    parsed.sort(key=lambda item: item[0])
    return tuple(path for _, path in parsed)


def _split_paths(data_dir: str | Path, split: str) -> tuple[Path, ...]:
    return tuple(_natural_sort([Path(path) for path in glob.glob(str(Path(data_dir) / f"{split}_*.bin"))]))


def _fingerprint_from_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return digest.hexdigest()


def _maybe_make_tqdm(*args, **kwargs):
    try:
        from tqdm.auto import tqdm

        return tqdm(*args, **kwargs)
    except Exception:
        return None


def _fingerprint_from_paths_with_progress(paths: Sequence[Path], *, desc: str) -> str:
    progress = _maybe_make_tqdm(
        total=len(paths),
        desc=desc,
        unit="shard",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        digest = hashlib.sha1()
        for path in paths:
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
            if progress is not None:
                progress.update(1)
        return digest.hexdigest()
    finally:
        if progress is not None:
            progress.close()


def fingerprint_source_shards(data_dir: str | Path) -> str:
    return _fingerprint_from_paths(_ordered_source_shards(data_dir))


def fingerprint_packed_index_artifact(data_dir: str | Path) -> str:
    data_path = Path(data_dir)
    meta_path = data_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    digest = hashlib.sha1(meta_path.read_bytes())
    for name in (
        "train.row_ptr.bin",
        "train.segments.bin",
        "train.virtual_shards.jsonl",
        "val.row_ptr.bin",
        "val.segments.bin",
        "val.virtual_shards.jsonl",
    ):
        path = data_path / name
        if not path.exists():
            continue
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return digest.hexdigest()


def _paths_for_split(data_dir: str | Path, split: str) -> dict[str, Path]:
    base = Path(data_dir)
    return {
        "row_ptr": base / f"{split}.row_ptr.bin",
        "segments": base / f"{split}.segments.bin",
        "virtual_shards": base / f"{split}.virtual_shards.jsonl",
    }


def _resolve_source_data_dir(index_dir: str | Path, override: str | None = None) -> Path:
    meta = _load_json(Path(index_dir) / "meta.json")
    raw = override or meta.get("source_data_dir")
    if not raw:
        raise ValueError(
            f"Packed-index artifact {index_dir} does not record source_data_dir and no override was provided."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(index_dir) / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Source shard directory not found: {path}")
    return path


@dataclass(frozen=True)
class SourceSegment:
    source_shard_idx: int
    source_token_start: int
    source_token_end: int

    @property
    def length(self) -> int:
        return int(self.source_token_end - self.source_token_start)


@dataclass(frozen=True)
class SourceDocument:
    docs_consumed: int
    tokens: np.ndarray
    segments: tuple[SourceSegment, ...]

    def copy(self) -> "SourceDocument":
        return SourceDocument(
            docs_consumed=int(self.docs_consumed),
            tokens=np.asarray(self.tokens, dtype=np.uint16).copy(),
            segments=tuple(
                SourceSegment(
                    source_shard_idx=int(seg.source_shard_idx),
                    source_token_start=int(seg.source_token_start),
                    source_token_end=int(seg.source_token_end),
                )
                for seg in self.segments
            ),
        )

    @property
    def length(self) -> int:
        return int(self.tokens.size)


@dataclass(frozen=True)
class SourceCursorState:
    source_shard_idx: int
    source_token_offset: int


@dataclass(frozen=True)
class BuilderResumeSignature:
    source_data_dir: str
    source_shards_fingerprint: str
    seq_len: int
    row_tokens: int
    buffer_docs: int
    virtual_shard_rows: int
    val_shards: int
    bos_token_id: int


@dataclass(frozen=True)
class BuilderResumeState:
    next_virtual_shard_idx: int
    docs_consumed: int
    tokens_cropped_total: int
    buffered_docs: tuple[SourceDocument, ...]
    cursor: SourceCursorState
    output_state: dict[str, dict[str, int]]


@dataclass(frozen=True)
class VirtualShardInfo:
    split: str
    local_shard_idx: int
    global_virtual_shard_idx: int
    shard_path: str
    row_start: int
    row_end: int

    @property
    def row_count(self) -> int:
        return int(self.row_end - self.row_start)


@dataclass
class PackingStats:
    docs_processed: int = 0
    tokens_cropped_total: int = 0


def _segment_length(segment: SourceSegment) -> int:
    return int(segment.source_token_end - segment.source_token_start)


def _segments_length(segments: Sequence[SourceSegment]) -> int:
    return int(sum(_segment_length(segment) for segment in segments))


def _crop_segments(segments: Sequence[SourceSegment], keep_tokens: int) -> tuple[SourceSegment, ...]:
    remaining = int(keep_tokens)
    kept: list[SourceSegment] = []
    for segment in segments:
        if remaining <= 0:
            break
        length = _segment_length(segment)
        if length <= remaining:
            kept.append(segment)
            remaining -= length
            continue
        kept.append(
            SourceSegment(
                source_shard_idx=int(segment.source_shard_idx),
                source_token_start=int(segment.source_token_start),
                source_token_end=int(segment.source_token_start + remaining),
            )
        )
        remaining = 0
        break
    if remaining != 0:
        raise ValueError(
            f"Requested crop of {keep_tokens} token(s), but only {_segments_length(segments)} token(s) are available."
        )
    return tuple(kept)


def _buffer_largest_that_fits(doc_buffer: Sequence[SourceDocument], remaining: int) -> int:
    best_idx = -1
    best_len = 0
    for index, doc in enumerate(doc_buffer):
        length = doc.length
        if length <= remaining and length > best_len:
            best_idx = index
            best_len = length
    return best_idx


def _buffer_shortest(doc_buffer: Sequence[SourceDocument]) -> int:
    best_idx = 0
    best_len = doc_buffer[0].length
    for index in range(1, len(doc_buffer)):
        length = doc_buffer[index].length
        if length < best_len:
            best_len = length
            best_idx = index
    return best_idx


class LengthIndexedDocBuffer:
    """Exact best-fit buffer with faster length-based selection."""

    def __init__(self, docs: Sequence[SourceDocument] | None = None) -> None:
        self._next_id = 0
        self._docs: dict[int, SourceDocument] = {}
        self._length_to_ids: dict[int, deque[int]] = {}
        self._sorted_lengths: list[int] = []
        if docs is not None:
            for doc in docs:
                self.append(doc)

    def __len__(self) -> int:
        return len(self._docs)

    def append(self, doc: SourceDocument) -> None:
        doc_copy = doc.copy()
        doc_id = int(self._next_id)
        self._next_id += 1
        self._docs[doc_id] = doc_copy
        length = int(doc_copy.length)
        bucket = self._length_to_ids.get(length)
        if bucket is None:
            bucket = deque()
            self._length_to_ids[length] = bucket
            bisect.insort(self._sorted_lengths, length)
        bucket.append(doc_id)

    def pop_largest_that_fits(self, remaining: int) -> SourceDocument | None:
        length_idx = bisect.bisect_right(self._sorted_lengths, int(remaining)) - 1
        if length_idx < 0:
            return None
        return self._pop_from_length(self._sorted_lengths[length_idx])

    def pop_shortest(self) -> SourceDocument:
        if not self._sorted_lengths:
            raise IndexError("pop from empty LengthIndexedDocBuffer")
        return self._pop_from_length(self._sorted_lengths[0])

    def snapshot(self) -> tuple[SourceDocument, ...]:
        return tuple(self._docs[doc_id].copy() for doc_id in sorted(self._docs))

    def _pop_from_length(self, length: int) -> SourceDocument:
        bucket = self._length_to_ids[int(length)]
        doc_id = bucket.popleft()
        doc = self._docs.pop(int(doc_id))
        if not bucket:
            del self._length_to_ids[int(length)]
            length_idx = bisect.bisect_left(self._sorted_lengths, int(length))
            if length_idx >= len(self._sorted_lengths) or self._sorted_lengths[length_idx] != int(length):
                raise ValueError(f"Length bucket bookkeeping drifted for length={length}.")
            self._sorted_lengths.pop(length_idx)
        return doc


class RawBOSDocumentStream:
    """Iterate logical BOS documents from a pretokenized raw shard stream."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
        start_state: SourceCursorState | None = None,
        docs_consumed_start: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.shard_paths = _ordered_source_shards(self.data_dir)
        self.bos_token_id = int(bos_token_id)
        self._memmaps: dict[int, np.memmap] = {}
        self._bos_offsets_cache: dict[int, np.ndarray] = {}
        self._cursor_shard_idx = int(start_state.source_shard_idx) if start_state is not None else 0
        self._cursor_token_offset = int(start_state.source_token_offset) if start_state is not None else 0
        self._docs_consumed = int(docs_consumed_start)
        self._validate_cursor()

    def _memmap(self, source_shard_idx: int) -> np.memmap:
        if source_shard_idx not in self._memmaps:
            self._memmaps[source_shard_idx] = np.memmap(
                self.shard_paths[source_shard_idx],
                dtype=np.uint16,
                mode="r",
            )
        return self._memmaps[source_shard_idx]

    def _validate_cursor(self) -> None:
        if self._cursor_shard_idx >= len(self.shard_paths):
            return
        shard_idx = self._cursor_shard_idx
        token_offset = self._cursor_token_offset
        while shard_idx < len(self.shard_paths):
            mm = self._memmap(shard_idx)
            if token_offset >= int(mm.size):
                shard_idx += 1
                token_offset = 0
                continue
            if int(mm[token_offset]) != self.bos_token_id:
                raise ValueError(
                    "Raw BOS document stream must resume at a BOS token boundary. "
                    f"Found token {int(mm[token_offset])} at source shard {shard_idx}, offset {token_offset}."
                )
            return

    def _bos_offsets(self, source_shard_idx: int) -> np.ndarray:
        if source_shard_idx not in self._bos_offsets_cache:
            mm = self._memmap(source_shard_idx)
            self._bos_offsets_cache[source_shard_idx] = np.flatnonzero(mm == self.bos_token_id).astype(
                np.int64,
                copy=False,
            )
        return self._bos_offsets_cache[source_shard_idx]

    def snapshot(self) -> SourceCursorState:
        return SourceCursorState(
            source_shard_idx=int(self._cursor_shard_idx),
            source_token_offset=int(self._cursor_token_offset),
        )

    def __iter__(self) -> "RawBOSDocumentStream":
        return self

    def __next__(self) -> SourceDocument:
        shard_idx = self._cursor_shard_idx
        token_offset = self._cursor_token_offset
        while shard_idx < len(self.shard_paths):
            mm = self._memmap(shard_idx)
            if token_offset >= int(mm.size):
                shard_idx += 1
                token_offset = 0
                continue
            if int(mm[token_offset]) != self.bos_token_id:
                raise ValueError(
                    "Encountered a non-BOS token where the next logical document should start. "
                    f"source_shard_idx={shard_idx}, token_offset={token_offset}, token={int(mm[token_offset])}"
                )
            break
        else:
            raise StopIteration

        token_parts: list[np.ndarray] = []
        segments: list[SourceSegment] = []
        at_doc_start = True

        while shard_idx < len(self.shard_paths):
            mm = self._memmap(shard_idx)
            if token_offset >= int(mm.size):
                shard_idx += 1
                token_offset = 0
                at_doc_start = False
                continue

            bos_offsets = self._bos_offsets(shard_idx)
            if at_doc_start:
                next_hit_idx = int(np.searchsorted(bos_offsets, token_offset, side="right"))
            else:
                next_hit_idx = int(np.searchsorted(bos_offsets, token_offset, side="left"))

            if next_hit_idx < int(bos_offsets.size):
                next_bos_offset = int(bos_offsets[next_hit_idx])
                if next_bos_offset == int(token_offset):
                    self._cursor_shard_idx = int(shard_idx)
                    self._cursor_token_offset = int(token_offset)
                    break
                token_parts.append(np.asarray(mm[token_offset:next_bos_offset], dtype=np.uint16).copy())
                segments.append(
                    SourceSegment(
                        source_shard_idx=int(shard_idx),
                        source_token_start=int(token_offset),
                        source_token_end=int(next_bos_offset),
                    )
                )
                self._cursor_shard_idx = int(shard_idx)
                self._cursor_token_offset = int(next_bos_offset)
                break

            token_parts.append(np.asarray(mm[token_offset:], dtype=np.uint16).copy())
            segments.append(
                SourceSegment(
                    source_shard_idx=int(shard_idx),
                    source_token_start=int(token_offset),
                    source_token_end=int(mm.size),
                )
            )
            shard_idx += 1
            token_offset = 0
            at_doc_start = False
        else:
            self._cursor_shard_idx = len(self.shard_paths)
            self._cursor_token_offset = 0

        if not token_parts:
            raise StopIteration

        self._docs_consumed += 1
        return SourceDocument(
            docs_consumed=int(self._docs_consumed),
            tokens=np.concatenate(token_parts).astype(np.uint16, copy=False),
            segments=tuple(segments),
        )


class BufferedDocCodec:
    """Serialize buffered source documents with both tokens and source segments."""

    @staticmethod
    def write(prefix: str, docs: Sequence[SourceDocument]) -> dict[str, str]:
        lengths = np.asarray([doc.length for doc in docs], dtype=DOC_LENGTHS_DTYPE)
        segment_counts = np.asarray([len(doc.segments) for doc in docs], dtype=DOC_SEGMENT_COUNTS_DTYPE)
        flat_tokens = (
            np.concatenate([np.asarray(doc.tokens, dtype=np.uint16) for doc in docs])
            if docs
            else np.empty((0,), dtype=np.uint16)
        )
        flat_segments = (
            np.asarray(
                [
                    (
                        int(segment.source_shard_idx),
                        int(segment.source_token_start),
                        int(segment.source_token_end),
                    )
                    for doc in docs
                    for segment in doc.segments
                ],
                dtype=SEGMENT_DTYPE,
            )
            if docs
            else np.empty((0,), dtype=SEGMENT_DTYPE)
        )

        lengths_path = prefix + ".buffer_lengths.bin"
        segment_counts_path = prefix + ".buffer_segment_counts.bin"
        tokens_path = prefix + ".buffer_tokens.bin"
        segments_path = prefix + ".buffer_segments.bin"
        atomic_write_array(lengths_path, lengths)
        atomic_write_array(segment_counts_path, segment_counts)
        atomic_write_array(tokens_path, flat_tokens)
        atomic_write_array(segments_path, flat_segments)
        return {
            "buffer_lengths_path": os.path.basename(lengths_path),
            "buffer_segment_counts_path": os.path.basename(segment_counts_path),
            "buffer_tokens_path": os.path.basename(tokens_path),
            "buffer_segments_path": os.path.basename(segments_path),
        }

    @staticmethod
    def read(state_dir: str | Path, payload: dict[str, Any]) -> tuple[SourceDocument, ...]:
        state_path = Path(state_dir)
        lengths = np.fromfile(state_path / payload["buffer_lengths_path"], dtype=DOC_LENGTHS_DTYPE)
        segment_counts = np.fromfile(state_path / payload["buffer_segment_counts_path"], dtype=DOC_SEGMENT_COUNTS_DTYPE)
        flat_tokens = np.fromfile(state_path / payload["buffer_tokens_path"], dtype=np.uint16)
        flat_segments = np.fromfile(state_path / payload["buffer_segments_path"], dtype=SEGMENT_DTYPE)

        if lengths.size != segment_counts.size:
            raise ValueError("Buffered-doc lengths and segment counts disagree on document count.")

        docs: list[SourceDocument] = []
        token_offset = 0
        segment_offset = 0
        docs_consumed = int(payload["docs_consumed"])
        buffered_doc_count = int(payload.get("buffered_doc_count", int(lengths.size)))
        if buffered_doc_count != int(lengths.size):
            raise ValueError(
                f"Resume state expected {buffered_doc_count} buffered docs, found {int(lengths.size)}."
            )

        for length_value, segment_count_value in zip(lengths.tolist(), segment_counts.tolist()):
            length = int(length_value)
            segment_count = int(segment_count_value)
            next_token_offset = token_offset + length
            next_segment_offset = segment_offset + segment_count
            token_slice = flat_tokens[token_offset:next_token_offset]
            segment_slice = flat_segments[segment_offset:next_segment_offset]
            docs.append(
                SourceDocument(
                    docs_consumed=docs_consumed,
                    tokens=np.asarray(token_slice, dtype=np.uint16).copy(),
                    segments=tuple(
                        SourceSegment(
                            source_shard_idx=int(segment["source_shard_idx"]),
                            source_token_start=int(segment["source_token_start"]),
                            source_token_end=int(segment["source_token_end"]),
                        )
                        for segment in segment_slice
                    ),
                )
            )
            token_offset = next_token_offset
            segment_offset = next_segment_offset

        if token_offset != int(flat_tokens.size):
            raise ValueError("Buffered-doc token payload contains trailing data.")
        if segment_offset != int(flat_segments.size):
            raise ValueError("Buffered-doc segment payload contains trailing data.")

        return tuple(docs)


class PackedIndexResumeStore:
    """Persist exact packed-index builder state at virtual-shard boundaries."""

    def __init__(self, out_dir: str | Path, signature: BuilderResumeSignature) -> None:
        self._out_dir = Path(out_dir)
        self._state_dir = self._out_dir / ".resume_state"
        self._signature = signature
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: BuilderResumeState) -> None:
        for backup in (False, True):
            suffix = ".backup" if backup else ""
            prefix = str(self._state_dir / f"state_{state.next_virtual_shard_idx:06d}{suffix}")
            paths_payload = BufferedDocCodec.write(prefix, state.buffered_docs)
            payload = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "next_virtual_shard_idx": int(state.next_virtual_shard_idx),
                "docs_consumed": int(state.docs_consumed),
                "tokens_cropped_total": int(state.tokens_cropped_total),
                "buffered_doc_count": int(len(state.buffered_docs)),
                "source_shard_idx": int(state.cursor.source_shard_idx),
                "source_token_offset": int(state.cursor.source_token_offset),
                "source_data_dir": self._signature.source_data_dir,
                "source_shards_fingerprint": self._signature.source_shards_fingerprint,
                "seq_len": int(self._signature.seq_len),
                "row_tokens": int(self._signature.row_tokens),
                "buffer_docs": int(self._signature.buffer_docs),
                "virtual_shard_rows": int(self._signature.virtual_shard_rows),
                "val_shards": int(self._signature.val_shards),
                "bos_token_id": int(self._signature.bos_token_id),
                "output_state": state.output_state,
                **paths_payload,
            }
            atomic_write_bytes(
                str(self._state_dir / f"state_{state.next_virtual_shard_idx:06d}{suffix}.json"),
                json.dumps(payload, indent=2).encode("utf-8"),
            )
        self._prune_all_but(int(state.next_virtual_shard_idx))

    def load(self, next_virtual_shard_idx: int) -> BuilderResumeState:
        last_error: BaseException | None = None
        for suffix in ("", ".backup"):
            json_path = self._state_dir / f"state_{int(next_virtual_shard_idx):06d}{suffix}.json"
            if not json_path.exists():
                continue
            try:
                payload = _load_json(json_path)
                self._validate_payload(payload, expected_next_virtual_shard_idx=int(next_virtual_shard_idx))
                buffered_docs = BufferedDocCodec.read(self._state_dir, payload)
                return BuilderResumeState(
                    next_virtual_shard_idx=int(payload["next_virtual_shard_idx"]),
                    docs_consumed=int(payload["docs_consumed"]),
                    tokens_cropped_total=int(payload["tokens_cropped_total"]),
                    buffered_docs=tuple(doc.copy() for doc in buffered_docs),
                    cursor=SourceCursorState(
                        source_shard_idx=int(payload["source_shard_idx"]),
                        source_token_offset=int(payload["source_token_offset"]),
                    ),
                    output_state={
                        str(split): {str(key): int(value) for key, value in counters.items()}
                        for split, counters in dict(payload["output_state"]).items()
                    },
                )
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"Failed to load resume state for virtual shard {next_virtual_shard_idx} from the active snapshot or backup."
            ) from last_error
        raise FileNotFoundError(
            f"No packed-index resume state found for virtual shard {next_virtual_shard_idx} under {self._state_dir}"
        )

    def cleanup_from_shard(
        self,
        start_virtual_shard_idx: int,
        *,
        keep_state_shard_idx: int | None = None,
        output_state: dict[str, dict[str, int]] | None = None,
    ) -> None:
        if output_state is not None:
            for split in ("val", "train"):
                split_state = output_state.get(split, {})
                paths = _paths_for_split(self._out_dir, split)
                self._truncate_or_remove(paths["row_ptr"], int(split_state.get("row_ptr_bytes", 0)))
                self._truncate_or_remove(paths["segments"], int(split_state.get("segments_bytes", 0)))
                self._truncate_or_remove(paths["virtual_shards"], int(split_state.get("virtual_shards_bytes", 0)))
        meta_path = self._out_dir / "meta.json"
        if meta_path.exists():
            meta_path.unlink()

        for name in os.listdir(self._state_dir):
            matched_idx = None
            if _STATE_JSON_RE.match(name):
                matched_idx = int(_STATE_JSON_RE.match(name).group(1))
            elif _STATE_BIN_RE.match(name):
                matched_idx = int(_STATE_BIN_RE.match(name).group(1))
            if matched_idx is None:
                continue
            if matched_idx >= int(start_virtual_shard_idx) and matched_idx != keep_state_shard_idx:
                os.remove(self._state_dir / name)

    def _truncate_or_remove(self, path: Path, size: int) -> None:
        if size <= 0:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            raise FileNotFoundError(f"Expected file to truncate during resume cleanup: {path}")
        with open(path, "r+b") as handle:
            handle.truncate(size)

    def _validate_payload(self, payload: dict[str, Any], expected_next_virtual_shard_idx: int) -> None:
        expected = {
            "source_data_dir": self._signature.source_data_dir,
            "source_shards_fingerprint": self._signature.source_shards_fingerprint,
            "seq_len": int(self._signature.seq_len),
            "row_tokens": int(self._signature.row_tokens),
            "buffer_docs": int(self._signature.buffer_docs),
            "virtual_shard_rows": int(self._signature.virtual_shard_rows),
            "val_shards": int(self._signature.val_shards),
            "bos_token_id": int(self._signature.bos_token_id),
        }
        for key, expected_value in expected.items():
            observed_value = payload.get(key)
            if observed_value != expected_value:
                raise ValueError(
                    f"Packed-index resume state mismatch for '{key}': expected {expected_value!r}, found {observed_value!r}."
                )
        observed_next = int(payload.get("next_virtual_shard_idx", -1))
        if observed_next != int(expected_next_virtual_shard_idx):
            raise ValueError(
                f"Packed-index resume state expected next_virtual_shard_idx={expected_next_virtual_shard_idx}, found {observed_next}."
            )

    def _prune_all_but(self, keep_next_virtual_shard_idx: int) -> None:
        for name in os.listdir(self._state_dir):
            matched_idx = None
            if _STATE_JSON_RE.match(name):
                matched_idx = int(_STATE_JSON_RE.match(name).group(1))
            elif _STATE_BIN_RE.match(name):
                matched_idx = int(_STATE_BIN_RE.match(name).group(1))
            if matched_idx is None or matched_idx == keep_next_virtual_shard_idx:
                continue
            os.remove(self._state_dir / name)


@dataclass
class _SplitOutputState:
    row_count: int = 0
    segment_count: int = 0
    local_shard_count: int = 0
    row_ptr_bytes: int = 0
    segments_bytes: int = 0
    virtual_shards_bytes: int = 0

    def to_resume_dict(self) -> dict[str, int]:
        return {
            "row_count": int(self.row_count),
            "segment_count": int(self.segment_count),
            "local_shard_count": int(self.local_shard_count),
            "row_ptr_bytes": int(self.row_ptr_bytes),
            "segments_bytes": int(self.segments_bytes),
            "virtual_shards_bytes": int(self.virtual_shards_bytes),
        }


class PackedIndexSink:
    """Collect packed rows and flush them into split-global index files."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        row_tokens: int,
        virtual_shard_rows: int,
        val_shards: int,
        start_virtual_shard_idx: int = 0,
        start_output_state: dict[str, dict[str, int]] | None = None,
        on_tokens_written: Callable[[int], None] | None = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._row_tokens = int(row_tokens)
        self._virtual_shard_rows = int(virtual_shard_rows)
        self._val_shards = int(val_shards)
        self._on_tokens_written = on_tokens_written
        self._global_virtual_shard_idx = int(start_virtual_shard_idx)
        self._split_states = {
            "val": _SplitOutputState(),
            "train": _SplitOutputState(),
        }
        if start_output_state is not None:
            for split in ("val", "train"):
                if split not in start_output_state:
                    continue
                state = start_output_state[split]
                self._split_states[split] = _SplitOutputState(
                    row_count=int(state.get("row_count", 0)),
                    segment_count=int(state.get("segment_count", 0)),
                    local_shard_count=int(state.get("local_shard_count", 0)),
                    row_ptr_bytes=int(state.get("row_ptr_bytes", 0)),
                    segments_bytes=int(state.get("segments_bytes", 0)),
                    virtual_shards_bytes=int(state.get("virtual_shards_bytes", 0)),
                )

        self.total_rows_written = int(sum(state.row_count for state in self._split_states.values()))
        self.total_tokens_written = int(self.total_rows_written * self._row_tokens)
        self.run_rows_written = 0
        self.run_tokens_written = 0

        self._current_rows = 0
        self._current_segments: list[SourceSegment] = []
        self._current_row_segment_ends: list[int] = []

    @property
    def current_shard_idx(self) -> int:
        return int(self._global_virtual_shard_idx)

    @property
    def current_shard_fill_rows(self) -> int:
        return int(self._current_rows)

    @property
    def current_shard_fill_tokens(self) -> int:
        return int(self._current_rows * self._row_tokens)

    @property
    def num_shards_total(self) -> int:
        return int(self._global_virtual_shard_idx)

    def snapshot_output_state(self) -> dict[str, dict[str, int]]:
        return {split: state.to_resume_dict() for split, state in self._split_states.items()}

    def set_on_tokens_written(self, callback: Callable[[int], None] | None) -> None:
        self._on_tokens_written = callback

    def append_row(self, segments: Sequence[SourceSegment]) -> int | None:
        row_length = _segments_length(segments)
        if row_length != self._row_tokens:
            raise ValueError(f"Packed row length mismatch: expected {self._row_tokens}, got {row_length}.")
        self._current_segments.extend(
            SourceSegment(
                source_shard_idx=int(segment.source_shard_idx),
                source_token_start=int(segment.source_token_start),
                source_token_end=int(segment.source_token_end),
            )
            for segment in segments
        )
        self._current_rows += 1
        self._current_row_segment_ends.append(len(self._current_segments))
        if self._on_tokens_written is not None:
            self._on_tokens_written(self._row_tokens)
        if self._current_rows == self._virtual_shard_rows:
            self._flush_current_shard()
            return int(self._global_virtual_shard_idx)
        return None

    def finalize(self) -> None:
        if self._current_rows == 0:
            return
        self._flush_current_shard()

    def _flush_current_shard(self) -> None:
        split = "val" if self._global_virtual_shard_idx < self._val_shards else "train"
        split_state = self._split_states[split]
        paths = _paths_for_split(self._out_dir, split)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        local_shard_idx = int(split_state.local_shard_count)
        global_virtual_shard_idx = int(self._global_virtual_shard_idx)
        shard_path = str((self._out_dir / f"{split}_{global_virtual_shard_idx:06d}.vrow").resolve())
        row_start = int(split_state.row_count)
        row_end = int(row_start + self._current_rows)

        if split_state.row_ptr_bytes == 0:
            with open(paths["row_ptr"], "ab") as handle:
                np.asarray([0], dtype=ROW_POINTER_DTYPE).tofile(handle)
            split_state.row_ptr_bytes += np.dtype(ROW_POINTER_DTYPE).itemsize

        if self._current_segments:
            with open(paths["segments"], "ab") as handle:
                np.asarray(
                    [
                        (
                            int(segment.source_shard_idx),
                            int(segment.source_token_start),
                            int(segment.source_token_end),
                        )
                        for segment in self._current_segments
                    ],
                    dtype=SEGMENT_DTYPE,
                ).tofile(handle)
            split_state.segments_bytes += len(self._current_segments) * SEGMENT_DTYPE.itemsize

        row_ptr_values = np.asarray(
            [split_state.segment_count + int(value) for value in self._current_row_segment_ends],
            dtype=ROW_POINTER_DTYPE,
        )
        with open(paths["row_ptr"], "ab") as handle:
            row_ptr_values.tofile(handle)
        split_state.row_ptr_bytes += int(row_ptr_values.size) * np.dtype(ROW_POINTER_DTYPE).itemsize

        manifest_record = {
            "split": split,
            "local_shard_idx": int(local_shard_idx),
            "global_virtual_shard_idx": int(global_virtual_shard_idx),
            "shard_path": shard_path,
            "row_start": int(row_start),
            "row_end": int(row_end),
            "row_count": int(self._current_rows),
        }
        line = json.dumps(manifest_record) + "\n"
        with open(paths["virtual_shards"], "a", encoding="utf-8") as handle:
            handle.write(line)
        split_state.virtual_shards_bytes += len(line.encode("utf-8"))

        split_state.row_count += int(self._current_rows)
        split_state.segment_count += len(self._current_segments)
        split_state.local_shard_count += 1

        self.total_rows_written += int(self._current_rows)
        self.total_tokens_written += int(self._current_rows * self._row_tokens)
        self.run_rows_written += int(self._current_rows)
        self.run_tokens_written += int(self._current_rows * self._row_tokens)
        self._global_virtual_shard_idx += 1

        self._current_rows = 0
        self._current_segments = []
        self._current_row_segment_ends = []


def pack_documents_into_index(
    token_docs: Iterator[SourceDocument],
    *,
    row_tokens: int,
    buffer_docs: int,
    sink: PackedIndexSink,
    stats: PackingStats,
    max_shards: int | None = None,
    initial_doc_buffer: Sequence[SourceDocument] | None = None,
    on_shard_completed: Callable[[BuilderResumeState], None] | None = None,
    on_buffer_state_changed: Callable[[int], None] | None = None,
    cursor_provider: Callable[[], SourceCursorState] | None = None,
) -> None:
    """Pack logical source documents into fixed-width rows under exact legacy semantics."""

    doc_buffer = LengthIndexedDocBuffer(initial_doc_buffer or ())
    if len(doc_buffer) > int(buffer_docs):
        raise ValueError(
            f"Resume state restored {len(doc_buffer)} buffered docs, which exceeds buffer_docs={buffer_docs}."
        )
    if max_shards is not None and int(max_shards) < 0:
        raise ValueError("max_shards must be >= 0 when provided.")

    docs_exhausted = False
    docs_consumed = int(stats.docs_processed)

    def emit_buffer_state() -> None:
        if on_buffer_state_changed is not None:
            on_buffer_state_changed(len(doc_buffer))

    def refill_buffer() -> None:
        nonlocal docs_consumed, docs_exhausted
        while len(doc_buffer) < int(buffer_docs) and not docs_exhausted:
            try:
                doc = next(token_docs)
            except StopIteration:
                docs_exhausted = True
                break
            doc_buffer.append(doc)
            docs_consumed = int(doc.docs_consumed)
            stats.docs_processed = docs_consumed
        emit_buffer_state()

    emit_buffer_state()
    if max_shards is not None and sink.current_shard_idx >= int(max_shards):
        return
    refill_buffer()
    while True:
        if max_shards is not None and sink.current_shard_idx >= int(max_shards):
            break
        if not doc_buffer:
            break

        row_segments: list[SourceSegment] = []
        pos = 0

        while pos < int(row_tokens):
            if not doc_buffer:
                refill_buffer()
                if not doc_buffer:
                    break

            remaining = int(row_tokens) - pos
            doc = doc_buffer.pop_largest_that_fits(remaining)

            if doc is not None:
                row_segments.extend(doc.segments)
                pos += doc.length
            else:
                doc = doc_buffer.pop_shortest()
                row_segments.extend(_crop_segments(doc.segments, remaining))
                stats.tokens_cropped_total += doc.length - remaining
                pos += remaining

            if len(doc_buffer) < int(buffer_docs):
                refill_buffer()

        if pos < int(row_tokens):
            break

        next_shard_idx = sink.append_row(tuple(row_segments))
        if next_shard_idx is not None:
            if on_shard_completed is not None:
                if cursor_provider is None:
                    raise ValueError("cursor_provider is required when on_shard_completed is used.")
                on_shard_completed(
                    BuilderResumeState(
                        next_virtual_shard_idx=int(next_shard_idx),
                        docs_consumed=int(docs_consumed),
                        tokens_cropped_total=int(stats.tokens_cropped_total),
                        buffered_docs=doc_buffer.snapshot(),
                        cursor=cursor_provider(),
                        output_state=sink.snapshot_output_state(),
                    )
                )
            if max_shards is not None and int(next_shard_idx) >= int(max_shards):
                break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an exact BOS packed-index artifact from pretokenized raw FineWebEdu shards."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing raw uint16 token shards such as val_000000.bin and train_000001.bin.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory where the packed-index artifact should be written.",
    )
    parser.add_argument("--seq_len", type=int, default=1024, help="Training sequence length.")
    parser.add_argument(
        "--buffer_docs",
        type=int,
        default=1000,
        help="How many logical BOS documents to keep in the best-fit packing buffer.",
    )
    parser.add_argument(
        "--shard_rows",
        type=int,
        default=0,
        help="Rows per virtual shard. 0 auto-sizes to about 100M output tokens per virtual shard.",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="If >0, stop after writing this many virtual shards total. 0 means no limit.",
    )
    parser.add_argument(
        "--val_shards",
        type=int,
        default=1,
        help="Number of initial virtual shards to label as validation.",
    )
    parser.add_argument(
        "--bos_token_id",
        type=int,
        default=DEFAULT_BOS_TOKEN_ID,
        help="Token id used as the BOS document marker in the source stream.",
    )
    parser.add_argument(
        "--resume_from_shard",
        type=int,
        default=0,
        help="Resume the builder from the beginning of this virtual shard using exact saved state.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq_len must be > 0")
    if args.buffer_docs <= 0:
        raise ValueError("--buffer_docs must be > 0")
    if args.shard_rows < 0:
        raise ValueError("--shard_rows must be >= 0")
    if args.max_shards < 0:
        raise ValueError("--max_shards must be >= 0")
    if args.val_shards < 0:
        raise ValueError("--val_shards must be >= 0")
    if args.resume_from_shard < 0:
        raise ValueError("--resume_from_shard must be >= 0")
    if args.bos_token_id < 0:
        raise ValueError("--bos_token_id must be >= 0")


def _fresh_builder_resume_state() -> BuilderResumeState:
    return BuilderResumeState(
        next_virtual_shard_idx=0,
        docs_consumed=0,
        tokens_cropped_total=0,
        buffered_docs=tuple(),
        cursor=SourceCursorState(source_shard_idx=0, source_token_offset=0),
        output_state={
            "val": _SplitOutputState().to_resume_dict(),
            "train": _SplitOutputState().to_resume_dict(),
        },
    )


def resolve_builder_resume_state(
    args: argparse.Namespace,
    *,
    resume_store: PackedIndexResumeStore,
) -> BuilderResumeState:
    if int(args.resume_from_shard) == 0:
        return _fresh_builder_resume_state()
    state = resume_store.load(int(args.resume_from_shard))
    resume_store.cleanup_from_shard(
        int(args.resume_from_shard),
        keep_state_shard_idx=int(args.resume_from_shard),
        output_state=state.output_state,
    )
    return state


def build_index_meta(
    *,
    args: argparse.Namespace,
    source_data_dir: Path,
    source_shards: Sequence[Path],
    source_shards_fingerprint: str,
    row_tokens: int,
    virtual_shard_rows: int,
    stats: PackingStats,
    sink: PackedIndexSink,
) -> dict[str, Any]:
    total_output_tokens = stats.tokens_cropped_total + sink.total_tokens_written
    crop_fraction = (
        float(stats.tokens_cropped_total) / float(total_output_tokens)
        if total_output_tokens > 0
        else 0.0
    )
    artifact_dir = Path(args.out_dir).expanduser().resolve()
    train_paths = _paths_for_split(artifact_dir, "train")
    val_paths = _paths_for_split(artifact_dir, "val")
    return {
        "format": PACKED_INDEX_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": str(source_data_dir),
        "source_shards_fingerprint": source_shards_fingerprint,
        "source_num_shards": int(len(source_shards)),
        "dtype": "uint16",
        "seq_len": int(args.seq_len),
        "row_tokens": int(row_tokens),
        "bos_token_id": int(args.bos_token_id),
        "eos_token_id": int(args.bos_token_id),
        "bos_is_eos": True,
        "packing_algo": PACKING_ALGO,
        "buffer_docs": int(args.buffer_docs),
        "virtual_shard_rows": int(virtual_shard_rows),
        "shard_rows": int(virtual_shard_rows),
        "max_shards": (None if args.max_shards <= 0 else int(args.max_shards)),
        "val_shards": int(args.val_shards),
        "num_shards_total": int(sink.num_shards_total),
        "rows_written_total": int(sink.total_rows_written),
        "tokens_written_total": int(sink.total_tokens_written),
        "run_rows_written": int(sink.total_rows_written),
        "run_tokens_written": int(sink.total_tokens_written),
        "tokens_cropped_total": int(stats.tokens_cropped_total),
        "crop_fraction": float(crop_fraction),
        "docs_processed": int(stats.docs_processed),
        "index_files": {
            "train": {
                "row_ptr": train_paths["row_ptr"].name,
                "segments": train_paths["segments"].name,
                "virtual_shards": train_paths["virtual_shards"].name,
            },
            "val": {
                "row_ptr": val_paths["row_ptr"].name,
                "segments": val_paths["segments"].name,
                "virtual_shards": val_paths["virtual_shards"].name,
            },
        },
        "notes": (
            "Rows are reconstructed from source shards using row pointers plus source token slices. "
            "Semantics exactly match bos_row_packed_bestfit: largest-fit then shortest-doc crop."
        ),
    }


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_progress = _maybe_make_tqdm(
        total=4,
        desc="build packed index",
        unit="stage",
        dynamic_ncols=True,
    )

    source_data_dir = Path(args.data_dir).expanduser().resolve()
    if stage_progress is not None:
        stage_progress.set_postfix_str("scan source shards")
    source_shards = _ordered_source_shards(source_data_dir)
    source_shards_fingerprint = _fingerprint_from_paths_with_progress(
        source_shards,
        desc="scan source shards",
    )
    if stage_progress is not None:
        stage_progress.update(1)

    row_tokens = int(args.seq_len) + 1
    auto_rows = max(1, 100_000_000 // row_tokens)
    virtual_shard_rows = auto_rows if int(args.shard_rows) <= 0 else int(args.shard_rows)
    signature = BuilderResumeSignature(
        source_data_dir=str(source_data_dir),
        source_shards_fingerprint=source_shards_fingerprint,
        seq_len=int(args.seq_len),
        row_tokens=int(row_tokens),
        buffer_docs=int(args.buffer_docs),
        virtual_shard_rows=int(virtual_shard_rows),
        val_shards=int(args.val_shards),
        bos_token_id=int(args.bos_token_id),
    )
    if stage_progress is not None:
        stage_progress.set_postfix_str("load resume state")
    resume_store = PackedIndexResumeStore(out_dir, signature)
    resume_state = resolve_builder_resume_state(args, resume_store=resume_store)
    if int(args.max_shards) > 0 and int(resume_state.next_virtual_shard_idx) > int(args.max_shards):
        raise ValueError(
            f"--max_shards={args.max_shards} is smaller than the resume point next_virtual_shard_idx="
            f"{resume_state.next_virtual_shard_idx}."
        )
    if stage_progress is not None:
        stage_progress.update(1)

    if stage_progress is not None:
        stage_progress.set_postfix_str("initialize builder")
    doc_stream = RawBOSDocumentStream(
        source_data_dir,
        bos_token_id=int(args.bos_token_id),
        start_state=resume_state.cursor,
        docs_consumed_start=int(resume_state.docs_consumed),
    )
    sink = PackedIndexSink(
        out_dir,
        row_tokens=int(row_tokens),
        virtual_shard_rows=int(virtual_shard_rows),
        val_shards=int(args.val_shards),
        start_virtual_shard_idx=int(resume_state.next_virtual_shard_idx),
        start_output_state=resume_state.output_state,
    )
    stats = PackingStats(
        docs_processed=int(resume_state.docs_consumed),
        tokens_cropped_total=int(resume_state.tokens_cropped_total),
    )
    if stage_progress is not None:
        stage_progress.update(1)

    progress = _maybe_make_tqdm(
        total=None,
        unit="tok",
        desc="pack rows",
        dynamic_ncols=True,
    )
    packing_state = {
        "buffered_docs": int(len(resume_state.buffered_docs)),
        "last_postfix": None,
    }

    def _refresh_packing_postfix(force: bool = False) -> None:
        if progress is None:
            return
        postfix = (
            f"docs={int(stats.docs_processed):,}, "
            f"shards={int(sink.num_shards_total):,}, "
            f"buffer={int(packing_state['buffered_docs']):,}, "
            f"cropped={int(stats.tokens_cropped_total):,}"
        )
        if force or postfix != packing_state["last_postfix"]:
            progress.set_postfix_str(postfix)
            packing_state["last_postfix"] = postfix

    token_refresh_budget = [0]

    def _on_tokens_written(tokens_written: int) -> None:
        if progress is not None:
            progress.update(int(tokens_written))
            token_refresh_budget[0] += int(tokens_written)
            if token_refresh_budget[0] >= int(row_tokens) * 64:
                token_refresh_budget[0] = 0
                _refresh_packing_postfix()

    def _on_buffer_state_changed(buffered_docs: int) -> None:
        packing_state["buffered_docs"] = int(buffered_docs)
        _refresh_packing_postfix()

    def _on_shard_completed(state: BuilderResumeState) -> None:
        resume_store.save(state)
        packing_state["buffered_docs"] = int(len(state.buffered_docs))
        _refresh_packing_postfix(force=True)

    sink.set_on_tokens_written(_on_tokens_written)
    _refresh_packing_postfix(force=True)

    try:
        pack_documents_into_index(
            iter(doc_stream),
            row_tokens=int(row_tokens),
            buffer_docs=int(args.buffer_docs),
            sink=sink,
            stats=stats,
            max_shards=(None if int(args.max_shards) <= 0 else int(args.max_shards)),
            initial_doc_buffer=resume_state.buffered_docs,
            on_shard_completed=_on_shard_completed,
            on_buffer_state_changed=_on_buffer_state_changed,
            cursor_provider=doc_stream.snapshot,
        )
    finally:
        if progress is not None:
            progress.close()
    if stage_progress is not None:
        stage_progress.set_postfix_str("finalize artifact")

    finalize_progress = _maybe_make_tqdm(
        total=2,
        desc="finalize artifact",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    if finalize_progress is not None:
        finalize_progress.set_postfix_str("flush remaining rows")
    sink.finalize()
    if finalize_progress is not None:
        finalize_progress.update(1)
        finalize_progress.set_postfix_str("write meta.json")

    meta = build_index_meta(
        args=args,
        source_data_dir=source_data_dir,
        source_shards=source_shards,
        source_shards_fingerprint=source_shards_fingerprint,
        row_tokens=int(row_tokens),
        virtual_shard_rows=int(virtual_shard_rows),
        stats=stats,
        sink=sink,
    )
    meta_path = out_dir / "meta.json"
    atomic_write_bytes(str(meta_path), json.dumps(meta, indent=2).encode("utf-8"))
    if finalize_progress is not None:
        finalize_progress.update(1)
        finalize_progress.close()
    if stage_progress is not None:
        stage_progress.update(1)
        stage_progress.set_postfix_str("done")
        stage_progress.close()
    return meta


def _read_virtual_shards_manifest(data_dir: str | Path, split: str) -> tuple[VirtualShardInfo, ...]:
    path = _paths_for_split(data_dir, split)["virtual_shards"]
    if not path.exists():
        return tuple()
    rows: list[VirtualShardInfo] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                VirtualShardInfo(
                    split=str(payload["split"]),
                    local_shard_idx=int(payload["local_shard_idx"]),
                    global_virtual_shard_idx=int(payload["global_virtual_shard_idx"]),
                    shard_path=str(payload["shard_path"]),
                    row_start=int(payload["row_start"]),
                    row_end=int(payload["row_end"]),
                )
            )
    return tuple(rows)


class PackedIndexView:
    """Read-only runtime view over a BOS packed-index artifact."""

    def __init__(self, data_dir: str | Path, *, source_data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.meta = _load_json(self.data_dir / "meta.json")
        if self.meta.get("format") != PACKED_INDEX_FORMAT:
            raise ValueError(f"{self.data_dir} is not a {PACKED_INDEX_FORMAT} artifact.")
        self.row_tokens = int(self.meta["row_tokens"])
        self.seq_len = int(self.meta["seq_len"])
        self.source_data_dir = _resolve_source_data_dir(self.data_dir, None if source_data_dir is None else str(source_data_dir))
        self.source_shard_paths = _ordered_source_shards(self.source_data_dir)
        self._source_memmaps: dict[int, np.memmap] = {}
        self._row_ptr: dict[str, np.memmap] = {}
        self._segments: dict[str, np.memmap] = {}
        self._virtual_shards = {
            split: _read_virtual_shards_manifest(self.data_dir, split)
            for split in ("val", "train")
        }

    def virtual_shards(self, split: str) -> tuple[VirtualShardInfo, ...]:
        return self._virtual_shards[str(split)]

    def rows_per_shard(self, split: str) -> tuple[int, ...]:
        return tuple(shard.row_count for shard in self.virtual_shards(split))

    def num_rows(self, split: str) -> int:
        shards = self.virtual_shards(split)
        if not shards:
            return 0
        return int(shards[-1].row_end)

    def row_pointer(self, split: str) -> np.memmap:
        split_key = str(split)
        if split_key not in self._row_ptr:
            self._row_ptr[split_key] = np.memmap(
                _paths_for_split(self.data_dir, split_key)["row_ptr"],
                dtype=ROW_POINTER_DTYPE,
                mode="r",
            )
        return self._row_ptr[split_key]

    def segments(self, split: str) -> np.memmap:
        split_key = str(split)
        if split_key not in self._segments:
            self._segments[split_key] = np.memmap(
                _paths_for_split(self.data_dir, split_key)["segments"],
                dtype=SEGMENT_DTYPE,
                mode="r",
            )
        return self._segments[split_key]

    def row_segments(self, split: str, row_id: int) -> tuple[SourceSegment, ...]:
        row_ptr = self.row_pointer(split)
        start = int(row_ptr[int(row_id)])
        end = int(row_ptr[int(row_id) + 1])
        segment_slice = self.segments(split)[start:end]
        return tuple(
            SourceSegment(
                source_shard_idx=int(segment["source_shard_idx"]),
                source_token_start=int(segment["source_token_start"]),
                source_token_end=int(segment["source_token_end"]),
            )
            for segment in segment_slice
        )

    def reconstruct_row(self, split: str, row_id: int) -> np.ndarray:
        row = np.empty((self.row_tokens,), dtype=np.uint16)
        pos = 0
        for segment in self.row_segments(split, row_id):
            mm = self._source_memmap(int(segment.source_shard_idx))
            chunk = np.asarray(mm[int(segment.source_token_start) : int(segment.source_token_end)], dtype=np.uint16)
            next_pos = pos + int(chunk.size)
            row[pos:next_pos] = chunk
            pos = next_pos
        if pos != self.row_tokens:
            raise ValueError(f"Packed-index row reconstruction expected {self.row_tokens} tokens, got {pos}.")
        return row

    def shard_for_row(self, split: str, row_id: int) -> VirtualShardInfo:
        shards = self.virtual_shards(split)
        shard_starts = [shard.row_start for shard in shards]
        index = bisect.bisect_right(shard_starts, int(row_id)) - 1
        if index < 0 or index >= len(shards):
            raise IndexError(f"Row {row_id} is out of range for split {split!r}.")
        shard = shards[index]
        if not (shard.row_start <= int(row_id) < shard.row_end):
            raise IndexError(f"Row {row_id} is out of range for split {split!r}.")
        return shard

    def row_from_virtual_location(self, split: str, shard_idx: int, local_row_idx: int) -> np.ndarray:
        shard = self.virtual_shards(split)[int(shard_idx)]
        global_row_id = int(shard.row_start + int(local_row_idx))
        return self.reconstruct_row(split, global_row_id)

    def _source_memmap(self, source_shard_idx: int) -> np.memmap:
        if source_shard_idx not in self._source_memmaps:
            self._source_memmaps[source_shard_idx] = np.memmap(
                self.source_shard_paths[source_shard_idx],
                dtype=np.uint16,
                mode="r",
            )
        return self._source_memmaps[source_shard_idx]


@dataclass(frozen=True)
class PackedIndexDatasetConfig:
    data_dir: str
    split: str = "train"
    batch_size: int = 16
    seq_len: int = 1024
    shuffle_blocks: bool = True
    seed: int = 1337
    max_blocks: int | None = None
    shard_by_rank: bool = True
    return_meta: bool = False
    source_data_dir: str | None = None


BatchType = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]


class MemmapBOSPackedIndexDataset(IterableDataset):
    """IterableDataset that reconstructs BOS packed rows from a compact index."""

    def __init__(self, cfg: PackedIndexDatasetConfig) -> None:
        super().__init__()
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if cfg.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.cfg = cfg
        self.view = PackedIndexView(cfg.data_dir, source_data_dir=cfg.source_data_dir)
        if int(self.view.seq_len) != int(cfg.seq_len):
            raise ValueError(
                f"seq_len mismatch: loader expects {cfg.seq_len}, packed-index artifact has {self.view.seq_len}."
            )
        self.row_tokens = int(self.view.row_tokens)
        self.shards = self.view.virtual_shards(cfg.split)
        if not self.shards:
            raise FileNotFoundError(f"No virtual shards found for split={cfg.split!r} under {cfg.data_dir}")
        self.rows_per_shard = [shard.row_count for shard in self.shards]
        self.block_rows = int(cfg.batch_size)
        self.blocks_per_shard = [row_count // self.block_rows for row_count in self.rows_per_shard]
        self.total_blocks = int(sum(self.blocks_per_shard))
        if self.total_blocks <= 0:
            raise ValueError(
                f"No valid blocks for split={cfg.split!r}: rows_per_shard={self.rows_per_shard!r}, "
                f"batch_size={cfg.batch_size}."
            )

    def __iter__(self) -> Iterator[BatchType]:
        cfg = self.cfg
        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        rank, world_size = _get_dist_info()
        shard_by_rank = cfg.shard_by_rank and world_size > 1

        share_shards = False
        if shard_by_rank:
            if len(self.shards) >= world_size:
                rank_shards = [idx for idx in range(len(self.shards)) if (idx % world_size) == rank]
            else:
                rank_shards = list(range(len(self.shards)))
                share_shards = True
        else:
            rank_shards = list(range(len(self.shards)))

        if not rank_shards:
            return iter(())

        ws, we = _split_work(len(rank_shards), worker_id, num_workers)
        shard_indices = rank_shards[ws:we]
        if not shard_indices:
            return iter(())

        rng = np.random.default_rng(int(cfg.seed) + 1000 * rank + worker_id)
        schedule: list[tuple[int, int]] = []
        for local_pos, shard_idx in enumerate(shard_indices):
            n_blocks = self.blocks_per_shard[shard_idx]
            if n_blocks <= 0:
                continue
            if share_shards:
                blocks = range(rank, n_blocks, world_size)
            else:
                blocks = range(n_blocks)
            schedule.extend((local_pos, block_idx) for block_idx in blocks)

        if not schedule:
            return iter(())

        blocks_yielded = 0
        while True:
            if cfg.shuffle_blocks:
                rng.shuffle(schedule)

            for local_pos, block_idx in schedule:
                shard_idx = shard_indices[local_pos]
                shard = self.shards[shard_idx]
                start_row = int(block_idx * self.block_rows)
                end_row = int(start_row + self.block_rows)
                rows = []
                for local_row_idx in range(start_row, end_row):
                    global_row_id = int(shard.row_start + local_row_idx)
                    rows.append(self.view.reconstruct_row(cfg.split, global_row_id))
                row_tensor = torch.from_numpy(np.stack(rows, axis=0).astype(np.int64, copy=False))
                x = row_tensor[:, :-1]
                y = row_tensor[:, 1:]

                if cfg.return_meta:
                    meta = {
                        "split": cfg.split,
                        "batch_size": int(cfg.batch_size),
                        "seq_len": int(cfg.seq_len),
                        "row_tokens": int(self.row_tokens),
                        "block_tokens": int(self.block_rows * self.row_tokens),
                        "shard_idx": int(shard_idx),
                        "shard_path": str(shard.shard_path),
                        "block_idx": int(block_idx),
                        "start": int(start_row * self.row_tokens),
                        "end": int(end_row * self.row_tokens),
                        "row_start": int(start_row),
                        "row_end": int(end_row),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "worker_id": int(worker_id),
                        "num_workers": int(num_workers),
                        "share_shards": bool(share_shards),
                    }
                    yield x, y, meta
                else:
                    yield x, y

                blocks_yielded += 1
                if cfg.max_blocks is not None and blocks_yielded >= int(cfg.max_blocks):
                    return


def make_bos_packed_index_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    *,
    seq_len: int = 1024,
    shuffle_blocks: bool = True,
    seed: int = 1337,
    num_workers: int = 0,
    max_blocks: int | None = None,
    shard_by_rank: bool = True,
    return_meta: bool = False,
    source_data_dir: str | None = None,
) -> DataLoader:
    cfg = PackedIndexDatasetConfig(
        data_dir=data_dir,
        split=split,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle_blocks=shuffle_blocks,
        seed=seed,
        max_blocks=max_blocks,
        shard_by_rank=shard_by_rank,
        return_meta=return_meta,
        source_data_dir=source_data_dir,
    )
    dataset = MemmapBOSPackedIndexDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    meta = run_builder(args)
    print("\nDone.")
    print(f"  out_dir: {Path(args.out_dir).expanduser().resolve()}")
    print(f"  virtual shards: {meta['num_shards_total']} (val_shards={meta['val_shards']})")
    print(f"  rows:           {meta['rows_written_total']:,}")
    print(f"  tokens:         {meta['tokens_written_total']:,}")
    print(f"  cropped:        {meta['tokens_cropped_total']:,} ({100.0 * meta['crop_fraction']:.2f}%)")
    print(f"  docs:           {meta['docs_processed']:,}")


__all__ = [
    "PACKED_INDEX_FORMAT",
    "PackedIndexView",
    "RawBOSDocumentStream",
    "SourceCursorState",
    "SourceDocument",
    "SourceSegment",
    "build_arg_parser",
    "fingerprint_packed_index_artifact",
    "fingerprint_source_shards",
    "main",
    "make_bos_packed_index_dataloader",
    "run_builder",
]
