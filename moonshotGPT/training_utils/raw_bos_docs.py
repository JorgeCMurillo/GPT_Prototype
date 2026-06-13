"""Helpers for reconstructing BOS-delimited documents from raw token shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import re
from typing import Sequence

import numpy as np


_SHARD_NAME_RE = re.compile(r"^(train|val)_(\d{6})\.bin$")


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


def ordered_raw_shard_paths(data_dir: str | Path, split: str | None = None) -> tuple[Path, ...]:
    """Return raw shard paths in numeric order, optionally filtered by split."""
    root = Path(data_dir).expanduser().resolve()
    paths = [Path(path) for path in glob.glob(str(root / "*.bin"))]
    parsed: list[tuple[int, Path]] = []
    for path in paths:
        match = _SHARD_NAME_RE.match(path.name)
        if match is None:
            continue
        path_split = match.group(1)
        if split is not None and path_split != str(split):
            continue
        parsed.append((int(match.group(2)), path))
    if not parsed:
        target = f" split={split!r}" if split is not None else ""
        raise FileNotFoundError(f"No raw shard files found under {root}{target}")
    parsed.sort(key=lambda item: item[0])
    return tuple(path for _, path in parsed)


class RawBOSDocumentStream:
    """Iterate logical BOS-delimited documents from an ordered shard list."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path] | str | Path,
        *,
        bos_token_id: int,
        start_state: SourceCursorState | None = None,
        docs_consumed_start: int = 0,
        skip_leading_non_bos: bool = False,
    ) -> None:
        if isinstance(shard_paths, (str, Path)):
            resolved_shard_paths = ordered_raw_shard_paths(shard_paths)
        else:
            resolved_shard_paths = tuple(Path(path).expanduser().resolve() for path in shard_paths)
        if not resolved_shard_paths:
            raise ValueError("RawBOSDocumentStream requires at least one shard path.")
        self.shard_paths = tuple(resolved_shard_paths)
        self.bos_token_id = int(bos_token_id)
        self._memmaps: dict[int, np.memmap] = {}
        self._bos_offsets_cache: dict[int, np.ndarray] = {}
        self._cursor_shard_idx = int(start_state.source_shard_idx) if start_state is not None else 0
        self._cursor_token_offset = int(start_state.source_token_offset) if start_state is not None else 0
        self._docs_consumed = int(docs_consumed_start)
        if start_state is None and skip_leading_non_bos:
            self._seek_to_next_bos()
        self._validate_cursor()

    def _memmap(self, source_shard_idx: int) -> np.memmap:
        if source_shard_idx not in self._memmaps:
            self._memmaps[source_shard_idx] = np.memmap(
                self.shard_paths[source_shard_idx],
                dtype=np.uint16,
                mode="r",
            )
        return self._memmaps[source_shard_idx]

    def _bos_offsets(self, source_shard_idx: int) -> np.ndarray:
        if source_shard_idx not in self._bos_offsets_cache:
            mm = self._memmap(source_shard_idx)
            self._bos_offsets_cache[source_shard_idx] = np.flatnonzero(mm == self.bos_token_id).astype(
                np.int64,
                copy=False,
            )
        return self._bos_offsets_cache[source_shard_idx]

    def _seek_to_next_bos(self) -> None:
        shard_idx = int(self._cursor_shard_idx)
        token_offset = int(self._cursor_token_offset)
        while shard_idx < len(self.shard_paths):
            mm = self._memmap(shard_idx)
            if token_offset >= int(mm.size):
                shard_idx += 1
                token_offset = 0
                continue
            bos_offsets = self._bos_offsets(shard_idx)
            next_hit_idx = int(np.searchsorted(bos_offsets, token_offset, side="left"))
            if next_hit_idx < int(bos_offsets.size):
                self._cursor_shard_idx = int(shard_idx)
                self._cursor_token_offset = int(bos_offsets[next_hit_idx])
                return
            shard_idx += 1
            token_offset = 0
        self._cursor_shard_idx = len(self.shard_paths)
        self._cursor_token_offset = 0

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
                    "Raw BOS document stream must start at a BOS token boundary. "
                    f"Found token {int(mm[token_offset])} at source shard {shard_idx}, offset {token_offset}."
                )
            return

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


__all__ = [
    "RawBOSDocumentStream",
    "SourceCursorState",
    "SourceDocument",
    "SourceSegment",
    "ordered_raw_shard_paths",
]
