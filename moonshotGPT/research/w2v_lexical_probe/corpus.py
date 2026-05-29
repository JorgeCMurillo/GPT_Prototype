"""Corpus readers for Word2Vec lexical-probe training.

The readers in this module all expose the same small surface:

- ``iter_documents()`` yields raw text documents.
- ``describe()`` returns serializable provenance metadata.
- ``source_name`` provides a compact name for run labels.

Keeping readers text-oriented lets the Word2Vec trainer own normalization and
tokenization, while corpus adapters only handle storage format details.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np
from transformers import AutoTokenizer

VALID_CORPUS_FORMATS = ("shard_bin", "text_dir", "hf_arrow")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_corpus_format(meta: dict) -> str:
    raw_format = str(meta.get("format", "")).strip()
    return raw_format or "bos_delimited_documents"


def describe_corpus_role(meta: dict) -> dict:
    """Classify the shard source so downstream runs can explain what they mean."""
    corpus_format = _normalize_corpus_format(meta)
    if corpus_format == "bos_row_packed_bestfit":
        return {
            "corpus_format": corpus_format,
            "baseline_role": "exposure_baseline",
            "interpretation": (
                "Uses BOS-delimited spans recovered from row-packed GPT training shards. "
                "This approximates the lexical neighborhoods the LM was exposed to during training, "
                "including packing and crop artifacts."
            ),
        }
    return {
        "corpus_format": corpus_format,
        "baseline_role": "corpus_baseline",
        "interpretation": (
            "Uses BOS-delimited source documents. This is a clean lexical baseline for the underlying corpus "
            "rather than the exact packed training exposure."
        ),
    }


@dataclass(frozen=True)
class ShardCorpusConfig:
    data_dir: str
    split: str = "train"
    max_shards: int = 1
    max_docs: int = 0
    tokenizer_name: Optional[str] = None
    use_fast_tokenizer: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ShardCorpusReader:
    """Read BOS-delimited token shards and yield detokenized documents."""

    def __init__(self, config: ShardCorpusConfig):
        self.config = config
        self.project_root = _project_root()
        self.data_dir = self._resolve_data_dir(config.data_dir)
        self.meta = self._load_meta()
        self.tokenizer_name = config.tokenizer_name or self.meta.get("tokenizer", "gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=config.use_fast_tokenizer,
        )
        self.bos_token_id = int(
            self.meta.get(
                "bos_token_id",
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id,
            )
        )
        self.shard_paths = self._list_shards()
        self.corpus_role = describe_corpus_role(self.meta)

    @property
    def source_name(self) -> str:
        return self.data_dir.name

    def _resolve_data_dir(self, raw_path: str) -> Path:
        requested = Path(raw_path).expanduser()
        candidates = (
            requested,
            self.project_root / requested,
            self.project_root / "data" / requested,
        )
        for candidate in candidates:
            if (candidate / "meta.json").exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find shard dataset directory for: {raw_path}")

    def _load_meta(self) -> dict:
        meta_path = self.data_dir / "meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _list_shards(self) -> list[Path]:
        pattern = f"{self.config.split}_*.bin"
        shards = sorted(self.data_dir.glob(pattern))
        if not shards:
            raise FileNotFoundError(
                f"No shards found for split='{self.config.split}' under {self.data_dir}"
            )
        if self.config.max_shards > 0:
            shards = shards[: self.config.max_shards]
        return shards

    def iter_documents(self) -> Iterator[str]:
        docs_yielded = 0
        for shard_path in self.shard_paths:
            tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
            bos_positions = np.flatnonzero(tokens == self.bos_token_id)
            if bos_positions.size == 0:
                continue

            boundaries = list(bos_positions[1:]) + [len(tokens)]
            for start, end in zip(bos_positions, boundaries):
                if self.config.max_docs > 0 and docs_yielded >= self.config.max_docs:
                    return
                doc_ids = tokens[start + 1 : end].tolist()
                if not doc_ids:
                    continue
                yield self.tokenizer.decode(doc_ids, clean_up_tokenization_spaces=False)
                docs_yielded += 1

    def describe(self) -> dict:
        description = {
            "data_dir": str(self.data_dir),
            "source_name": self.source_name,
            "split": self.config.split,
            "max_shards": self.config.max_shards,
            "max_docs": self.config.max_docs,
            "tokenizer": self.tokenizer_name,
            "bos_token_id": self.bos_token_id,
            "shards": [str(path) for path in self.shard_paths],
        }
        description.update(self.corpus_role)
        if "crop_fraction" in self.meta:
            description["crop_fraction"] = float(self.meta["crop_fraction"])
        if "tokens_cropped_total" in self.meta:
            description["tokens_cropped_total"] = int(self.meta["tokens_cropped_total"])
        return description


@dataclass(frozen=True)
class TextLineCorpusConfig:
    data_dir: str
    glob_pattern: str = "*.train"
    max_docs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class TextLineCorpusReader:
    """Read raw text files and treat each non-empty line as one document."""

    corpus_role = {
        "corpus_format": "raw_text_lines",
        "baseline_role": "text_corpus_lexical_baseline",
        "interpretation": (
            "Uses raw text files as line-level documents for a word-level SGNS "
            "lexical baseline."
        ),
    }

    def __init__(self, config: TextLineCorpusConfig):
        self.config = config
        self.project_root = _project_root()
        self.data_dir = self._resolve_data_dir(config.data_dir)
        self.files = self._list_files()

    @property
    def source_name(self) -> str:
        return self.data_dir.name

    def _resolve_data_dir(self, raw_path: str) -> Path:
        requested = Path(raw_path).expanduser()
        candidates = (
            requested,
            self.project_root / requested,
            self.project_root / "data" / requested,
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find text corpus directory for: {raw_path}")

    def _list_files(self) -> list[Path]:
        files = sorted(self.data_dir.glob(self.config.glob_pattern))
        if not files:
            raise FileNotFoundError(
                f"No text files matching {self.config.glob_pattern!r} under {self.data_dir}"
            )
        return files

    def iter_documents(self) -> Iterator[str]:
        docs_yielded = 0
        for path in self.files:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if self.config.max_docs > 0 and docs_yielded >= self.config.max_docs:
                        return
                    text = line.strip()
                    if not text:
                        continue
                    yield text
                    docs_yielded += 1

    def describe(self) -> dict:
        description = {
            "data_dir": str(self.data_dir),
            "source_name": self.source_name,
            "glob_pattern": self.config.glob_pattern,
            "max_docs": self.config.max_docs,
            "files": [str(path) for path in self.files],
        }
        description.update(self.corpus_role)
        return description


@dataclass(frozen=True)
class ArrowTextCorpusConfig:
    arrow_path: str
    text_column: str = "text"
    max_docs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ArrowTextCorpusReader:
    """Read a Hugging Face Arrow dataset text column as reusable documents."""

    corpus_role = {
        "corpus_format": "hf_arrow_text_column",
        "baseline_role": "arrow_text_lexical_baseline",
        "interpretation": (
            "Uses one Hugging Face Arrow text-column row as one document for a "
            "word-level SGNS lexical baseline."
        ),
    }

    def __init__(self, config: ArrowTextCorpusConfig):
        self.config = config
        self.project_root = _project_root()
        self.arrow_path = self._resolve_arrow_path(config.arrow_path)
        try:
            from datasets import Dataset
        except Exception as exc:
            raise RuntimeError(
                "The `datasets` package is required for corpus_format='hf_arrow'."
            ) from exc
        self.dataset = Dataset.from_file(str(self.arrow_path))
        if config.text_column not in self.dataset.column_names:
            columns = ", ".join(self.dataset.column_names)
            raise ValueError(
                f"Text column {config.text_column!r} not found in {self.arrow_path}. "
                f"Available columns: {columns}"
            )

    @property
    def source_name(self) -> str:
        return self.arrow_path.stem

    def _resolve_arrow_path(self, raw_path: str) -> Path:
        requested = Path(raw_path).expanduser()
        candidates = (
            requested,
            self.project_root / requested,
            self.project_root / "data" / requested,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find Arrow corpus file for: {raw_path}")

    def iter_documents(self) -> Iterator[str]:
        docs_yielded = 0
        for row in self.dataset:
            if self.config.max_docs > 0 and docs_yielded >= self.config.max_docs:
                return
            text = str(row.get(self.config.text_column) or "").strip()
            if not text:
                continue
            yield text
            docs_yielded += 1

    def describe(self) -> dict:
        description = {
            "arrow_path": str(self.arrow_path),
            "source_name": self.source_name,
            "text_column": self.config.text_column,
            "max_docs": self.config.max_docs,
            "num_rows": int(len(self.dataset)),
        }
        description.update(self.corpus_role)
        return description


class CorpusReader(Protocol):
    @property
    def source_name(self) -> str:
        ...

    def iter_documents(self) -> Iterator[str]:
        ...

    def describe(self) -> dict:
        ...


CorpusConfig = ShardCorpusConfig | TextLineCorpusConfig | ArrowTextCorpusConfig


def build_corpus_reader(
    *,
    corpus_format: str,
    data_dir: str,
    split: str = "train",
    max_shards: int = 0,
    max_docs: int = 0,
    tokenizer_name: Optional[str] = None,
    use_fast_tokenizer: bool = True,
    glob_pattern: str = "*.train",
    text_column: str = "text",
) -> tuple[CorpusConfig, CorpusReader]:
    """Build a corpus config and reader for a supported storage format."""

    normalized = str(corpus_format).strip().lower()
    if normalized == "shard":
        normalized = "shard_bin"
    if normalized not in VALID_CORPUS_FORMATS:
        valid = ", ".join(VALID_CORPUS_FORMATS)
        raise ValueError(f"Unknown corpus_format {corpus_format!r}; expected one of: {valid}")

    if normalized == "shard_bin":
        config = ShardCorpusConfig(
            data_dir=data_dir,
            split=split,
            max_shards=max_shards,
            max_docs=max_docs,
            tokenizer_name=tokenizer_name,
            use_fast_tokenizer=use_fast_tokenizer,
        )
        return config, ShardCorpusReader(config)

    if normalized == "text_dir":
        config = TextLineCorpusConfig(
            data_dir=data_dir,
            glob_pattern=glob_pattern,
            max_docs=max_docs,
        )
        return config, TextLineCorpusReader(config)

    config = ArrowTextCorpusConfig(
        arrow_path=data_dir,
        text_column=text_column,
        max_docs=max_docs,
    )
    return config, ArrowTextCorpusReader(config)


__all__ = [
    "ArrowTextCorpusConfig",
    "ArrowTextCorpusReader",
    "CorpusConfig",
    "CorpusReader",
    "ShardCorpusConfig",
    "ShardCorpusReader",
    "TextLineCorpusConfig",
    "TextLineCorpusReader",
    "VALID_CORPUS_FORMATS",
    "build_corpus_reader",
    "describe_corpus_role",
]
