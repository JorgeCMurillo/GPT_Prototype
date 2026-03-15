"""Corpus readers for detokenizing GPT-2 shard datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from transformers import AutoTokenizer


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


__all__ = ["ShardCorpusConfig", "ShardCorpusReader", "describe_corpus_role"]
