"""Backend-neutral Word2Vec run persistence for lexical probing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Vocabulary:
    id_to_token: list[str]
    counts: list[int]
    total_tokens_seen: int
    total_tokens_retained: int

    def __post_init__(self):
        if len(self.id_to_token) != len(self.counts):
            raise ValueError("id_to_token and counts must have the same length")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def to_json_dict(self) -> dict:
        return {
            "id_to_token": self.id_to_token,
            "counts": self.counts,
            "total_tokens_seen": self.total_tokens_seen,
            "total_tokens_retained": self.total_tokens_retained,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "Vocabulary":
        return cls(
            id_to_token=list(payload["id_to_token"]),
            counts=[int(value) for value in payload["counts"]],
            total_tokens_seen=int(payload["total_tokens_seen"]),
            total_tokens_retained=int(payload["total_tokens_retained"]),
        )


@dataclass(frozen=True)
class Word2VecTrainingConfig:
    embedding_dim: int = 300
    window_size: int = 5
    negative_samples: int = 5
    min_count: int = 5
    epochs: int = 3
    learning_rate: float = 0.025
    batch_words: int = 10000
    seed: int = 42
    checkpoint_every_epochs: int = 0
    workers: int = 1
    sample: float = 1e-3
    sg: int = 1
    hs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LoadedWord2VecRun:
    run_dir: Path
    vocabulary: Vocabulary
    token_to_id: dict[str, int]
    vectors: np.ndarray
    training_config: dict
    summary: dict
    backend: str

    def vector_for_token(self, token: str) -> np.ndarray | None:
        token_id = self.token_to_id.get(token)
        if token_id is None:
            return None
        return self.vectors[token_id]

    def mean_pool(self, tokens: Sequence[str]) -> tuple[np.ndarray | None, list[str]]:
        kept_tokens = [token for token in tokens if token in self.token_to_id]
        if not kept_tokens:
            return None, []
        token_ids = [self.token_to_id[token] for token in kept_tokens]
        pooled = self.vectors[token_ids].mean(axis=0)
        return pooled.astype(np.float32, copy=False), kept_tokens


def _require_gensim_keyed_vectors():
    try:
        from gensim.models import KeyedVectors
    except Exception as exc:
        raise RuntimeError(
            "gensim is required for gensim-based Word2Vec runs. Install it with `pip install gensim`."
        ) from exc
    return KeyedVectors


def build_vocabulary_from_keyed_vectors(
    keyed_vectors,
    total_tokens_seen: int | None = None,
) -> Vocabulary:
    id_to_token = list(keyed_vectors.index_to_key)
    counts = [int(keyed_vectors.get_vecattr(token, "count")) for token in id_to_token]
    total_tokens_retained = int(sum(counts))
    if total_tokens_seen is None:
        total_tokens_seen = total_tokens_retained
    return Vocabulary(
        id_to_token=id_to_token,
        counts=counts,
        total_tokens_seen=int(total_tokens_seen),
        total_tokens_retained=total_tokens_retained,
    )


def save_word2vec_run(
    run_dir: Path,
    keyed_vectors,
    vocabulary: Vocabulary,
    training_config: Word2VecTrainingConfig,
    summary: dict,
    gensim_model=None,
) -> None:
    _require_gensim_keyed_vectors()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary.to_json_dict(), f, indent=2)
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    keyed_vectors.save(str(run_dir / "vectors.kv"))
    if gensim_model is not None:
        gensim_model.save(str(run_dir / "gensim.model"))


def _load_new_gensim_run(run_path: Path, vocabulary: Vocabulary, summary: dict) -> LoadedWord2VecRun:
    KeyedVectors = _require_gensim_keyed_vectors()
    keyed_vectors = KeyedVectors.load(str(run_path / "vectors.kv"), mmap="r")
    config = dict(summary.get("training_config", {}))
    vectors = np.asarray(keyed_vectors.vectors, dtype=np.float32)
    return LoadedWord2VecRun(
        run_dir=run_path,
        vocabulary=vocabulary,
        token_to_id=vocabulary.token_to_id,
        vectors=vectors,
        training_config=config,
        summary=summary,
        backend=str(summary.get("backend", "gensim")),
    )


def _load_legacy_pytorch_run(run_path: Path, vocabulary: Vocabulary, summary: dict) -> LoadedWord2VecRun:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "torch is required to load legacy PyTorch Word2Vec runs."
        ) from exc

    checkpoint = torch.load(run_path / "model.pt", map_location="cpu")
    config = dict(checkpoint.get("training_config", {}))
    state_dict = checkpoint["model_state_dict"]
    vectors_tensor = state_dict["input_embeddings.weight"]
    vectors = vectors_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if "training_config" not in summary:
        summary = dict(summary)
        summary["training_config"] = config
    return LoadedWord2VecRun(
        run_dir=run_path,
        vocabulary=vocabulary,
        token_to_id=vocabulary.token_to_id,
        vectors=vectors,
        training_config=config,
        summary=summary,
        backend=str(summary.get("backend", "pytorch_legacy")),
    )


def load_word2vec_run(run_dir: str | Path) -> LoadedWord2VecRun:
    run_path = Path(run_dir).expanduser().resolve()
    with (run_path / "vocab.json").open("r", encoding="utf-8") as f:
        vocabulary = Vocabulary.from_json_dict(json.load(f))
    with (run_path / "train_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if (run_path / "vectors.kv").exists():
        return _load_new_gensim_run(run_path, vocabulary, summary)
    if (run_path / "model.pt").exists():
        return _load_legacy_pytorch_run(run_path, vocabulary, summary)
    raise FileNotFoundError(
        f"Could not find a supported Word2Vec payload in {run_path}; expected vectors.kv or model.pt."
    )


__all__ = [
    "LoadedWord2VecRun",
    "Vocabulary",
    "Word2VecTrainingConfig",
    "build_vocabulary_from_keyed_vectors",
    "load_word2vec_run",
    "save_word2vec_run",
]
