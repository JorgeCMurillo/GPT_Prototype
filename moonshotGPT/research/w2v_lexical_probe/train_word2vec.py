"""Train a Word2Vec lexical probe and evaluate it on EWoK.

How it works:
- Resolve a training corpus in one supported format:
  - BOS-delimited token shards (`shard_bin`)
  - raw line-oriented text directories (`text_dir`)
  - Hugging Face Arrow files with a text column (`hf_arrow`)
- Stream raw text documents from the corpus reader.
- Normalize the decoded text with the shared Word2Vec text pipeline.
- Build a gensim vocabulary from the resulting tokenized documents.
- Train a skip-gram negative-sampling model with CPU worker threads.
- Optionally save intermediate checkpoints after fixed epoch intervals.
- Optionally run the standard Word2Vec EWoK evaluator after training.
- Save the trained vectors, vocabulary, and run summary under the output root.

Corpus framing:
- `fineweb_edu_10B`-style shards act as a clean source-corpus lexical baseline.
- `fineweb_edu_10B_bosrow`-style shards act as an exposure baseline because they
  reflect the packed BOS-row format seen by GPT training, including packing and
  crop artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from gensim import __version__ as GENSIM_VERSION
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    _GENSIM_IMPORT_ERROR = None
except Exception as exc:
    GENSIM_VERSION = None
    Word2Vec = None
    CallbackAny2Vec = object
    _GENSIM_IMPORT_ERROR = exc

from .corpus import (
    VALID_CORPUS_FORMATS,
    CorpusConfig,
    CorpusReader,
    ShardCorpusConfig,
    build_corpus_reader,
)
from .eval_ewok_word2vec import evaluate_and_write_outputs, evaluate_loaded_run, load_ewok_eval_data
from .model import (
    LoadedWord2VecRun,
    Word2VecTrainingConfig,
    build_vocabulary_from_keyed_vectors,
    save_word2vec_run,
)
from .text import TextNormalizationConfig, WordTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "runs" / "research" / "w2v_lexical_probe"
_DEFAULT_TRAIN_CHUNK_WORDS = 5_000_000


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count // 2))


def _available_cpu_count() -> int:
    return os.cpu_count() or 1


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _require_gensim() -> None:
    if _GENSIM_IMPORT_ERROR is not None or Word2Vec is None:
        raise RuntimeError(
            "gensim is required to train Word2Vec runs. Install it with `pip install gensim`."
        ) from _GENSIM_IMPORT_ERROR


class _TokenizedDocumentIterable:
    """Reusable iterable over normalized word-token documents."""

    def __init__(
        self,
        reader: CorpusReader,
        tokenizer: WordTokenizer,
        *,
        total_documents: int | None = None,
        report_every_seconds: float | None = None,
        progress_desc: str | None = None,
        heartbeat_label: str = "w2v",
    ):
        self.reader = reader
        self.tokenizer = tokenizer
        self.total_documents = total_documents
        self.report_every_seconds = report_every_seconds
        self.progress_desc = progress_desc
        self.heartbeat_label = heartbeat_label
        self.last_pass_stats: dict[str, int] | None = None

    def __iter__(self):
        docs_seen = 0
        total_tokens_seen = 0
        last_report_time = time.monotonic()

        raw_documents = self.reader.iter_documents()
        progress = None
        if self.progress_desc is not None:
            progress = tqdm(
                raw_documents,
                desc=self.progress_desc,
                total=self.total_documents,
                unit="doc",
                leave=False,
                dynamic_ncols=True,
            )
            raw_documents = progress

        for text in raw_documents:
            tokens = self.tokenizer.tokenize(text)
            if not tokens:
                continue
            docs_seen += 1
            total_tokens_seen += len(tokens)
            if progress is not None and docs_seen % 256 == 0:
                progress.set_postfix(docs=docs_seen, toks=total_tokens_seen)
            if self.report_every_seconds is not None and self.report_every_seconds > 0:
                now = time.monotonic()
                if now - last_report_time >= self.report_every_seconds:
                    print(
                        f"[info] {self.heartbeat_label} heartbeat:"
                        f" docs={docs_seen}"
                        f" tokens={total_tokens_seen}"
                    )
                    last_report_time = now
            yield tokens

        if progress is not None:
            progress.set_postfix(docs=docs_seen, toks=total_tokens_seen)
            progress.close()

        self.last_pass_stats = {
            "documents_seen": int(docs_seen),
            "total_tokens_seen": int(total_tokens_seen),
        }


def _format_slug(reader: CorpusReader) -> str:
    return _safe_slug(str(reader.describe().get("corpus_format", "corpus")))


def _build_run_name(reader: CorpusReader, config: Word2VecTrainingConfig, corpus_config: CorpusConfig) -> str:
    parts = [
        "sgns",
        _safe_slug(reader.source_name),
    ]
    if not isinstance(corpus_config, ShardCorpusConfig):
        parts.append(_format_slug(reader))
    parts.extend(
        [
            f"d{config.embedding_dim}",
            f"w{config.window_size}",
            f"neg{config.negative_samples}",
            f"mc{config.min_count}",
            f"ep{config.epochs}",
            f"seed{config.seed}",
        ]
    )
    if isinstance(corpus_config, ShardCorpusConfig):
        parts.insert(-2, f"sh{corpus_config.max_shards}")
    if corpus_config.max_docs > 0:
        parts.append(f"docs{corpus_config.max_docs}")
    return "_".join(parts)


def _build_run_summary(
    *,
    run_name: str,
    reader: CorpusReader,
    corpus_config: CorpusConfig,
    text_config: TextNormalizationConfig,
    training_config: Word2VecTrainingConfig,
    vocab_stats: dict,
    training_stats: dict,
    checkpoint_epoch: int | None = None,
    periodic_eval_summary: dict | None = None,
) -> dict:
    summary = {
        "backend": "gensim",
        "gensim_version": GENSIM_VERSION,
        "workers_used": int(training_config.workers),
        "run_name": run_name,
        "corpus": reader.describe(),
        "corpus_config": corpus_config.to_dict(),
        "normalization": text_config.to_dict(),
        "training_config": training_config.to_dict(),
        "vocabulary": vocab_stats,
        "training": training_stats,
    }
    if periodic_eval_summary is not None:
        summary["periodic_eval"] = periodic_eval_summary
    if checkpoint_epoch is not None:
        summary["checkpoint"] = {
            "epoch": int(checkpoint_epoch),
            "kind": "intermediate",
        }
    return summary


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_last_jsonl_record(path: Path, *, record_type: str | None = None) -> dict | None:
    if not path.exists():
        return None
    last_match = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            if record_type is not None and record.get("type") != record_type:
                continue
            last_match = record
    return last_match


def _coerce_epoch_durations(payload) -> list[float]:
    if not isinstance(payload, list):
        return []
    out = []
    for value in payload:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _load_resume_state(resume_run_dir: str | Path) -> tuple[object, dict]:
    resume_path = Path(resume_run_dir).expanduser().resolve()
    model_path = resume_path / "gensim.model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot resume from {resume_path}: expected {model_path.name}. "
            "Resume requires a run/checkpoint saved with the full gensim model."
        )

    summary = _load_json_if_exists(resume_path / "train_summary.json")
    training_summary = summary.get("training", {}) if isinstance(summary, dict) else {}
    interval_metrics_path = resume_path / "ewok_interval_metrics.jsonl"
    prior_periodic = summary.get("periodic_eval", {}) if isinstance(summary, dict) else {}
    if isinstance(prior_periodic, dict):
        prior_interval_path = prior_periodic.get("interval_metrics_path")
        if prior_interval_path:
            interval_metrics_path = Path(prior_interval_path).expanduser()
    last_interval = _load_last_jsonl_record(interval_metrics_path, record_type="ewok_interval_eval")

    prior_epochs_completed = int(training_summary.get("epochs_completed", 0))
    estimated_raw_words = int(training_summary.get("raw_words", 0))
    estimated_effective_words = int(training_summary.get("effective_words", 0))

    model = Word2Vec.load(str(model_path))

    if estimated_raw_words <= 0 and prior_epochs_completed > 0:
        estimated_raw_words = int(model.corpus_total_words) * prior_epochs_completed
    if estimated_effective_words <= 0 and prior_epochs_completed > 0:
        estimated_effective_words = estimated_raw_words

    prior_words_trained_total = estimated_raw_words
    if isinstance(last_interval, dict):
        prior_words_trained_total = max(
            prior_words_trained_total,
            int(last_interval.get("words_trained_total", 0)),
        )

    return model, {
        "run_dir": resume_path,
        "summary": summary if isinstance(summary, dict) else {},
        "prior_epochs_completed": prior_epochs_completed,
        "prior_effective_words": estimated_effective_words,
        "prior_raw_words": estimated_raw_words,
        "prior_epoch_durations": _coerce_epoch_durations(training_summary.get("epoch_durations_seconds")),
        "prior_interval_eval_count": int(last_interval.get("eval_index", 0)) if isinstance(last_interval, dict) else 0,
        "prior_words_trained_total": int(prior_words_trained_total),
    }


def _validate_resume_configuration(
    *,
    resume_state: dict,
    corpus_config: CorpusConfig,
    text_config: TextNormalizationConfig,
    training_config: Word2VecTrainingConfig,
    eval_every_words: int,
    ewok_variant: str,
    ewok_text_preprocessing: str,
) -> None:
    summary = resume_state.get("summary", {})
    if not isinstance(summary, dict):
        return

    mismatches = []
    prior_corpus = summary.get("corpus_config", {})
    if isinstance(prior_corpus, dict):
        for key, current_value in corpus_config.to_dict().items():
            if key in prior_corpus and prior_corpus.get(key) != current_value:
                mismatches.append(
                    f"corpus_config.{key}: resume={prior_corpus.get(key)!r} current={current_value!r}"
                )

    prior_norm = summary.get("normalization", {})
    if isinstance(prior_norm, dict):
        for key, current_value in text_config.to_dict().items():
            if key in prior_norm and prior_norm.get(key) != current_value:
                mismatches.append(
                    f"normalization.{key}: resume={prior_norm.get(key)!r} current={current_value!r}"
                )

    prior_train = summary.get("training_config", {})
    if isinstance(prior_train, dict):
        train_checks = {
            "embedding_dim": training_config.embedding_dim,
            "window_size": training_config.window_size,
            "negative_samples": training_config.negative_samples,
            "min_count": training_config.min_count,
            "sample": training_config.sample,
            "sg": training_config.sg,
            "hs": training_config.hs,
        }
        for key, current_value in train_checks.items():
            if key in prior_train and prior_train.get(key) != current_value:
                mismatches.append(
                    f"training_config.{key}: resume={prior_train.get(key)!r} current={current_value!r}"
                )

    prior_periodic = summary.get("periodic_eval", {})
    if eval_every_words > 0 and isinstance(prior_periodic, dict):
        prior_every = int(prior_periodic.get("every_words", 0))
        if prior_every > 0 and prior_every != int(eval_every_words):
            mismatches.append(
                f"periodic_eval.every_words: resume={prior_every!r} current={int(eval_every_words)!r}"
            )
        prior_variant = str(prior_periodic.get("ewok_variant", "fast")).strip().lower()
        if prior_variant and prior_variant != str(ewok_variant).strip().lower():
            mismatches.append(
                f"periodic_eval.ewok_variant: resume={prior_variant!r} current={str(ewok_variant).strip().lower()!r}"
            )
        prior_preprocessing = str(prior_periodic.get("ewok_text_preprocessing", "probe")).strip().lower()
        if prior_preprocessing and prior_preprocessing != str(ewok_text_preprocessing).strip().lower():
            mismatches.append(
                "periodic_eval.ewok_text_preprocessing:"
                f" resume={prior_preprocessing!r} current={str(ewok_text_preprocessing).strip().lower()!r}"
            )

    if mismatches:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(
            "Resume configuration does not match the saved run:\n"
            f"{mismatch_text}"
        )


def _resolve_batch_words(args: argparse.Namespace) -> int:
    if args.batch_words is not None:
        if args.batch_size is not None:
            print(
                "[warn] --batch_size is deprecated and ignored because --batch_words was also provided."
            )
        return int(args.batch_words)
    if args.batch_size is not None:
        print("[warn] --batch_size is deprecated; use --batch_words instead.")
        return int(args.batch_size)
    return 10000


def _warn_deprecated_args(args: argparse.Namespace) -> None:
    if args.device is not None:
        print("[warn] --device is deprecated and ignored; gensim training is CPU-threaded.")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _resolve_train_chunk_words(eval_every_words: int) -> int:
    if eval_every_words > 0:
        return max(1, min(_DEFAULT_TRAIN_CHUNK_WORDS, int(eval_every_words)))
    return _DEFAULT_TRAIN_CHUNK_WORDS


def _snapshot_training_stats(
    *,
    model,
    completed_epochs: int,
    epoch_durations: list[float],
    train_result: tuple[int, int] | None = None,
    epoch_offset: int = 0,
    prior_effective_words: int = 0,
    prior_raw_words: int = 0,
    prior_epoch_durations: list[float] | None = None,
) -> dict:
    combined_epoch_durations = list(prior_epoch_durations or []) + [float(value) for value in epoch_durations]
    stats = {
        "epochs_completed": int(epoch_offset + completed_epochs),
        "epoch_durations_seconds": combined_epoch_durations,
        "corpus_count": int(model.corpus_count),
        "corpus_total_words": int(model.corpus_total_words),
    }
    if train_result is not None:
        stats["effective_words"] = int(prior_effective_words + int(train_result[0]))
        stats["raw_words"] = int(prior_raw_words + int(train_result[1]))
    elif prior_effective_words > 0 or prior_raw_words > 0:
        stats["effective_words"] = int(prior_effective_words)
        stats["raw_words"] = int(prior_raw_words)
    return stats


def _build_live_loaded_run(
    *,
    run_dir: Path,
    vocabulary,
    training_config: Word2VecTrainingConfig,
    summary: dict,
    model,
) -> LoadedWord2VecRun:
    vectors = np.asarray(model.wv.vectors, dtype=np.float32)
    return LoadedWord2VecRun(
        run_dir=run_dir,
        vocabulary=vocabulary,
        token_to_id=vocabulary.token_to_id,
        vectors=vectors,
        training_config=training_config.to_dict(),
        summary=summary,
        backend="gensim",
    )


def _save_epoch_checkpoint(
    *,
    run_dir: Path,
    run_name: str,
    reader: CorpusReader,
    corpus_config: CorpusConfig,
    text_config: TextNormalizationConfig,
    training_config: Word2VecTrainingConfig,
    vocabulary,
    vocab_stats: dict,
    training_stats: dict,
    model,
    epoch_num: int,
    periodic_eval_summary: dict | None = None,
) -> None:
    checkpoint_dir = run_dir / "checkpoints" / f"epoch_{epoch_num:04d}"
    checkpoint_summary = _build_run_summary(
        run_name=run_name,
        reader=reader,
        corpus_config=corpus_config,
        text_config=text_config,
        training_config=training_config,
        vocab_stats=vocab_stats,
        training_stats=training_stats,
        checkpoint_epoch=epoch_num,
        periodic_eval_summary=periodic_eval_summary,
    )
    save_word2vec_run(
        checkpoint_dir,
        model.wv,
        vocabulary,
        training_config,
        checkpoint_summary,
        gensim_model=model,
    )
    print(f"[info] saved checkpoint to {checkpoint_dir}")


def _iter_training_chunks(
    reader: CorpusReader,
    tokenizer: WordTokenizer,
    *,
    chunk_word_limit: int,
    total_documents: int | None,
    epoch_idx: int,
    total_epochs: int,
):
    docs_iterable = reader.iter_documents()
    progress = tqdm(
        docs_iterable,
        desc=f"w2v epoch {epoch_idx}/{total_epochs}",
        total=total_documents,
        unit="doc",
        leave=False,
        dynamic_ncols=True,
    )
    chunk_documents: list[list[str]] = []
    chunk_words = 0
    docs_seen = 0
    chunk_idx = 0

    for text in progress:
        tokens = tokenizer.tokenize(text)
        if not tokens:
            continue
        docs_seen += 1
        chunk_documents.append(tokens)
        chunk_words += len(tokens)
        if docs_seen % 256 == 0:
            progress.set_postfix(docs=docs_seen, chunk_words=chunk_words)
        if chunk_words >= chunk_word_limit:
            chunk_idx += 1
            yield chunk_documents, {
                "chunk_index": chunk_idx,
                "chunk_words": int(chunk_words),
                "chunk_docs": int(len(chunk_documents)),
                "docs_seen": int(docs_seen),
            }
            chunk_documents = []
            chunk_words = 0

    if chunk_documents:
        chunk_idx += 1
        yield chunk_documents, {
            "chunk_index": chunk_idx,
            "chunk_words": int(chunk_words),
            "chunk_docs": int(len(chunk_documents)),
            "docs_seen": int(docs_seen),
        }

    progress.set_postfix(docs=docs_seen, chunk_words=chunk_words)
    progress.close()


def _run_periodic_ewok_eval(
    *,
    model,
    run_dir: Path,
    vocabulary,
    training_config: Word2VecTrainingConfig,
    summary: dict,
    ewok_df,
    ewok_source,
    margin_eps: float,
    output_path: Path,
    eval_index: int,
    target_words: int,
    words_trained_total: int,
    epoch_num: int,
    filter_agent_names: bool,
    ewok_variant: str,
    ewok_text_preprocessing: str,
) -> dict:
    live_run = _build_live_loaded_run(
        run_dir=run_dir,
        vocabulary=vocabulary,
        training_config=training_config,
        summary=summary,
        model=model,
    )
    metrics_by_method = evaluate_loaded_run(
        live_run,
        ewok_df=ewok_df,
        return_per_item=False,
        margin_eps=margin_eps,
        return_all_methods=True,
        filter_agent_names=filter_agent_names,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    babylm_metrics = metrics_by_method["babylm_completion_choice"]
    ewok_context_metrics = metrics_by_method["ewok_context_sensitivity"]
    record = {
        "type": "ewok_interval_eval",
        "eval_index": int(eval_index),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "words_trained_total": int(words_trained_total),
        "scheduled_eval_words": int(target_words),
        "epoch": int(epoch_num),
        "margin_eps": float(margin_eps),
        "filter_ewok_agent_names": bool(filter_agent_names),
        "ewok_variant": ewok_variant,
        "ewok_text_preprocessing": ewok_text_preprocessing,
        "ewok_source": str(ewok_source),
        "domain_scores_official": babylm_metrics["domain_scores_official"],
        "domain_scores_full": babylm_metrics["domain_scores_full"],
        "domain_margin_stats": babylm_metrics["domain_margin_stats"],
        "ewok_context_sensitivity_domain_scores_official": ewok_context_metrics["domain_scores_official"],
        "ewok_context_sensitivity_domain_scores_full": ewok_context_metrics["domain_scores_full"],
        "metrics_by_method": metrics_by_method,
    }
    _append_jsonl(output_path, record)
    average_stats = babylm_metrics["domain_margin_stats"].get("average", {})
    print(
        "[info] periodic EWoK evaluation complete:"
        f" eval_index={eval_index}"
        f" words={words_trained_total}"
        f" acc_combined={average_stats.get('acc_combined', float('nan')):.4f}"
    )
    return record


class _EpochCallback(CallbackAny2Vec):
    def __init__(
        self,
        *,
        run_dir: Path,
        run_name: str,
        reader: CorpusReader,
        corpus_config: CorpusConfig,
        text_config: TextNormalizationConfig,
        training_config: Word2VecTrainingConfig,
        vocabulary,
        vocab_stats: dict,
        epoch_offset: int = 0,
        prior_effective_words: int = 0,
        prior_raw_words: int = 0,
        prior_epoch_durations: list[float] | None = None,
    ):
        self.run_dir = run_dir
        self.run_name = run_name
        self.reader = reader
        self.corpus_config = corpus_config
        self.text_config = text_config
        self.training_config = training_config
        self.vocabulary = vocabulary
        self.vocab_stats = vocab_stats
        self.epoch_offset = int(epoch_offset)
        self.prior_effective_words = int(prior_effective_words)
        self.prior_raw_words = int(prior_raw_words)
        self.prior_epoch_durations = list(prior_epoch_durations or [])
        self.current_epoch = 0
        self.epoch_durations: list[float] = []
        self._epoch_started_at: float | None = None

    def on_train_begin(self, model) -> None:
        print(
            "[info] starting gensim Word2Vec training:"
            f" epochs={self.training_config.epochs}"
            f" resume_epoch_offset={self.epoch_offset}"
            f" workers={self.training_config.workers}"
            f" batch_words={self.training_config.batch_words}"
        )

    def on_epoch_begin(self, model) -> None:
        self.current_epoch += 1
        self._epoch_started_at = time.monotonic()
        absolute_epoch = self.epoch_offset + self.current_epoch
        total_epochs = self.epoch_offset + self.training_config.epochs
        print(f"[info] w2v epoch {absolute_epoch}/{total_epochs} starting")

    def on_epoch_end(self, model) -> None:
        elapsed = 0.0
        if self._epoch_started_at is not None:
            elapsed = time.monotonic() - self._epoch_started_at
        self.epoch_durations.append(float(elapsed))
        absolute_epoch = self.epoch_offset + self.current_epoch
        total_epochs = self.epoch_offset + self.training_config.epochs
        print(
            f"[info] w2v epoch {absolute_epoch}/{total_epochs} finished"
            f" elapsed={elapsed:.1f}s"
        )
        if self.training_config.checkpoint_every_epochs <= 0:
            return
        if self.current_epoch % self.training_config.checkpoint_every_epochs != 0:
            return
        _save_epoch_checkpoint(
            run_dir=self.run_dir,
            run_name=self.run_name,
            reader=self.reader,
            corpus_config=self.corpus_config,
            text_config=self.text_config,
            training_config=self.training_config,
            vocabulary=self.vocabulary,
            vocab_stats=self.vocab_stats,
            training_stats=_snapshot_training_stats(
                model=model,
                completed_epochs=self.current_epoch,
                epoch_durations=self.epoch_durations,
                epoch_offset=self.epoch_offset,
                prior_effective_words=self.prior_effective_words,
                prior_raw_words=self.prior_raw_words,
                prior_epoch_durations=self.prior_epoch_durations,
            ),
            model=model,
            epoch_num=absolute_epoch,
        )


def _train_with_periodic_eval(
    *,
    model,
    run_dir: Path,
    run_name: str,
    reader: CorpusReader,
    corpus_config: CorpusConfig,
    text_config: TextNormalizationConfig,
    training_config: Word2VecTrainingConfig,
    vocabulary,
    vocab_stats: dict,
    tokenizer: WordTokenizer,
    total_documents: int | None,
    eval_every_words: int,
    eval_margin_eps: float,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
    initial_epochs_completed: int = 0,
    prior_effective_words: int = 0,
    prior_raw_words: int = 0,
    prior_epoch_durations: list[float] | None = None,
    prior_words_trained_total: int = 0,
    prior_interval_eval_count: int = 0,
) -> tuple[dict, dict]:
    ewok_df, ewok_source = load_ewok_eval_data(ewok_variant)
    interval_prefix = "ewok" if ewok_variant == "fast" else f"ewok_{ewok_variant}"
    interval_suffix = "" if ewok_text_preprocessing == "probe" else f"_{ewok_text_preprocessing}prep"
    interval_metrics_name = f"{interval_prefix}{interval_suffix}_interval_metrics.jsonl"
    interval_metrics_path = run_dir / interval_metrics_name
    if interval_metrics_path.exists() and prior_interval_eval_count <= 0 and prior_words_trained_total <= 0:
        interval_metrics_path.unlink()

    chunk_word_limit = _resolve_train_chunk_words(eval_every_words)
    session_training_words_target = max(1, int(model.corpus_total_words) * int(training_config.epochs))
    initial_alpha = float(model.alpha)
    final_alpha = float(model.min_alpha)

    epoch_durations: list[float] = []
    total_effective_words = 0
    total_raw_words = 0
    words_trained_total = int(prior_words_trained_total)
    words_trained_session = 0
    next_eval_words = ((words_trained_total // int(eval_every_words)) + 1) * int(eval_every_words)
    eval_records_written = int(prior_interval_eval_count)

    print(
        "[info] periodic EWoK evaluation enabled:"
        f" variant={ewok_variant}"
        f" text_preprocessing={ewok_text_preprocessing}"
        f" every_words={eval_every_words}"
        f" train_chunk_words={chunk_word_limit}"
        f" filter_agent_names={filter_agent_names}"
    )

    for relative_epoch_num in range(1, training_config.epochs + 1):
        epoch_num = initial_epochs_completed + relative_epoch_num
        epoch_started_at = time.monotonic()
        print(f"[info] w2v epoch {epoch_num}/{initial_epochs_completed + training_config.epochs} starting")
        for chunk_documents, chunk_meta in _iter_training_chunks(
            reader,
            tokenizer,
            chunk_word_limit=chunk_word_limit,
            total_documents=total_documents,
            epoch_idx=epoch_num,
            total_epochs=initial_epochs_completed + training_config.epochs,
        ):
            start_progress = words_trained_session / session_training_words_target
            end_progress = min(1.0, (words_trained_session + chunk_meta["chunk_words"]) / session_training_words_target)
            start_alpha = initial_alpha - (initial_alpha - final_alpha) * start_progress
            end_alpha = initial_alpha - (initial_alpha - final_alpha) * end_progress
            train_result = model.train(
                chunk_documents,
                total_examples=chunk_meta["chunk_docs"],
                total_words=chunk_meta["chunk_words"],
                epochs=1,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
            )
            total_effective_words += int(train_result[0])
            total_raw_words += int(train_result[1])
            words_trained_total += int(chunk_meta["chunk_words"])
            words_trained_session += int(chunk_meta["chunk_words"])

            while next_eval_words > 0 and words_trained_total >= next_eval_words:
                current_training_stats = _snapshot_training_stats(
                    model=model,
                    completed_epochs=relative_epoch_num,
                    epoch_durations=epoch_durations,
                    train_result=(total_effective_words, total_raw_words),
                    epoch_offset=initial_epochs_completed,
                    prior_effective_words=prior_effective_words,
                    prior_raw_words=prior_raw_words,
                    prior_epoch_durations=prior_epoch_durations,
                )
                current_periodic_eval = {
                    "enabled": True,
                    "every_words": int(eval_every_words),
                    "interval_metrics_path": str(interval_metrics_path),
                    "evaluations_completed": int(eval_records_written),
                    "filter_agent_names": bool(filter_agent_names),
                    "ewok_variant": ewok_variant,
                    "ewok_text_preprocessing": ewok_text_preprocessing,
                }
                current_summary = _build_run_summary(
                    run_name=run_name,
                    reader=reader,
                    corpus_config=corpus_config,
                    text_config=text_config,
                    training_config=training_config,
                    vocab_stats=vocab_stats,
                    training_stats=current_training_stats,
                    periodic_eval_summary=current_periodic_eval,
                )
                eval_records_written += 1
                _run_periodic_ewok_eval(
                    model=model,
                    run_dir=run_dir,
                    vocabulary=vocabulary,
                    training_config=training_config,
                    summary=current_summary,
                    ewok_df=ewok_df,
                    ewok_source=ewok_source,
                    margin_eps=eval_margin_eps,
                    output_path=interval_metrics_path,
                    eval_index=eval_records_written,
                    target_words=next_eval_words,
                    words_trained_total=words_trained_total,
                    epoch_num=epoch_num,
                    filter_agent_names=filter_agent_names,
                    ewok_variant=ewok_variant,
                    ewok_text_preprocessing=ewok_text_preprocessing,
                )
                next_eval_words += int(eval_every_words)

        elapsed = time.monotonic() - epoch_started_at
        epoch_durations.append(float(elapsed))
        print(f"[info] w2v epoch {epoch_num}/{initial_epochs_completed + training_config.epochs} finished elapsed={elapsed:.1f}s")
        if training_config.checkpoint_every_epochs > 0 and epoch_num % training_config.checkpoint_every_epochs == 0:
            _save_epoch_checkpoint(
                run_dir=run_dir,
                run_name=run_name,
                reader=reader,
                corpus_config=corpus_config,
                text_config=text_config,
                training_config=training_config,
                vocabulary=vocabulary,
                vocab_stats=vocab_stats,
                training_stats=_snapshot_training_stats(
                    model=model,
                    completed_epochs=relative_epoch_num,
                    epoch_durations=epoch_durations,
                    train_result=(total_effective_words, total_raw_words),
                    epoch_offset=initial_epochs_completed,
                    prior_effective_words=prior_effective_words,
                    prior_raw_words=prior_raw_words,
                    prior_epoch_durations=prior_epoch_durations,
                ),
                model=model,
                epoch_num=epoch_num,
                periodic_eval_summary={
                    "enabled": True,
                    "every_words": int(eval_every_words),
                    "interval_metrics_path": str(interval_metrics_path),
                    "evaluations_completed": int(eval_records_written),
                    "filter_agent_names": bool(filter_agent_names),
                    "ewok_variant": ewok_variant,
                    "ewok_text_preprocessing": ewok_text_preprocessing,
                },
            )

    return (
        _snapshot_training_stats(
            model=model,
            completed_epochs=training_config.epochs,
            epoch_durations=epoch_durations,
            train_result=(total_effective_words, total_raw_words),
            epoch_offset=initial_epochs_completed,
            prior_effective_words=prior_effective_words,
            prior_raw_words=prior_raw_words,
            prior_epoch_durations=prior_epoch_durations,
        ),
        {
            "enabled": True,
            "every_words": int(eval_every_words),
            "interval_metrics_path": str(interval_metrics_path),
            "evaluations_completed": int(eval_records_written),
            "filter_agent_names": bool(filter_agent_names),
            "ewok_variant": ewok_variant,
            "ewok_text_preprocessing": ewok_text_preprocessing,
        },
    )


def _resolve_max_docs(args: argparse.Namespace) -> int:
    max_docs = int(args.max_docs)
    max_lines = getattr(args, "max_lines", None)
    if max_lines is not None:
        max_lines = int(max_lines)
        if max_lines < 0:
            raise ValueError("--max_lines must be >= 0")
        if max_docs > 0 and max_docs != max_lines:
            print("[warn] --max_lines is ignored because --max_docs was also provided.")
        elif max_docs == 0:
            print("[warn] --max_lines is deprecated; use --max_docs instead.")
            max_docs = max_lines
    if max_docs < 0:
        raise ValueError("--max_docs must be >= 0")
    return max_docs


def _log_resolved_corpus(
    *,
    reader: CorpusReader,
    corpus_config: CorpusConfig,
    training_config: Word2VecTrainingConfig,
) -> None:
    description = reader.describe()
    source = description.get("data_dir") or description.get("arrow_path") or reader.source_name
    print(
        "[info] resolved corpus:"
        f" format={description.get('corpus_format', 'unknown')}"
        f" source={source}"
    )
    if isinstance(corpus_config, ShardCorpusConfig):
        print(
            "[info] shard selection:"
            f" split={corpus_config.split}"
            f" shards={len(description.get('shards', []))}"
            f" max_shards={corpus_config.max_shards}"
        )
    elif "files" in description:
        print(
            "[info] text file selection:"
            f" files={len(description['files'])}"
            f" glob={description.get('glob_pattern')}"
        )
    elif "num_rows" in description:
        print(
            "[info] arrow selection:"
            f" rows={description.get('num_rows')}"
            f" text_column={description.get('text_column')}"
        )
    if int(description.get("max_docs", 0) or 0) > 0:
        print(f"[info] document cap: {description['max_docs']}")
    print(
        "[info] worker configuration:"
        f" {training_config.workers}/{_available_cpu_count()}"
        f" ({(100.0 * training_config.workers / _available_cpu_count()):.0f}%)"
    )
    print(
        "[info] corpus framing:"
        f" role={description.get('baseline_role', 'unknown')}"
    )
    interpretation = description.get("interpretation")
    if interpretation:
        print(f"[info] {interpretation}")


def train(args: argparse.Namespace) -> Path:
    _require_gensim()
    _warn_deprecated_args(args)

    batch_words = _resolve_batch_words(args)
    if batch_words <= 0:
        raise ValueError("--batch_words must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.sample < 0:
        raise ValueError("--sample must be >= 0")
    if args.eval_every_words < 0:
        raise ValueError("--eval_every_words must be >= 0")
    if args.eval_margin_eps < 0:
        raise ValueError("--eval_margin_eps must be >= 0")
    max_docs = _resolve_max_docs(args)
    print(f"[info] loading {args.corpus_format} corpus from {args.data_dir}")
    corpus_config, reader = build_corpus_reader(
        corpus_format=args.corpus_format,
        data_dir=args.data_dir,
        split=args.split,
        max_shards=args.max_train_shards,
        max_docs=max_docs,
        tokenizer_name=args.tokenizer,
        glob_pattern=args.glob_pattern,
        text_column=args.text_column,
    )
    text_config = TextNormalizationConfig()

    resume_model = None
    resume_state = None
    if args.resume_run_dir is not None:
        resume_model, resume_state = _load_resume_state(args.resume_run_dir)
        if args.learning_rate != build_arg_parser().get_default("learning_rate"):
            print(
                "[warn] --learning_rate is ignored when resuming;"
                f" using saved model alpha={float(resume_model.alpha):.6f}"
            )

    learning_rate = float(resume_model.alpha) if resume_model is not None else float(args.learning_rate)
    training_config = Word2VecTrainingConfig(
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        min_count=args.min_count,
        epochs=args.epochs,
        learning_rate=learning_rate,
        batch_words=batch_words,
        seed=args.seed,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        workers=args.workers,
        sample=args.sample,
    )

    _set_all_seeds(training_config.seed)

    if resume_state is not None:
        _validate_resume_configuration(
            resume_state=resume_state,
            corpus_config=corpus_config,
            text_config=text_config,
            training_config=training_config,
            eval_every_words=args.eval_every_words,
            ewok_variant=args.ewok_variant,
            ewok_text_preprocessing=args.ewok_text_preprocessing,
        )

    _log_resolved_corpus(
        reader=reader,
        corpus_config=corpus_config,
        training_config=training_config,
    )
    if resume_state is not None:
        print(
            "[info] resuming Word2Vec run:"
            f" run_dir={resume_state['run_dir']}"
            f" prior_epochs={resume_state['prior_epochs_completed']}"
            f" prior_words={resume_state['prior_words_trained_total']}"
        )
        if args.output_root != build_arg_parser().get_default("output_root") or args.run_name is not None:
            print("[warn] --output_root/--run_name are ignored when --resume_run_dir is used.")

    total_documents = max_docs if max_docs > 0 else None
    word_tokenizer = WordTokenizer(text_config)

    if resume_model is None:
        vocab_documents = _TokenizedDocumentIterable(
            reader,
            word_tokenizer,
            total_documents=total_documents,
            report_every_seconds=(args.vocab_report_every_secs if args.vocab_report_every_secs > 0 else None),
            progress_desc="w2v vocab",
            heartbeat_label="vocab",
        )

        if args.vocab_report_every_secs > 0:
            print(f"[info] vocab heartbeat interval={args.vocab_report_every_secs:g}s")

        model = Word2Vec(
            sentences=None,
            vector_size=training_config.embedding_dim,
            window=training_config.window_size,
            min_count=training_config.min_count,
            sample=training_config.sample,
            seed=training_config.seed,
            workers=training_config.workers,
            alpha=training_config.learning_rate,
            sg=training_config.sg,
            hs=training_config.hs,
            negative=training_config.negative_samples,
            epochs=training_config.epochs,
            batch_words=training_config.batch_words,
            compute_loss=False,
        )

        print("[info] building Word2Vec vocabulary from detokenized documents")
        model.build_vocab(vocab_documents)
        run_name = args.run_name or _build_run_name(reader, training_config, corpus_config)
        run_dir = Path(args.output_root).expanduser().resolve() / run_name
    else:
        model = resume_model
        model.workers = training_config.workers
        model.batch_words = training_config.batch_words
        run_dir = resume_state["run_dir"]
        run_name = str(resume_state["summary"].get("run_name", run_dir.name))

    vocabulary = build_vocabulary_from_keyed_vectors(
        model.wv,
        total_tokens_seen=int(model.corpus_total_words),
    )
    vocab_stats = {
        "documents_seen": int(model.corpus_count),
        "vocab_size": vocabulary.size,
        "total_tokens_seen": int(model.corpus_total_words),
        "total_tokens_retained": vocabulary.total_tokens_retained,
        "min_count": training_config.min_count,
    }
    print(
        "[info] vocabulary ready:"
        f" docs={vocab_stats['documents_seen']}"
        f" tokens={vocab_stats['total_tokens_seen']}"
        f" vocab_size={vocabulary.size}"
    )
    if vocabulary.size == 0:
        raise ValueError("Vocabulary is empty after applying min_count; lower min_count or use more data.")

    run_dir.mkdir(parents=True, exist_ok=True)

    prior_epochs_completed = int(resume_state["prior_epochs_completed"]) if resume_state is not None else 0
    prior_effective_words = int(resume_state["prior_effective_words"]) if resume_state is not None else 0
    prior_raw_words = int(resume_state["prior_raw_words"]) if resume_state is not None else 0
    prior_epoch_durations = list(resume_state["prior_epoch_durations"]) if resume_state is not None else []
    prior_words_trained_total = int(resume_state["prior_words_trained_total"]) if resume_state is not None else 0
    prior_interval_eval_count = int(resume_state["prior_interval_eval_count"]) if resume_state is not None else 0

    if args.eval_every_words > 0:
        training_stats, periodic_eval_summary = _train_with_periodic_eval(
            model=model,
            run_dir=run_dir,
            run_name=run_name,
            reader=reader,
            corpus_config=corpus_config,
            text_config=text_config,
            training_config=training_config,
            vocabulary=vocabulary,
            vocab_stats=vocab_stats,
            tokenizer=word_tokenizer,
            total_documents=total_documents,
            eval_every_words=args.eval_every_words,
            eval_margin_eps=args.eval_margin_eps,
            filter_agent_names=args.filter_ewok_agent_names,
            ewok_variant=args.ewok_variant,
            ewok_text_preprocessing=args.ewok_text_preprocessing,
            initial_epochs_completed=prior_epochs_completed,
            prior_effective_words=prior_effective_words,
            prior_raw_words=prior_raw_words,
            prior_epoch_durations=prior_epoch_durations,
            prior_words_trained_total=prior_words_trained_total,
            prior_interval_eval_count=prior_interval_eval_count,
        )
    else:
        callback = _EpochCallback(
            run_dir=run_dir,
            run_name=run_name,
            reader=reader,
            corpus_config=corpus_config,
            text_config=text_config,
            training_config=training_config,
            vocabulary=vocabulary,
            vocab_stats=vocab_stats,
            epoch_offset=prior_epochs_completed,
            prior_effective_words=prior_effective_words,
            prior_raw_words=prior_raw_words,
            prior_epoch_durations=prior_epoch_durations,
        )
        train_documents = _TokenizedDocumentIterable(
            reader,
            word_tokenizer,
            total_documents=total_documents,
        )
        train_result = model.train(
            train_documents,
            total_examples=model.corpus_count,
            epochs=training_config.epochs,
            callbacks=[callback],
        )
        training_stats = _snapshot_training_stats(
            model=model,
            completed_epochs=callback.current_epoch,
            epoch_durations=callback.epoch_durations,
            train_result=train_result,
            epoch_offset=prior_epochs_completed,
            prior_effective_words=prior_effective_words,
            prior_raw_words=prior_raw_words,
            prior_epoch_durations=prior_epoch_durations,
        )
        periodic_eval_summary = None

    summary = _build_run_summary(
        run_name=run_name,
        reader=reader,
        corpus_config=corpus_config,
        text_config=text_config,
        training_config=training_config,
        vocab_stats=vocab_stats,
        training_stats=training_stats,
        periodic_eval_summary=periodic_eval_summary,
    )
    save_word2vec_run(run_dir, model.wv, vocabulary, training_config, summary, gensim_model=model)
    output_payload = {
        "run_dir": str(run_dir),
        "vocab_size": vocabulary.size,
        "workers_used": training_config.workers,
    }
    if periodic_eval_summary is not None:
        output_payload["ewok_interval_metrics"] = periodic_eval_summary["interval_metrics_path"]
    if args.eval_after_train:
        print(
            "[info] running final EWoK evaluation:"
            f" variant={args.ewok_variant}"
            f" preprocessing={args.ewok_text_preprocessing}"
        )
        final_eval_paths = evaluate_and_write_outputs(
            run_dir,
            margin_eps=args.eval_margin_eps,
            ewok_variant=args.ewok_variant,
            ewok_text_preprocessing=args.ewok_text_preprocessing,
            write_per_item=args.write_per_item,
            filter_agent_names=args.filter_ewok_agent_names,
            write_prediction_df=args.write_prediction_df,
        )
        output_payload["ewok_metrics"] = str(final_eval_paths["metrics"])
        if args.write_prediction_df:
            output_payload["ewok_word2vec_predictions"] = str(final_eval_paths["prediction_df"])
    if resume_state is not None:
        output_payload["resumed_from"] = str(resume_state["run_dir"])
    print(json.dumps(output_payload, indent=2))
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Word2Vec lexical probe and optionally evaluate it on EWoK.")
    parser.add_argument(
        "--corpus_format",
        choices=VALID_CORPUS_FORMATS,
        default="shard_bin",
        help="Training corpus storage format: token shards, text files, or a Hugging Face Arrow file",
    )
    parser.add_argument("--data_dir", required=True, help="Corpus directory/file path for the selected --corpus_format")
    parser.add_argument("--split", default="train", help="Shard split to read when --corpus_format=shard_bin")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer name/path from shard meta.json")
    parser.add_argument("--max_train_shards", type=int, default=0, help="Shard cap for shard_bin; 0 means all shards")
    parser.add_argument("--max_docs", type=int, default=0, help="Optional cap on documents/lines/rows; 0 means all")
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Deprecated alias for --max_docs when using line or Arrow corpora",
    )
    parser.add_argument("--glob_pattern", default="*.train", help="File glob for --corpus_format=text_dir")
    parser.add_argument("--text_column", default="text", help="Text column for --corpus_format=hf_arrow")
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--negative_samples", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--batch_words", type=int, default=None, help="gensim batch size measured in words")
    parser.add_argument("--batch_size", type=int, default=None, help="Deprecated alias for --batch_words")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Number of CPU worker threads for gensim")
    parser.add_argument("--sample", type=float, default=1e-3, help="Downsampling threshold for frequent tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_every_epochs",
        type=int,
        default=0,
        help="Write an intermediate checkpoint every N epochs; 0 disables intermediate checkpoints",
    )
    parser.add_argument(
        "--eval_every_words",
        type=int,
        default=0,
        help="Run in-memory EWoK evaluation every N post-normalization Word2Vec training words; 0 disables it",
    )
    parser.add_argument(
        "--eval_margin_eps",
        type=float,
        default=1e-6,
        help="Near-tie threshold used for periodic EWoK evaluations",
    )
    parser.add_argument(
        "--filter_ewok_agent_names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop recurring EWoK filler agent names during periodic EWoK evaluation only (default: on)",
    )
    parser.add_argument(
        "--ewok_variant",
        choices=("fast", "full"),
        default="fast",
        help="Which EWoK dataset to use for periodic evaluation: fast subset or full filtered set (default: fast)",
    )
    parser.add_argument(
        "--ewok_text_preprocessing",
        choices=("probe", "paper"),
        default="probe",
        help="How to tokenize EWoK text during periodic evaluation: probe-style tokenizer or paper-style lowercase/punctuation-strip split",
    )
    parser.add_argument(
        "--eval_after_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the standard EWoK Word2Vec evaluator after saving the final vectors (default: off)",
    )
    parser.add_argument("--write_per_item", action="store_true", help="With --eval_after_train, also write EWoK per-item JSONL")
    parser.add_argument(
        "--write_prediction_df",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --eval_after_train, write merged EWoK prediction CSVs (default: on)",
    )
    parser.add_argument("--device", default=None, help="Deprecated no-op; gensim training is CPU-threaded")
    parser.add_argument(
        "--vocab_report_every_secs",
        type=float,
        default=180.0,
        help="Emit an explicit vocabulary heartbeat every N seconds during vocab building; 0 disables it",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory where runs are written")
    parser.add_argument("--run_name", default=None, help="Optional override for the run directory name")
    parser.add_argument(
        "--resume_run_dir",
        default=None,
        help="Optional run/checkpoint directory containing gensim.model to resume training in place",
    )
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
