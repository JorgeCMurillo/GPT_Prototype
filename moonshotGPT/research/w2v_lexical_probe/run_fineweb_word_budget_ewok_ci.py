"""Run word-budget Word2Vec EWoK experiments on FineWeb-Edu shards.

The experiment asks: if Word2Vec only sees N normalized words from FineWeb-Edu,
how much EWoK signal does it recover, and how variable is that estimate?

For each requested word budget and replicate, this script:

1. Samples a deterministic random document order from token shards.
2. Streams normalized word-token documents until the word budget is reached.
3. Trains a gensim SGNS Word2Vec model.
4. Evaluates the saved vectors on EWoK.
5. Aggregates domain scores across replicates and plots confidence intervals.

Example:

```bash
python -m research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci \
  --data_dir data/processed/fineweb_edu_10B \
  --word_budget 10000000 \
  --word_budget 100000000 \
  --replicates 5 \
  --workers 44 \
  --ewok_variant full
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from gensim import __version__ as GENSIM_VERSION
    from gensim.models import Word2Vec
    _GENSIM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in minimal envs
    GENSIM_VERSION = None
    Word2Vec = None
    _GENSIM_IMPORT_ERROR = exc

from .compare_ewok_word2vec_runs import (
    BABYLM_COMPLETION_CHOICE,
    DOMAIN_ORDER,
    EWOK_CONTEXT_SENSITIVITY,
    VALID_METHODS,
    VALID_SCORE_KINDS,
    extract_scores,
)
from .corpus import ShardCorpusConfig, ShardCorpusReader
from .eval_ewok_word2vec import evaluate_and_write_outputs, get_ewok_output_paths
from .model import Word2VecTrainingConfig, build_vocabulary_from_keyed_vectors, save_word2vec_run
from .text import TextNormalizationConfig, WordTokenizer


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "runs" / "research" / "w2v_lexical_probe" / "fineweb_word_budget_ewok_ci"
DEFAULT_WORD_BUDGETS = (10_000_000, 100_000_000)
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class WordBudgetSampleConfig:
    data_dir: str
    split: str
    max_train_shards: int
    word_budget: int
    sample_seed: int
    shuffle_documents: bool
    tokenizer_name: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class RandomShardWordBudgetCorpus:
    """Repeatable random document-order sample from BOS-delimited token shards."""

    def __init__(
        self,
        *,
        reader: ShardCorpusReader,
        word_tokenizer: WordTokenizer,
        config: WordBudgetSampleConfig,
    ):
        if config.word_budget <= 0:
            raise ValueError("word_budget must be > 0")
        self.reader = reader
        self.word_tokenizer = word_tokenizer
        self.config = config
        self.last_pass_stats: dict[str, int] | None = None

    def __iter__(self) -> Iterator[list[str]]:
        rng = np.random.default_rng(self.config.sample_seed)
        shard_order = np.arange(len(self.reader.shard_paths))
        rng.shuffle(shard_order)

        words_yielded = 0
        docs_seen = 0
        docs_yielded = 0
        shards_seen = 0

        for shard_pos in shard_order:
            if words_yielded >= self.config.word_budget:
                break
            shards_seen += 1
            shard_path = self.reader.shard_paths[int(shard_pos)]
            for text in self._iter_shard_documents(shard_path, rng):
                docs_seen += 1
                tokens = self.word_tokenizer.tokenize(text)
                if not tokens:
                    continue
                remaining = self.config.word_budget - words_yielded
                if remaining <= 0:
                    break
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]
                words_yielded += len(tokens)
                docs_yielded += 1
                yield tokens
                if words_yielded >= self.config.word_budget:
                    break

        self.last_pass_stats = {
            "word_budget": int(self.config.word_budget),
            "words_yielded": int(words_yielded),
            "documents_seen": int(docs_seen),
            "documents_yielded": int(docs_yielded),
            "shards_seen": int(shards_seen),
        }

    def _iter_shard_documents(self, shard_path: Path, rng) -> Iterator[str]:
        token_stream = np.memmap(shard_path, dtype=np.uint16, mode="r")
        bos_positions = np.flatnonzero(token_stream == self.reader.bos_token_id)
        if bos_positions.size == 0:
            return

        starts = bos_positions
        ends = np.concatenate([bos_positions[1:], np.array([len(token_stream)], dtype=bos_positions.dtype)])
        doc_order = np.arange(len(starts))
        if self.config.shuffle_documents:
            rng.shuffle(doc_order)

        for doc_idx in doc_order:
            start = int(starts[int(doc_idx)]) + 1
            end = int(ends[int(doc_idx)])
            if end <= start:
                continue
            doc_ids = token_stream[start:end].tolist()
            if doc_ids:
                yield self.reader.tokenizer.decode(doc_ids, clean_up_tokenization_spaces=False)

    def describe(self) -> dict:
        description = self.reader.describe()
        description["sample"] = {
            "strategy": "random_shard_order_random_doc_order_word_budget",
            "word_budget": int(self.config.word_budget),
            "sample_seed": int(self.config.sample_seed),
            "shuffle_documents": bool(self.config.shuffle_documents),
        }
        return description


def _require_gensim() -> None:
    if _GENSIM_IMPORT_ERROR is not None or Word2Vec is None:
        raise RuntimeError("gensim is required to train Word2Vec runs.") from _GENSIM_IMPORT_ERROR


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "run"


def _format_word_count(value: int) -> str:
    value = int(value)
    if value % 1_000_000_000 == 0:
        return f"{value // 1_000_000_000}B"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_run_name(args: argparse.Namespace, *, budget: int, replicate_index: int, sample_seed: int) -> str:
    corpus_name = _safe_slug(Path(args.data_dir).expanduser().name)
    return "_".join(
        [
            "sgns",
            corpus_name,
            f"words{_format_word_count(budget)}",
            f"rep{replicate_index:02d}",
            f"d{args.embedding_dim}",
            f"w{args.window_size}",
            f"neg{args.negative_samples}",
            f"mc{args.min_count}",
            f"ep{args.epochs}",
            f"seed{sample_seed}",
        ]
    )


def _train_replicate(
    args: argparse.Namespace,
    *,
    budget: int,
    replicate_index: int,
    sample_seed: int,
) -> Path:
    _require_gensim()
    _set_all_seeds(sample_seed)

    run_name = _build_run_name(
        args,
        budget=budget,
        replicate_index=replicate_index,
        sample_seed=sample_seed,
    )
    run_dir = Path(args.output_root).expanduser().resolve() / f"words_{_format_word_count(budget)}" / run_name
    output_paths = get_ewok_output_paths(run_dir, args.ewok_variant, args.ewok_text_preprocessing)
    if args.skip_existing and output_paths["metrics"].exists():
        print(f"[info] skipping existing replicate: {run_dir}")
        return run_dir
    if run_dir.exists() and not args.overwrite and not args.skip_existing:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Pass --skip_existing or --overwrite."
        )
    if args.dry_run:
        print(f"[dry-run] would train budget={budget} replicate={replicate_index} seed={sample_seed} -> {run_dir}")
        return run_dir

    shard_config = ShardCorpusConfig(
        data_dir=args.data_dir,
        split=args.split,
        max_shards=args.max_train_shards,
        max_docs=0,
        tokenizer_name=args.tokenizer,
    )
    reader = ShardCorpusReader(shard_config)
    text_config = TextNormalizationConfig()
    word_tokenizer = WordTokenizer(text_config)
    sample_config = WordBudgetSampleConfig(
        data_dir=args.data_dir,
        split=args.split,
        max_train_shards=args.max_train_shards,
        word_budget=budget,
        sample_seed=sample_seed,
        shuffle_documents=args.shuffle_documents,
        tokenizer_name=args.tokenizer,
    )
    corpus = RandomShardWordBudgetCorpus(
        reader=reader,
        word_tokenizer=word_tokenizer,
        config=sample_config,
    )
    training_config = Word2VecTrainingConfig(
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        min_count=args.min_count,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_words=args.batch_words,
        seed=sample_seed,
        workers=args.workers,
        sample=args.sample,
    )

    print(
        "[info] training replicate:"
        f" budget={_format_word_count(budget)}"
        f" replicate={replicate_index}/{args.replicates}"
        f" seed={sample_seed}"
        f" run_dir={run_dir}"
    )
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

    vocab_started = time.monotonic()
    model.build_vocab(corpus)
    vocab_elapsed = time.monotonic() - vocab_started
    vocabulary = build_vocabulary_from_keyed_vectors(
        model.wv,
        total_tokens_seen=int(model.corpus_total_words),
    )
    if vocabulary.size == 0:
        raise ValueError("Vocabulary is empty; lower --min_count or increase --word_budget.")

    train_started = time.monotonic()
    train_result = model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=training_config.epochs,
    )
    train_elapsed = time.monotonic() - train_started
    training_stats = {
        "epochs_completed": int(training_config.epochs),
        "epoch_durations_seconds": [float(train_elapsed)],
        "corpus_count": int(model.corpus_count),
        "corpus_total_words": int(model.corpus_total_words),
        "effective_words": int(train_result[0]),
        "raw_words": int(train_result[1]),
    }
    vocab_stats = {
        "documents_seen": int(model.corpus_count),
        "vocab_size": int(vocabulary.size),
        "total_tokens_seen": int(model.corpus_total_words),
        "total_tokens_retained": int(vocabulary.total_tokens_retained),
        "min_count": int(training_config.min_count),
        "vocab_build_seconds": float(vocab_elapsed),
    }
    summary = {
        "backend": "gensim",
        "gensim_version": GENSIM_VERSION,
        "workers_used": int(training_config.workers),
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus.describe(),
        "corpus_config": sample_config.to_dict(),
        "normalization": text_config.to_dict(),
        "training_config": training_config.to_dict(),
        "vocabulary": vocab_stats,
        "training": training_stats,
        "experiment": {
            "kind": "fineweb_word_budget_ewok_ci",
            "budget_words": int(budget),
            "replicate_index": int(replicate_index),
            "sample_seed": int(sample_seed),
        },
    }
    save_word2vec_run(run_dir, model.wv, vocabulary, training_config, summary, gensim_model=model)
    evaluate_and_write_outputs(
        run_dir,
        margin_eps=args.eval_margin_eps,
        ewok_variant=args.ewok_variant,
        ewok_text_preprocessing=args.ewok_text_preprocessing,
        write_per_item=args.write_per_item,
        filter_agent_names=args.filter_ewok_agent_names,
        write_prediction_df=args.write_prediction_df,
    )
    return run_dir


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    return 1.96


def summarize_domain_scores(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["word_budget"]), str(row["domain"])), []).append(float(row["score"]))

    def sort_key(item: tuple[tuple[int, str], list[float]]) -> tuple[int, int, str]:
        budget, domain = item[0]
        if domain in DOMAIN_ORDER:
            return budget, DOMAIN_ORDER.index(domain), domain
        return budget, len(DOMAIN_ORDER), domain

    summaries = []
    for (budget, domain), values in sorted(grouped.items(), key=sort_key):
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean()) if n else float("nan")
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
        ci_radius = float(_t_critical_95(n) * sem)
        summaries.append(
            {
                "word_budget": budget,
                "budget_label": _format_word_count(budget),
                "domain": domain,
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "ci95_low": max(0.0, mean - ci_radius),
                "ci95_high": min(1.0, mean + ci_radius),
                "ci95_radius": ci_radius,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_replicate_scores(
    *,
    run_dirs: dict[tuple[int, int], Path],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for (budget, replicate_index), run_dir in sorted(run_dirs.items()):
        metrics_path = get_ewok_output_paths(
            run_dir,
            args.ewok_variant,
            args.ewok_text_preprocessing,
        )["metrics"]
        if args.dry_run:
            continue
        payload = _load_json(metrics_path)
        scores = extract_scores(payload, method=args.method, score_kind=args.score_kind)
        for domain in DOMAIN_ORDER:
            if domain not in scores:
                continue
            rows.append(
                {
                    "word_budget": int(budget),
                    "budget_label": _format_word_count(budget),
                    "replicate": int(replicate_index),
                    "seed": int(args.seed + replicate_index - 1),
                    "run_dir": str(run_dir),
                    "domain": domain,
                    "score": float(scores[domain]),
                    "method": args.method,
                    "score_kind": args.score_kind,
                    "ewok_variant": args.ewok_variant,
                }
            )
    return rows


def _format_average_summary(summary_rows: list[dict]) -> str | None:
    average_rows = [row for row in summary_rows if row["domain"] == "average"]
    if not average_rows:
        return None

    lines = ["Average score"]
    for row in sorted(average_rows, key=lambda item: int(item["word_budget"])):
        lines.append(
            "{budget}: {mean:.3f} (95% CI [{low:.3f}, {high:.3f}])".format(
                budget=row["budget_label"],
                mean=float(row["mean"]),
                low=float(row["ci95_low"]),
                high=float(row["ci95_high"]),
            )
        )
    return "\n".join(lines)


def _plot_summary(summary_rows: list[dict], output_path: Path, *, title: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib is unavailable; skipping plot: {exc}")
        return None

    domains = [domain for domain in DOMAIN_ORDER if domain != "average"]
    if any(row["domain"] == "average" for row in summary_rows):
        domains.append("average")
    budgets = sorted({int(row["word_budget"]) for row in summary_rows})
    by_key = {(int(row["word_budget"]), str(row["domain"])): row for row in summary_rows}

    x = np.arange(len(domains), dtype=np.float64)
    width = min(0.8 / max(1, len(budgets)), 0.34)
    fig_w = max(12.0, 0.75 * len(domains) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    for idx, budget in enumerate(budgets):
        offset = (idx - (len(budgets) - 1) / 2.0) * width
        means = []
        yerr = [[], []]
        for domain in domains:
            row = by_key.get((budget, domain))
            if row is None:
                means.append(np.nan)
                yerr[0].append(0.0)
                yerr[1].append(0.0)
                continue
            means.append(row["mean"])
            yerr[0].append(row["mean"] - row["ci95_low"])
            yerr[1].append(row["ci95_high"] - row["mean"])
        ax.errorbar(
            x + offset,
            means,
            yerr=np.asarray(yerr),
            fmt="o",
            capsize=4,
            linewidth=1.4,
            markersize=5,
            label=_format_word_count(budget),
        )

    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        0.985,
        0.505,
        "Chance (50%)",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=9,
        color="#b22222",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_ylabel("EWoK score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=35, ha="right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(title="Word budget")
    average_summary = _format_average_summary(summary_rows)
    if average_summary:
        ax.text(
            0.015,
            0.98,
            average_summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#bdbdbd",
                "alpha": 0.92,
            },
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FineWeb-Edu word-budget W2V replicates and plot EWoK CIs.")
    parser.add_argument("--data_dir", default="data/processed/fineweb_edu_10B", help="FineWeb-Edu shard dataset directory")
    parser.add_argument("--split", default="train", help="Shard split prefix")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer name/path from meta.json")
    parser.add_argument("--max_train_shards", type=int, default=0, help="Shard cap; 0 means all shards")
    parser.add_argument("--word_budget", type=int, action="append", default=None, help="Normalized word budget. Repeat for multiple budgets.")
    parser.add_argument("--replicates", type=int, default=5, help="Replicates per word budget")
    parser.add_argument("--seed", type=int, default=42, help="Base sample/training seed")
    parser.add_argument("--shuffle_documents", action=argparse.BooleanOptionalAction, default=True, help="Shuffle document order within sampled shards")
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--negative_samples", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--batch_words", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample", type=float, default=1e-3)
    parser.add_argument("--ewok_variant", choices=("fast", "full"), default="full")
    parser.add_argument("--ewok_text_preprocessing", choices=("probe", "paper"), default="probe")
    parser.add_argument("--method", choices=VALID_METHODS, default=BABYLM_COMPLETION_CHOICE)
    parser.add_argument("--score_kind", choices=VALID_SCORE_KINDS, default="pair_average")
    parser.add_argument("--eval_margin_eps", type=float, default=1e-6)
    parser.add_argument("--filter_ewok_agent_names", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_per_item", action="store_true")
    parser.add_argument("--write_prediction_df", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--skip_existing", action="store_true", help="Reuse runs that already have EWoK metrics")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing run artifacts")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without training")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.replicates <= 0:
        raise ValueError("--replicates must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.batch_words <= 0:
        raise ValueError("--batch_words must be > 0")
    word_budgets = args.word_budget or list(DEFAULT_WORD_BUDGETS)
    if any(int(budget) <= 0 for budget in word_budgets):
        raise ValueError("All --word_budget values must be > 0")

    run_dirs: dict[tuple[int, int], Path] = {}
    for budget in word_budgets:
        for replicate_index in range(1, args.replicates + 1):
            sample_seed = int(args.seed + replicate_index - 1)
            run_dirs[(int(budget), replicate_index)] = _train_replicate(
                args,
                budget=int(budget),
                replicate_index=replicate_index,
                sample_seed=sample_seed,
            )

    output_root = Path(args.output_root).expanduser().resolve()
    experiment_tag = f"{args.method}_{args.score_kind}_{args.ewok_variant}"
    replicate_rows = _load_replicate_scores(run_dirs=run_dirs, args=args)
    if args.dry_run:
        print(json.dumps({"planned_runs": {str(key): str(path) for key, path in run_dirs.items()}}, indent=2))
        return

    summary_rows = summarize_domain_scores(replicate_rows)
    replicate_csv = _write_csv(
        output_root / f"{experiment_tag}_replicate_scores.csv",
        replicate_rows,
        [
            "word_budget",
            "budget_label",
            "replicate",
            "seed",
            "run_dir",
            "domain",
            "score",
            "method",
            "score_kind",
            "ewok_variant",
        ],
    )
    summary_csv = _write_csv(
        output_root / f"{experiment_tag}_domain_ci95.csv",
        summary_rows,
        [
            "word_budget",
            "budget_label",
            "domain",
            "n",
            "mean",
            "sd",
            "sem",
            "ci95_low",
            "ci95_high",
            "ci95_radius",
        ],
    )
    plot_path = _plot_summary(
        summary_rows,
        output_root / f"{experiment_tag}_domain_ci95.png",
        title=f"FineWeb-Edu Word2Vec on EWoK ({args.method}, {args.score_kind})",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(Path(args.data_dir).expanduser()),
        "word_budgets": [int(value) for value in word_budgets],
        "replicates": int(args.replicates),
        "seed": int(args.seed),
        "ewok_variant": args.ewok_variant,
        "ewok_text_preprocessing": args.ewok_text_preprocessing,
        "method": args.method,
        "score_kind": args.score_kind,
        "replicate_scores_csv": str(replicate_csv),
        "summary_csv": str(summary_csv),
        "plot_png": str(plot_path) if plot_path is not None else None,
        "plot_svg": str(plot_path.with_suffix(".svg")) if plot_path is not None else None,
        "run_dirs": {
            f"{_format_word_count(budget)}:rep{replicate}": str(path)
            for (budget, replicate), path in sorted(run_dirs.items())
        },
    }
    manifest_path = output_root / f"{experiment_tag}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
