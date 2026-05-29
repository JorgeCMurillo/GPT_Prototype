# Word2Vec Lexical Probe

This package trains and evaluates a Word2Vec lexical baseline for EWoK. Its job
is not to replace GPT training. Its job is to answer a narrower question:

How much EWoK signal can be recovered from corpus-level lexical co-occurrence
alone?

That makes the probe useful as a baseline when you later interpret GPT-2 or
attribution results. If a language model only matches this baseline, then much
of its behavior may be explainable by lexical statistics. If it clearly exceeds
the baseline, then the interesting gap is likely coming from richer contextual
or compositional behavior.

## Current Layout

```text
research/w2v_lexical_probe/
  README.md
  __init__.py
  corpus.py
  text.py
  model.py
  train_word2vec.py
  eval_ewok_word2vec.py
  compare_ewok_word2vec_runs.py
  run_fineweb_word_budget_ewok_ci.py
  eval_google_word2vec.py
  plot_ewok_interval_metrics.py
  plot_gpt2_vs_word2vec_domains.py
  gpt_w2v_babylm_completion_analysis.py
  list_ewok_agent_names.py
  gpt2_medium_vs_word2vec_babylm_completion.ipynb
```

## File Guide

- `corpus.py`
  Reads token shards, reconstructs text, and yields normalized sentences for
  Word2Vec training.
- `text.py`
  Normalization and token-cleanup rules shared by training and evaluation.
- `model.py`
  Run-format helpers for saving and loading Word2Vec runs in a consistent local
  format.
- `train_word2vec.py`
  Main training entrypoint for gensim Word2Vec on shard, raw-text, and Arrow
  corpora. Can optionally run the standard EWoK evaluator after training.
- `eval_ewok_word2vec.py`
  Evaluates a saved lexical-probe run on EWoK and writes prediction artifacts.
- `compare_ewok_word2vec_runs.py`
  Compares saved EWoK metrics from multiple Word2Vec runs by domain.
- `run_fineweb_word_budget_ewok_ci.py`
  Trains repeated random FineWeb-Edu word-budget W2V samples, evaluates EWoK,
  and plots per-domain confidence intervals.
- `eval_google_word2vec.py`
  Imports Google News pretrained vectors into the same local run format and
  evaluates them on EWoK.
- `plot_ewok_interval_metrics.py`
  Plots periodic EWoK metrics logged during Word2Vec training.
- `plot_gpt2_vs_word2vec_domains.py`
  Produces a domain-by-domain comparison between a Word2Vec run and a GPT-2
  metrics file.
- `gpt_w2v_babylm_completion_analysis.py`
  Shared notebook helpers for joined GPT-vs-Word2Vec item-level analysis.
- `list_ewok_agent_names.py`
  Lightweight utility for scanning recurring agent names in EWoK text.
- `gpt2_medium_vs_word2vec_babylm_completion.ipynb`
  Notebook that uses the shared comparison helpers to inspect GPT/Word2Vec
  agreement and failure cases.

## What This Package Produces

Typical lexical-probe runs live under:

```text
runs/research/w2v_lexical_probe/<run_name>/
```

Common artifacts are:

- `vectors.kv`
- `vocab.json`
- `train_summary.json`
- `ewok_interval_metrics.jsonl` when periodic EWoK evaluation is enabled
- EWoK prediction exports written by `eval_ewok_word2vec.py`

Legacy PyTorch-format runs may still contain `model.pt`. The evaluator can still
read those older runs.

## Dataset Choices

You will usually train on one of two corpus views:

- `data/processed/fineweb_edu_10B`
  Use this when you want a cleaner lexical baseline for the underlying source
  corpus.
- `data/processed/bos_aligned_proto/fineweb_edu_10B_bosrow`
  Use this when you want an exposure-oriented lexical baseline that more closely
  reflects BOS-packed LM training rows.

The BOS-row view is intentionally less natural as text because it preserves
packing and cropping artifacts from LM supervision. That makes it less ideal as
plain prose, but sometimes more relevant when the research question is about
what the model was actually exposed to.

The same trainer also supports BabyLM-style text corpora:

- `--corpus_format text_dir`
  Treats each non-empty line in files matching `--glob_pattern` as a document.
- `--corpus_format hf_arrow`
  Treats each row in an Arrow file's `--text_column` as a document.

This lets you ask the intended question directly: train a Word2Vec model on a
particular dataset, then evaluate that trained lexical model on EWoK.

## Setup

Run from `/home/jorge/tokenPred/moonshotGPT`.

Install `gensim` if it is not already available:

```bash
pip install gensim
```

EWoK sources expected by this package live at the repo root:

- `ewok_fast_jsonl.zip`
  Fast subset used by default.
- `ewok_full_jsonl.zip`
  Full filtered benchmark.

## Main Commands

### Train on a FineWeb-Edu view

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --corpus_format shard_bin \
  --data_dir data/processed/fineweb_edu_10B
```

Small explicit example:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --corpus_format shard_bin \
  --data_dir data/processed/fineweb_edu_10B \
  --max_train_shards 1 \
  --max_docs 2000 \
  --embedding_dim 128 \
  --epochs 1 \
  --workers 8
```

Exposure-oriented BOS-row example:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --corpus_format shard_bin \
  --data_dir data/processed/bos_aligned_proto/fineweb_edu_10B_bosrow \
  --max_train_shards 0 \
  --max_docs 0 \
  --workers 44 \
  --vocab_report_every_secs 120
```

### Train on BabyLM-Style Corpora and Evaluate on EWoK

Raw BabyLM text files:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --corpus_format text_dir \
  --data_dir /home/jorge/tokenPred/babylm_10m/train_files/train_10M \
  --glob_pattern "*.train" \
  --eval_after_train \
  --ewok_variant full \
  --write_per_item
```

BabyLM-Cosmo-Fine Arrow cache:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --corpus_format hf_arrow \
  --data_dir /home/jorge/.cache/huggingface/datasets/ltg___babylm-2024-baby-cosmo-fine-10m/default/0.0.0/5179e7ac0b6be2083ed03444a3a8c3d2c96061a2/babylm-2024-baby-cosmo-fine-10m-train.arrow \
  --text_column text \
  --eval_after_train \
  --ewok_variant full
```

### Evaluate a saved run on EWoK

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name>
```

### Compare Word2Vec Runs on EWoK

```bash
python -m research.w2v_lexical_probe.compare_ewok_word2vec_runs \
  --run raw_babylm=/path/to/raw_babylm_w2v_run \
  --run babycosmo=/path/to/babycosmo_w2v_run \
  --run fineweb=/path/to/fineweb_w2v_run \
  --ewok_variant full
```

### Estimate FineWeb Word-Budget Variation on EWoK

Run five random 10M-word and 100M-word FineWeb-Edu W2V samples, evaluate each
on EWoK, and plot per-domain 95% confidence intervals:

```bash
python -m research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci \
  --data_dir data/processed/fineweb_edu_10B \
  --word_budget 10000000 \
  --word_budget 100000000 \
  --replicates 5 \
  --workers 44 \
  --ewok_variant full
```

By default this uses the BabyLM-style completion-choice score. For the EWoK
context-sensitivity score, pass `--method ewok_context_sensitivity`. To reuse
the existing 10M/100M runs and add 50M/200M to the same plot, include all four
budgets and `--skip_existing`:

```bash
python -m research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci \
  --data_dir data/processed/fineweb_edu_10B \
  --word_budget 10000000 \
  --word_budget 50000000 \
  --word_budget 100000000 \
  --word_budget 200000000 \
  --replicates 5 \
  --epochs 4 \
  --workers 44 \
  --ewok_variant full \
  --method ewok_context_sensitivity \
  --score_kind pair_average \
  --skip_existing
```

After that finishes, generate the paper-style combined context-sensitivity plot
from the same saved metrics without retraining:

```bash
python -m research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci \
  --data_dir data/processed/fineweb_edu_10B \
  --word_budget 10000000 \
  --word_budget 50000000 \
  --word_budget 100000000 \
  --word_budget 200000000 \
  --replicates 5 \
  --epochs 4 \
  --workers 44 \
  --ewok_variant full \
  --method ewok_context_sensitivity \
  --score_kind combined \
  --skip_existing
```

Useful smoke run:

```bash
python -m research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci \
  --data_dir data/processed/fineweb_edu_10B \
  --word_budget 50000 \
  --replicates 2 \
  --embedding_dim 64 \
  --epochs 1 \
  --workers 4 \
  --ewok_variant fast
```

### Evaluate pretrained Google Word2Vec

```bash
python -m research.w2v_lexical_probe.eval_google_word2vec \
  --model word2vec-google-news-300
```

### Plot interval metrics

```bash
python -m research.w2v_lexical_probe.plot_ewok_interval_metrics \
  --interval_metrics runs/research/w2v_lexical_probe/<run_name>/ewok_interval_metrics.jsonl
```

## Important Flags

- `--workers`
  Number of gensim CPU worker threads.
- `--batch_words`
  Gensim batching knob.
- `--sample`
  Downsampling threshold for frequent tokens.
- `--checkpoint_every_epochs`
  Save intermediate checkpoints under `checkpoints/`.
- `--eval_every_words`
  Run in-memory EWoK evaluation every N training words.
- `--ewok_variant`
  Choose `fast` or `full`.
- `--ewok_text_preprocessing`
  Switch between probe preprocessing and the simpler paper-style tokenization.
- `--vocab_report_every_secs`
  Emit an explicit heartbeat during vocabulary building.

## Relation to the Rest of moonshotGPT

This package is most useful in three situations:

- when you want a lexical lower-bound before interpreting GPT-2 EWoK behavior;
- when you want a quick exposure-oriented baseline on BOS-packed data;
- when you want to compare domain-level EWoK profiles between GPT and a simpler
  co-occurrence model.

If your goal is retained-data attribution rather than lexical baselines, the
maintained path is still
`research/bos_aligned_proto/analysis/attribution/`.
