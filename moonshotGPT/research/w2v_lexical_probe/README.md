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
  Main training entrypoint for gensim Word2Vec on FineWeb-derived corpora.
- `eval_ewok_word2vec.py`
  Evaluates a saved lexical-probe run on EWoK and writes prediction artifacts.
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

## Setup

Run from the repo root's `moonshotGPT/` directory.

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
  --data_dir data/processed/fineweb_edu_10B
```

Small explicit example:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
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
  --data_dir data/processed/bos_aligned_proto/fineweb_edu_10B_bosrow \
  --max_train_shards 0 \
  --max_docs 0 \
  --workers 44 \
  --vocab_report_every_secs 120
```

### Evaluate a saved run on EWoK

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name>
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
