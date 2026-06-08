# GPT_Prototype

A portable research repo for training GPT-style language models and studying
which retained training data supports EWoK performance.

The current project center is data attribution and result analysis. Rho-1 token
filtering remains implemented and documented, but recent tests suggest that
token-level filtering is not the primary issue behind EWoK behavior. The main
loop now is to train or collect runs, evaluate EWoK and companion benchmarks,
reconstruct the training examples seen by a checkpoint, and use attribution or
small intervention runs to understand which rows help or hurt.

Most active code lives under [`moonshotGPT/`](moonshotGPT/).

## Start Here

- [`moonshotGPT/README.md`](moonshotGPT/README.md)
  for the current end-to-end workflow and repo map.
- [`moonshotGPT/research/bos_aligned_proto/README.md`](moonshotGPT/research/bos_aligned_proto/README.md)
  for BOS-packed training, maintained post-hoc evaluation, and the current
  EWoK attribution path.
- [`moonshotGPT/research/bos_aligned_proto/analysis/README.md`](moonshotGPT/research/bos_aligned_proto/analysis/README.md)
  for checkpoint/result analysis.
- [`moonshotGPT/research/bos_aligned_proto/analysis/attribution/README.md`](moonshotGPT/research/bos_aligned_proto/analysis/attribution/README.md)
  for the shared TRAK/TrackStar attribution workflow.
- [`moonshotGPT/training_utils/README.md`](moonshotGPT/training_utils/README.md)
  for rho-1 details, alignment notes, and historical findings.

## Current Research Loop

1. Build or load FineWeb-Edu token shards or BOS-packed data views.
2. Train a baseline, retained-data, or continued-pretraining variant.
3. Evaluate checkpoints on EWoK and companion benchmarks such as HellaSwag,
   BLiMP, and CORE-style tasks.
4. Reconstruct exposed training examples from run artifacts.
5. Score candidate examples with attribution methods such as TrackStar or TRAK.
6. Inspect top and bottom rows, compare against current results, and use the
   findings to plan the next retained-data or intervention run.

## Rho-1 Status

Rho-1 was useful for testing whether loss-guided token retention could be
implemented cleanly. It remains available as a supporting experiment path, but
it should not be treated as the headline hypothesis for EWoK. The tests so far
suggest that data composition, example structure, and retained-row effects are
more important to investigate than simple token-level filtering.

For rho-specific commands and implementation notes, use
[`moonshotGPT/training_utils/README.md`](moonshotGPT/training_utils/README.md).

## Quick Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-research.txt
```

Use the dev requirements for tests. Use the research requirements for
attribution, Bergson/TRAK-related tooling, and exploratory analysis.

## Typical Commands

Run project commands from `moonshotGPT/`:

```bash
cd moonshotGPT
```

Tokenize a FineWeb-Edu sample:

```bash
python fineweb.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --text_field text \
  --tokenizer gpt2 \
  --out_dir data/processed/fineweb_edu_10B
```

Train a baseline GPT-2-style run:

```bash
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir data/processed/fineweb_edu_10B \
  --micro_batch_size 4 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16
```

Run post-hoc checkpoint evaluation:

```bash
python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  runs/research/bos_aligned_proto/<run_name> \
  --step 16000
```

Run an attribution pass:

```bash
python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir runs/research/bos_aligned_proto/<run_name> \
  --data_dir data/processed/bos_aligned_proto/<data_view> \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32
```

These are portable examples. Adjust dataset sizes, run names, devices, and
environment wrappers for the machine you are using.

## Repository Map

- `moonshotGPT/fineweb.py`
  tokenizes FineWeb/FineWeb-Edu into memmapped train/validation shards.
- `moonshotGPT/train_gpt2_finewebedu_bin.py`
  top-level GPT-2-style trainer for contiguous token-stream runs.
- `moonshotGPT/evaluation/`
  shared benchmark evaluators and reusable evaluation runner.
- `moonshotGPT/research/bos_aligned_proto/`
  BOS-packed training, post-hoc evaluation, and attribution research package.
- `moonshotGPT/research/bos_aligned_proto/analysis/attribution/`
  maintained candidate reconstruction and TRAK/TrackStar attribution workflow.
- `moonshotGPT/training_utils/`
  shared training helpers, rho-1 utilities, and resume/debug support.
- `moonshotGPT/data/`
  preferred location for processed shards and reference-loss artifacts.
- `moonshotGPT/runs/`
  preferred location for modern run outputs.
- `moonshotGPT/experiments/`
  legacy run outputs and historical comparison artifacts.
- `moonshotGPT/tests/`
  regression coverage for training, evaluation, BOS packing, attribution, and
  supporting utilities.

## Data And Artifacts

This repo should version code, docs, tests, configs, notebooks, and small
benchmark fixtures. Keep generated runtime artifacts out of git, especially:

- model checkpoints and optimizer states;
- generated training runs under `moonshotGPT/runs/`;
- legacy generated runs under `moonshotGPT/experiments/`;
- processed token shards under `moonshotGPT/data/processed/`;
- reference-loss binaries under `moonshotGPT/data/ref_loss/`;
- local logs, caches, API keys, and machine-specific paths.

For public or portable use, prefer relative paths in docs and commands. Put
machine-specific locations in shell variables, local config files, or ignored
run scripts.
