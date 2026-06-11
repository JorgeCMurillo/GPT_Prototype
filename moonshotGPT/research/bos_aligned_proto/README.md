# BOS-Aligned Research Prototype

This package is the BOS-packed research branch of `moonshotGPT`. It exists so
we can experiment with BOS-aligned supervision, BOS-packed data views, and
post-hoc attribution workflows without destabilizing the main top-level trainer.

If your main question is:

Which training rows seem to help or hurt EWoK?

this folder is the most relevant part of the repo. It contains the maintained
post-hoc evaluation and attribution pipeline used to study that question on
BOS-packed runs.

## Current Layout

```text
research/bos_aligned_proto/
  README.md
  __init__.py
  bos_train.log
  pipeline/
  training/
  evaluation/
  analysis/
  docs/
  experiments/
```

## Folder Guide

### `pipeline/`

Data-preparation and loading code for BOS-packed training views.

- `bos_packed_index.py`
  Canonical packed-index implementation for new work.
- `build_bos_packed_index.py`
  CLI entrypoint for building exact packed-index artifacts.
- `prepare_finewebedu_bos_rows.py`
  Legacy materialized BOS-row preprocessor kept for reference and parity checks.
- `bos_row_loader.py`
  Runtime loader for materialized BOS-row shards.

The repo now supports two BOS data formats:

- materialized BOS rows on disk;
- exact packed-index artifacts that reconstruct rows from the original token
  shards.

### `training/`

Unified BOS research trainer and supporting helpers.

- `config.py`
  CLI surface and `TrainConfig` definition.
- `trainer.py`
  Main BOS research training entrypoint.
- `checkpoints.py`
  Checkpoint resolution and resume helpers.
- `eval_hooks.py`
  Evaluation cleanup and plotting helpers.
- `reporting.py`
  Shared JSON and JSONL reporting utilities.

Rho-specific gap diagnostics such as student-vs-reference delta summaries
belong in the training layer, not in `pipeline/`. The pipeline owns artifact
construction and loading; rho diagnostics are training-time measurements derived
from `token_loss`, aligned `ref_loss`, and the rho retention mask.

### `evaluation/`

Research-local shims around the shared benchmark evaluators.

- `core.py`
- `ewok.py`
- `ewok_category.py`
- `hellaswag.py`

These files let the BOS prototype reuse the shared benchmark logic while keeping
imports and BOS-specific aggregation close to the research package.

### `analysis/`

Post-training inspection and attribution.

- `run_checkpoint_evals.py`
  Maintained post-hoc benchmark runner for finished checkpoints or Hugging Face
  model ids.
- `plot_ewok_baseline_full_mean.py`
  Plots standalone checkpoint EWoK metrics.
- `plot_ewok_checkpoint_baseline_compare.py`
  Overlays a checkpoint baseline on training-run EWoK curves.
- `attribution/`
  Reusable TRAK and TrackStar attribution package.
- `notebooks/`
  Exploratory notebook workflows.

If you are deciding where to start, `analysis/` is usually the right answer.

### `docs/`

- `compare_ab_runs.md`
  A/B comparison checklist and run protocol notes.

### `experiments/`

Legacy outputs, comparison plots, and helper plotting scripts that were kept in
place for reference.

Current notable items include:

- historical BOS run directories;
- comparison plot folders;
- `plot_ewok_rho_reference_comparison.py`;
- `step_metrics_gpt_medium.json`.

## Typical Workflow

### 1. Build a BOS data view

For new large work, prefer the packed-index path:

```bash
python -m research.bos_aligned_proto.pipeline.build_bos_packed_index \
  --help
```

For older parity or reference work, the legacy materialized BOS-row path is
still available via `prepare_finewebedu_bos_rows.py`.

### 2. Train a BOS-packed run

```bash
python -m research.bos_aligned_proto.training.trainer --help
```

The trainer supports both:

- `--loader_kind stream`
  Plain token-stream baseline behavior.
- `--loader_kind bos_packed_index`
  Exact BOS-packed supervision backed by the packed index.

### 3. Run an architecture speed probe

To compare GPT-2 blocks against randomly initialized Llama-style blocks while
keeping tokenization, data, optimizer, and eval settings fixed:

For a GPT-2-medium-sized probe on the raw GPT-2-tokenized FineWeb-Edu 10B
shards, use the stream loader. In this mode `--data_dir` is the original shard
directory, so `--source_data_dir` is not needed:

```bash
python -m research.bos_aligned_proto.experiments.run_architecture_speed_benchmark \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --loader_kind stream \
  --num_processes 8 \
  --mixed_precision bf16 \
  --n_embd 1024 --n_head 16 --n_layer 24 \
  --seq_len 1024 \
  --micro_batch_size 4 \
  --total_batch_tokens 491520 \
  --max_train_steps 200
```

For a BOS packed-index artifact, pass the packed-index directory as `--data_dir`
and the original raw token shards as `--source_data_dir`:

```bash
python -m research.bos_aligned_proto.experiments.run_architecture_speed_benchmark \
  --data_dir /path/to/bos_packed_index_artifact \
  --source_data_dir /path/to/raw_token_shards \
  --loader_kind bos_packed_index \
  --num_processes 8 \
  --mixed_precision bf16 \
  --n_embd 1024 --n_head 16 --n_layer 24 \
  --micro_batch_size 4 \
  --total_batch_tokens 491520 \
  --max_train_steps 200
```

The harness launches eval-free `gpt2`, shape-matched `llama_shape`, and
parameter-matched `llama_param` runs by default, then writes:

- `architecture_speed_manifest.json`
- `architecture_speed_summary.csv`
- `architecture_speed_summary.json`

Use `--dry_run` to print the exact trainer commands without launching them.

If you use `/home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_100B`,
also pass `--tokenizer_name_or_path gpt2`; that older shard directory does not
have a `meta.json` tokenizer field for the trainer to read.

Muon+Polar Express supports batched matrix updates through
`--muon_batch_updates` and optimizer-step profiling through
`--profile_optimizer_steps`. On the 2026-06-08 Llama-medium-size smoke suite
(`d1024`, `h16`, `L24`, intermediate `2816`, `total_batch_tokens=81920`,
GPU 3), batching moved Muon from the older ~15% end-to-end overhead versus
AdamW down to roughly 3-4%:

- AdamW median training step: `5.955s`;
- batched Muon median training step: `6.174s`;
- batched Muon median optimizer step: `214.5ms`;
- scalar Muon median optimizer step: `247.9ms`.

The transpose-compatible Muon batching experiment did not show a practical
speedup over exact-shape batching in the optimizer-only benchmark
(`170.48ms` vs `170.67ms` median), while increasing peak CUDA allocation
(`4.88GB` vs `3.86GB`). That path has been removed; exact-shape batching is the
lower-memory default unless a later model shape shows a real win.

For Llama Liger-kernel probes, the architecture benchmark can run both the
startup-inclusive Triton compile window and a post-warmup steady-state window.
Use `--liger_modes off,on`, keep `--profile_optimizer_steps` enabled, and leave
`--compile_window_steps 20 --warmup_steps 20` to report the first 20 steps and
the steps after 20 separately in `architecture_speed_summary.csv`. New runs also
include `max_cuda_memory_allocated_mb` and `max_cuda_memory_reserved_mb` so the
Liger/no-Liger comparison records peak process CUDA memory:

```bash
python -m research.bos_aligned_proto.experiments.run_architecture_speed_benchmark \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_100B \
  --loader_kind stream \
  --tokenizer_name_or_path gpt2 \
  --variants llama_param \
  --optimizers muon_pe \
  --liger_modes off,on \
  --num_processes 1 \
  --max_train_steps 100 \
  --warmup_steps 20 \
  --compile_window_steps 20 \
  --profile_optimizer_steps
```

For a longer learning-efficiency probe, keep validation/EWoK enabled and save
periodic checkpoints. This example compares AdamW against Muon+Polar Express
on the same GPT-2-medium architecture for 5k optimizer steps:

```bash
python -m research.bos_aligned_proto.experiments.run_architecture_speed_benchmark \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --loader_kind stream \
  --variants gpt2 \
  --optimizers adamw,muon_pe \
  --num_processes 2 \
  --mixed_precision bf16 \
  --n_embd 1024 --n_head 16 --n_layer 24 \
  --seq_len 1024 \
  --micro_batch_size 4 \
  --total_batch_tokens 491520 \
  --max_train_steps 5000 \
  --muon_lr 0.02 \
  --muon_momentum 0.95 \
  --muon_weight_decay 0.1 \
  --muon_ns_steps 5 \
  --eval_every 250 \
  --ewok_every 250 \
  --save_every 1000 \
  --save_final_checkpoint \
  --no-skip_final_ewok
```

### 4. Run post-hoc evaluation

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --step 16000
```

### 5. Run attribution

Start in:

- `analysis/attribution/run_trackstar.py` for the Bergson-backed path;
- `analysis/attribution/run_trak.py` for the TRAK baseline.

The shared higher-level workflow is:

1. resolve checkpoints;
2. rebuild or load the BOS row manifest;
3. map exposure logs to candidate training rows;
4. build EWoK targets;
5. score candidates with the chosen backend;
6. export row and domain summaries for inspection.

## How This Fits the Main Repo

The top-level `moonshotGPT` directory still owns the general GPT-2 training
stack, top-level benchmark data, and rho-1 utilities. This package is where the
research becomes more targeted:

- BOS-packed supervision experiments;
- checkpoint-local post-hoc evaluation;
- EWoK-focused attribution;
- exploratory notebook analysis tied to BOS exposure logs.

If you only read one more README after this one, make it
`analysis/attribution/README.md`.
