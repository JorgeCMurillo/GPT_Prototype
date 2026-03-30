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

### 3. Run post-hoc evaluation

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  runs/research/bos_aligned_proto/<run_name> \
  --step 16000
```

### 4. Run attribution

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
