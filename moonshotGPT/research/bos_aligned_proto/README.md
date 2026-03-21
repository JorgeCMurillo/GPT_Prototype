# BOS-Aligned Research Prototype for moonshotGPT

This folder contains an isolated research prototype to test nanochat-style BOS-aligned supervision in moonshotGPT without changing GPT-2 model architecture or benchmark scripts.

It now lives under `research/` to keep experimental work separate from the baseline training pipeline.

Current package layout:

```text
research/bos_aligned_proto/
  README.md
  __init__.py
  pipeline/
    __init__.py
    bos_packed_index.py
    bos_row_loader.py
    build_bos_packed_index.py
    prepare_finewebedu_bos_rows.py
  evaluation/
    __init__.py
    core.py
    ewok.py
    ewok_category.py
    hellaswag.py
  training/
    __init__.py
    checkpoints.py
    config.py
    eval_hooks.py
    reporting.py
    trainer.py
  docs/
    compare_ab_runs.md
  analysis/
    run_checkpoint_evals.py
    notebooks/
      mine_synthetic_minimal_pairs.ipynb
  experiments/   # legacy outputs kept in place
  bos_train.log  # legacy log kept in place
```

## Package layout

### `pipeline/`

- `pipeline/bos_packed_index.py`
  - Canonical BOS data path for new work.
  - Builds and reads compact exact packed-row indexes over the existing raw
    token shards instead of duplicating BOS row tokens on disk.
  - Preserves the same `bos_row_packed_bestfit` semantics as the older
    materialized BOS-row preprocessor:
    - pick the largest doc that fits remaining row space;
    - if none fit, pick the shortest doc and crop to exact remaining space.
  - Also provides the index-backed BOS dataloader used by the unified trainer.
- `pipeline/build_bos_packed_index.py`
  - Thin CLI entrypoint for building exact BOS packed-index artifacts from the
    raw token-stream dataset.
- `pipeline/prepare_finewebedu_bos_rows.py`
  - Offline preprocessor for FineWeb-Edu.
  - Builds row-packed `uint16` shards where each row has length `seq_len + 1` and starts with BOS.
  - Packing logic matches nanochat-style behavior:
    - pick the largest doc that fits remaining row space;
    - if none fit, pick the shortest doc and crop to exact remaining space.
  - This is now the legacy materialized BOS-row path kept for reference and
    parity checks, not the recommended path for new large runs.
- `pipeline/bos_row_loader.py`
  - Runtime memmap loader for the row-packed format.
  - Produces `(x, y)` with shape `(B, T)` from row data.
  - Supports optional metadata for exposure logging.

### `training/`

- `training/config.py`
  - Central definition of training settings.
  - Owns the `TrainConfig` dataclass and the CLI parser.
  - If you add or rename a training flag, this is the first file to update.
- `training/checkpoints.py`
  - Resume-path resolution and checkpoint metadata validation helpers.
  - Keeps trainer-state load/save code out of the main training loop.
- `training/reporting.py`
  - Shared JSON, JSONL, and metric-serialization helpers.
  - Useful any time you need to add a new on-disk metric artifact.
- `training/eval_hooks.py`
  - Evaluation cleanup and EWOK plot-refresh helpers.
  - Keeps plotting-heavy code separate from the main training loop.
- `training/trainer.py`
  - Canonical research training entrypoint and unified trainer implementation.
  - Supports both:
    - `--loader_kind stream`
      for the plain contiguous token-stream baseline
    - `--loader_kind bos_packed_index`
      for exact nanochat-style BOS packed rows backed by a compact index
  - Reads a `TrainConfig`, builds the runtime objects, and runs training, validation,
    HellaSwag, CORE, EWoK, checkpointing, and metric logging.
  - EWoK logging now records both BabyLM completion-choice scoring and the original
    EWoK paper context-sensitivity scoring.
  - Writes `run_config.json` at startup so resolved run settings are saved next to metrics.
- `training/__init__.py`
  - Marks the training directory as a Python package.

### `evaluation/`

- `evaluation/core.py`
  - Research-local shim around the shared DCLM CORE evaluator.
  - Supports local bundle overrides, local-files-only mode, and nanochat-style bundle download fallback.
- `evaluation/ewok.py`
  - Research-local shim that keeps BOS imports under `research.bos_aligned_proto.evaluation`
    while reusing the shared top-level EWoK evaluator.
- `evaluation/ewok_category.py`
  - BOS-specific helpers for EWoK category aggregation and category-subplot plotting.
- `evaluation/hellaswag.py`
  - Research-local shim around the shared HellaSwag evaluator.

### `docs/`

- `docs/compare_ab_runs.md`
  - A/B run protocol and comparison checklist.

### `analysis/`

- `analysis/run_checkpoint_evals.py`
  - Post-hoc benchmark runner for a single resolved BOS checkpoint.
  - Accepts a run directory, a direct `ckpt_*_stepXXXXXXX/` path, or `--hf-model <model_id>`.
  - Runs evaluations in priority order: CORE first, then HellaSwag, then EWoK, then BLiMP.
  - The EWoK stage stores only the mean-reduction `domain_scores_full` outputs
    for BabyLM completion choice and EWoK paper context sensitivity.
  - Writes resumable standalone outputs under `posthoc_eval/<checkpoint_name>/` for local checkpoints,
    or `runs/research/bos_aligned_proto/posthoc_hf_eval/<model_slug>/` for Hugging Face models.
- `analysis/notebooks/mine_synthetic_minimal_pairs.ipynb`
  - Notebook-based analysis for synthetic minimal pairs and run inspection.

### `experiments/`

- Legacy outputs and comparison plots kept in place for reference.

## Training files at a glance

### `training/config.py`

This file does not run training. Its job is to define the knobs for a run and parse them from the command line into one `TrainConfig` object.

Use this file when you want to:

- add a new CLI argument;
- change a default value;
- see the full set of supported training options.

One example is plotting policy: the config now includes `include_ewok_sum_plots`, which defaults to `False`, so auto-generated EWOK plots are mean-only unless you opt in to sum-reduction plots.

### `training/trainer.py`

This is now the canonical executable research training script. It takes the
parsed config and does the actual work of:

- setting up the accelerator and runtime;
- constructing tokenizer, model, optimizer, and dataloaders;
- running the training loop;
- writing checkpoints and metrics;
- running validation, HellaSwag, CORE, and EWoK evaluation.

The intended split is simple:

- `config.py` defines and parses training settings.
- `trainer.py` uses those settings to execute the run.
- `checkpoints.py`, `reporting.py`, and `eval_hooks.py` hold the cross-cutting helper logic.

## Recommended Data Backends

The research trainer now supports two data backends behind one CLI:

- `stream`
  - Reads the raw `train_*.bin` / `val_*.bin` token stream directly.
  - This is the plain contiguous-token baseline and is closest to the older
    non-BOS moonshot training path.
- `bos_packed_index`
  - Reads a compact exact BOS packed-row index plus the original raw token
    shards.
  - This preserves the exact `bos_row_packed_bestfit` row semantics used by
    the older materialized BOS-row pipeline, but without duplicating all row
    tokens on disk.

For new BOS-aligned runs, `bos_packed_index` is the recommended path.

## Plotting defaults

By default, the end-of-run auto-generated EWOK plots only include mean-reduction plots. EWOK sum metrics are still written into `step_metrics.json`, but sum-based PNGs are skipped unless you opt in.

The auto-generated analysis also no longer creates the EWOK "all domains" overlay plot, since that combined view was not very useful in practice.

To include EWOK sum plots in the auto-generated analysis output, pass:

```bash
--include_ewok_sum_plots
```

This flag is defined in `training/config.py` and is forwarded to `plot_step_metrics.py` by the training entrypoint.

Training-time EWoK analysis PNGs are written to:

- `runs/research/bos_aligned_proto/<run_name>/plots_from_step_metrics/`

The main exception is:

- `runs/research/bos_aligned_proto/<run_name>/loss_curve.png`

`loss_curve.png` is refreshed when `save_plot()` runs, which means:

- at `save_every` optimizer-step intervals, if `save_every > 0`;
- once again at the end of training.

The in-memory train-loss history itself is still updated every optimizer step, and
validation-loss history is updated every `eval_every` steps.

## CORE evaluation

The BOS trainer now supports nanochat-style DCLM CORE evaluation through the shared `evaluation/core.py` module.

Relevant CLI flags in `training/config.py`:

- `--core_every`
  - Run periodic CORE evaluation every N optimizer steps.
  - If `0`, both periodic and final CORE evaluation are disabled.
- `--core_max_per_task`
  - Limit how many examples are scored per CORE task.
  - Use `-1` to evaluate the full local bundle.
- `--core_bundle_dir`
  - Point to an existing local `eval_bundle/` directory.
- `--core_local_files_only`
  - Never auto-download the bundle; disable CORE cleanly if the bundle is missing.

When CORE is enabled, BOS runs write:

- `core_metrics.jsonl`
  - Detailed per-run CORE records with raw accuracies, centered scores, per-task counts, and `core_metric`.
- `scalars.jsonl`
  - Compact `type="core"` summary rows for plotting and quick inspection.
- `step_metrics.json`
  - Nested `core` entries with bundle source, `max_per_task`, raw task scores, centered scores, and the aggregate metric.
- `run_config.json`
  - A copy of the resolved run config, including the CORE flags.

## Post-hoc checkpoint evaluation

If you want to evaluate one finished BOS checkpoint after training, use the
analysis-layer runner:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name>
```

This resolves the latest checkpoint under the run by default and writes outputs
to:

- `runs/research/bos_aligned_proto/<run_name>/posthoc_eval/<checkpoint_name>/`

Useful variants:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --step 30000
```

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name>/ckpt_final_step0030000
```

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  --hf-model gpt2-medium
```

When using `--hf-model`, the evaluator now prints whether model/tokenizer files
appear cached locally and whether it is about to download from the Hugging Face
Hub. This status mode is enabled by default and can be turned off with:

```bash
--no-show-hf-download-status
```

The task order is always normalized to:

1. CORE
2. HellaSwag
3. EWoK
4. BLiMP

That way the most important aggregate metric is produced first, and reruns can
skip already-completed tasks unless you pass `--force`.

The analysis runner defaults to `--device cuda`, so it will fail loudly instead
of quietly evaluating on CPU. If you intentionally want CPU fallback for a
smoke test, pass `--device auto` or `--device cpu`.

When you use `--hf-model`, outputs default to:

- `runs/research/bos_aligned_proto/posthoc_hf_eval/<model_slug>/`

So for GPT-2 Medium, the default output directory is:

- `runs/research/bos_aligned_proto/posthoc_hf_eval/gpt2-medium/`

## EWoK scoring modes and outputs

The shared EWoK evaluator now computes two scoring views in one pass:

- BabyLM completion-choice scoring
  - Hold context fixed and compare the correct target against the distractor.
  - This is the training script's primary EWoK view for plots and category breakdowns.
- EWoK paper context-sensitivity scoring
  - Hold target fixed and compare the correct context against the distractor.
  - This is logged alongside the BabyLM view for comparison.

The most explicit keys in `step_metrics.json` are now:

- BabyLM completion-choice:
  - `eval_babylm_completion_choice_official_*`
  - `eval_babylm_completion_choice_full_*`
  - `eval_babylm_completion_choice_margin_stats_*`
  - `eval_babylm_completion_choice_by_category_full_*`
- EWoK paper context sensitivity:
  - `eval_ewok_paper_context_sensitivity_official_*`
  - `eval_ewok_paper_context_sensitivity_full_*`
  - `eval_ewok_paper_context_sensitivity_margin_stats_*`

Per-item `ewok_items.jsonl` records also carry explicit prefixes:

- `babylm_completion_choice_*`
- `ewok_paper_context_sensitivity_*`

Backward-compatible aliases are still written for the BabyLM view, including older
keys such as `eval_full_*`, `eval_margin_stats_*`, `eval_by_category_full_*`,
`margin_official_m1`, and `correct_official`.

`plot_step_metrics.py` and the BOS in-training analysis plots now prefer the explicit
BabyLM-prefixed keys when present, but still fall back to the older aliases so
historical runs continue to plot correctly.

## Data formats

### `--loader_kind stream`

Expected files in `--data_dir`:

- `train_*.bin`
- `val_*.bin`
- optionally `meta.json`

This is the plain contiguous token-stream dataset used by the baseline loader.

### `--loader_kind bos_packed_index`

Expected files in `--data_dir`:

- `meta.json`
- `train.row_ptr.bin`
- `train.segments.bin`
- `train.virtual_shards.jsonl`
- `val.row_ptr.bin`
- `val.segments.bin`
- `val.virtual_shards.jsonl`

The packed-index `meta.json` includes fields such as:

- `format: "bos_row_packed_bestfit_index_v1"`
- `source_data_dir`
- `source_shards_fingerprint`
- `seq_len`
- `row_tokens`
- `packing_algo`
- `buffer_docs`
- `virtual_shard_rows`
- `rows_written_total`
- `tokens_cropped_total`
- `crop_fraction`

Important: the packed-index artifact is still `seq_len`-specific. The loader
validates `seq_len` against `meta.json`.

Recommended data layout:

```text
data/
  processed/
    fineweb_edu_100B/
    bos_aligned_proto/
      fineweb_edu_100B_bospackedindex/
```

## End-to-end commands

Run from `tokenPred/moonshotGPT`.

### 1) Build exact BOS packed index from raw token shards

```bash
python -m research.bos_aligned_proto.pipeline.build_bos_packed_index \
  --data_dir data/processed/fineweb_edu_100B \
  --out_dir data/processed/bos_aligned_proto/fineweb_edu_100B_bospackedindex \
  --seq_len 1024 \
  --buffer_docs 1000 \
  --shard_rows 97656 \
  --val_shards 1
```

This preserves the old `bos_row_packed_bestfit` packing behavior exactly, but
stores only a compact index plus synthetic virtual-shard metadata.

### 2) Train with the baseline stream loader

```bash
python -m research.bos_aligned_proto.training.trainer \
  --loader_kind stream \
  --data_dir data/processed/fineweb_edu_100B \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

### 3) Train BOS prototype from the packed index

3090-oriented example (`micro_batch_size=10`):

```bash
python -m research.bos_aligned_proto.training.trainer \
  --loader_kind bos_packed_index \
  --data_dir data/processed/bos_aligned_proto/fineweb_edu_100B_bospackedindex \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

### 4) Train BOS GPT-2 medium from the packed index

Start with a smaller `micro_batch_size` and increase only if GPU memory allows:

```bash
python -m research.bos_aligned_proto.training.trainer \
  --loader_kind bos_packed_index \
  --data_dir data/processed/bos_aligned_proto/fineweb_edu_100B_bospackedindex \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --micro_batch_size 2 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

The trainer now creates run names that reflect the backend, for example:

- `babygpt_fineweb_stream_*`
- `babygpt_fineweb_bospackedindex_*`

Historical outputs already under `research/bos_aligned_proto/experiments/`
remain as legacy artifacts.

### Legacy materialized BOS rows

If you need the older fully materialized BOS-row pipeline for comparison,
`prepare_finewebedu_bos_rows.py` and the historical `*_bosrow` datasets are
still valid references. They are just no longer the recommended large-run path
now that the exact packed-index workflow exists.

## Notes and tradeoffs

- Runtime is fast (memmap + reshape) and stays close to moonshot loader behavior.
- Preprocessing is heavier than baseline because it performs best-fit packing.
- Some tokens are intentionally cropped; this is tracked in `meta.json`.
