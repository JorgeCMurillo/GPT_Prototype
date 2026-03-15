# BOS-Aligned Research Prototype for moonshotGPT

This folder contains an isolated research prototype to test nanochat-style BOS-aligned supervision in moonshotGPT without changing GPT-2 model architecture or benchmark scripts.

It now lives under `research/` to keep experimental work separate from the baseline training pipeline.

Current package layout:

```text
research/bos_aligned_proto/
  README.md
  __init__.py
  data/
    __init__.py
    bos_row_loader.py
    prepare_finewebedu_bos_rows.py
  evaluation/
    __init__.py
    core.py
    ewok.py
    ewok_category.py
    hellaswag.py
  training/
    __init__.py
    config.py
    train_gpt2_finewebedu_bos_bin.py
  docs/
    compare_ab_runs.md
  analysis/
    notebooks/
      mine_synthetic_minimal_pairs.ipynb
  experiments/   # legacy outputs kept in place
  bos_train.log  # legacy log kept in place
```

## Package layout

### `data/`

- `data/prepare_finewebedu_bos_rows.py`
  - Offline preprocessor for FineWeb-Edu.
  - Builds row-packed `uint16` shards where each row has length `seq_len + 1` and starts with BOS.
  - Packing logic matches nanochat-style behavior:
    - pick the largest doc that fits remaining row space;
    - if none fit, pick the shortest doc and crop to exact remaining space.
- `data/bos_row_loader.py`
  - Runtime memmap loader for the row-packed format.
  - Produces `(x, y)` with shape `(B, T)` from row data.
  - Supports optional metadata for exposure logging.

### `training/`

- `training/config.py`
  - Central definition of training settings.
  - Owns the `TrainConfig` dataclass and the CLI parser.
  - If you add or rename a training flag, this is the first file to update.
- `training/train_gpt2_finewebedu_bos_bin.py`
  - Main training entrypoint.
  - Reads a `TrainConfig`, builds the runtime objects, and runs training, validation,
    HellaSwag, CORE, EWoK, checkpointing, and metric logging.
  - EWoK logging now records both BabyLM completion-choice scoring and the original
    EWoK paper context-sensitivity scoring.
  - Writes `run_config.json` at startup so resolved BOS run settings are saved next to metrics.
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

### `training/train_gpt2_finewebedu_bos_bin.py`

This is the executable training script. It takes the parsed config and does the actual work of:

- setting up the accelerator and runtime;
- constructing tokenizer, model, optimizer, and dataloaders;
- running the training loop;
- writing checkpoints and metrics;
- running validation, HellaSwag, CORE, and EWoK evaluation.

The intended split is simple:

- `config.py` defines and parses training settings.
- `train_gpt2_finewebedu_bos_bin.py` uses those settings to execute the run.

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

## Data format

Expected files in `--data_dir`:

- `train_*.bin`
- `val_*.bin`
- `meta.json`

`meta.json` includes baseline-compatible fields plus BOS-specific fields such as:

- `format: "bos_row_packed_bestfit"`
- `seq_len`
- `row_tokens`
- `packing_algo`
- `batch_docs`
- `buffer_docs`
- `rows_written_total`
- `tokens_cropped_total`
- `crop_fraction`

Important: this dataset is `seq_len`-specific. Loader validates `seq_len` against `meta.json`.

## End-to-end commands

Run from `tokenPred/moonshotGPT`.

### 1) Build BOS row-packed dataset

```bash
python -m research.bos_aligned_proto.data.prepare_finewebedu_bos_rows \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --text_field text \
  --tokenizer gpt2 \
  --out_dir fineweb_edu_10B_bosrow \
  --seq_len 1024 \
  --batch_docs 256 \
  --buffer_docs 1000 \
  --val_shards 1
```

### 2) Train BOS prototype

3090-oriented example (`micro_batch_size=10`):

```bash
python -m research.bos_aligned_proto.training.train_gpt2_finewebedu_bos_bin \
  --data_dir fineweb_edu_10B_bosrow \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

### 3) Train BOS GPT-2 medium

Start with a smaller `micro_batch_size` and increase only if GPU memory allows:

```bash
python -m research.bos_aligned_proto.training.train_gpt2_finewebedu_bos_bin \
  --data_dir fineweb_edu_10B_bosrow \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --micro_batch_size 2 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

Future runs go to `runs/research/bos_aligned_proto/babygpt_fineweb_bosrow_*` by default.
Historical outputs already under `research/bos_aligned_proto/experiments/` remain as legacy artifacts.

## Notes and tradeoffs

- Runtime is fast (memmap + reshape) and stays close to moonshot loader behavior.
- Preprocessing is heavier than baseline because it performs best-fit packing.
- Some tokens are intentionally cropped; this is tracked in `meta.json`.
