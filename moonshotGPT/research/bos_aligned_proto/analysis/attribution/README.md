# BOS Attribution Package

This package is the maintained post-training attribution pipeline for the
BOS-aligned prototype.

Its job is to answer a specific question:

Which BOS-packed training rows appear most responsible for the EWoK behavior
observed at a given checkpoint, and how does that attribution change across
training?

The package is intentionally organized so that the outer workflow stays stable
across attribution methods:

`config -> checkpoints -> row manifest -> exposures -> candidates -> EWoK targets -> backend -> export -> compare`

Everything before and after `backend` is shared. The backend is the only part
that changes between methods.

## What Lives Here

### `common/`

Shared pipeline code that both attribution methods use:

- checkpoint discovery and model/tokenizer loading
- BOS row manifest construction
- exposure-log parsing
- candidate row selection
- EWoK target construction
- export helpers
- checkpoint-to-checkpoint comparisons
- shared CLI defaults and validation

### `trak/`

The classic TRAK backend.

- runner entrypoint: `analysis/attribution/run_trak.py`
- backend implementation: `analysis/attribution/trak/backend.py`
- config parser: `analysis/attribution/trak/config.py`

This path depends on the `traker` Python package.

### `trackstar/`

The Bergson-backed attribution backend.

- runner entrypoint: `analysis/attribution/run_trackstar.py`
- backend implementation: `analysis/attribution/trackstar/backend.py`
- Bergson candidate dataset adapter: `bergson_datasets.py`
- Bergson query logic for the custom EWoK loss: `bergson_queries.py`
- config parser: `config.py`

This path uses EleutherAI Bergson programmatically and keeps the repo’s own run
structure, candidate selection, and export format.

Important TrackStar-specific note:

- TrackStar now defaults to projected candidate/query gradients with
  `use_fast_jl=True` and `proj_dim=16`

This is a Bergson-specific default, not a TRAK-style projection setting. See
`analysis/attribution/trackstar/README.md` for the audit note explaining why
the Bergson projection dimension has different memory semantics.

## Core Concepts

### Run Directory

`--run_dir` should point at a completed BOS training run directory that contains
checkpoint folders such as:

- `ckpt_periodic_step0001000/`
- `ckpt_final_step00030000/`

and an `exposures/` directory with per-rank JSONL exposure logs.

### Data Directory

`--data_dir` should point at the BOS training-data view that matches the run you
are analyzing.

The attribution code now supports two BOS data formats:

- materialized BOS rows
- exact BOS packed-index artifacts

For the older materialized BOS-row path, the pipeline expects:

- `meta.json`
- `train_*.bin`

For the newer packed-index path, the pipeline expects:

- `meta.json`
- `train.row_ptr.bin`
- `train.segments.bin`
- `train.virtual_shards.jsonl`

and reconstructs rows from the packed index plus the original raw token shards
recorded in `meta.json` or overridden through the training config.

The row manifest is built from these files and is used to map exposure offsets
back to stable global row ids in either format.

### Candidate Rows

Candidate rows are always BOS-packed training rows selected from exposure logs.
The pipeline does not score the full dataset by default. It scores a checkpoint-
local candidate subset chosen from rows that were actually exposed.

For packed-index runs, the exposed shard identities are synthetic virtual shards
such as `train_000123.vrow`, but the attribution contract stays the same:

- `row_id`
- `shard_path`
- `local_row_idx`

still refer to stable BOS-packed training rows.

Supported strategies:

- `between_checkpoints`
  Use rows exposed after the previous checkpoint and up to the current one.
- `up_to_step`
  Use all rows exposed up to the current checkpoint.
- `recent_window`
  Use rows exposed in the most recent `--recent_window_steps`.
- `new_since_prev`
  Use rows first seen between the previous and current checkpoints.

If the candidate pool is too large, it is deterministically subsampled with
`--max_candidate_rows` and `--candidate_seed`.

### EWoK Targets

The target side is built from the fast EWoK bundle.

Each target item consists of four paired conditional scores:

- `s11 = log P(T1 | C1)`
- `s12 = log P(T2 | C1)`
- `s22 = log P(T2 | C2)`
- `s21 = log P(T1 | C2)`

The pipeline supports two score views:

- `babylm_completion_choice`
  Uses margins `m1 = s11 - s12` and `m2 = s22 - s21`
- `ewok_paper_context_sensitivity`
  Uses margins `m1 = s11 - s21` and `m2 = s22 - s12`

The final target score is the negative paired softplus loss:

`L = 0.5 * [softplus(-m1 / tau) + softplus(-m2 / tau)]`

where `tau` is `--temperature`.

### What "Target" Means

In this pipeline, a `target` is one EWoK evaluation item, not one training
row.

Each target is a paired-comparison example with:

- `C1`, `T1`
- `C2`, `T2`

So when the code says things like:

- `target_count`
- `max_targets`
- `top rows per target`

it is always referring to EWoK items.

The candidate side is separate:

- candidate rows are BOS-packed training rows from your exposure-selected pool

That means the main score matrix for one checkpoint has shape:

`[num_targets, num_candidate_rows]`

For example, if you ran:

- `--max_targets 32`
- `--max_candidate_rows 512`

then the checkpoint-local score matrix is:

- `32 x 512`

and:

- each row of that matrix corresponds to one EWoK item
- each column corresponds to one BOS candidate row

### Score Reduction

`--score_reduction` controls how token log-probabilities are reduced inside each
target:

- `mean`
  Average over target tokens
- `sum`
  Sum over target tokens

### Target Scope

`--ewok_target_scope` controls which groups are exported:

- `overall`
- `per_domain`
- `both`

This affects exported grouping and summaries. The main runner still computes
item-level target rows, then aggregates downstream where needed.

## Which Runner To Use

### `run_trak.py`

Use this when you want the original TRAK-style attribution path.

Good fit:

- single-process runs
- a stable baseline
- smaller or moderate candidate sets
- environments where `traker` is already available

### `run_trackstar.py`

Use this when you want the Bergson-backed path.

Good fit:

- larger candidate sets
- reusable checkpoint-local gradient indexes
- CUDA-first execution
- multi-GPU launches via `torchrun`
- optional `accelerate launch` compatibility via environment detection

## Notebook Analysis

If you want to inspect finished attribution outputs without reading CSV/JSONL
files by hand, use:

- `analysis/notebooks/analyze_attribution_outputs.ipynb`

That notebook is designed as a guided viewer for TrackStar and TRAK exports. It
starts from one output directory, discovers the files that exist there, explains
what each file means, and comes with prebuilt checks for:

- artifact inventory
- per-step summaries
- hardest EWoK targets by softplus loss
- globally influential candidate rows
- rows that recur across many target-level top-k lists
- domain-specialized rows
- one-target and one-row drilldowns

By default it points at the local `trackstar_step16000` example output. In most
cases the only thing you need to change is the `OUTPUT_DIR` cell.

Two practical notes:

- Most of the notebook's reusable logic lives in
  `analysis/attribution/common/notebook_analysis.py`.
- The bundled `trackstar_step16000` example is a smoke-sized run, so
  `max_targets` truncation can leave it with only a subset of EWoK domains.
  The domain leaderboard cell now defaults to the first domain actually
  exported by the loaded folder rather than assuming a fixed domain name.

## Dependencies

### Shared

Use the repo’s `babylm` environment when possible:

```bash
conda run -n babylm python -m ...
```

### TRAK

The TRAK backend requires the `traker` package to be importable.

The code expects one of these workflows to have happened already:

- `pip install traker`
- `pip install 'traker[fast]'`

### TrackStar / Bergson

The TrackStar backend requires the `bergson` Python package to be importable.

The code is written for the EleutherAI Bergson repo:

- https://github.com/EleutherAI/bergson/

The clone location does not matter to the attribution code. What matters is
that the same Python environment used for attribution can import `bergson`.

A simple local workflow is:

```bash
cd /home/jorge/tokenPred
git clone https://github.com/EleutherAI/bergson.git

conda run -n babylm pip install -e /home/jorge/tokenPred/bergson
conda run -n babylm python -c "import bergson; print(bergson.__file__)"
```

If you prefer another location such as `/home/jorge/src/bergson`, that is also
fine. The important step is the editable install into `babylm`:

```bash
conda run -n babylm pip install -e /absolute/path/to/bergson
```

If the verification command prints a file path instead of raising
`ModuleNotFoundError`, the TrackStar runner should be able to import Bergson.

## Common CLI Arguments

Both runners support the same outer configuration surface:

- `--run_dir`
  Finished BOS run directory.
- `--data_dir`
  BOS training data directory for the analyzed run.
  This can be either a materialized BOS-row dataset or an exact BOS packed-index
  artifact.
- `--exp_name`
  Name of the analysis subdirectory created under
  `run_dir/analysis/attribution/`.
- `--output_dir`
  Override the output directory.
- `--cache_dir`
  Override the cache directory.
- `--checkpoint_steps`
  Optional explicit checkpoint steps. If omitted, all discovered checkpoints are
  used.
- `--candidate_strategy`
  Candidate selection strategy.
- `--max_candidate_rows`
  Cap on checkpoint-local candidate rows.
- `--candidate_seed`
  Seed for deterministic candidate subsampling.
- `--recent_window_steps`
  Window size used by `recent_window`.
- `--ewok_score_view`
  One of `babylm_completion_choice` or
  `ewok_paper_context_sensitivity`.
- `--ewok_target_scope`
  One of `overall`, `per_domain`, or `both`.
- `--score_reduction`
  One of `mean` or `sum`.
- `--temperature`
  Softplus temperature for the paired EWoK objective.
- `--topk`
  Number of top rows per target written to the exported CSVs.
- `--bottomk`
  Optional number of bottom rows per target written to `bottom_rows_*.csv`.
  Set this when you want exact per-target negative examples without saving the
  full dense matrix.
- `--write_dense_scores`
  If set, write the full dense score matrix for each checkpoint.
- `--device`
  One of `cuda`, `auto`, or `cpu`.
- `--distributed`
  One of `none`, `ddp`, or `fsdp`.
  Only `run_trackstar.py` supports distributed execution.
- `--batch_size`
  Batch size for row featurization and target batching.
- `--proj_dim`
  Projection dimension passed to the backend where applicable.
- `--use_fast_jl`
  Toggle fast JL-style projection behavior in the backend path that supports it.
- `--max_targets`
  Limit the number of EWoK items. Useful for smoke tests.

## How Row Summaries Work

The `row_summary_stepXXXXXXXX.csv` and `domain_summary_stepXXXXXXXX.csv` files
are not per-target tables. They are aggregated views over a target set.

For one checkpoint, let:

- `s_t(r)` be the attribution score for target `t` and candidate row `r`

Then the row-level summary metrics are:

- `mean_score`
  `mean_t s_t(r)` over the summarized target set.
- `mean_abs_score`
  `mean_t |s_t(r)|` over the summarized target set.
- `positive_score_sum`
  `sum_t max(s_t(r), 0)`.
- `negative_score_sum`
  `sum_t min(s_t(r), 0)`.
- `max_abs_score`
  `max_t |s_t(r)|`.
- `target_count`
  Number of targets included in that summary.

This means:

- `row_summary_step...csv` uses the overall target set for that checkpoint
- `domain_summary_step...csv` uses only the targets inside each domain group

So the summaries are meant to answer questions like:

- Which training rows matter broadly across many EWoK items?
- Which rows spike strongly for at least one item?
- Which rows are mostly positive, mostly negative, or mixed?

They are not meant to replace the per-target `top_rows_*.csv` or
`bottom_rows_*.csv` tables.

## How To Run TRAK

Run from the repo root:

```bash
cd /home/jorge/tokenPred/moonshotGPT
```

Single-checkpoint smoke test:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trak_smoke \
  --checkpoint_steps 30000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --topk 20 \
  --bottomk 20 \
  --device cuda
```

All-checkpoint run with a larger candidate pool:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trak_full \
  --candidate_strategy between_checkpoints \
  --max_candidate_rows 50000 \
  --topk 100 \
  --device cuda
```

Important runtime note:

- `run_trak.py` is single-process only.
- If you launch it with `torchrun` or `accelerate launch` and more than one
  process is detected, the runner will fail loudly.

## How To Run TrackStar

TrackStar uses the same outer pipeline but swaps in the Bergson backend.

Run from the repo root:

```bash
cd /home/jorge/tokenPred/moonshotGPT
```

Single-GPU smoke test:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 30000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --topk 20 \
  --bottomk 20 \
  --device cuda \
  --distributed none
```

Multi-GPU with `torchrun`:

```bash
cd /home/jorge/tokenPred/moonshotGPT
torchrun --nproc_per_node=4 -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_ddp \
  --checkpoint_steps 30000 \
  --max_candidate_rows 50000 \
  --device cuda \
  --distributed ddp
```

Compatible `accelerate launch` pattern:

```bash
cd /home/jorge/tokenPred/moonshotGPT
accelerate launch --num_processes 4 -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_accelerate \
  --checkpoint_steps 30000 \
  --device cuda \
  --distributed ddp
```

Important runtime notes:

- `torchrun` is the primary multi-GPU path.
- `accelerate launch` works because the runner honors common Accelerate
  environment variables, not because it uses `Accelerator`.
- `--distributed ddp|fsdp` requires CUDA and a multi-process launcher.
- `--device cuda` fails loudly if CUDA is unavailable.
- `--device auto` allows CPU fallback.
- TrackStar now prints stage-by-stage status messages so the terminal is not
  blank during long index-build or query-scoring phases.

## What Each Method Actually Computes

### TRAK

For each checkpoint:

1. Load the checkpoint into the model.
2. Build the checkpoint-local candidate dataset from exposed row ids.
3. Featurize candidate rows with TRAK.
4. Score EWoK targets against those candidate features.
5. Export the score matrix and summaries.

### TrackStar

For each checkpoint:

1. Load the checkpoint into the model.
2. Build the checkpoint-local candidate dataset from exposed row ids.
3. Build or reuse a Bergson gradient index under the cache directory.
4. Compute query gradients for the custom paired EWoK softplus loss.
5. Score candidate rows against those query gradients.
6. Assemble the same exported score matrix shape used by the TRAK path.

The TrackStar path is still item-level in the main exported run. Internally the
query layer can also reduce gradients by domain or overall, but the primary run
path keeps the export contract aligned with TRAK.

## Output Layout

By default, outputs are written under:

`<run_dir>/analysis/attribution/<exp_name>/`

The main files are:

- `config.json`
  Fully resolved config used for the run.
- `checkpoint_manifest.json`
  Ordered checkpoint list.
- `target_items.jsonl`
  Item-level EWoK target metadata.
- `top_rows_stepXXXXXXXX.csv`
  Top positively scoring rows per target.
- `bottom_rows_stepXXXXXXXX.csv`
  Bottom negatively scoring rows per target when `--bottomk` is enabled.
- `row_summary_stepXXXXXXXX.csv`
  Overall row-level summaries for each checkpoint.
- `domain_summary_stepXXXXXXXX.csv`
  Per-domain row-level summaries when domain grouping is present.
- `target_diagnostics_stepXXXXXXXX.jsonl`
  Item-level margin and softplus diagnostics.
- `dense_scores_stepXXXXXXXX.npy`
  Full score matrix when `--write_dense_scores` is enabled.
  This is larger than `bottom_rows_*.csv`, but it preserves every target-row
  score so you can recompute arbitrary top-k and bottom-k views after the run.
- `checkpoint_compare.csv`
  Adjacent-checkpoint comparison summary.
- `run_summary.json`
  Overall run-level summary and artifact manifest.

## Export Metric Glossary

This section is the plain-language reference for the exported columns written by
`common/export.py`.

### `top_rows_stepXXXXXXXX.csv` and `bottom_rows_stepXXXXXXXX.csv`

Each row in these files is one `(target, candidate_row)` pair.

- `checkpoint_step`
  Training step for the checkpoint being analyzed.
- `target_id`
  Stable identifier for one EWoK item.
- `domain`
  EWoK domain for that target item.
- `row_id`
  Stable global BOS row id.
- `score`
  Attribution score for that exact `(target, row)` pair.
  Higher positive values mean the row is more aligned with the target-side
  query gradient. More negative values mean the row is more opposed.
- `rank`
  Position within that target's exported ranking.
  In `top_rows`, rank 1 is the highest score. In `bottom_rows`, rank 1 is the
  lowest score.
- `m1`
  First EWoK margin for that target.
- `m2`
  Second EWoK margin for that target.
- `softplus_loss`
  Paired softplus loss for that target.
- `combined_margin`
  `0.5 * (m1 + m2)` for that target.
- `shard_path`
  Path to the BOS shard that contains the candidate row.
- `local_row_idx`
  Row index within that shard.

Important note:

- `m1`, `m2`, `softplus_loss`, and `combined_margin` are target-level
  diagnostics, so they repeat across all exported rows for the same `target_id`
  by design.

### `row_summary_stepXXXXXXXX.csv` and `domain_summary_stepXXXXXXXX.csv`

Each row in these files is one candidate BOS row aggregated over a target set.

- `checkpoint_step`
  Training step for the checkpoint being analyzed.
- `group`
  Summary group name.
  `overall` means all exported targets for the checkpoint.
  `domain:...` means only targets in that domain.
- `row_id`
  Stable global BOS row id.
- `mean_score`
  Mean signed attribution score over the summarized target set.
- `mean_abs_score`
  Mean absolute attribution score over the summarized target set.
- `positive_score_sum`
  Sum of positive contributions only.
- `negative_score_sum`
  Sum of negative contributions only.
- `max_abs_score`
  Largest absolute contribution this row had for any target in the summarized
  set.
- `target_count`
  Number of targets in the summarized set.
- `shard_path`
  Path to the BOS shard that contains the row.
- `local_row_idx`
  Row index within that shard.

### `target_diagnostics_stepXXXXXXXX.jsonl`

Each line is one EWoK target item.

- `s11_mean`, `s12_mean`, `s22_mean`, `s21_mean`
  Mean token-level log-prob reductions for the four paired conditionals.
- `s11_sum`, `s12_sum`, `s22_sum`, `s21_sum`
  Sum token-level log-prob reductions for the same four conditionals.
- `margin_1`, `margin_2`
  The two EWoK margins induced by `--ewok_score_view`.
- `combined_margin`
  `0.5 * (margin_1 + margin_2)`.
- `softplus_loss`
  `0.5 * [softplus(-m1 / tau) + softplus(-m2 / tau)]`.
- `score`
  Negative softplus loss used as the target-side scalar score.

## Cache Layout

The default cache directory is:

`<output_dir>/cache/`

TRAK stores checkpoint-local feature artifacts there.

TrackStar stores checkpoint-local cache artifacts under:

`cache/stepXXXXXXXX/trackstar/`

This includes reusable candidate index metadata and related backend artifacts so
query scoring can be rerun without rebuilding the candidate index when the
checkpoint and candidate set are unchanged.

## Recommended Workflow

For a new run:

1. Start with a smoke test.
2. Use `--checkpoint_steps` to target one checkpoint.
3. Use a small `--max_candidate_rows`.
4. Use a small `--max_targets`.
5. Inspect `top_rows_*.csv` and `target_diagnostics_*.jsonl`.
6. Scale up once the outputs look sane.

For checkpoint-to-checkpoint analysis:

1. Keep `--exp_name` stable for a coherent output directory.
2. Run multiple checkpoints in order.
3. Inspect `row_summary_*.csv` for each checkpoint.
4. Use `checkpoint_compare.csv` to quantify overlap and sign flips.

## Troubleshooting

### `--device cuda was requested but CUDA is not available`

The runner is designed to fail loudly instead of silently dropping to CPU.

Use one of these options:

- run in an environment with working CUDA
- switch to `--device auto`
- switch to `--device cpu` if you explicitly want CPU

### TrackStar says Bergson is missing

Make sure the environment can import `bergson`.

The backend does not use the Bergson CLI. It imports Bergson directly from
Python, so the package must be installed into the same environment used to run
the attribution command.

### Multi-process launch fails on TRAK

That is expected. Only TrackStar currently supports distributed execution.

### A run is too slow

Start by reducing:

- `--max_candidate_rows`
- `--max_targets`
- number of `--checkpoint_steps`

Then scale back up after the artifacts look reasonable.

## Quick Pointers

- Start with this file for the package overview.
- Read `trak/README.md` for TRAK-specific details.
- Read `trackstar/README.md` for Bergson-specific details.
