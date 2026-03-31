# Attribution Package

This package is the maintained post-training attribution pipeline for the
BOS-aligned prototype.

Its guiding question is:

Which exposed training examples look most helpful or harmful for EWoK at a
given checkpoint?

The outer workflow stays stable across attribution methods:

`config -> checkpoints -> training-example manifest -> exposures -> candidates -> EWoK targets -> backend -> export -> compare`

Everything before and after `backend` is shared. The backend is the only part
that changes between TRAK and TrackStar.

## Current Layout

```text
analysis/attribution/
  README.md
  __init__.py
  build_matched_cpt_pools.py
  run_trak.py
  run_trackstar.py
  common/
  trak/
  trackstar/
```

## Top-Level Entry Points

- `run_trak.py`
  CLI runner for the TRAK backend.
- `run_trackstar.py`
  CLI runner for the Bergson-backed TrackStar path.
- `build_matched_cpt_pools.py`
  Utility for turning attribution-ranked candidates into matched treated/control
  continued-pretraining datasets.

TrackStar-specific continued-pretraining ablation utilities live under
`trackstar/`:

- `trackstar/run_cpt_ablation.py`
  Paired continued-pretraining runner for matched treated/control pools.
- `trackstar/plot_cpt_ablation.py`
  Plotter for treated/control margin curves and treated-minus-control effects.
- `trackstar/cpt_ablation.py`
  Shared helper layer for planning runs, preparing BOS-row training views,
  evaluating the step-0 baseline, and aggregating outputs.

Both runners share the same high-level contract:

- resolve checkpoints from a finished BOS run;
- reconstruct the faithful training-example universe that matches that run;
- select candidate examples from exposure logs;
- build EWoK query targets;
- score the checkpoint-local candidate set;
- export reusable CSV and JSONL summaries.

## `common/` File Guide

The shared pipeline logic lives in `common/`:

- `config_base.py`
  Shared CLI defaults, validation, and configuration pieces used by both
  backends.
- `checkpoints.py`
  Checkpoint discovery, model loading, and tokenizer resolution.
- `row_dataset.py`
  Backward-compatible wrapper around the generic training-example manifest.
- `training_examples.py`
  Shared manifest/dataset layer for BOS-packed rows and stream-trained windows.
- `training_metadata.py`
  Run-aware logic for recovering whether a checkpoint trained on BOS-packed rows
  or exact stream windows.
- `exposures.py`
  Exposure-log parsing and step-window filtering.
- `candidates.py`
  Candidate-example selection and deterministic subsampling.
- `ewok_targets.py`
  EWoK target loading and paired-loss target construction.
- `export.py`
  Writers for row summaries, domain summaries, diagnostics, and run summaries.
- `compare.py`
  Checkpoint-to-checkpoint comparison helpers.
- `notebook_analysis.py`
  Convenience loaders used by the attribution notebook.
- `ewok_query_specs/`
  Example JSON files for restricting EWoK queries by domain, context type,
  context difference, target difference, or explicit row index.

## Supported Data Views

`--data_dir` should point at the training-data view that matches the run you are
analyzing.

Two faithful candidate modes are supported:

- BOS-packed rows:
  materialized BOS-row datasets with `meta.json` and `train_*.bin`, or
  packed-index datasets with `meta.json`, `train.row_ptr.bin`,
  `train.segments.bin`, and `train.virtual_shards.jsonl`.
- stream windows:
  raw token-stream shards with `meta.json` and `train_*.bin`, where candidates
  are reconstructed as the exact `seq_len + 1` windows that produced SGD
  updates during training.

The runner infers the right candidate unit from the finished run plus
`--data_dir`. For stream runs, this is intentionally more faithful than
post-hoc document reconstruction: candidates are the real training windows, even
if some windows cross document boundaries.

## Core Concepts

### Candidate examples

Candidates are not the full dataset by default. They are BOS-packed training
examples selected from exposure logs near the checkpoint you are analyzing.

Supported strategies include:

- `between_checkpoints`
- `up_to_step`
- `recent_window`
- `new_since_prev`

If the pool is too large, it is deterministically subsampled with
`--max_candidate_rows` and `--candidate_seed`.

### EWoK targets

Each target is one EWoK item, not one training row.

The pipeline builds paired query targets from the four conditional scores:

- `s11 = log P(T1 | C1)`
- `s12 = log P(T2 | C1)`
- `s22 = log P(T2 | C2)`
- `s21 = log P(T1 | C2)`

Two score views are supported:

- `babylm_completion_choice`
  Uses `m1 = s11 - s12` and `m2 = s22 - s21`.
- `ewok_paper_context_sensitivity`
  Uses `m1 = s11 - s21` and `m2 = s22 - s12`.

`--score_reduction` controls whether target token log-probabilities are reduced
with `mean` or `sum`.

You can also restrict the EWoK query set with:

- `--ewok_variant fast|full`
- `--ewok_filter_spec <path-to-json>`

The filter spec files under
[ewok_query_specs/](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs)
show the supported schema and the currently observed category values.

### Export groups

`--ewok_target_scope` controls whether exports are grouped:

- overall;
- per domain;
- both.

## Exported Artifacts

Finished runs typically write files such as:

- `top_rows_stepXXXXXXXX.csv`
- `bottom_rows_stepXXXXXXXX.csv`
- `row_summary_stepXXXXXXXX.csv`
- `domain_summary_stepXXXXXXXX.csv`
- `target_diagnostics_stepXXXXXXXX.jsonl`
- `target_items.jsonl`
- `run_summary.json`

These exports are designed so downstream notebooks can stay thin and mostly
read-only. The CSVs now include generic candidate metadata such as
`candidate_id`, `candidate_kind`, `local_example_idx`, and token offsets, while
keeping the legacy `row_id` aliases for older notebook code.

When `--write_dense_scores` is enabled, the runner also writes
`dense_scores_stepXXXXXXXX.npy`, which exposes the full query-by-candidate score
matrix. That file is what enables later per-query candidate selection.

## Building Fair Continued-Pretraining Pools

`build_matched_cpt_pools.py` is the downstream utility for small
continued-pretraining experiments where you want the main contrast to be
TrackStar score, not easy confounders like shard mixture or token budget.

The utility now supports both BOS-packed candidates and stream-window
candidates. In both cases it materializes the selected examples into fixed-row
datasets for ablation, so the treated/control pools preserve the exact selected
training examples during continued pretraining. It:

- selects treated rows by one score view;
- matches controls on `candidate_kind` and exact `token_count`;
- prefers controls from the same `shard_path`;
- breaks ties by choosing the closest `local_example_idx`;
- prefers controls whose score is at or below `--max_control_score`;
- writes a balance report so you can audit how well the treated/control split
  stayed matched on observable metadata.

Supported score views are:

- `positive_pooled`
  Uses the row-summary column `positive_score_sum`, defined as:

$$
S_+(x_i) = \sum_j max(s(x_i, q_j), 0)
$$

- `mean_score`
  Uses the row-summary mean over all selected EWoK queries.
- `mean_abs_score`
  Uses the row-summary mean absolute score.
- `per_query`
  Uses one row of `dense_scores_stepXXXXXXXX.npy`, selected by `--target_id`.
  This mode requires that the attribution run was exported with
  `--write_dense_scores`.

For stream-trained checkpoints, the builder infers `seq_len` from the row
summary token counts by default, and you can override it with `--seq_len` if
needed.

Example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.build_matched_cpt_pools \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/outputs \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --step 16000 \
  --output_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/matched_positive_pooled \
  --score_mode positive_pooled \
  --num_treated 2048 \
  --max_control_score 0.0
```

## Running The CPT Ablation

Once you have a matched pool, the TrackStar ablation runner can take one
checkpoint plus that matched-pool root and launch paired continued-pretraining
arms:

- `treated`
  High-score rows selected from TrackStar outputs.
- `control`
  Low-score rows matched on observable metadata such as token count and shard
  membership.

V1 of this runner uses weights-only continuation from the checkpoint. It does
not resume the original optimizer state. The runner prepares BOS-row training
views for the BOS trainer and synthesizes a `val_000000.bin` split by
concatenating the train shards, so the matched-pool artifacts do not need to be
edited in place.

Default training settings are:

- `micro_batch_size = 4`
- `total_batch_tokens = 32768`
- effective global batch target = `32` sequences at `seq_len = 1024`
- `num_epochs = 3`
- `ewok_every = steps_per_epoch`
- `hellaswag_every = 0`
- `core_every = 0`

The main score view assumed by this workflow is still:

$$
S_+(x_i) = \sum_j max(s(x_i, q_j), 0)
$$

Example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_cpt_ablation \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name>/ckpt_final_step0016000 \
  --matched_pool_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/matched_positive_pooled \
  --output_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/cpt_ablation_step16000 \
  --learning_rates 1e-5,2e-5,4e-5,8e-5 \
  --seeds 42,43,44
```

The runner writes:

- `ablation_manifest.json`
- `baseline/baseline_ewok_items.jsonl`
- `baseline/baseline_summary.json`
- `ablation_runs.json`
- `ablation_curves.jsonl`
- `ablation_summary.json`
- `plots/`

`ablation_curves.jsonl` stores per-eval-point records with a baseline margin and
delta from baseline, while `ablation_summary.json` stores the final paired
effect:

$$
\Delta_{\text{arm}}(t) = M_{\text{arm}}(t) - M_{\text{base}}
$$

$$
\Delta\Delta(t) = \Delta_{\text{treated}}(t) - \Delta_{\text{control}}(t)
$$

which is the same as:

$$
M_{\text{treated}}(t) - M_{\text{control}}(t)
$$

If you already have the aggregated JSONL outputs and just want plots, use:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.plot_cpt_ablation \
  --ablation_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/cpt_ablation_step16000 \
  --group_by average \
  --reduction mean
```

## Which Runner To Use

### Use `run_trak.py` when:

- you want the simpler baseline backend;
- you are running single-process attribution;
- you want parity with earlier TRAK-style experiments.

### Use `run_trackstar.py` when:

- you want the Bergson-backed path;
- you want the backend that is currently most aligned with the repo’s
  retained-data and EWoK analysis direction;
- you want cosine-style gradient-alignment scores for candidate examples.

## Example Commands

TrackStar example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

TRAK example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --exp_name trak_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

## Where To Read Next

- `trak/README.md` for the TRAK backend.
- `trackstar/README.md` for the Bergson-backed backend.
- `../notebooks/README.md` for notebook-based inspection on top of exported
  artifacts.
