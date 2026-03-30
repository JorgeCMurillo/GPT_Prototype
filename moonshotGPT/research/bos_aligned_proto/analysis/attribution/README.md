# BOS Attribution Package

This package is the maintained post-training attribution pipeline for the
BOS-aligned prototype.

Its guiding question is:

Which exposed BOS-packed training rows look most helpful or harmful for EWoK at
a given checkpoint?

The outer workflow stays stable across attribution methods:

`config -> checkpoints -> row manifest -> exposures -> candidates -> EWoK targets -> backend -> export -> compare`

Everything before and after `backend` is shared. The backend is the only part
that changes between TRAK and TrackStar.

## Current Layout

```text
analysis/attribution/
  README.md
  __init__.py
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

Both runners share the same high-level contract:

- resolve checkpoints from a finished BOS run;
- reconstruct the BOS training-row universe that matches that run;
- select candidate rows from exposure logs;
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
  BOS row manifest construction for both materialized BOS rows and packed-index
  artifacts.
- `exposures.py`
  Exposure-log parsing and step-window filtering.
- `candidates.py`
  Candidate-row selection and deterministic subsampling.
- `ewok_targets.py`
  EWoK target loading and paired-loss target construction.
- `export.py`
  Writers for row summaries, domain summaries, diagnostics, and run summaries.
- `compare.py`
  Checkpoint-to-checkpoint comparison helpers.
- `notebook_analysis.py`
  Convenience loaders used by the attribution notebook.

## Supported Data Views

`--data_dir` should point at the BOS training-data view that matches the run you
are analyzing.

Two formats are supported:

- materialized BOS-row datasets with `meta.json` and `train_*.bin`;
- packed-index datasets with `meta.json`, `train.row_ptr.bin`,
  `train.segments.bin`, and `train.virtual_shards.jsonl`.

The row manifest normalizes both formats into stable row identities so exposure
records can still be traced back to candidate training rows.

## Core Concepts

### Candidate rows

Candidates are not the full dataset by default. They are BOS-packed training
rows selected from exposure logs near the checkpoint you are analyzing.

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
read-only.

## Which Runner To Use

### Use `run_trak.py` when:

- you want the simpler baseline backend;
- you are running single-process attribution;
- you want parity with earlier TRAK-style experiments.

### Use `run_trackstar.py` when:

- you want the Bergson-backed path;
- you want the backend that is currently most aligned with the repo’s
  retained-data and EWoK analysis direction;
- you want cosine-style gradient-alignment scores for candidate rows.

## Example Commands

TrackStar example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir runs/research/bos_aligned_proto/<run_name> \
  --data_dir data/processed/bos_aligned_proto/<data_view> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

TRAK example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir runs/research/bos_aligned_proto/<run_name> \
  --data_dir data/processed/bos_aligned_proto/<data_view> \
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
