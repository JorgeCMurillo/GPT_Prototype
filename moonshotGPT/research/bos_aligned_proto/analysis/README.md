# BOS Analysis

This directory is the post-training analysis layer for the BOS-aligned
prototype. It does not own training or data packing. Its job is to read finished
artifacts and answer questions like:

- how did a checkpoint perform?;
- how does one checkpoint compare with another?;
- which training rows seem most aligned with better EWoK behavior?;
- which investigations are still exploratory enough to belong in notebooks?

If you are here to study retained data and EWoK, this is the right level of the
repo to start from.

## Current Layout

```text
analysis/
  README.md
  __init__.py
  run_checkpoint_evals.py
  plot_ewok_baseline_full_mean.py
  plot_ewok_checkpoint_baseline_compare.py
  attribution/
  notebooks/
```

## File Guide

### `run_checkpoint_evals.py`

Maintained post-hoc benchmark runner for a resolved BOS checkpoint or a Hugging
Face model id.

What it does:

- resolves one checkpoint;
- runs CORE, HellaSwag, EWoK, and BLiMP in a fixed order;
- writes standalone outputs that can be inspected later without rerunning
  training.

Typical command:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  runs/research/bos_aligned_proto/<run_name> \
  --step 16000
```

### `plot_ewok_baseline_full_mean.py`

Plots a standalone checkpoint `ewok_metrics.json` file. This is useful when you
want a checkpoint baseline view without looking at a whole training curve.

### `plot_ewok_checkpoint_baseline_compare.py`

Overlays a checkpoint EWoK baseline on a run’s `step_metrics.json` trajectory.
This is useful when you want to compare a post-hoc checkpoint evaluation against
the training-time curve.

### `attribution/`

The maintained attribution package.

This is where the reusable implementation lives for:

- checkpoint resolution;
- BOS row reconstruction;
- exposure-log parsing;
- candidate row selection;
- EWoK target construction;
- backend scoring with TRAK or TrackStar;
- export of summaries and checkpoint-to-checkpoint comparisons.

If you want the main answer to "which rows seem to improve EWoK?", start here.

### `notebooks/`

Exploratory workflows that sit on top of finished outputs. These notebooks are
useful for inspection and idea generation, but they are not the source of truth
for the maintained attribution pipeline.

## How This Fits the Workflow

The broader BOS prototype has three layers:

1. `pipeline/` and `training/` create data views, checkpoints, and exposure logs.
2. `evaluation/` measures benchmark behavior.
3. `analysis/` reads those artifacts afterward and explains or compares them.

This directory is intentionally that third layer.

## Where To Start

- If you want standalone benchmark outputs for a checkpoint, start with
  `run_checkpoint_evals.py`.
- If you want the structured attribution pipeline, read
  `attribution/README.md`.
- If you want the Bergson-backed backend specifically, read
  `attribution/trackstar/README.md`.
- If you want a looser exploratory workflow, open `notebooks/README.md`.
