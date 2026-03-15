# BOS TRAK Package

This package implements checkpoint-local TRAK attribution for the BOS-aligned
prototype. The intended question is: which BOS-packed training rows most support
the EWoK behavior observed at a given checkpoint, and how does that support
change over training?

## Pipeline Overview

The package follows a fixed flow:

`config -> checkpoints -> row manifest -> exposures -> candidates -> EWoK targets -> model output / TRAK -> export -> compare`

The orchestration entrypoint is `run_trak.py`. The only module that should
directly depend on the `trak` library is `model_output.py`.

For live TRAK validation, use the `babylm` conda environment. It already
contains a compatible `trak` installation along with the PyTorch and
Transformers versions used by this repo.

## File Map

### `__init__.py`

Small public entrypoint that re-exports the main config object and top-level run
function for callers that want a stable import surface.

### `config.py`

Defines the `TRAKConfig` dataclass and CLI parsing. This is the single place
that owns defaults, validation, and output/cache path resolution.

### `checkpoints.py`

Discovers checkpoints from completed BOS runs and loads models, tokenizers, or
raw checkpoint weights when the runner needs them.

### `row_dataset.py`

Builds a deterministic manifest over BOS-packed shard files and exposes a finite
per-row dataset. This is the bridge from on-disk packed data to stable global
row IDs that the rest of the pipeline can reason about.

### `exposures.py`

Parses exposure JSONL logs and converts token offsets back into row IDs using
the row manifest. It answers questions such as which rows were exposed by a
given checkpoint or first appeared between two checkpoints.

### `candidates.py`

Implements checkpoint-local candidate selection strategies such as
`between_checkpoints`, `up_to_step`, `recent_window`, and `new_since_prev`. It
also owns deterministic subsampling when the exposed row set is too large.

### `ewok_targets.py`

Loads the fast EWoK bundle, normalizes item metadata, defines item-level target
records, and prepares tokenized batches for the paired comparisons used during
scoring.

### `model_output.py`

Contains all direct integration with `trak`. It defines the train-side BOS-row
scalar, the EWoK target-side scalar, optional dependency handling for `trak`,
and the backend that drives featurization and scoring for one checkpoint.

### `run_trak.py`

Coordinates the full run. It resolves checkpoints, builds manifests and
exposure indexes, selects candidate rows, launches the backend, and writes
per-checkpoint outputs plus final comparisons.

### `export.py`

Turns checkpoint results into durable artifacts such as JSON manifests, target
diagnostics, top-row CSVs, row summaries, and optional dense score dumps.

### `compare.py`

Compares checkpoint summaries after export. It is responsible for overlap,
correlation, sign-flip, and newly influential-row comparisons across steps.

## Reader Guide

- Start with `run_trak.py` if you want to understand the end-to-end flow.
- Read `model_output.py` next if you want the actual scoring logic and TRAK
  integration details.
- Read `row_dataset.py`, `exposures.py`, and `candidates.py` together if you
  want to understand how training rows are selected.
