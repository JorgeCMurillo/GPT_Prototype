# BOS Analysis Layout

This directory holds analysis code for the BOS-aligned prototype after training
artifacts already exist. The goal is to keep post-training inspection,
attribution, and exploratory work separate from the training and evaluation
entrypoints elsewhere in the repo.

In practice, `analysis/` is where we put code that answers questions like:

- What patterns show up in finished runs?
- Which training rows appear to matter for downstream behavior?
- Which ideas are still exploratory enough to live in notebooks?

## Current Layout

### `notebooks/`

Notebook-based exploratory work. This is the right place for quick
investigations, visualization drafts, and one-off analyses that are still being
shaped. Notebooks should consume existing outputs rather than becoming the
source of truth for reusable analysis logic.

### `trak/`

The structured attribution package for BOS-row TRAK analysis. This folder is
where the reusable implementation lives for:

- resolving checkpoints from a finished run
- mapping exposure logs back to BOS-packed training rows
- constructing EWoK targets
- computing checkpoint-local TRAK scores
- exporting summaries and checkpoint-to-checkpoint comparisons

If you want the maintained analysis pipeline rather than an exploratory notebook,
start here.

### `__init__.py`

Package marker for the `analysis` namespace. It exists so the analysis code can
be imported from elsewhere in the research package.

## How This Fits the Bigger Workflow

The broader BOS-aligned prototype has three different layers:

- training code creates checkpoints, exposure logs, and other run artifacts
- evaluation code measures model behavior on tasks such as EWoK
- analysis code reads those artifacts afterward and tries to explain or compare
  what happened

This directory is intentionally in that third layer. It should not own training
logic, data packing, or benchmark definitions unless analysis genuinely needs a
read-only view of those components.

## Where To Start

- If you want quick orientation to the analysis area, read `trak/README.md`.
- If you want the implementation entrypoint for attribution, read
  `trak/run_trak.py`.
- If you want a looser exploratory workflow, look in `notebooks/`.
