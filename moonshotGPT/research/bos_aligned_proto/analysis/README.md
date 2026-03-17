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

### `run_checkpoint_evals.py`

Post-hoc benchmark runner for BOS checkpoints. This is the maintained way to
take a finished run, a specific `ckpt_*_stepXXXXXXX/` folder, or a Hugging Face
model id and compute standalone evaluation artifacts after training.

The script resolves one checkpoint, loads the model, and runs evaluations in a
fixed priority order:

- CORE first
- HellaSwag second
- EWoK third
- BLiMP fourth

For EWoK, the post-hoc runner records only the mean-reduction `domain_scores_full`
outputs for:

- BabyLM completion choice
- EWoK paper context sensitivity

By default it writes outputs under:

- `runs/research/bos_aligned_proto/<run_name>/posthoc_eval/<checkpoint_name>/`
- `runs/research/bos_aligned_proto/posthoc_hf_eval/<model_slug>/` for `--hf-model`

and skips tasks whose output files already exist, so reruns can continue after
an interruption.

The evaluator defaults to `--device cuda` so it does not silently fall back to
CPU. On shared machines, a good default launch pattern is:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals ...
```

Examples:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  runs/research/bos_aligned_proto/<run_name> \
  --step 30000
```

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  --hf-model gpt2-medium
```

In `--hf-model` mode, the runner prints cache/download status before loading and
enables Hugging Face download progress bars by default, so long waits are easier
to distinguish from a stalled process. Disable that with:

```bash
--no-show-hf-download-status
```

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

- If you want post-training benchmark numbers for a checkpoint, start with
  `run_checkpoint_evals.py`.
- If you want quick orientation to the analysis area, read `trak/README.md`.
- If you want the implementation entrypoint for attribution, read
  `trak/run_trak.py`.
- If you want a looser exploratory workflow, look in `notebooks/`.
