# Analysis Notebooks

This folder contains the exploratory notebook layer for the BOS-aligned
prototype.

Current notebooks:

- `analyze_attribution_outputs.ipynb`
- `ewok_score_columns_material_dynamics.ipynb`
- `mine_synthetic_minimal_pairs.ipynb`

These notebooks are intentionally downstream of the maintained analysis code.
They should consume finished outputs rather than replace the reusable pipeline in
`analysis/attribution/`.

## `analyze_attribution_outputs.ipynb`

This is the main viewer notebook for finished attribution exports.

It is designed for the stage after a run has already written files such as:

- `top_rows_stepXXXXXXXX.csv`
- `bottom_rows_stepXXXXXXXX.csv`
- `row_summary_stepXXXXXXXX.csv`
- `domain_summary_stepXXXXXXXX.csv`
- `target_diagnostics_stepXXXXXXXX.jsonl`
- `target_items.jsonl`
- `run_summary.json`

The notebook is meant to be readable even if you do not already know the
TrackStar or Bergson internals. It starts from one output directory, discovers
what files are present, and walks through useful checks like:

- hardest targets by paired softplus loss;
- most influential positive rows overall;
- most negative rows overall;
- rows that recur across many targets;
- domain specialization patterns;
- inspection of one target item;
- inspection of one candidate row;
- checkpoint-to-checkpoint comparisons when multiple steps were exported.

Helpful supporting code lives in:

- `research/bos_aligned_proto/analysis/attribution/common/notebook_analysis.py`

That helper module keeps the notebook itself relatively thin.

## `ewok_score_columns_material_dynamics.ipynb`

This is a narrower exploratory notebook for inspecting EWoK score columns and
concept-level comparisons inside the `material-dynamics` domain.

It computes per-item score columns like `S11`, `S12`, `S22`, and `S21`, derives
combined margin and boolean-pair metrics, and then focuses on concept-level
patterns inside one EWoK domain rather than on the full attribution export
stack.

## `mine_synthetic_minimal_pairs.ipynb`

This notebook is an exploratory data-generation workbench, not a maintained
pipeline.

Its purpose is to:

1. read BOS exposure logs from a finished run;
2. decode the row text that the model actually saw;
3. mine literal physical or material event snippets;
4. turn those snippets into `(C1, T1)` seeds;
5. use GPT generation to expand them into EWoK-style minimal pairs;
6. verify, repair, and summarize the resulting candidates.

It is tightly tied to a local research environment and currently assumes things
like:

- access to a specific completed BOS run;
- local exposure logs for that run;
- the default GPT-2 tokenizer;
- a locally configured API-key file path inside the notebook;
- later-stage outputs under `data_augmentation/`.

That means this notebook is best read as exploratory research scaffolding rather
than as a stable general-purpose tool.

## Which Notebook To Open

- Open `analyze_attribution_outputs.ipynb` when you already have attribution
  outputs and want to understand them.
- Open `ewok_score_columns_material_dynamics.ipynb` when you want focused
  `material-dynamics` score-column analysis.
- Open `mine_synthetic_minimal_pairs.ipynb` when you are exploring synthetic
  EWoK-style data generation from BOS exposure text.
