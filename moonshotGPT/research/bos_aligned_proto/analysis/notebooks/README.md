# Analysis Notebooks

This folder currently contains two notebooks:

- `analyze_attribution_outputs.ipynb`
- `mine_synthetic_minimal_pairs.ipynb`

## `analyze_attribution_outputs.ipynb`

This notebook is a guided viewer for exported attribution outputs, especially
TrackStar runs.

It is meant for the stage after a run has already finished and written files
such as:

- `top_rows_stepXXXXXXXX.csv`
- `bottom_rows_stepXXXXXXXX.csv`
- `row_summary_stepXXXXXXXX.csv`
- `domain_summary_stepXXXXXXXX.csv`
- `target_diagnostics_stepXXXXXXXX.jsonl`
- `target_items.jsonl`
- `run_summary.json`

The notebook is intentionally written for someone who does not already know the
internal TrackStar or Bergson code. It starts from one output directory,
discovers the files that exist there, explains what each file means, and then
walks through useful checks such as:

- hardest targets by softplus loss
- most influential rows overall
- most negative rows overall
- rows that recur across many targets
- domain specialization heatmaps
- inspection of one target item
- inspection of one candidate row
- checkpoint-to-checkpoint comparisons when multiple steps are present

By default it points at the local successful `trackstar_step16000` run, but the
only value you need to change is `OUTPUT_DIR`.

Two small orientation notes that help when reading the notebook:

- Most of the reusable loading, ranking, and plotting helpers live in
  `analysis/attribution/common/notebook_analysis.py`, so the notebook itself is
  intentionally thin.
- The notebook can now decode exported rows from both materialized BOS-row
  datasets and exact BOS packed-index artifacts, so newer `*.vrow` shard paths
  are expected and valid.
- The bundled `trackstar_step16000` example is a smoke-sized run with
  `max_targets` truncation, so it may expose only a subset of EWoK domains.
  The domain leaderboard cell now auto-selects the first exported domain from
  the loaded folder.

## `mine_synthetic_minimal_pairs.ipynb`

This notebook is not just a decoding demo. It is a working notebook for mining
synthetic EWoK-style minimal pairs out of BOS exposure text, then pushing those
candidates through generation, verification, repair, and summary analysis.

### Big Picture

The notebook is trying to bootstrap new counterfactual evaluation items from the
actual BOS-row training stream. The rough idea is:

1. read exposure logs from a completed BOS run
2. decode the exact row text that specific micro-batches came from
3. mine literal physical/material event snippets from that text
4. turn those snippets into `(C1, T1)` candidate seeds
5. ask GPT-5.2 to expand them into EWoK-style `(C1, C2, T1, T2)` minimal pairs
6. verify the generated pairs against structural and semantic constraints
7. optionally repair failed-but-salvageable pairs
8. inspect concept coverage and pipeline yield with summary plots

So the notebook is really a synthetic data-generation workbench, not just a
viewer for exposure logs.

### What The Notebook Actually Does

The notebook has a few major phases.

#### 1. Exposure log inspection and decoding

The early cells define utilities for:

- discovering `exposures_rank*.jsonl` files
- iterating per-step exposure records
- selecting micro-batches by rank and step
- memory-mapping shard files from `start` / `end` token offsets
- decoding BOS-packed rows with the default GPT-2 tokenizer

This gives the notebook a way to recover the exact row text that the model saw
during selected training steps.

#### 2. Step-based text extraction

The next utility layer lets the notebook:

- select explicit steps or step ranges
- decode exposure text for those steps
- return nested structures grouped by step or rank
- extract text-only payloads for downstream mining

That is the bridge from raw exposure metadata to a large list of decoded row
strings.

#### 3. Literal material/physical event mining

The middle of the notebook shifts from decoding to mining. It defines a fairly
large set of heuristics for:

- sentence splitting and cleanup
- literal-vs-idiomatic filtering
- active concept detection
- material-group detection
- swappable noun detection
- physical-frame filtering

At the moment the active concept priority is narrowed to:

- `stir`
- `wrinkle`
- `hang`
- `squeeze`

The goal is to mine plausible literal physical situations from exposure text,
not arbitrary verb mentions.

#### 4. Candidate `(C1, T1)` construction and filtering

After literal sentence mining, the notebook constructs seed pairs using two
strategies:

- adjacent-sentence mining
- split-sentence mining

Those raw candidates are then filtered for properties like:

- having a swappable key noun
- keeping the noun grounded in the context
- looking like a literal physical/material frame
- matching the currently allowed active verbs

This stage is trying to keep only seeds that are realistic enough to become
good counterfactual minimal pairs later.

#### 5. GPT-5.2 generation of EWoK-style pairs

The notebook then sets up the OpenAI client and uses GPT-5.2 with a long prompt
to transform mined `(C1_RAW, T1_RAW)` seeds into structured minimal pairs:

- `C1`
- `C2`
- `T1`
- `T2`

The prompt is geared toward material-dynamics style reasoning and emphasizes:

- one-sentence structure for each field
- minimal paraphrase when possible
- strong crossed plausibility
- explicit focus on the prioritized concepts

This part of the notebook is where the mined raw text becomes candidate
benchmark items.

#### 6. Verifier and repair pass

The notebook does not trust model generations blindly. Later cells define a
verifier that checks things like:

- JSON extractability
- one-sentence structure
- constrained edit spans
- context swap consistency
- banned explanatory phrasing

It then partitions results into:

- verifier-passed outputs
- model rejects
- structurally failed but potentially repairable items

There is also an editor/repair prompt intended to salvage failed-but-promising
items by making small localized edits, mostly to `T2`.

#### 7. Export-oriented summaries and coverage plots

The final section looks more like dataset curation analytics. It computes:

- concept counts in the merged final pairs
- concept counts earlier in the mining/filter pipeline
- stage-by-stage pipeline yield
- bar plots for concept distribution and pipeline attrition

This is useful for checking whether the generation process is balanced and
whether the notebook is producing enough usable items.

### Important Assumptions and Hardcoded Dependencies

This notebook is exploratory and currently has several hardcoded assumptions:

- it points at a specific completed BOS run under
  `research/bos_aligned_proto/experiments/...steps18000`
- it expects exposure logs to exist for that run
- it decodes with the default GPT-2 tokenizer
- it expects an OpenAI API key at
  `/home/jorge/tokenPred/moonshotGPT/openai_key.txt`
- it calls `client.responses.create(..., model=\"gpt-5.2\")`
- later statistics cells look for merged pair files under
  `/home/jorge/tokenPred/moonshotGPT/data_augmentation/`

In other words, this notebook is a research workbench tied to a specific local
environment, not yet a fully packaged analysis pipeline.

### How To Read It

The cleanest way to understand the notebook is to read it in this order:

1. the opening markdown cells for the exposure/micro-batch mental model
2. the exposure decode utilities
3. the mining and filtering cells
4. the GPT generation prompt cells
5. the verifier and repair cells
6. the final coverage and pipeline summary plots

If you jump straight to the later plotting cells, the notebook can look like a
dataset report. In reality, most of the notebook is about building that dataset
from exposure text in the first place.

### Current Status

This notebook is clearly in-progress and iterative:

- some large cells are commented out snapshots of earlier ideas
- some export/repair code appears partially staged rather than fully cleaned up
- multiple sections overwrite shared variables like `prompt`, `pairs`, and
  related payloads

That is normal for exploratory notebook work, but it also means the notebook is
best treated as a research prototype and reference workflow, not as a stable
production pipeline.
