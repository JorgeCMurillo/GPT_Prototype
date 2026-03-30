# moonshotGPT

`moonshotGPT` is a GPT-2-style training and analysis workspace centered on one
question:

How do we identify which retained training data most improves EWoK?

The repo still contains the machinery for baseline training, optional rho-1
token filtering, benchmark evaluation, and debug tooling. The current emphasis,
though, is no longer "prove rho-1 works." That implementation has already been
tested. The main goal now is to train or collect candidate retained-data views,
measure benchmark behavior, and run attribution analyses that explain which
examples appear to support EWoK performance.

Rho-1 details now live in [`training_utils/README.md`](training_utils/README.md)
so the root README can stay focused on the broader workflow.

## Current Goal

The practical research loop in this repo is:

1. tokenize or load FineWeb-Edu training data;
2. train a baseline model or a retained-data variant;
3. evaluate EWoK and companion benchmarks;
4. run attribution to rank which exposed training rows look most helpful or
   harmful for EWoK;
5. inspect those rows and use the results to guide the next retained-data
   iteration.

If you are new to the repo and want the maintained attribution path, start with
`research/bos_aligned_proto/analysis/attribution/`.

## Repo Map

### Top-level training and analysis scripts

- `fineweb.py`
  Builds `train_*.bin` and `val_*.bin` token shards from FineWeb-Edu.
- `train_gpt2_finewebedu_bin.py`
  Main GPT-2-style trainer for contiguous token-stream runs.
- `compute_ref_loss_shards.py`
  Precomputes per-token reference losses for optional rho-1 filtering.
- `plot_step_metrics.py`
  Plots training-time benchmark and optimization logs from `step_metrics.json`.
- `analyze_ref_loss.py`
  Small utilities for inspecting precomputed reference-loss shards.
- `analyze_rank_overlap.py`
  Utilities for comparing rank-level exposure overlap.
- `compare_dataloaders.py`
  Debug helper for checking dataloader behavior across implementations.
- `run_training_parity_debug.py`
  Short parity and tiny-overfit debug entrypoint built on
  `training_utils/debug_parity.py`.

### Key folders

- `evaluation/`
  Shared benchmark evaluators and the reusable evaluation runner.
- `training_utils/`
  Shared helpers for rho-1, resume-safe log trimming, and parity debugging.
- `research/`
  Research packages layered on top of the main training stack. The most relevant
  subfolders are `research/bos_aligned_proto/` for BOS-packed training plus
  attribution, and `research/w2v_lexical_probe/` for Word2Vec lexical
  baselines.
- `data/`
  Preferred location for processed token shards and reference-loss shards.
- `runs/`
  Modern run outputs for debug runs and research packages.
- `experiments/`
  Legacy top-level run outputs kept for reference.
- `tests/`
  Regression coverage for training, BOS packing, rho-1, evaluators, and
  attribution helpers.
- `data_augmentation/`
  Synthetic EWoK-style item generation artifacts used by exploratory notebook
  workflows.
- `eval_bundle/`, `blimp_fast/`, `ewok_full_jsonl/`
  Local benchmark data and evaluation bundles.

## Recommended Workflow

### 1. Build or point at tokenized data

Run commands from the repo root's `moonshotGPT/` directory:

```bash
cd moonshotGPT
```

Typical tokenization command:

```bash
python fineweb.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-100BT \
  --split train \
  --text_field text \
  --tokenizer gpt2 \
  --out_dir data/processed/fineweb_edu_100B \
  --shard_tokens 100000000 \
  --val_shards 1
```

### 2. Train a baseline or retained-data run

Baseline GPT-2 Medium example:

```bash
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir data/processed/fineweb_edu_100B \
  --micro_batch_size 4 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16 \
  --num_workers 0 \
  --shuffle_blocks
```

Optional rho-1 supporting step:

```bash
accelerate launch --num_processes 8 compute_ref_loss_shards.py \
  --data_dir data/processed/fineweb_edu_100B \
  --out_dir data/ref_loss/fineweb_edu_100B/gpt2m_T1024_B4 \
  --split train \
  --seq_len 1024 \
  --batch_size 4 \
  --ref_model openai-community/gpt2-medium \
  --tokenizer gpt2 \
  --out_dtype float16 \
  --mixed_precision bf16
```

If you want the rho-1 path, keep the details in
[`training_utils/README.md`](training_utils/README.md) nearby. That document now
holds the original motivation, the masking math, and the findings from the
completed rho-1 tests.

### 3. Evaluate benchmark behavior

There are two main evaluation surfaces:

- training-time evaluation from `train_gpt2_finewebedu_bin.py`;
- post-hoc evaluation from
  `research/bos_aligned_proto/analysis/run_checkpoint_evals.py`.

The BOS post-hoc runner is the maintained path when you want standalone
checkpoint artifacts for later comparison or attribution:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.run_checkpoint_evals \
  runs/research/bos_aligned_proto/<run_name> \
  --step 16000
```

### 4. Run attribution for EWoK

The most developed attribution stack currently lives under
`research/bos_aligned_proto/analysis/attribution/`.

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

### 5. Inspect outputs and iterate

Useful destinations after an attribution run:

- `research/bos_aligned_proto/analysis/notebooks/analyze_attribution_outputs.ipynb`
  for guided output inspection;
- `research/bos_aligned_proto/analysis/attribution/common/notebook_analysis.py`
  for reusable loading and ranking helpers;
- `plot_step_metrics.py`
  for training curves and EWoK trend plots.

## Data Layout

Preferred derived-data layout:

```text
data/
  processed/
    fineweb_edu_100B/
    bos_aligned_proto/
  ref_loss/
    fineweb_edu_100B/
      gpt2m_T1024_B4/
```

Legacy top-level paths such as `ref_loss_gpt2m_T1024_B4/` are still present for
older runs, but new work should prefer the `data/` tree.

## Notes on Rho-1

Rho-1 remains available, tested, and documented, but it is now an auxiliary
piece of the repo rather than the headline.

The current framing is:

- rho-1 was useful for testing whether loss-guided token retention could be
  implemented cleanly;
- those experiments informed later retained-data questions;
- the present research priority is attribution: explain which retained rows seem
  to help EWoK, not simply whether a token filter can be turned on.

## Where To Start

- If you want the main training entrypoint, open `train_gpt2_finewebedu_bin.py`.
- If you want rho-1 details and findings, open `training_utils/README.md`.
- If you want BOS-packed training plus maintained attribution tooling, open
  `research/bos_aligned_proto/README.md`.
- If you want the attribution package directly, open
  `research/bos_aligned_proto/analysis/attribution/README.md`.
- If you want the lexical baseline side project, open
  `research/w2v_lexical_probe/README.md`.
