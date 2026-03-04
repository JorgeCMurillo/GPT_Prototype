# moonshotGPT: GPT-2 Medium on FineWeb-Edu with Optional Rho-1 Token Filtering

This repository trains GPT-2 style models on tokenized FineWeb-Edu shards, evaluates on EWoK/HellaSwag, and logs data exposure metadata for later analysis. The main training entrypoint is `train_gpt2_finewebedu_bin.py`.

The modified plan in this repo is:
- first, verify that rho-1 style training is implemented correctly and does not collapse performance,
- then, use related methods to discard unnecessary data while preserving GPT-2 Medium EWoK performance,
- then, run attribution analyses to identify which retained data improves EWoK.

## What This Repo Is Trying To Do
At a high level, you are training a student model (GPT-2 Medium architecture) on FineWeb-Edu, while optionally using a fixed reference model (default: OpenAI GPT-2 Medium) to guide which training tokens to keep. The near-term goal is a correctness and performance check for rho-1 behavior, not yet final dataset compression.

Long-term, this becomes a data selection pipeline: move from token-level filtering to sequence/dataset-level pruning so a smaller subset can still recover strong EWoK behavior, and then attribute what in that subset matters most.

## 60-Second TL;DR
- `fineweb.py` builds `train_*.bin`/`val_*.bin` uint16 shards from FineWeb-Edu.
- `train_gpt2_finewebedu_bin.py` can train baseline GPT-2 Medium on those shards.
- `compute_ref_loss_shards.py` precomputes per-token reference losses from `openai-community/gpt2-medium`.
- Enabling rho in training (`--rho_ref_loss_dir ...`) filters which tokens contribute to loss.
- A nonzero `--rho_ref_loss_cap` is the key knob for explicitly excluding tokens the reference model finds too hard.
- Resume is supported via `--resume_from_run`, including log trimming + data-stream fast-forward alignment.

## Glossary
- **Student model**: the model being optimized in this repo (for example GPT-2 Medium config: `n_embd=1024, n_head=16, n_layer=24`).
- **Reference model**: a fixed model used only to provide per-token reference losses (default docs target: `openai-community/gpt2-medium`).
- **Token loss**: negative log-likelihood for a single target token.
- **rho**: token-selection mechanism that keeps only a fraction of candidate tokens for optimization.
- **keep fraction (`rho_keep_frac`)**: fraction of candidate tokens retained each batch.
- **warmup (`rho_warmup_steps`)**: early steps where rho masking is disabled and all tokens are used.
- **cap (`rho_ref_loss_cap`)**: optional ceiling on reference loss; tokens above it are excluded from rho candidates.
- **exposure logs**: per-rank JSONL logs showing which shard/block spans were fed to the model.
- **checkpoint resume**: load model/optimizer/trainer state, trim logs to checkpoint step, then fast-forward dataloader stream.

## How Rho-1 Works In This Script
In one sentence: rho mode computes token scores from student/reference losses, keeps top-scoring tokens among valid candidates, and averages loss over only those kept tokens.

In this repo, the intent for your rho-1 experiment is: remove tokens that are likely unhelpful for this phase, especially when both student/reference indicate high difficulty (via a nonzero reference-loss cap), then check whether quality remains close to baseline.

### Rho Math
1) Per-token student loss:

$$\ell_s(t) = -\log p_{\theta}(x_t \mid x_{<t})$$

What this means: the student pays high loss when token \(x_t\) is hard to predict given prior context.

2) Per-token reference loss (precomputed):

$$\ell_r(t) = -\log p_{\phi}(x_t \mid x_{<t})$$

What this means: this is the same quantity, but measured under the fixed reference model.

3) Scoring modes:

- `delta` mode:
- $$s(t) = \ell_s(t) - \ell_r(t)
$$
- `ref_only` mode:
$$s(t) = -\ell_r(t)$$

What this means: `delta` prioritizes tokens where the student underperforms the reference; `ref_only` prioritizes tokens the reference finds easier.

4) Candidate set with optional cap:

$$
C = \{t : \text{ref\_valid}(t)=1 \land (\ell_r(t) \le c \text{ if } c>0 \text{ else True})\}
$$
where \(c\) is `--rho_ref_loss_cap`.

What this means: cap-enabling is the explicit mechanism for dropping very hard-for-reference tokens before top-k selection.

5) Top-k keep rule:

$$
k = \lceil \rho \cdot |C| \rceil
$$

where $\rho$ is `--rho_keep_frac`, and

$$
m(t)=1 \text{ if } t \in \text{TopK}_{C}(s, k), \text{ else } 0
$$

What this means: only the highest-scored candidate tokens are kept for gradient signal.

6) Optimized loss:

$$
L = \frac{\sum_t m(t)\,\ell_s(t)}{\max(1,\sum_t m(t))}
$$

What this means: the update ignores dropped tokens by masking them out of the mean.

7) Warmup behavior:

If `step < --rho_warmup_steps`, then effectively:

$$
m(t)=1 \quad \forall t
$$

What this means: rho filtering starts only after warmup.

8) Interpretation for your "both hard" objective:

Setting a nonzero `--rho_ref_loss_cap` is what explicitly excludes high-reference-loss tokens from candidate selection; without cap, rho still ranks candidates but does not pre-drop those high-reference-loss tokens.

## Setup
Run commands from this directory:

```bash
cd tokenPred/moonshotGPT
```

Minimal dependencies used by these scripts include:
- `torch`
- `accelerate`
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `matplotlib`
- `pandas`

If needed, initialize Accelerate once:

```bash
accelerate config
```

## End-to-End Commands

### 1) Tokenize FineWeb-Edu
Why you run this now: training expects memmapped `train_*.bin` and `val_*.bin` shards plus `meta.json`.

```bash
python fineweb.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --text_field text \
  --tokenizer gpt2 \
  --out_dir fineweb_edu_10B \
  --shard_tokens 100000000 \
  --val_shards 1
```

### 2) Baseline GPT-2 Medium Training (No Rho)
Why you run this now: establish a reference run before token filtering.

```bash
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir fineweb_edu_10B \
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

### 3) Precompute Reference Loss Shards (OpenAI GPT-2 Medium)
Why you run this now: rho mode needs precomputed per-token reference losses aligned to your train shards.

```bash
accelerate launch --num_processes 8 compute_ref_loss_shards.py \
  --data_dir fineweb_edu_10B \
  --out_dir ref_loss_gpt2m_T1024_B4 \
  --split train \
  --seq_len 1024 \
  --batch_size 4 \
  --ref_model openai-community/gpt2-medium \
  --tokenizer gpt2 \
  --out_dtype float16 \
  --mixed_precision bf16
```

Important:
- Keep `--batch_size` in this step equal to training `--micro_batch_size`.
- Keep `--seq_len` equal between precompute and training.

### 4) Rho-1 Experiment Run (Cap-Enabled)
Why you run this now: this is the modified training mode that keeps only selected tokens for optimization and explicitly excludes high-reference-loss tokens.

```bash
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir fineweb_edu_10B \
  --micro_batch_size 4 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16 \
  --rho_ref_loss_dir ref_loss_gpt2m_T1024_B4 \
  --rho_keep_frac 0.7 \
  --rho_warmup_steps 500 \
  --rho_mode delta \
  --rho_ref_loss_cap 3.0
```

Notes:
- `rho_ref_loss_cap=3.0` is an example threshold; tune empirically.
- Start by checking whether performance remains close to baseline while filtering works as expected.

### 5) Resume Training
Why you run this now: continue an interrupted run from an existing run folder or a specific checkpoint folder.

```bash
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir fineweb_edu_10B \
  --micro_batch_size 4 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16 \
  --rho_ref_loss_dir ref_loss_gpt2m_T1024_B4 \
  --rho_keep_frac 0.7 \
  --rho_warmup_steps 500 \
  --rho_mode delta \
  --rho_ref_loss_cap 3.0 \
  --resume_from_run experiments/<your_run_or_ckpt_dir>
```

Resume behavior in this script:
- loads model + optimizer + trainer state,
- trims logs above checkpoint step (with backups),
- fast-forwards dataloader stream in data-only mode to align replay state.

### 6) Plot Metrics
Why you run this now: summarize EWoK/HellaSwag and training traces from `step_metrics.json`.

```bash
python plot_step_metrics.py \
  --metrics experiments/<your_run>/step_metrics.json
```

## Outputs And Where To Look
Each run creates an `experiments/<run_name>/` directory with:
- `step_metrics.json` (main structured metrics history)
- `scalars.jsonl` (step-level scalar logs)
- `ewok_items.jsonl` (per-item EWoK logs)
- `hellaswag_metrics.jsonl` (HellaSwag summaries)
- `exposures/exposures_rank*.jsonl` (data exposure traces by rank)
- `ckpt_*_stepXXXXXXX/` (model/tokenizer/optimizer/trainer state checkpoints)
- `plots_from_step_metrics/` (generated analysis plots)

## Troubleshooting
### 1) Ref-loss alignment / batch-size mismatch
- Symptom: rho preflight or runtime alignment errors.
- Fix: make sure these match exactly:
  - training `--micro_batch_size` == precompute `--batch_size`
  - training `--seq_len` == precompute `--seq_len`
  - training data shard set == precompute shard set

### 2) Missing EWoK source
- Symptom: EWoK loader errors in `ewok_eval.py`.
- Fix: ensure one of these exists:
  - `ewok_fast_jsonl.zip` in repo root,
  - directory pointed to by `EWOK_SRC`,
  - fallback legacy source path if applicable.

### 3) Resume compatibility failures
- Symptom: resume preflight mismatch errors.
- Fix: keep replay-critical settings consistent when resuming:
  - seed, micro-batch size, seq_len, grad accumulation, world size, worker count, data dir, shuffle setting.

### 4) Mixed precision / hardware caveats
- Symptom: bf16 not supported warnings or launch instability.
- Fix:
  - use `--mixed_precision bf16` when supported,
  - otherwise fallback to `fp16` or `no`.

## Research Roadmap
### Stage 1: Validate rho-1 implementation
Success criteria:
- logs show rho is active with intended keep fraction behavior,
- no alignment/resume inconsistencies,
- performance stays reasonably close to baseline.

### Stage 2: Move to sequence-level data pruning
Success criteria:
- produce smaller retained dataset variants,
- recover strong GPT-2 Medium EWoK behavior on reduced data.

### Stage 3: Attribution of EWoK improvements
Success criteria:
- identify which retained data characteristics or subsets are linked to EWoK gains,
- produce reproducible evidence from exposure + evaluation logs.

## Public Interface Notes
- This README adds documentation only.
- No code API or CLI changes are required to use this workflow.
