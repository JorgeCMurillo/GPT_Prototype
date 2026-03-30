# Training Utilities

This folder holds reusable helpers shared by the main top-level trainer. It is
also the right place to keep the rho-1 documentation now that the repo’s main
landing page is focused on attribution analysis rather than on the rho-1
experiment itself.

## Current Layout

```text
training_utils/
  README.md
  __init__.py
  rho1.py
  resume_trim.py
  debug_parity.py
```

## File Guide

- `rho1.py`
  Shared rho-1 configuration, preflight validation, reference-loss alignment
  checks, lazy memmap loading, token masking, and per-step metric aggregation.
- `resume_trim.py`
  Backup-first helpers for trimming logs to a checkpoint boundary before resume,
  plus replay-oriented helpers for reconstructing dataloader state safely.
- `debug_parity.py`
  Utilities for caching a small fixed batch set, replaying it for tiny-overfit
  or parity checks, and writing detailed JSONL traces for comparison.

## Original Rho-1 Goal

The original rho-1 question in this repo was:

Can we retain only a subset of tokens, chosen using student and reference
losses, without destroying downstream benchmark behavior?

The intended use case was not just raw compression. It was to test whether
loss-guided token retention could remove obviously unhelpful supervision while
preserving EWoK-relevant learning signal.

At a high level, the implementation compares per-token student and reference
losses, builds a candidate set of tokens that are allowed to contribute to the
update, keeps only the highest-scoring fraction, and averages the training loss
over those kept tokens.

## Rho-1 Math

Per-token student loss:

$$
\ell_s(t) = -\log p_{\theta}(x_t \mid x_{<t})
$$

Per-token reference loss:

$$
\ell_r(t) = -\log p_{\phi}(x_t \mid x_{<t})
$$

The main scoring mode used in the early experiments was `delta`:

$$
s(t) = \ell_s(t) - \ell_r(t)
$$

With an optional reference-loss cap `c`, the candidate set is:

$$
C = \{ t : \mathrm{ref\_valid}(t)=1 \land (\ell_r(t) \le c \text{ if } c>0 \text{ else True}) \}
$$

If `rho_keep_frac = \rho`, the trainer keeps:

$$
k = \lceil \rho \cdot |C| \rceil
$$

highest-scoring candidate tokens and optimizes:

$$
L = \frac{\sum_t m(t)\,\ell_s(t)}{\max(1,\sum_t m(t))}
$$

where `m(t)` is `1` for kept tokens and `0` otherwise.

Warmup is handled by disabling masking for the first
`rho_warmup_steps` optimization steps.

## Current Status and Findings

Rho-1 is implemented, tested, and still available, but it is no longer the main
story of the repo.

What we learned from the completed rho-1 pass:

- the implementation was worth building because it gave us a clean retained-data
  control knob and validated the reference-loss alignment machinery;
- in practice, rho-1 weakened some model performance on variable swapping;
- experiments that pushed harder toward "hard tokens only" weakened performance
  a lot;
- that suggests easier tokens are important for learning variable swapping, and
  likely contribute useful supervision for other domains too.

That is why the repo’s main focus has shifted. The next question is not merely
"can we filter tokens?" It is:

Which retained data actually helps EWoK, and how can we attribute that effect?

## Minimal Rho-1 Usage

Precompute reference losses:

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

Run training with rho-1 enabled:

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
  --rho_ref_loss_dir data/ref_loss/fineweb_edu_100B/gpt2m_T1024_B4 \
  --rho_keep_frac 0.7 \
  --rho_warmup_steps 500 \
  --rho_mode delta \
  --rho_ref_loss_cap 3.0
```

## Other Utilities Here

### Resume trimming

`resume_trim.py` exists because checkpoint resume is not just "load a model and
continue." The log files and exposure traces also need to agree with the resume
point.

The trimming helpers therefore:

- back up the original file once;
- trim JSON and JSONL records beyond a checkpoint step;
- optionally trim exposure logs too;
- rewrite files atomically so interrupted resume prep does not leave partial
  logs behind.

### Parity debugging

`debug_parity.py` supports short debugging runs where you want to:

- cache a fixed set of batches;
- replay them forever;
- compare microstep traces between two training implementations;
- run tiny-overfit checks with detailed JSONL outputs.

That support is used by `run_training_parity_debug.py` and by the corresponding
tests under `tests/`.
