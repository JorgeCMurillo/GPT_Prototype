# GPT_Prototype

A focused research repo for training and evaluating GPT-2 style models on FineWeb-Edu token shards.

This repo is centered on `moonshotGPT`, including:
- baseline GPT-2 training,
- optional rho-based token filtering,
- EWoK/HellaSwag evaluation,
- checkpointing, resume support, and run-level analysis artifacts.

## Start Here
If you only need to run experiments, use the detailed runner-first guide:
- [`moonshotGPT/README.md`](moonshotGPT/README.md)

That README includes end-to-end commands for:
- tokenization,
- baseline GPT-2 Medium training,
- reference-loss precompute,
- rho runs,
- resume,
- plotting.

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Prepare data
```bash
cd moonshotGPT
python fineweb.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --out_dir fineweb_edu_10B
```

### 3) Train baseline GPT-2 Medium
```bash
cd moonshotGPT
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir fineweb_edu_10B \
  --micro_batch_size 4 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16
```

### 4) Optional: precompute reference-loss shards (required for rho)
```bash
cd moonshotGPT
accelerate launch --num_processes 8 compute_ref_loss_shards.py \
  --data_dir fineweb_edu_10B \
  --out_dir ref_loss_gpt2m_T1024_B4 \
  --split train \
  # must match training --seq_len
  --seq_len 1024 \
  # must match training --micro_batch_size
  --batch_size 4 \
  --ref_model openai-community/gpt2-medium \
  --tokenizer gpt2 \
  --out_dtype float16 \
  --mixed_precision bf16
```

Alignment note:
- Keep precompute `--batch_size` equal to training `--micro_batch_size`.
- Keep precompute `--seq_len` equal to training `--seq_len`.

### 5) Optional: run rho-enabled training
```bash
cd moonshotGPT
accelerate launch --num_processes 8 train_gpt2_finewebedu_bin.py \
  --data_dir fineweb_edu_10B \
  # must match precompute --batch_size
  --micro_batch_size 4 \
  # must match precompute --seq_len
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --mixed_precision bf16 \
  # must be generated from same shard set
  --rho_ref_loss_dir ref_loss_gpt2m_T1024_B4 \
  --rho_keep_frac 0.7 \
  --rho_warmup_steps 500 \
  --rho_mode delta
```

Cap note:
- `--rho_ref_loss_cap` is optional and off by default.
- Add `--rho_ref_loss_cap <positive_value>` only when you explicitly want cap-based pre-filtering.

## Repository Map
- `moonshotGPT/fineweb.py`: streams + tokenizes FineWeb/FineWeb-Edu into `train_*.bin` / `val_*.bin` and `meta.json`.
- `moonshotGPT/train_gpt2_finewebedu_bin.py`: main trainer (baseline + optional rho path, eval, checkpoints, resume).
- `moonshotGPT/compute_ref_loss_shards.py`: precomputes per-token reference loss shards for rho training.
- `moonshotGPT/shard_loader.py`: memmapped shard loader used by training.
- `moonshotGPT/ewok_eval.py`: EWoK evaluation logic used during/after training.
- `moonshotGPT/hellaswag_eval.py`: optional HellaSwag evaluation utilities.
- `moonshotGPT/plot_step_metrics.py`: generates analysis plots from `step_metrics.json`.
- `moonshotGPT/training_utils/resume_trim.py`: resume preflight trimming and dataloader replay helpers.

## Typical Outputs
Training runs write under `moonshotGPT/experiments/<run_name>/`:
- `step_metrics.json`
- `scalars.jsonl`
- `ewok_items.jsonl`
- `hellaswag_metrics.jsonl` (if enabled)
- `exposures/exposures_rank*.jsonl`
- `ckpt_*_stepXXXXXXX/`
- `plots_from_step_metrics/`

## Data + Large Files
This repo is intended to version code and docs, not generated artifacts.
Keep large runtime outputs out of git, especially:
- `moonshotGPT/experiments/`
- token shard datasets (for example `fineweb_edu_10B/`)
- reference-loss binaries (for example `ref_loss_gpt2m_*/`)

## Notes
- Default reference model for rho workflows is `openai-community/gpt2-medium`.
- For stable rho alignment, keep training `--micro_batch_size` equal to precompute `--batch_size`, and keep `--seq_len` matched.
