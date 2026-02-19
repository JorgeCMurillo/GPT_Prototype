# GPT_Prototype

Lightweight GPT-2 training/evaluation prototype for FineWeb-Edu token shards, with:

- Distributed training via `accelerate`
- Periodic validation and checkpoints
- EWoK evaluation (including per-item logs and category metrics)
- Optional HellaSwag evaluation
- Run-level metric plotting

## Repository Layout

- `moonshotGPT/fineweb.py`
  - Pretokenizes FineWeb/FineWeb-Edu text into GPT-2 `uint16` shard files (`train_*.bin`, `val_*.bin`) plus `meta.json`.
- `moonshotGPT/train_gpt2_finewebedu_bin.py`
  - Main trainer for GPT-2 style causal LM on `.bin` token shards.
- `moonshotGPT/shard_loader.py`
  - Data loader for `train_*.bin` / `val_*.bin` shards.
- `moonshotGPT/ewok_eval.py`
  - EWoK evaluator used during/after training.
- `moonshotGPT/hellaswag_eval.py` (optional)
  - HellaSwag evaluation utilities.
- `moonshotGPT/plot_step_metrics.py` (optional)
  - Generates plots from `step_metrics.json`.
- `moonshotGPT/ewok_fast_jsonl.zip`
  - Password-protected EWoK JSONL data archive.

## How It Works

`fineweb.py` (data prep):

1. Streams documents from FineWeb/FineWeb-Edu.
2. Tokenizes with GPT-2 tokenizer and prepends BOS per document.
3. Writes a flat stream of `uint16` token IDs into shard files.
4. Emits `meta.json` describing the dataset/shard configuration.

`train_gpt2_finewebedu_bin.py`:

1. Builds a GPT-2 style model (`GPT2Config`) and AdamW optimizer.
2. Streams training/validation batches from shard files in `--data_dir`.
3. Trains with gradient accumulation to match a global token budget (`--total_batch_tokens`).
4. Logs scalar metrics (`scalars.jsonl`) and evaluation metrics (`step_metrics.json`).
5. Runs periodic:
   - Validation loss
   - EWoK eval (sum/mean variants + per-category metrics)
   - HellaSwag eval (if enabled and available)
6. Saves periodic/final checkpoints and optional analysis plots.

## Data Requirements

`--data_dir` must contain:

- `meta.json`
- `train_*.bin`
- `val_*.bin`

You can generate this directory with `moonshotGPT/fineweb.py`.

## Prepare FineWeb-Edu Data

Example:

```bash
cd moonshotGPT
python fineweb.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --out_dir /home/jorge/tokenPred/moonshotGPT/fineweb_edu_10B
```

## EWoK Data Loading

`ewok_eval.py` loads EWoK from the first available source:

1. `EWOK_SRC` (if set; can be a `.zip` or directory)
2. `moonshotGPT/ewok_fast_jsonl.zip`
3. `moonshotGPT/ewok_fast/`
4. Legacy absolute path fallback

For encrypted zip files, password is read from:

- `EWOK_ZIP_PASSWORD` (default is `ew2026`)

EWoK zip behavior:

- You do **not** need to manually unzip `ewok_fast_jsonl.zip`.
- `ewok_eval.py` reads `.jsonl` files directly from the zip in memory.
- It does not auto-extract files to disk.

## Install Requirements

You will need at least:

- `torch`
- `accelerate`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib` (optional for plots)

## Run Training

Example (8 processes, bf16):

```bash
cd moonshotGPT
accelerate launch --num_processes=8 --mixed_precision=bf16 train_gpt2_finewebedu_bin.py --data_dir=/home/jorge/tokenPred/moonshotGPT/fineweb_edu_10B
```

Example with explicit depth (GPT-2 small default is `--n_layer 12`):

```bash
cd moonshotGPT
accelerate launch --num_processes=8 --mixed_precision=bf16 train_gpt2_finewebedu_bin.py \
  --data_dir=/home/jorge/tokenPred/moonshotGPT/fineweb_edu_10B \
  --n_layer=12
```

## Useful Flags

- `--n_layer` model depth (default `12`)
- `--eval_every` validation frequency
- `--ewok_every` EWoK eval frequency
- `--hellaswag_every` HellaSwag eval frequency
- `--save_every` checkpoint frequency
- `--exposure_every` exposure metadata logging frequency

## Outputs

Each run writes to `experiments/<run_name>/` with files like:

- `scalars.jsonl`
- `step_metrics.json`
- `ewok_items.jsonl`
- `hellaswag_metrics.jsonl` (if enabled)
- `loss_curve.png`
- `ckpt_*` checkpoint directories
