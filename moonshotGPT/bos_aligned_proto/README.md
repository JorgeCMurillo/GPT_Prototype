# BOS-Aligned Prototype for moonshotGPT

This folder contains an isolated prototype to test nanochat-style BOS-aligned supervision in moonshotGPT without changing GPT-2 model architecture or benchmark scripts.

## What is included

- `prepare_finewebedu_bos_rows.py`
  - Offline preprocessor for FineWeb-Edu.
  - Builds row-packed `uint16` shards where each row has length `seq_len + 1` and starts with BOS.
  - Packing logic matches nanochat-style behavior:
    - pick the largest doc that fits remaining row space;
    - if none fit, pick the shortest doc and crop to exact remaining space.
- `bos_row_loader.py`
  - Runtime memmap loader for the row-packed format.
  - Produces `(x, y)` with shape `(B, T)` from row data.
  - Supports optional metadata for exposure logging.
- `train_gpt2_finewebedu_bos_bin.py`
  - Parallel trainer entrypoint with the same evaluation/logging flow as baseline,
    but using the BOS row loader.
- `compare_ab_runs.md`
  - A/B run protocol and comparison checklist.

## Data format

Expected files in `--data_dir`:

- `train_*.bin`
- `val_*.bin`
- `meta.json`

`meta.json` includes baseline-compatible fields plus BOS-specific fields such as:

- `format: "bos_row_packed_bestfit"`
- `seq_len`
- `row_tokens`
- `packing_algo`
- `batch_docs`
- `buffer_docs`
- `rows_written_total`
- `tokens_cropped_total`
- `crop_fraction`

Important: this dataset is `seq_len`-specific. Loader validates `seq_len` against `meta.json`.

## End-to-end commands

Run from `tokenPred/moonshotGPT`.

### 1) Build BOS row-packed dataset

```bash
python -m bos_aligned_proto.prepare_finewebedu_bos_rows \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --text_field text \
  --tokenizer gpt2 \
  --out_dir fineweb_edu_10B_bosrow \
  --seq_len 1024 \
  --batch_docs 256 \
  --buffer_docs 1000 \
  --val_shards 1
```

### 2) Train BOS prototype

3090-oriented example (`micro_batch_size=10`):

```bash
python -m bos_aligned_proto.train_gpt2_finewebedu_bos_bin \
  --data_dir fineweb_edu_10B_bosrow \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

### 3) Train BOS GPT-2 medium

Start with a smaller `micro_batch_size` and increase only if GPU memory allows:

```bash
python -m bos_aligned_proto.train_gpt2_finewebedu_bos_bin \
  --data_dir fineweb_edu_10B_bosrow \
  --n_embd 1024 \
  --n_head 16 \
  --n_layer 24 \
  --micro_batch_size 2 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

Outputs go to `experiments/babygpt_fineweb_bosrow_*`.

## Notes and tradeoffs

- Runtime is fast (memmap + reshape) and stays close to moonshot loader behavior.
- Preprocessing is heavier than baseline because it performs best-fit packing.
- Some tokens are intentionally cropped; this is tracked in `meta.json`.
