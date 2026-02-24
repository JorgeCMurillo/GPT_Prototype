# A/B Experiment Plan: Baseline vs BOS Row-Packed

Run from `tokenPred/moonshotGPT`.

## Goal

Measure impact of BOS-aligned supervision on training dynamics and benchmark performance while keeping model/eval pipeline unchanged.

## A) Baseline run

```bash
python -m train_gpt2_finewebedu_bin \
  --data_dir fineweb_edu_10B \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

## B) BOS run

### 1) Build BOS dataset

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

### 2) Train BOS model

```bash
python -m bos_aligned_proto.train_gpt2_finewebedu_bos_bin \
  --data_dir fineweb_edu_10B_bosrow \
  --micro_batch_size 10 \
  --seq_len 1024 \
  --total_batch_tokens 491520 \
  --max_train_steps 20000 \
  --num_workers 0
```

## Keep these knobs matched

- `seed`
- `micro_batch_size`
- `seq_len`
- `total_batch_tokens`
- `max_train_steps`
- `eval_every`, `hellaswag_every`, `ewok_every`
- `learning_rate`, `warmup_iters`, `learning_rate_decay_frac`

## Artifacts to compare

- `scalars.jsonl`
- `step_metrics.json`
- `exposures/exposures_rank*.jsonl`
- `ewok_items.jsonl`
- `hellaswag_metrics.jsonl`

BOS-only diagnostics:

- `meta.json`:
  - `tokens_cropped_total`
  - `crop_fraction`
  - `rows_written_total`

## Result table template

| Metric | Baseline | BOS Row-Packed | Delta |
|---|---:|---:|---:|
| Final train loss (opt-step mean) |  |  |  |
| Val loss @ matched step |  |  |  |
| HellaSwag accuracy |  |  |  |
| HellaSwag accuracy_norm |  |  |  |
| EWoK official sum metric |  |  |  |
| EWoK official mean metric |  |  |  |
| Tokens/sec (reported) |  |  |  |
| Grad norm stability |  |  |  |
| Crop fraction (BOS only) | N/A |  |  |

## Interpretation checklist

- Did BOS alignment improve benchmark quality at matched compute?
- Did BOS alignment change optimization stability (loss smoothness, grad norms)?
- Was runtime throughput materially worse/better?
- Is any observed gain plausibly explained by data discard (`crop_fraction`)?
