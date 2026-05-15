#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jorge/tokenPred/moonshotGPT"
DATA_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth"
TOKENIZER_NAME="$ROOT/experiments/babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000/ckpt_periodic_step0008000"

SYNTH_CSV="${SYNTH_CSV:-$DATA_ROOT/cardinal_api_gpt52_pass_through_v2_10pct_tokens39219_seed42.csv}"
MIX_CSV="${MIX_CSV:-$DATA_ROOT/spatial_relations_natural_mix_gpt52_pass_through_v2_nat902_synth098_tok400000_seed42.csv}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_gpt52_pass_through_v2_nat902_synth098_tok400000_full_loss_3epoch_synth_eval}"

cd "$ROOT"

python research/bos_aligned_proto/spatial_synth/generate_natural_synth_mix_csv.py \
  --synthetic-csv "$SYNTH_CSV" \
  --natural-data-dir "$ROOT/data/processed/fineweb_edu_10B" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --target-tokens 400000 \
  --synthetic-token-ratio 0.098 \
  --seed 42 \
  --out "$MIX_CSV"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
TEMPLATE_PRESET=cardinal_v1 \
LOSS_MODE=full \
EWOK_VARIANT=fast \
SYNTHETIC_SPATIAL_EVAL=three_tier \
SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET=cardinal_v1 \
EPOCHS="${EPOCHS:-3}" \
EPOCH_EVAL="${EPOCH_EVAL:-0.25}" \
LRS="${LRS:-1e-5 4e-5 8e-5}" \
DIFFICULTY=all \
DATA_TAG="natural_mix_gpt52_pass_through_v2_nat902_synth098_tok400000_seed42" \
DATA_PATH="$MIX_CSV" \
OUT_DIR="$OUT_DIR" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}" \
EWOK_BATCH_SIZE="${EWOK_BATCH_SIZE:-2}" \
SYNTHETIC_SPATIAL_EVAL_BATCH_SIZE="${SYNTHETIC_SPATIAL_EVAL_BATCH_SIZE:-2}" \
bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh
