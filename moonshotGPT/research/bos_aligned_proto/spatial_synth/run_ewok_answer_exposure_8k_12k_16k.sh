#!/usr/bin/env bash
set -euo pipefail

# Intentional EWoK answer-exposure / memorization probe.
# The training CSV contains only the correct BabyLM pairs:
#   C_1,T_1 = Context1 + Target1
#   C_2,T_2 = Context2 + Target2
#
# This is not a fair held-out evaluation setup. It asks: if a GPT-2-medium-sized
# checkpoint is directly fed the correct EWoK answers, how fast do accuracy and
# margins move, especially for spatial-relations?
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_ewok_answer_exposure_8k_12k_16k.sh
#
# Common overrides:
#   EWOK_VARIANT=full EPOCHS=3 LRS="8e-5 2e-4" bash research/bos_aligned_proto/spatial_synth/run_ewok_answer_exposure_8k_12k_16k.sh
#   DOMAINS="spatial-relations" bash research/bos_aligned_proto/spatial_synth/run_ewok_answer_exposure_8k_12k_16k.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
cd "$ROOT"

CKPT_BASE="$ROOT/experiments/babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000"
CKPT_8K="$CKPT_BASE/ckpt_periodic_step0008000"
CKPT_12K="$CKPT_BASE/ckpt_periodic_step0012000"
CKPT_16K="$CKPT_BASE/ckpt_periodic_step0016000"

SEED="${SEED:-42}"
EWOK_VARIANT="${EWOK_VARIANT:-fast}"
SIDES="${SIDES:-both}"
DOMAINS="${DOMAINS:-}"
DOMAIN_TAG="all_domains"
if [[ -n "$DOMAINS" ]]; then
  DOMAIN_TAG="$(echo "$DOMAINS" | tr ' ,' '__' | tr -cd 'A-Za-z0-9_.-')"
fi
DATA_TAG="${DATA_TAG:-ewok_answer_exposure_${EWOK_VARIANT}_${SIDES}_${DOMAIN_TAG}}"
DATA_PATH="${DATA_PATH:-$ROOT/runs/research/bos_aligned_proto/spatial_synth/${DATA_TAG}.csv}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_${DATA_TAG}}"

# These checkpoints include tokenizer files, avoiding network/cache dependence.
TOKENIZER_NAME="${TOKENIZER_NAME:-$CKPT_8K}"

LRS="${LRS:-4e-5 8e-5}"
EPOCHS="${EPOCHS:-3}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
LOSS_MODE="${LOSS_MODE:-completion}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EWOK_BATCH_SIZE="${EWOK_BATCH_SIZE:-8}"
VAL_FRAC="${VAL_FRAC:-0.0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

for ckpt in "$CKPT_8K" "$CKPT_12K" "$CKPT_16K"; do
  if [[ ! -d "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

domain_args=()
if [[ -n "$DOMAINS" ]]; then
  read -r -a domain_args <<< "$DOMAINS"
  domain_args=(--domains "${domain_args[@]}")
fi

if [[ ! -f "$DATA_PATH" ]]; then
  python research/bos_aligned_proto/spatial_synth/generate_ewok_answer_exposure_csv.py \
    --ewok-variant "$EWOK_VARIANT" \
    --sides "$SIDES" \
    "${domain_args[@]}" \
    --out "$DATA_PATH"
else
  echo "Using existing data: $DATA_PATH"
fi

max_examples_args=()
if [[ -n "$MAX_TRAIN_EXAMPLES" ]]; then
  max_examples_args=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi

# shellcheck disable=SC2086
python research/bos_aligned_proto/spatial_synth/train_spatial_relations_causal_lm.py \
  --data "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --checkpoints "$CKPT_8K" "$CKPT_12K" "$CKPT_16K" \
  --learning-rates $LRS \
  --difficulty all \
  --no-balance-mixed \
  "${max_examples_args[@]}" \
  --val-frac "$VAL_FRAC" \
  --epochs "$EPOCHS" \
  --epoch-eval "$EPOCH_EVAL" \
  --block-size "$BLOCK_SIZE" \
  --loss-mode "$LOSS_MODE" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --ewok-batch-size "$EWOK_BATCH_SIZE" \
  --ewok-variant "$EWOK_VARIANT" \
  $EXTRA_TRAIN_ARGS
