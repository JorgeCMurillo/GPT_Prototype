#!/usr/bin/env bash
set -euo pipefail

# Run synthetic spatial-relations causal-LM fine-tuning from the 8k/12k/16k
# BabyGPT checkpoints, with EWoK BabyLM completion full-mean evaluation.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k.sh
#
# Override defaults, for example:
#   N=30000 EPOCHS=2 LRS="8e-5 2e-4" LOSS_MODE=completion bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
cd "$ROOT"

CKPT_BASE="$ROOT/experiments/babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000"
CKPT_8K="$CKPT_BASE/ckpt_periodic_step0008000"
CKPT_12K="$CKPT_BASE/ckpt_periodic_step0012000"
CKPT_16K="$CKPT_BASE/ckpt_periodic_step0016000"

SEED="${SEED:-42}"
N="${N:-10000}"
DIFFICULTY="${DIFFICULTY:-mixed}"
DATA_TAG="${DATA_TAG:-implicit_v3}"
if [[ -z "${TEMPLATE_PRESET:-}" ]]; then
  if [[ "$DATA_TAG" == *v21* ]]; then
    TEMPLATE_PRESET="v21"
  elif [[ "$DATA_TAG" == *v20* ]]; then
    TEMPLATE_PRESET="v20"
  elif [[ "$DATA_TAG" == *v19* ]]; then
    TEMPLATE_PRESET="v19"
  elif [[ "$DATA_TAG" == *v15* ]]; then
    TEMPLATE_PRESET="v15"
  elif [[ "$DATA_TAG" == *v14* ]]; then
    TEMPLATE_PRESET="v14"
  elif [[ "$DATA_TAG" == *v13* ]]; then
    TEMPLATE_PRESET="v13"
  elif [[ "$DATA_TAG" == *v3* ]]; then
    TEMPLATE_PRESET="v3"
  elif [[ "$DATA_TAG" == *v12* ]]; then
    TEMPLATE_PRESET="v12"
  elif [[ "$DATA_TAG" == *v11* ]]; then
    TEMPLATE_PRESET="v11"
  elif [[ "$DATA_TAG" == *v10* ]]; then
    TEMPLATE_PRESET="v10"
  elif [[ "$DATA_TAG" == *v9* ]]; then
    TEMPLATE_PRESET="v9"
  elif [[ "$DATA_TAG" == *v8* ]]; then
    TEMPLATE_PRESET="v8"
  elif [[ "$DATA_TAG" == *v7* ]]; then
    TEMPLATE_PRESET="v7"
  elif [[ "$DATA_TAG" == *v6* ]]; then
    TEMPLATE_PRESET="v6"
  elif [[ "$DATA_TAG" == *v5* ]]; then
    TEMPLATE_PRESET="v5"
  else
    TEMPLATE_PRESET="v4"
  fi
fi
DATA_PATH="${DATA_PATH:-$ROOT/runs/research/bos_aligned_proto/spatial_synth/spatial_relations_synth_${DATA_TAG}_n${N}_seed${SEED}_${DIFFICULTY}.csv}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k}"

# These checkpoints include the GPT-2 tokenizer files. Using CKPT_8K avoids a
# network/cache dependency while still using the same GPT-2 tokenizer family.
TOKENIZER_NAME="${TOKENIZER_NAME:-$CKPT_8K}"

LRS="${LRS:-8e-5 2e-4}"
EPOCHS="${EPOCHS:-3}"
EPOCH_EVAL="${EPOCH_EVAL:-0.5}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
LOSS_MODE="${LOSS_MODE:-full}"
COMPLETION_LOSS_RATIO="${COMPLETION_LOSS_RATIO:-0.7}"
MIXED_FULL_LOSS_RATIO="${MIXED_FULL_LOSS_RATIO:-0.7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EWOK_BATCH_SIZE="${EWOK_BATCH_SIZE:-8}"
EWOK_VARIANT="${EWOK_VARIANT:-full}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

for ckpt in "$CKPT_8K" "$CKPT_12K" "$CKPT_16K"; do
  if [[ ! -d "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

if [[ ! -f "$DATA_PATH" ]]; then
  python research/bos_aligned_proto/spatial_synth/generate_spatial_relations_csv.py \
    --n "$N" \
    --seed "$SEED" \
    --difficulty "$DIFFICULTY" \
    --template-preset "$TEMPLATE_PRESET" \
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
  --difficulty "$DIFFICULTY" \
  "${max_examples_args[@]}" \
  --epochs "$EPOCHS" \
  --epoch-eval "$EPOCH_EVAL" \
  --block-size "$BLOCK_SIZE" \
  --loss-mode "$LOSS_MODE" \
  --mixed-full-loss-ratio "$MIXED_FULL_LOSS_RATIO" \
  --completion-loss-ratio "$COMPLETION_LOSS_RATIO" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --ewok-batch-size "$EWOK_BATCH_SIZE" \
  --ewok-variant "$EWOK_VARIANT" \
  $EXTRA_TRAIN_ARGS
