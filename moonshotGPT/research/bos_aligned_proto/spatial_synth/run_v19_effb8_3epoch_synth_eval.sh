#!/usr/bin/env bash
set -euo pipefail

# Run the current keeper recipe:
#   v19 = v14 + matched left/right paired contrasts
#   effective batch = 8
#   full causal-LM loss
#   3 epochs
#   EWoK fast + synthetic three-tier spatial eval every quarter epoch
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v19_effb8_3epoch_synth_eval.sh
#
# Useful overrides:
#   LRS="8e-5" SYNTHETIC_SPATIAL_EVAL_N_PER_TIER=100 bash research/bos_aligned_proto/spatial_synth/run_v19_effb8_3epoch_synth_eval.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
DATA_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth"
OUT_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training"

SEED="${SEED:-42}"
N="${N:-10000}"
DIFFICULTY="${DIFFICULTY:-mixed}"
DATA_TAG="${DATA_TAG:-implicit_v19_v14_lr_paired_n${N}_seed${SEED}_${DIFFICULTY}}"
DATA_PATH="${DATA_PATH:-$DATA_ROOT/spatial_relations_synth_${DATA_TAG}.csv}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/ckpt_8k_12k_16k_fast_eval_v19_v14_lr_paired_n${N}_effb8_full_loss_3epoch_synth_eval}"

TEMPLATE_PRESET="${TEMPLATE_PRESET:-v19}"
LOSS_MODE="${LOSS_MODE:-full}"
EWOK_VARIANT="${EWOK_VARIANT:-fast}"
EPOCHS="${EPOCHS:-3}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"
LRS="${LRS:-4e-5 8e-5}"

# Effective batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS = 1 * 8.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

SYNTHETIC_SPATIAL_EVAL="${SYNTHETIC_SPATIAL_EVAL:-three_tier}"
SYNTHETIC_SPATIAL_EVAL_N_PER_TIER="${SYNTHETIC_SPATIAL_EVAL_N_PER_TIER:-300}"
SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET="${SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET:-v19}"

TEMPLATE_PRESET="$TEMPLATE_PRESET" \
LOSS_MODE="$LOSS_MODE" \
EWOK_VARIANT="$EWOK_VARIANT" \
EPOCHS="$EPOCHS" \
EPOCH_EVAL="$EPOCH_EVAL" \
LRS="$LRS" \
N="$N" \
SEED="$SEED" \
DIFFICULTY="$DIFFICULTY" \
DATA_TAG="$DATA_TAG" \
DATA_PATH="$DATA_PATH" \
OUT_DIR="$OUT_DIR" \
PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
SYNTHETIC_SPATIAL_EVAL="$SYNTHETIC_SPATIAL_EVAL" \
SYNTHETIC_SPATIAL_EVAL_N_PER_TIER="$SYNTHETIC_SPATIAL_EVAL_N_PER_TIER" \
SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET="$SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET" \
bash "$BASE_RUNNER"
