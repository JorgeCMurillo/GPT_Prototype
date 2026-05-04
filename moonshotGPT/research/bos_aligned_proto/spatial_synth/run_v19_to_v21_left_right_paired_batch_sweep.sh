#!/usr/bin/env bash
set -euo pipefail

# v14 + matched left/right contrast-pair sweeps:
#   v19 = v14 + paired left/right contrast at normal weight
#   v20 = v14 + paired left/right contrast at medium weight
#   v21 = v14 + paired left/right contrast at high weight
#
# Each version is run across effective batch sizes 8/16/32 by default.
# Effective batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS. To preserve the
# older batch-8 condition, this keeps GRAD_ACCUM_STEPS=8 and maps effective
# batches 8/16/32 to per-device batches 1/2/4.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v19_to_v21_left_right_paired_batch_sweep.sh
#
# Useful subsets:
#   VERSIONS="v19" bash research/bos_aligned_proto/spatial_synth/run_v19_to_v21_left_right_paired_batch_sweep.sh
#   VERSIONS="v19 v20" EFFECTIVE_BATCHES="8 32" bash research/bos_aligned_proto/spatial_synth/run_v19_to_v21_left_right_paired_batch_sweep.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
DATA_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth"
OUT_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training"

VERSIONS="${VERSIONS:-v19 v20 v21}"
EFFECTIVE_BATCHES="${EFFECTIVE_BATCHES:-8 16 32}"
LRS="${LRS:-4e-5 8e-5}"
N="${N:-10000}"
EPOCHS="${EPOCHS:-1}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
SEED="${SEED:-42}"
DIFFICULTY="${DIFFICULTY:-mixed}"

per_device_for_effective_batch() {
  local effective_batch="$1"
  if (( effective_batch % GRAD_ACCUM_STEPS != 0 )); then
    echo "Effective batch $effective_batch is not divisible by GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS" >&2
    exit 1
  fi
  local per_device=$(( effective_batch / GRAD_ACCUM_STEPS ))
  if (( per_device < 1 )); then
    echo "Effective batch $effective_batch gives invalid per-device batch $per_device" >&2
    exit 1
  fi
  echo "$per_device"
}

for version in $VERSIONS; do
  case "$version" in
    v19|v20|v21) ;;
    *)
      echo "Unknown version: $version" >&2
      exit 1
      ;;
  esac

  for effective_batch in $EFFECTIVE_BATCHES; do
    per_device_batch="$(per_device_for_effective_batch "$effective_batch")"
    data_tag="implicit_${version}_v14_lr_paired_n${N}_seed${SEED}_${DIFFICULTY}"
    data_path="$DATA_ROOT/spatial_relations_synth_${data_tag}.csv"
    out_dir="$OUT_ROOT/ckpt_8k_12k_16k_fast_eval_${version}_v14_lr_paired_n${N}_effb${effective_batch}_full_loss_1epoch"

    echo "=== Running $version: N=$N, effective_batch=$effective_batch (per_device=$per_device_batch, grad_accum=$GRAD_ACCUM_STEPS) ==="
    TEMPLATE_PRESET="$version" \
    LOSS_MODE="${LOSS_MODE:-full}" \
    EWOK_VARIANT="${EWOK_VARIANT:-fast}" \
    EPOCHS="$EPOCHS" \
    EPOCH_EVAL="$EPOCH_EVAL" \
    LRS="$LRS" \
    N="$N" \
    SEED="$SEED" \
    DIFFICULTY="$DIFFICULTY" \
    DATA_TAG="$data_tag" \
    DATA_PATH="$data_path" \
    OUT_DIR="$out_dir" \
    PER_DEVICE_BATCH_SIZE="$per_device_batch" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    bash "$BASE_RUNNER"
  done
done
