#!/usr/bin/env bash
set -euo pipefail

# v14 follow-up sweeps:
#   v16 = v14, N=10k, effective-batch sweep 8/16/32
#   v17 = v14, N=15k, effective-batch 32 only
#   v18 = v14, N=15k, effective-batch sweep 8/16/32
#
# Effective batch here is PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS. To keep
# continuity with the earlier runs, this script keeps GRAD_ACCUM_STEPS=8 and
# uses PER_DEVICE_BATCH_SIZE=1/2/4 for effective batches 8/16/32.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v16_to_v18_v14_batch_data_sweep.sh
#
# Useful subsets:
#   VERSIONS="v16" bash research/bos_aligned_proto/spatial_synth/run_v16_to_v18_v14_batch_data_sweep.sh
#   VERSIONS="v18" EFFECTIVE_BATCHES="8 16" bash research/bos_aligned_proto/spatial_synth/run_v16_to_v18_v14_batch_data_sweep.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
DATA_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth"
OUT_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training"

VERSIONS="${VERSIONS:-v16 v17 v18}"
LRS="${LRS:-4e-5 8e-5}"
EPOCHS="${EPOCHS:-1}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EFFECTIVE_BATCHES="${EFFECTIVE_BATCHES:-8 16 32}"
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

run_setting() {
  local version="$1"
  local n="$2"
  local effective_batch="$3"
  local per_device_batch
  per_device_batch="$(per_device_for_effective_batch "$effective_batch")"

  local data_tag="implicit_${version}_v14_n${n}_seed${SEED}_${DIFFICULTY}"
  local data_path="$DATA_ROOT/spatial_relations_synth_${data_tag}.csv"
  local out_dir="$OUT_ROOT/ckpt_8k_12k_16k_fast_eval_${version}_v14_n${n}_effb${effective_batch}_full_loss_1epoch"

  echo "=== Running $version: v14, N=$n, effective_batch=$effective_batch (per_device=$per_device_batch, grad_accum=$GRAD_ACCUM_STEPS) ==="
  TEMPLATE_PRESET="v14" \
  LOSS_MODE="${LOSS_MODE:-full}" \
  EWOK_VARIANT="${EWOK_VARIANT:-fast}" \
  EPOCHS="$EPOCHS" \
  EPOCH_EVAL="$EPOCH_EVAL" \
  LRS="$LRS" \
  N="$n" \
  SEED="$SEED" \
  DIFFICULTY="$DIFFICULTY" \
  DATA_TAG="$data_tag" \
  DATA_PATH="$data_path" \
  OUT_DIR="$out_dir" \
  PER_DEVICE_BATCH_SIZE="$per_device_batch" \
  GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
  bash "$BASE_RUNNER"
}

for version in $VERSIONS; do
  case "$version" in
    v16)
      for effective_batch in $EFFECTIVE_BATCHES; do
        run_setting "$version" 10000 "$effective_batch"
      done
      ;;
    v17)
      run_setting "$version" 15000 32
      ;;
    v18)
      for effective_batch in $EFFECTIVE_BATCHES; do
        run_setting "$version" 15000 "$effective_batch"
      done
      ;;
    *)
      echo "Unknown version: $version" >&2
      exit 1
      ;;
  esac
done
