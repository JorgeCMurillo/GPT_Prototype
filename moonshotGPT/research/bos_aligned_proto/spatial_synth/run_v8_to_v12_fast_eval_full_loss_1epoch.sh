#!/usr/bin/env bash
set -euo pipefail

# Run hypothesis variants against the v6 baseline setup:
#   v8  = v6 + reciprocal close/far
#   v9  = v6 + low-weight distance contrast
#   v10 = v6 + direct turn-around left/right side flips
#   v11 = v6 + cardinal direction guardrail examples
#   v12 = v6 + symmetric-vs-inverse relation-type contrasts
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v8_to_v12_fast_eval_full_loss_1epoch.sh
#
# Run a subset:
#   VERSIONS="v8 v10" bash research/bos_aligned_proto/spatial_synth/run_v8_to_v12_fast_eval_full_loss_1epoch.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"

VERSIONS="${VERSIONS:-v8 v9 v10 v11 v12}"
LRS="${LRS:-4e-5 8e-5}"

for version in $VERSIONS; do
  case "$version" in
    v8) tag="implicit_v8_v6base_distance_reciprocal_full_loss_1epoch" ;;
    v9) tag="implicit_v9_v6base_distance_lite_full_loss_1epoch" ;;
    v10) tag="implicit_v10_v6base_turn_around_lr_full_loss_1epoch" ;;
    v11) tag="implicit_v11_v6base_cardinal_guard_full_loss_1epoch" ;;
    v12) tag="implicit_v12_v6base_relation_type_contrast_full_loss_1epoch" ;;
    *)
      echo "Unknown version: $version" >&2
      exit 1
      ;;
  esac

  echo "=== Running $version ($tag) ==="
  TEMPLATE_PRESET="$version" \
  LOSS_MODE="${LOSS_MODE:-full}" \
  EWOK_VARIANT="${EWOK_VARIANT:-fast}" \
  EPOCHS="${EPOCHS:-1}" \
  LRS="$LRS" \
  DATA_TAG="$tag" \
  OUT_DIR="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_${version}_v6base_full_loss_1epoch" \
  bash "$BASE_RUNNER"
done
