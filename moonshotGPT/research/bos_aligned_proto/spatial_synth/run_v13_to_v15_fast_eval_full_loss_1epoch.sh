#!/usr/bin/env bash
set -euo pipefail

# Run the next v8-based hypothesis variants:
#   v13 = v8 + turn-left/right order variants
#   v14 = v8 + direct turn-around left/right side flips
#   v15 = v8 + direct turn-around left/right side flips + turn-left/right order variants
#
# This keeps the same LR defaults as the recent one-epoch hypothesis runners,
# but evaluates every quarter epoch so we can see whether shorter training is
# already enough before the full epoch finishes.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v13_to_v15_fast_eval_full_loss_1epoch.sh
#
# Run a subset:
#   VERSIONS="v13 v15" bash research/bos_aligned_proto/spatial_synth/run_v13_to_v15_fast_eval_full_loss_1epoch.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"

VERSIONS="${VERSIONS:-v13 v14 v15}"
LRS="${LRS:-4e-5 8e-5}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"

for version in $VERSIONS; do
  case "$version" in
    v13) tag="implicit_v13_v8base_turn_lr_order_full_loss_1epoch" ;;
    v14) tag="implicit_v14_v8base_turn_around_lr_full_loss_1epoch" ;;
    v15) tag="implicit_v15_v8base_turn_around_lr_order_full_loss_1epoch" ;;
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
  EPOCH_EVAL="$EPOCH_EVAL" \
  LRS="$LRS" \
  DATA_TAG="$tag" \
  OUT_DIR="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_${version}_v8base_full_loss_1epoch" \
  bash "$BASE_RUNNER"
done
