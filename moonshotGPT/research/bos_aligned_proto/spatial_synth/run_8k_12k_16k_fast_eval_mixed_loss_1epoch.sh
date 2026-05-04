#!/usr/bin/env bash
set -euo pipefail

# Run a shorter 8k/12k/16k BabyGPT checkpoint sweep with EWoK fast evaluation
# and mixed sequence-level 70/30 loss. This is meant as a quick test after completion-only
# loss showed very sharp synthetic-loss drops.
#
# Defaults:
#   - 1 epoch
#   - learning rates 4e-5 and 8e-5
#   - mixed loss with 70% full-loss examples / 30% completion-only examples
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_mixed_loss_1epoch.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"

export LOSS_MODE="${LOSS_MODE:-mixed}"
export MIXED_FULL_LOSS_RATIO="${MIXED_FULL_LOSS_RATIO:-0.7}"
export EWOK_VARIANT="${EWOK_VARIANT:-fast}"
export EPOCHS="${EPOCHS:-1}"
export LRS="${LRS:-4e-5 8e-5}"
export DATA_TAG="${DATA_TAG:-implicit_v4_pass_through_mixed_loss_1epoch}"
export OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_mixed_loss_1epoch}"

bash "$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
