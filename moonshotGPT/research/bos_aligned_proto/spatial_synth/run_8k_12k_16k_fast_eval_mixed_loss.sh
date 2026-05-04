#!/usr/bin/env bash
set -euo pipefail

# Run the 8k/12k/16k BabyGPT checkpoint sweep with EWoK fast evaluation and
# mixed sequence-level loss: 70% of examples use full loss, and 30% use
# completion-only loss.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_mixed_loss.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"

export LOSS_MODE="${LOSS_MODE:-mixed}"
export MIXED_FULL_LOSS_RATIO="${MIXED_FULL_LOSS_RATIO:-0.7}"
export EWOK_VARIANT="${EWOK_VARIANT:-fast}"
export DATA_TAG="${DATA_TAG:-implicit_v4_pass_through_mixed_loss}"
export OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_mixed_loss}"

bash "$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
