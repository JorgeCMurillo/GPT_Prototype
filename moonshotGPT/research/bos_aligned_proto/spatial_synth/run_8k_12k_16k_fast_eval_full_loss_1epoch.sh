#!/usr/bin/env bash
set -euo pipefail

# Run a short 8k/12k/16k BabyGPT checkpoint sweep with EWoK fast evaluation
# and standard full causal-LM loss on every synthetic token.
#
# This isolates whether the recent spatial drop is caused by completion/mixed
# masking, or by the newer, more complex synthetic examples themselves.
#
# Defaults:
#   - 1 epoch
#   - learning rates 4e-5 and 8e-5
#   - full loss on all tokens
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_full_loss_1epoch.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"

export LOSS_MODE="${LOSS_MODE:-full}"
export EWOK_VARIANT="${EWOK_VARIANT:-fast}"
export EPOCHS="${EPOCHS:-1}"
export LRS="${LRS:-4e-5 8e-5}"
export DATA_TAG="${DATA_TAG:-implicit_v4_pass_through_full_loss_1epoch}"
export OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_full_loss_1epoch}"

bash "$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
