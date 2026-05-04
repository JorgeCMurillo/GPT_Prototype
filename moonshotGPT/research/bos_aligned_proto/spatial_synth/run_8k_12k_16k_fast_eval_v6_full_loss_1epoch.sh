#!/usr/bin/env bash
set -euo pipefail

# Run the v6 synthetic spatial data with full causal-LM loss and EWoK fast eval.
# v6 adds front/back-to-left/right turn contrasts to the v5 generator.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_v6_full_loss_1epoch.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"

export TEMPLATE_PRESET="${TEMPLATE_PRESET:-v6}"
export LOSS_MODE="${LOSS_MODE:-full}"
export EWOK_VARIANT="${EWOK_VARIANT:-fast}"
export EPOCHS="${EPOCHS:-1}"
export LRS="${LRS:-4e-5 8e-5}"
export DATA_TAG="${DATA_TAG:-implicit_v6_turn_lr_contrast_full_loss_1epoch}"
export OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_v6_full_loss_1epoch}"

bash "$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
