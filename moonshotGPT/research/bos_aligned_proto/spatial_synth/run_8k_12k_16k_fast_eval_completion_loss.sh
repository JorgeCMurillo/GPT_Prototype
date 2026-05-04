#!/usr/bin/env bash
set -euo pipefail

# Run the 8k/12k/16k BabyGPT checkpoint sweep with EWoK fast evaluation and
# completion-only loss masking. This keeps the full scenario in the input, but
# trains only on the generated final completion sentence.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_completion_loss.sh
#
# Override defaults, for example:
#   N=30000 EPOCHS=2 LRS="8e-5 2e-4" bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval_completion_loss.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"

export LOSS_MODE="${LOSS_MODE:-completion}"
export EWOK_VARIANT="${EWOK_VARIANT:-fast}"
export DATA_TAG="${DATA_TAG:-implicit_v4_pass_through_completion_loss}"
export OUT_DIR="${OUT_DIR:-$ROOT/runs/research/bos_aligned_proto/spatial_synth_training/ckpt_8k_12k_16k_fast_eval_completion_loss}"

bash "$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
