#!/usr/bin/env bash
set -euo pipefail

# Fast first-answer Llama+Muon Liger A/B.
#
# Example:
#   tmux new -s liger-first-answer
#   GPU=4 scripts/run_liger_first_answer_speed.sh
#
# Extra args are forwarded to run_architecture_speed_benchmark.py, so you can
# override defaults. For estimates to track overrides, prefer env vars:
#   GPU=4 MAX_TRAIN_STEPS=60 scripts/run_liger_first_answer_speed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="${GPU:-4}"
CONDA_ENV="${CONDA_ENV:-babylm}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-runs/research/bos_aligned_proto/liger_first_answer_speed_${STAMP}}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/processed/fineweb_edu_100B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-gpt2}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-81920}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40}"
SEQ_LEN="${SEQ_LEN:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
COMPILE_WINDOW_STEPS="${COMPILE_WINDOW_STEPS:-20}"

# Rough wall-clock defaults from prior gas-20 Llama+Muon smoke runs. The extra
# startup range covers Liger/Triton first-use compilation and normal launch jitter.
EST_STEP_SECONDS_LOW="${EST_STEP_SECONDS_LOW:-5}"
EST_STEP_SECONDS_HIGH="${EST_STEP_SECONDS_HIGH:-8}"
EST_STARTUP_SECONDS_LOW="${EST_STARTUP_SECONDS_LOW:-120}"
EST_STARTUP_SECONDS_HIGH="${EST_STARTUP_SECONDS_HIGH:-300}"

format_duration() {
  local seconds="$1"
  local minutes=$(((seconds + 30) / 60))
  if (( minutes < 60 )); then
    printf "%dm" "${minutes}"
  else
    printf "%dh%02dm" "$((minutes / 60))" "$((minutes % 60))"
  fi
}

TOKENS_PER_MICROSTEP=$((NUM_PROCESSES * MICRO_BATCH_SIZE * SEQ_LEN))
GRAD_ACCUM_STEPS=$(((TOTAL_BATCH_TOKENS + TOKENS_PER_MICROSTEP - 1) / TOKENS_PER_MICROSTEP))
VARIANT_COUNT=2
EST_LOW_SECONDS=$((VARIANT_COUNT * MAX_TRAIN_STEPS * EST_STEP_SECONDS_LOW + EST_STARTUP_SECONDS_LOW))
EST_HIGH_SECONDS=$((VARIANT_COUNT * MAX_TRAIN_STEPS * EST_STEP_SECONDS_HIGH + EST_STARTUP_SECONDS_HIGH))

cd "${REPO_ROOT}"

echo "GPU=${GPU}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "OUT=${OUT}"
echo "DATA_DIR=${DATA_DIR}"
echo "TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "TOTAL_BATCH_TOKENS=${TOTAL_BATCH_TOKENS}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "GRAD_ACCUM_STEPS~${GRAD_ACCUM_STEPS}"
echo "TIMING_WINDOWS=first ${COMPILE_WINDOW_STEPS} steps, post ${WARMUP_STEPS} warmup steps"
echo "ESTIMATED_RUNTIME~$(format_duration "${EST_LOW_SECONDS}")-$(format_duration "${EST_HIGH_SECONDS}") for both variants"
echo "NOTE: Extra CLI args are forwarded, but this estimate reflects the env/default values above."

CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" python \
  research/bos_aligned_proto/experiments/run_architecture_speed_benchmark.py \
  --launcher python \
  --variants llama_param \
  --optimizers muon_pe \
  --liger_modes off,on \
  --data_dir "${DATA_DIR}" \
  --loader_kind stream \
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
  --output_dir "${OUT}" \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --total_batch_tokens "${TOTAL_BATCH_TOKENS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --compile_window_steps "${COMPILE_WINDOW_STEPS}" \
  --profile_optimizer_steps \
  --eval_every 0 \
  --hellaswag_every 0 \
  --core_every 0 \
  --ewok_every 0 \
  --save_every 0 \
  --exposure_every 0 \
  --no-save_final_checkpoint \
  --skip_final_ewok \
  "$@"
