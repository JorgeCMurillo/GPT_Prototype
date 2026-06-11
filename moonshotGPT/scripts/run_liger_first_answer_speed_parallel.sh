#!/usr/bin/env bash
set -euo pipefail

# Parallel fast first-answer Llama+Muon Liger A/B.
#
# Runs no-Liger and Liger at the same time on two GPUs, then summarizes both
# into one architecture_speed_summary.csv.
#
# Example:
#   tmux new -s liger-first-answer-parallel
#   GPU_NOLIGER=4 GPU_LIGER=5 scripts/run_liger_first_answer_speed_parallel.sh
#   GPU_NOLIGER=4 GPU_LIGER=5 scripts/run_liger_first_answer_speed_parallel.sh --dry_run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_NOLIGER="${GPU_NOLIGER:-4}"
GPU_LIGER="${GPU_LIGER:-5}"
CONDA_ENV="${CONDA_ENV:-babylm}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-runs/research/bos_aligned_proto/liger_first_answer_parallel_${STAMP}}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/processed/fineweb_edu_100B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-gpt2}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-81920}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40}"
SEQ_LEN="${SEQ_LEN:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
COMPILE_WINDOW_STEPS="${COMPILE_WINDOW_STEPS:-20}"

EST_STEP_SECONDS_LOW="${EST_STEP_SECONDS_LOW:-5}"
EST_STEP_SECONDS_HIGH="${EST_STEP_SECONDS_HIGH:-8}"
EST_STARTUP_SECONDS_LOW="${EST_STARTUP_SECONDS_LOW:-120}"
EST_STARTUP_SECONDS_HIGH="${EST_STARTUP_SECONDS_HIGH:-300}"

DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == "--dry_run" || "${1:-}" == "--preview" ]]; then
  DRY_RUN=1
  shift
fi
EXTRA_TRAINER_ARGS=("$@")

format_duration() {
  local seconds="$1"
  local minutes=$(((seconds + 30) / 60))
  if (( minutes < 60 )); then
    printf "%dm" "${minutes}"
  else
    printf "%dh%02dm" "$((minutes / 60))" "$((minutes % 60))"
  fi
}

TOKENS_PER_MICROSTEP=$((MICRO_BATCH_SIZE * SEQ_LEN))
GRAD_ACCUM_STEPS=$(((TOTAL_BATCH_TOKENS + TOKENS_PER_MICROSTEP - 1) / TOKENS_PER_MICROSTEP))
EST_LOW_SECONDS=$((MAX_TRAIN_STEPS * EST_STEP_SECONDS_LOW + EST_STARTUP_SECONDS_LOW))
EST_HIGH_SECONDS=$((MAX_TRAIN_STEPS * EST_STEP_SECONDS_HIGH + EST_STARTUP_SECONDS_HIGH))

cd "${REPO_ROOT}"
mkdir -p "${OUT}/logs" "${OUT}/runs"

echo "GPU_NOLIGER=${GPU_NOLIGER}"
echo "GPU_LIGER=${GPU_LIGER}"
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
echo "ESTIMATED_PARALLEL_RUNTIME~$(format_duration "${EST_LOW_SECONDS}")-$(format_duration "${EST_HIGH_SECONDS}")"
echo "DRY_RUN=${DRY_RUN}"

common_args=(
  research/bos_aligned_proto/training/trainer.py
  --loader_kind stream
  --data_dir "${DATA_DIR}"
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
  --seed 42
  --mixed_precision bf16
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --total_batch_tokens "${TOTAL_BATCH_TOKENS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --seq_len "${SEQ_LEN}"
  --vocab_size 0
  --model_arch llama
  --n_embd 1024
  --n_head 16
  --n_layer 24
  --llama_intermediate_size 2816
  --llama_num_key_value_heads 0
  --rope_theta 10000.0
  --num_workers 0
  --learning_rate 0.0006
  --warmup_iters 700
  --learning_rate_decay_frac 0.0
  --optimizer muon_pe
  --weight_decay 0.1
  --beta1 0.9
  --beta2 0.95
  --muon_lr 0.02
  --muon_momentum 0.95
  --muon_weight_decay 0.1
  --muon_ns_steps 5
  --grad_clip 1.0
  --eval_every 0
  --hellaswag_every 0
  --core_every 0
  --ewok_every 0
  --save_every 0
  --exposure_every 0
  --no-save_final_checkpoint
  --skip_final_ewok
  --muon_nesterov
  --muon_split_qkv
  --muon_batch_updates
  --profile_optimizer_steps
)

run_variant() {
  local label="$1"
  local gpu="$2"
  local liger_flag="$3"
  local experiments_dir="${OUT}/runs/${label}"
  local log_path="${OUT}/logs/${label}.log"

  echo "[${label}] GPU=${gpu} log=${log_path}"
  cmd=(conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${common_args[@]}" \
    --experiments_dir "${experiments_dir}" \
    "${liger_flag}" \
    "${EXTRA_TRAINER_ARGS[@]}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_path}" 2>&1
}

run_variant llama_param_i2816_noliger "${GPU_NOLIGER}" --no-use_liger_kernel &
pid_noliger=$!
run_variant llama_param_i2816_liger "${GPU_LIGER}" --use_liger_kernel &
pid_liger=$!

status_noliger=0
status_liger=0
wait "${pid_noliger}" || status_noliger=$?
wait "${pid_liger}" || status_liger=$?

echo "[llama_param_i2816_noliger] exit code: ${status_noliger}"
echo "[llama_param_i2816_liger] exit code: ${status_liger}"

if (( status_noliger != 0 || status_liger != 0 )); then
  echo "One or both variants failed. See logs under ${OUT}/logs." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only; no training launched and no summary generated."
  exit 0
fi

noliger_run="$(find "${OUT}/runs/llama_param_i2816_noliger" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
liger_run="$(find "${OUT}/runs/llama_param_i2816_liger" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

conda run -n "${CONDA_ENV}" python \
  research/bos_aligned_proto/experiments/run_architecture_speed_benchmark.py \
  --summarize_only \
  --output_dir "${OUT}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --compile_window_steps "${COMPILE_WINDOW_STEPS}" \
  --run_dir "llama_param_i2816_noliger=${noliger_run}" \
  --run_dir "llama_param_i2816_liger=${liger_run}"

echo "Summary: ${OUT}/architecture_speed_summary.csv"
