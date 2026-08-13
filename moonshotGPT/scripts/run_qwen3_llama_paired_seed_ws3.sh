#!/usr/bin/env bash
set -euo pipefail

# Run one paired Qwen3/Llama Muon seed concurrently on two three-GPU groups.
# Both jobs process exactly 491,520 tokens per optimizer step and save
# checkpoints every 1,000 steps for post-hoc EWoK-fast and CORE curves.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAINER="$PROJECT_ROOT/research/bos_aligned_proto/training/trainer.py"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/processed/fineweb_edu_100B}"

SEED="${SEED:-123}"
QWEN_GPUS="${QWEN_GPUS:-0,1,2}"
LLAMA_GPUS="${LLAMA_GPUS:-3,4,5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-491520}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
QWEN_PORT="${QWEN_PORT:-29731}"
LLAMA_PORT="${LLAMA_PORT:-29732}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/research/bos_aligned_proto/qwen3_llama_paired_seed${SEED}_ws3_${STAMP}}"
QWEN_EXPERIMENTS="$OUTPUT_ROOT/runs/qwen3_liger_muon"
LLAMA_EXPERIMENTS="$OUTPUT_ROOT/runs/llama_liger_muon"

mkdir -p "$OUTPUT_ROOT/logs" "$QWEN_EXPERIMENTS" "$LLAMA_EXPERIMENTS"
cd "$PROJECT_ROOT"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "seed=$SEED"
  echo "qwen_physical_gpus=$QWEN_GPUS"
  echo "llama_physical_gpus=$LLAMA_GPUS"
  echo "world_size_per_model=3"
  echo "total_batch_tokens=$TOTAL_BATCH_TOKENS"
  echo "max_train_steps=$MAX_TRAIN_STEPS"
  echo "eval_every=$EVAL_EVERY"
  echo "save_every=$SAVE_EVERY"
  echo "qwen_micro_batch_size=5"
  echo "qwen_grad_accum_steps_expected=32"
  echo "llama_micro_batch_size=4"
  echo "llama_grad_accum_steps_expected=40"
} > "$OUTPUT_ROOT/paired_run_config.txt"

common_args=(
  --loader_kind stream
  --data_dir "$DATA_DIR"
  --tokenizer_name_or_path gpt2
  --seed "$SEED"
  --mixed_precision bf16
  --total_batch_tokens "$TOTAL_BATCH_TOKENS"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --seq_len 1024
  --vocab_size 0
  --n_embd 1024
  --n_head 16
  --n_layer 24
  --use_liger_kernel
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
  --muon_nesterov
  --muon_split_qkv
  --muon_batch_updates
  --grad_clip 1.0
  --eval_every "$EVAL_EVERY"
  --hellaswag_every 0
  --core_every 0
  --ewok_every 0
  --save_every "$SAVE_EVERY"
  --exposure_every 0
  --no-save_final_checkpoint
  --skip_final_ewok
  --profile_optimizer_steps
)

qwen_cmd=(
  "$ACCELERATE_BIN" launch
  --num_processes 3
  --main_process_port "$QWEN_PORT"
  "$TRAINER"
  "${common_args[@]}"
  --experiments_dir "$QWEN_EXPERIMENTS"
  --micro_batch_size 5
  --model_arch qwen3
  --qwen_intermediate_size 3152
  --qwen_num_key_value_heads 8
  --qwen_head_dim 64
  --qwen_tie_word_embeddings
  --rope_theta 1000000
)

llama_cmd=(
  "$ACCELERATE_BIN" launch
  --num_processes 3
  --main_process_port "$LLAMA_PORT"
  "$TRAINER"
  "${common_args[@]}"
  --experiments_dir "$LLAMA_EXPERIMENTS"
  --micro_batch_size 4
  --model_arch llama
  --llama_intermediate_size 2816
  --llama_num_key_value_heads 0
  --llama_tie_word_embeddings
  --rope_theta 10000
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q' "$QWEN_GPUS"
  printf ' %q' "${qwen_cmd[@]}"
  printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%q' "$LLAMA_GPUS"
  printf ' %q' "${llama_cmd[@]}"
  printf '\n'
  echo "output_root=$OUTPUT_ROOT"
  exit 0
fi

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv -l 10 > "$OUTPUT_ROOT/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!

qwen_pid=""
llama_pid=""
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  if [[ -n "$qwen_pid" ]]; then
    kill "$qwen_pid" 2>/dev/null || true
  fi
  if [[ -n "$llama_pid" ]]; then
    kill "$llama_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date --iso-8601=seconds)] launching Qwen3 seed $SEED on GPUs $QWEN_GPUS"
CUDA_VISIBLE_DEVICES="$QWEN_GPUS" PYTHONUNBUFFERED=1 \
  "${qwen_cmd[@]}" > "$OUTPUT_ROOT/logs/qwen3.log" 2>&1 &
qwen_pid=$!

echo "[$(date --iso-8601=seconds)] launching Llama seed $SEED on GPUs $LLAMA_GPUS"
CUDA_VISIBLE_DEVICES="$LLAMA_GPUS" PYTHONUNBUFFERED=1 \
  "${llama_cmd[@]}" > "$OUTPUT_ROOT/logs/llama.log" 2>&1 &
llama_pid=$!

qwen_status=0
llama_status=0
wait "$qwen_pid" || qwen_status=$?
qwen_pid=""
wait "$llama_pid" || llama_status=$?
llama_pid=""

echo "qwen_exit_code=$qwen_status" | tee "$OUTPUT_ROOT/completion_status.txt"
echo "llama_exit_code=$llama_status" | tee -a "$OUTPUT_ROOT/completion_status.txt"
echo "output_root=$OUTPUT_ROOT" | tee -a "$OUTPUT_ROOT/completion_status.txt"

if (( qwen_status != 0 || llama_status != 0 )); then
  exit 1
fi
