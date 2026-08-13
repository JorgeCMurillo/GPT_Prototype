#!/usr/bin/env bash
set -uo pipefail

# Sequential four-GPU Qwen3+Liger+Muon benchmarks. The two variants keep the
# global optimizer-step budget fixed at 491,520 tokens while varying only the
# per-GPU microbatch and corresponding gradient accumulation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
TRAINER="$PROJECT_ROOT/research/bos_aligned_proto/training/trainer.py"
SUMMARIZER="$PROJECT_ROOT/research/bos_aligned_proto/experiments/run_architecture_speed_benchmark.py"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/processed/fineweb_edu_100B}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/research/bos_aligned_proto/qwen3_liger_muon_microbatch_ws4_${STAMP}}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1,2,3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-491520}"
MAIN_PROCESS_PORT_BASE="${MAIN_PROCESS_PORT_BASE:-29670}"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs"
cd "$PROJECT_ROOT"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=$PHYSICAL_GPUS"
  echo "world_size=4"
  echo "data_dir=$DATA_DIR"
  echo "tokenizer=gpt2"
  echo "max_train_steps=$MAX_TRAIN_STEPS"
  echo "total_batch_tokens=$TOTAL_BATCH_TOKENS"
  echo "model_arch=qwen3"
  echo "parameters=359404544"
  echo "shape=d1024_L24_q16_kv8_head64_mlp3152"
  echo "optimizer=muon_pe"
  echo "liger=true"
} > "$OUTPUT_ROOT/benchmark_config.txt"

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv -l 2 > "$OUTPUT_ROOT/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT

run_dirs=()
failed=0

run_variant() {
  local microbatch="$1"
  local label="mbs${microbatch}"
  local port="$2"
  local experiments_dir="$OUTPUT_ROOT/runs/$label"
  local log_path="$OUTPUT_ROOT/logs/$label.log"

  mkdir -p "$experiments_dir"
  echo "[$(date --iso-8601=seconds)] starting $label on physical GPUs $PHYSICAL_GPUS"

  if "$ACCELERATE_BIN" launch \
      --num_processes 4 \
      --main_process_port "$port" \
      "$TRAINER" \
      --loader_kind stream \
      --data_dir "$DATA_DIR" \
      --tokenizer_name_or_path gpt2 \
      --experiments_dir "$experiments_dir" \
      --seed 42 \
      --mixed_precision bf16 \
      --micro_batch_size "$microbatch" \
      --total_batch_tokens "$TOTAL_BATCH_TOKENS" \
      --max_train_steps "$MAX_TRAIN_STEPS" \
      --seq_len 1024 \
      --vocab_size 0 \
      --model_arch qwen3 \
      --n_embd 1024 \
      --n_head 16 \
      --n_layer 24 \
      --qwen_intermediate_size 3152 \
      --qwen_num_key_value_heads 8 \
      --qwen_head_dim 64 \
      --qwen_tie_word_embeddings \
      --rope_theta 1000000 \
      --use_liger_kernel \
      --num_workers 0 \
      --learning_rate 0.0006 \
      --warmup_iters 700 \
      --learning_rate_decay_frac 0.0 \
      --optimizer muon_pe \
      --weight_decay 0.1 \
      --beta1 0.9 \
      --beta2 0.95 \
      --muon_lr 0.02 \
      --muon_momentum 0.95 \
      --muon_weight_decay 0.1 \
      --muon_ns_steps 5 \
      --muon_nesterov \
      --muon_split_qkv \
      --muon_batch_updates \
      --grad_clip 1.0 \
      --eval_every "$MAX_TRAIN_STEPS" \
      --hellaswag_every 0 \
      --core_every 0 \
      --ewok_every 0 \
      --save_every 0 \
      --exposure_every 0 \
      --no-save_final_checkpoint \
      --skip_final_ewok \
      --profile_optimizer_steps > "$log_path" 2>&1; then
    local run_dir
    run_dir="$(find "$experiments_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    run_dirs+=("$label=$run_dir")
    echo "[$(date --iso-8601=seconds)] completed $label: $run_dir"
  else
    echo "[$(date --iso-8601=seconds)] failed $label; see $log_path"
    failed=1
  fi
}

run_variant 4 "$MAIN_PROCESS_PORT_BASE"
if (( failed == 0 )); then
  run_variant 6 "$((MAIN_PROCESS_PORT_BASE + 1))"
fi

if (( ${#run_dirs[@]} > 0 )); then
  summary_args=(
    "$SUMMARIZER"
    --summarize_only
    --output_dir "$OUTPUT_ROOT"
    --warmup_steps 20
    --compile_window_steps 20
  )
  for run_dir in "${run_dirs[@]}"; do
    summary_args+=(--run_dir "$run_dir")
  done
  "$PYTHON_BIN" "${summary_args[@]}" > "$OUTPUT_ROOT/logs/summarize.log" 2>&1 || failed=1
fi

echo "[$(date --iso-8601=seconds)] benchmark finished; failed=$failed"
echo "output_root=$OUTPUT_ROOT"
exit "$failed"
