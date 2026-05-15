#!/usr/bin/env bash
set -euo pipefail

# v22 natural/synthetic dose sweep:
#   nat100_synth00 = continued fine-tuning on natural text only
#   nat95_synth05  = 95% natural tokens, 5% v19 synthetic spatial tokens
#   nat90_synth10  = 90% natural tokens, 10% v19 synthetic spatial tokens
#   nat80_synth20  = 80% natural tokens, 20% v19 synthetic spatial tokens
#   nat60_synth40  = 60% natural tokens, 40% v19 synthetic spatial tokens
#
# Mixtures are built by token budget, then passed through the existing spatial
# fine-tuning/eval runner unchanged. Use DIFFICULTY=all because natural rows do
# not have easy/medium/hard labels.
#
# Usage:
#   bash research/bos_aligned_proto/spatial_synth/run_v22_natural_synth_mix_sweep.sh
#
# Useful subsets:
#   SYNTH_RATIOS="0.05 0.10" bash research/bos_aligned_proto/spatial_synth/run_v22_natural_synth_mix_sweep.sh
#   TARGET_TOKENS=200000 LRS="4e-5" bash research/bos_aligned_proto/spatial_synth/run_v22_natural_synth_mix_sweep.sh

ROOT="/home/jorge/tokenPred/moonshotGPT"
BASE_RUNNER="$ROOT/research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh"
MIX_GENERATOR="$ROOT/research/bos_aligned_proto/spatial_synth/generate_natural_synth_mix_csv.py"
DATA_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth"
OUT_ROOT="$ROOT/runs/research/bos_aligned_proto/spatial_synth_training"

SYNTH_CSV="${SYNTH_CSV:-$DATA_ROOT/spatial_relations_synth_implicit_v19_v14_lr_paired_n10000_seed42_mixed.csv}"
NATURAL_DATA_DIR="${NATURAL_DATA_DIR:-$ROOT/data/processed/fineweb_edu_10B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$ROOT/experiments/babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000/ckpt_periodic_step0008000}"

SYNTH_RATIOS="${SYNTH_RATIOS:-0 0.05 0.10 0.20 0.40}"
TARGET_TOKENS="${TARGET_TOKENS:-400000}"
SEED="${SEED:-42}"
NATURAL_SKIP_DOCS="${NATURAL_SKIP_DOCS:-0}"
MIN_NATURAL_DOC_TOKENS="${MIN_NATURAL_DOC_TOKENS:-16}"
MAX_NATURAL_DOC_TOKENS="${MAX_NATURAL_DOC_TOKENS:-1024}"

LRS="${LRS:-4e-5 8e-5}"
EPOCHS="${EPOCHS:-3}"
EPOCH_EVAL="${EPOCH_EVAL:-0.25}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EWOK_VARIANT="${EWOK_VARIANT:-fast}"
EWOK_BATCH_SIZE="${EWOK_BATCH_SIZE:-8}"
SYNTHETIC_SPATIAL_EVAL_N_PER_TIER="${SYNTHETIC_SPATIAL_EVAL_N_PER_TIER:-300}"

ratio_tag() {
  case "$1" in
    0|0.0|0.00) echo "nat100_synth00" ;;
    .05|0.05|0.050) echo "nat95_synth05" ;;
    .10|0.10|0.100|0.1) echo "nat90_synth10" ;;
    .20|0.20|0.200|0.2) echo "nat80_synth20" ;;
    .40|0.40|0.400|0.4) echo "nat60_synth40" ;;
    *)
      local clean="${1//./p}"
      echo "synth${clean}"
      ;;
  esac
}

for ratio in $SYNTH_RATIOS; do
  tag="$(ratio_tag "$ratio")"
  data_path="$DATA_ROOT/spatial_relations_natural_mix_v22_${tag}_tok${TARGET_TOKENS}_seed${SEED}.csv"
  out_dir="$OUT_ROOT/ckpt_8k_12k_16k_fast_eval_v22_natural_mix_${tag}_tok${TARGET_TOKENS}_full_loss_${EPOCHS}epoch_synth_eval"

  echo "=== Building v22 mixture $tag: synthetic ratio=$ratio, target_tokens=$TARGET_TOKENS ==="
  python "$MIX_GENERATOR" \
    --synthetic-csv "$SYNTH_CSV" \
    --natural-data-dir "$NATURAL_DATA_DIR" \
    --tokenizer-name "$TOKENIZER_NAME" \
    --target-tokens "$TARGET_TOKENS" \
    --synthetic-token-ratio "$ratio" \
    --seed "$SEED" \
    --natural-skip-docs "$NATURAL_SKIP_DOCS" \
    --min-natural-doc-tokens "$MIN_NATURAL_DOC_TOKENS" \
    --max-natural-doc-tokens "$MAX_NATURAL_DOC_TOKENS" \
    --out "$data_path"

  echo "=== Running v22 mixture $tag ==="
  TEMPLATE_PRESET="v19" \
  LOSS_MODE="full" \
  EWOK_VARIANT="$EWOK_VARIANT" \
  EWOK_BATCH_SIZE="$EWOK_BATCH_SIZE" \
  SYNTHETIC_SPATIAL_EVAL="three_tier" \
  SYNTHETIC_SPATIAL_EVAL_N_PER_TIER="$SYNTHETIC_SPATIAL_EVAL_N_PER_TIER" \
  SYNTHETIC_SPATIAL_EVAL_TEMPLATE_PRESET="v19" \
  EPOCHS="$EPOCHS" \
  EPOCH_EVAL="$EPOCH_EVAL" \
  LRS="$LRS" \
  SEED="$SEED" \
  DIFFICULTY="all" \
  DATA_TAG="natural_mix_v22_${tag}_tok${TARGET_TOKENS}_seed${SEED}" \
  DATA_PATH="$data_path" \
  OUT_DIR="$out_dir" \
  PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
  bash "$BASE_RUNNER"
done
