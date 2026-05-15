#!/usr/bin/env bash
set -euo pipefail

cd /home/jorge/tokenPred/moonshotGPT

python research/bos_aligned_proto/spatial_synth/generate_cardinal_api_text.py \
  --model gpt-5.2 \
  --target-tokens 39219 \
  --request-size 16 \
  --max-requests 220 \
  --seed 42 \
  --resume \
  --sleep 20 \
  --rate-limit-sleep 120 \
  --max-api-attempts 100 \
  --out /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/spatial_synth/cardinal_api_gpt52_10pct_tokens39219_seed42.csv \
  --jsonl-out /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/spatial_synth/cardinal_api_gpt52_10pct_tokens39219_seed42.jsonl
