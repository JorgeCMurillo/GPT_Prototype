#!/usr/bin/env python3
"""Measure close/far likelihoods without a descriptive spatial context."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.ewok import per_token_unconditional_log_likelihood, per_token_conditional_log_likelihood, resolve_bos_token_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    model.config.use_cache = False
    pairs = [
        ('bare_word', 'close', 'far'),
        ('leading_space_word', ' close', ' far'),
        ('word_with_period', 'close.', 'far.'),
        ('full_target_sentence', 'The objects are close.', 'The objects are far.'),
    ]
    texts = [text for _, a, b in pairs for text in [a, b]]
    with torch.inference_mode():
        unconditional = per_token_unconditional_log_likelihood(model, tokenizer, texts, device='cuda', batch_size=8)
        conditional = per_token_conditional_log_likelihood(model, tokenizer, ['The objects are'] * 2, ['close', 'far'], device='cuda', batch_size=2)
    rows = []
    for i, (pair_id, close, far) in enumerate(pairs + [('word_after_neutral_prefix', 'close', 'far')]):
        token_scores = unconditional[2 * i:2 * i + 2] if i < len(pairs) else conditional
        entries = {}
        for concept, text, scores in zip(['close', 'far'], [close, far], token_scores):
            scores = scores.detach().float().cpu()
            assert torch.isfinite(scores).all() and len(scores)
            scored_text = text if i < len(pairs) else ' ' + text
            ids = tokenizer.encode(scored_text, add_special_tokens=False)
            assert len(ids) == len(scores)
            entries[concept] = {'text': text, 'token_count': len(scores), 'tokens': tokenizer.convert_ids_to_tokens(ids), 'token_log_probs': scores.tolist(), 'mean_log_likelihood': float(scores.mean()), 'sum_log_likelihood': float(scores.sum())}
        delta = entries['close']['mean_log_likelihood'] - entries['far']['mean_log_likelihood']
        sum_delta = entries['close']['sum_log_likelihood'] - entries['far']['sum_log_likelihood']
        rows.append({'pair_id': pair_id, 'context': None if i < len(pairs) else 'The objects are', **entries, 'close_minus_far_mean': delta, 'preferred_by_mean': 'close' if delta > 0 else 'far' if delta < 0 else 'tie', 'close_to_far_sequence_likelihood_ratio': math.exp(sum_delta)})
    result = {
        'model': args.model, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'dtype': 'float32', 'primary_reduction': 'mean', 'log_base': 'e',
        'bos_token_id': resolve_bos_token_id(tokenizer),
        'note': 'No EOS appended. All sequences start with the same evaluator BOS. Ratios use summed log probabilities; they are not obtained by exponentiating mean differences for multi-token targets.',
        'results': rows,
    }
    (out / 'scores.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
