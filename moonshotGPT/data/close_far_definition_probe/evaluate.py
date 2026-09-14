#!/usr/bin/env python3
"""Score the definition probe with the repository's EWoK likelihood routine."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from evaluation.ewok import per_token_conditional_log_likelihood
from data.spatial_bias_report.generate import generate_one as generate_bias_table


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    return {
        'n_probes': n,
        'completion_choice_accuracy': sum(r['completion_choice_accuracy'] for r in rows) / n,
        'close_accuracy': sum(r['close_correct'] for r in rows) / n,
        'far_accuracy': sum(r['far_correct'] for r in rows) / n,
        'paired_success': sum(r['paired_success'] for r in rows) / n,
        'context_sensitivity_accuracy': sum(r['context_sensitivity_accuracy'] for r in rows) / n,
        'context_sensitivity_paired_success': sum(r['context_sensitivity_paired_success'] for r in rows) / n,
        'exact_ties': sum(r['close_margin'] == 0 for r in rows) + sum(r['far_margin'] == 0 for r in rows),
        'near_ties_1e6': sum(abs(r['close_margin']) < 1e-6 for r in rows) + sum(abs(r['far_margin']) < 1e-6 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--include-sum', action='store_true',
                        help='Also report summed-token scores; mean-token scoring remains primary.')
    args = parser.parse_args()
    reductions = ['mean', 'sum'] if args.include_sum else ['mean']
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    probe_path = ROOT / 'generated/probes.jsonl'
    probes = [json.loads(line) for line in probe_path.read_text().splitlines()]
    assert len(probes) == 72 and len({p['probe_id'] for p in probes}) == 72
    assert torch.cuda.is_available(), 'This evaluation requires the requested free GPU.'
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager',
    ).to('cuda').eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    print(f'Loaded {sum(p.numel() for p in model.parameters()):,} parameters on {torch.cuda.get_device_name(0)}', flush=True)
    contexts, targets, keys = [], [], []
    for probe in probes:
        for c, t in [(1, 1), (1, 2), (2, 2), (2, 1)]:
            context, target = probe[f'Context{c}'], probe[f'Target{t}']
            ctx_ids = tokenizer.encode(context, add_special_tokens=False)
            joined_ids = tokenizer.encode(context + ' ' + target, add_special_tokens=False)
            assert joined_ids[:len(ctx_ids)] == ctx_ids, 'Context tokenization boundary changed'
            assert len(joined_ids) > len(ctx_ids)
            contexts.append(context)
            targets.append(target)
            keys.append((probe['probe_id'], f'S{c}{t}'))
    with torch.inference_mode():
        token_scores = per_token_conditional_log_likelihood(
            model, tokenizer, contexts, targets, device='cuda', batch_size=args.batch_size,
        )
    assert len(token_scores) == 288
    lookup = {}
    with (out / 'token_scores.jsonl').open('w') as stream:
        for (probe_id, combination), scores in zip(keys, token_scores):
            values = scores.detach().float().cpu()
            assert len(values) and torch.isfinite(values).all()
            lookup[(probe_id, combination)] = values
            stream.write(json.dumps({'probe_id': probe_id, 'combination': combination, 'target_token_log_probs': values.tolist()}) + '\n')
    results = []
    for reduction in reductions:
        for probe in probes:
            scores = {k: float(getattr(lookup[(probe['probe_id'], k)], reduction)()) for k in ['S11', 'S12', 'S22', 'S21']}
            close_margin = scores['S11'] - scores['S12']
            far_margin = scores['S22'] - scores['S21']
            k1 = scores['S11'] - scores['S21']
            k2 = scores['S22'] - scores['S12']
            results.append({
                **probe, 'score_reduction': reduction, **scores,
                **{f'{k}_target_token_count': len(lookup[(probe['probe_id'], k)]) for k in scores},
                'close_margin': close_margin, 'far_margin': far_margin,
                'close_correct': close_margin > 0, 'far_correct': far_margin > 0,
                'completion_choice_accuracy': ((close_margin > 0) + (far_margin > 0)) / 2,
                'paired_success': close_margin > 0 and far_margin > 0,
                'context_sensitivity_close_margin': k1, 'context_sensitivity_far_margin': k2,
                'context_sensitivity_accuracy': ((k1 > 0) + (k2 > 0)) / 2,
                'context_sensitivity_paired_success': k1 > 0 and k2 > 0,
            })
    write_csv(out / 'item_scores.csv', results)
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    grouped = []
    for reduction in reductions:
        selected = [r for r in results if r['score_reduction'] == reduction]
        for factors in [(), ('direction',), ('direction', 'adjective_pair_id'), ('direction', 'modifier_id'), ('direction', 'definition_structure_id'), ('direction', 'context_stem_id'), ('direction', 'adjective_pair_id', 'modifier_id')]:
            groups = defaultdict(list)
            for row in selected:
                groups[tuple(row[f] for f in factors)].append(row)
            for values, group in groups.items():
                grouped.append({'score_reduction': reduction, 'grouping': '+'.join(factors) or 'all', 'group': '|'.join(str(x) for x in values) or 'all', **summarize(group)})
    write_csv(out / 'grouped_scores.csv', grouped)
    summaries, consistency = {}, []
    for reduction in reductions:
        selected = [r for r in results if r['score_reduction'] == reduction]
        by_direction = {d: summarize([r for r in selected if r['direction'] == d]) for d in ['word_to_definition', 'definition_to_word']}
        groups = defaultdict(list)
        for row in selected:
            groups[row['definition_variant_id']].append(row)
        for variant, group in groups.items():
            assert len(group) == 4
            forward = [r for r in group if r['direction'] == 'word_to_definition']
            reverse = next(r for r in group if r['direction'] == 'definition_to_word')
            consistency.append({
                'score_reduction': reduction, 'definition_variant_id': variant,
                'forward_stems_passing_both': sum(r['paired_success'] for r in forward),
                'all_forward_stems_pass': all(r['paired_success'] for r in forward),
                'reverse_pass': reverse['paired_success'],
                'all_four_probes_pass': all(r['paired_success'] for r in group),
            })
        summaries[reduction] = {
            'overall': summarize(selected), 'by_direction': by_direction,
            'direction_balanced_accuracy': sum(v['completion_choice_accuracy'] for v in by_direction.values()) / 2,
            'matched_variants_passing_all_four': sum(r['all_four_probes_pass'] for r in consistency if r['score_reduction'] == reduction),
            'matched_variants': len(groups),
        }
    write_csv(out / 'consistency.csv', consistency)
    summary = {
        'model': str(Path(args.model).resolve()), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'probe_sha256': hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        'components_sha256': hashlib.sha256((ROOT / 'components.json').read_bytes()).hexdigest(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name': torch.cuda.get_device_name(0), 'dtype': 'float32',
        'attention_implementation': 'eager', 'batch_size': args.batch_size,
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'primary_reduction': 'mean', 'score_reductions': reductions, 'scores': summaries,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    generate_bias_table(out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
