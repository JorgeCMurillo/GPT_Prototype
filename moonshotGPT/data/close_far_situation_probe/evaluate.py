#!/usr/bin/env python3
"""Evaluate situation probes with mean target-token EWoK likelihood scoring."""

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
from data.close_far_definition_probe.evaluate import summarize, write_csv
from data.spatial_bias_report.generate import generate_one as generate_bias_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dataset', choices=['situations', 'context_diagnostics'], default='situations')
    args = parser.parse_args()
    dataset_root = ROOT if args.dataset == 'situations' else ROOT / 'context_diagnostics'
    expected_probes = 468 if args.dataset == 'situations' else 228
    probe_path = dataset_root / 'generated/probes.jsonl'
    probes = [json.loads(line) for line in probe_path.read_text().splitlines()]
    assert len(probes) == expected_probes and len({r['probe_id'] for r in probes}) == expected_probes
    assert torch.cuda.is_available(), 'CUDA is required for this evaluation.'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager',
    ).to('cuda').eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    # Identical context/target strings are scored once and shared across matches.
    pairs = list(dict.fromkeys((r[f'Context{c}'], r[f'Target{t}']) for r in probes for c, t in [(1, 1), (1, 2), (2, 2), (2, 1)]))
    for context, target in pairs:
        ctx = tokenizer.encode(context, add_special_tokens=False)
        joined = tokenizer.encode(context + ' ' + target, add_special_tokens=False)
        assert joined[:len(ctx)] == ctx and len(joined) > len(ctx)
    print(f'Loaded model; scoring {len(pairs)} unique context/target pairs on {torch.cuda.get_device_name(0)}.', flush=True)
    lookup = {}
    with torch.inference_mode(), (out / 'token_scores.jsonl').open('w') as stream:
        for start in range(0, len(pairs), 128):
            chunk = pairs[start:start + 128]
            values = per_token_conditional_log_likelihood(
                model, tokenizer, [p[0] for p in chunk], [p[1] for p in chunk],
                device='cuda', batch_size=args.batch_size,
            )
            for pair, scores in zip(chunk, values):
                scores = scores.detach().float().cpu()
                assert len(scores) and torch.isfinite(scores).all()
                lookup[pair] = (float(scores.mean()), len(scores))
                stream.write(json.dumps({'context': pair[0], 'target': pair[1], 'target_token_log_probs': scores.tolist()}) + '\n')
            print(f'Scored {min(start + 128, len(pairs))}/{len(pairs)} unique pairs.', flush=True)
    results = []
    for probe in probes:
        scores, counts = {}, {}
        for c, t in [(1, 1), (1, 2), (2, 2), (2, 1)]:
            value, count = lookup[(probe[f'Context{c}'], probe[f'Target{t}'])]
            scores[f'S{c}{t}'] = value
            counts[f'S{c}{t}_target_token_count'] = count
        m1, m2 = scores['S11'] - scores['S12'], scores['S22'] - scores['S21']
        k1, k2 = scores['S11'] - scores['S21'], scores['S22'] - scores['S12']
        results.append({
            **probe, 'score_reduction': 'mean', **scores, **counts,
            'close_margin': m1, 'far_margin': m2,
            'close_correct': m1 > 0, 'far_correct': m2 > 0,
            'completion_choice_accuracy': ((m1 > 0) + (m2 > 0)) / 2,
            'paired_success': m1 > 0 and m2 > 0,
            'context_sensitivity_close_margin': k1, 'context_sensitivity_far_margin': k2,
            'context_sensitivity_accuracy': ((k1 > 0) + (k2 > 0)) / 2,
            'context_sensitivity_paired_success': k1 > 0 and k2 > 0,
        })
    assert len(results) == expected_probes
    write_csv(out / 'item_scores.csv', results)
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    grouped = []
    factors_to_group = [('condition_id',), ('condition_id', 'entity_type'), ('condition_id', 'scenario_id'), ('condition_id', 'name_id'), ('condition_id', 'object_id'), ('evidence_type',)]
    if args.dataset == 'context_diagnostics':
        factors_to_group += [('context_form_id', 'reference_form_id', 'body_comparison_available')]
    for factors in factors_to_group:
        groups = defaultdict(list)
        for row in results:
            groups[tuple(row[k] for k in factors)].append(row)
        for key, group in groups.items():
            grouped.append({'grouping': '+'.join(factors), 'group': '|'.join(str(v) for v in key), **summarize(group)})
    write_csv(out / 'grouped_scores.csv', grouped)
    by_id = {r['probe_id']: r for r in results}
    match_sources = ['control_matches', 'substitution_matches'] if args.dataset == 'situations' else ['diagnostic_matches']
    for source in match_sources:
        with (dataset_root / 'generated' / f'{source}.csv').open(newline='') as stream:
            matches = list(csv.DictReader(stream))
        comparisons = []
        for match in matches:
            reference_key = 'baseline_probe_id' if source == 'control_matches' else 'reference_probe_id'
            reference, variant = by_id[match[reference_key]], by_id[match['variant_probe_id']]
            if match.get('control') == 'body_reference':
                assert reference['Context2'] == variant['Context2']
                assert reference['S21'] == variant['S21'] and reference['S22'] == variant['S22']
            comparisons.append({
                **match,
                'reference_accuracy': reference['completion_choice_accuracy'],
                'variant_accuracy': variant['completion_choice_accuracy'],
                'accuracy_delta': variant['completion_choice_accuracy'] - reference['completion_choice_accuracy'],
                'close_margin_delta': variant['close_margin'] - reference['close_margin'],
                'far_margin_delta': variant['far_margin'] - reference['far_margin'],
                'both_probes_pass': reference['paired_success'] and variant['paired_success'],
                'same_decisions': (reference['close_correct'], reference['far_correct']) == (variant['close_correct'], variant['far_correct']),
            })
        write_csv(out / f'{source}_scores.csv', comparisons)
    groups = defaultdict(list)
    for row in results:
        groups[row['matched_group_id']].append(row)
    consistency = [{'matched_group_id': key, 'n_conditions': len(group), 'conditions_passing_both': sum(r['paired_success'] for r in group), 'all_conditions_pass': all(r['paired_success'] for r in group)} for key, group in groups.items()]
    write_csv(out / 'consistency.csv', consistency)
    by_condition = {}
    for condition in dict.fromkeys(r['condition_id'] for r in results):
        subset = [r for r in results if r['condition_id'] == condition]
        scenes = defaultdict(list)
        for row in subset:
            scenes[row['scenario_id']].append(row)
        by_condition[condition] = {
            **summarize(subset),
            'scenario_balanced_accuracy': sum(summarize(s)['completion_choice_accuracy'] for s in scenes.values()) / len(scenes),
        }
    summary = {
        'model': str(Path(args.model).resolve()), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'probe_sha256': hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        'components_sha256': hashlib.sha256((dataset_root / 'components.json').read_bytes()).hexdigest(),
        'dataset': args.dataset,
        'primary_reduction': 'mean', 'dtype': 'float32', 'attention_implementation': 'eager',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name': torch.cuda.get_device_name(0), 'batch_size': args.batch_size,
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'unique_context_target_pairs': len(pairs), 'overall': summarize(results),
        'by_condition': by_condition,
        'scenario_balanced_accuracy': sum(v['scenario_balanced_accuracy'] for v in by_condition.values()) / len(by_condition),
        'assignments_passing_all_conditions': sum(r['all_conditions_pass'] for r in consistency),
    }
    if args.dataset == 'context_diagnostics':
        summary['placement_by_context'] = {
            form: summarize([r for r in results if r['context_form_id'] == form and r['reference_form_id'] == 'placement'])
            for form in ['compact', 'setting', 'endpoint']
        }
        summary['body_comparison_by_context'] = {
            form: {
                ref: summarize([r for r in results if r['context_form_id'] == form and r['reference_form_id'] == ref and r['body_comparison_available']])
                for ref in ['placement', 'person_direct']
            }
            for form in ['compact', 'setting', 'endpoint']
        }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    generate_bias_table(out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
