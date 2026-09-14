#!/usr/bin/env python3
"""Calibrate saved mean-token conditional scores with target-only BOS scores."""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.ewok import per_token_unconditional_log_likelihood, resolve_bos_token_id
from data.close_far_definition_probe.evaluate import write_csv
from data.close_far_situation_probe.analyze_token_contributions import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    records, sources = [], {}
    model_path = None
    for dataset, folder in DATASETS.items():
        manifest = json.loads((folder / 'summary.json').read_text())
        if model_path is None:
            model_path = manifest['model']
        assert manifest['model'] == model_path and manifest['dtype'] == 'float32'
        path = folder / 'item_scores.jsonl'
        sources[dataset] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['score_reduction'] == 'mean':
                row['dataset'] = dataset
                row['analysis_group'] = row['direction'] if dataset == 'definition' else row['condition_id']
                records.append(row)
    assert len(records) == 768
    assert torch.cuda.is_available()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    model.config.use_cache = False
    targets = list(dict.fromkeys(r[k] for r in records for k in ['Target1', 'Target2']))
    print(f'Scoring {len(targets)} unique targets from BOS; reusing 768 probes of saved conditional scores.', flush=True)
    lookup, target_rows = {}, []
    with torch.inference_mode():
        for start in range(0, len(targets), 64):
            chunk = targets[start:start + 64]
            values = per_token_unconditional_log_likelihood(model, tokenizer, chunk, device='cuda', batch_size=8)
            for target, scores in zip(chunk, values):
                scores = scores.detach().float().cpu()
                assert torch.isfinite(scores).all() and len(scores)
                ids = tokenizer.encode(target, add_special_tokens=False)
                assert len(ids) == len(scores)
                row = {'target': target, 'token_count': len(scores), 'mean_log_likelihood': float(scores.mean()), 'tokens': tokenizer.convert_ids_to_tokens(ids), 'token_log_probs': scores.tolist()}
                lookup[target] = row
                target_rows.append(row)
    (out / 'target_priors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in target_rows))
    scored = []
    for r in records:
        b1, b2 = lookup[r['Target1']], lookup[r['Target2']]
        pm1 = (r['S11'] - b1['mean_log_likelihood']) - (r['S12'] - b2['mean_log_likelihood'])
        pm2 = (r['S22'] - b2['mean_log_likelihood']) - (r['S21'] - b1['mean_log_likelihood'])
        assert abs(pm1 + pm2 - r['close_margin'] - r['far_margin']) < 1e-10
        scored.append({
            'dataset': r['dataset'], 'group': r['analysis_group'], 'probe_id': r['probe_id'],
            'scenario_id': r.get('scenario_id'), 'entity_type': r.get('entity_type'),
            'body_comparison_available': r.get('body_comparison_available'),
            'context_form_id': r.get('context_form_id'), 'reference_form_id': r.get('reference_form_id'),
            'Context1': r['Context1'], 'Context2': r['Context2'], 'Target1': r['Target1'], 'Target2': r['Target2'],
            'B1_mean': b1['mean_log_likelihood'], 'B2_mean': b2['mean_log_likelihood'],
            'target_prior_gap_T1_minus_T2': b1['mean_log_likelihood'] - b2['mean_log_likelihood'],
            'raw_close_correct': r['close_correct'], 'raw_far_correct': r['far_correct'],
            'raw_accuracy': r['completion_choice_accuracy'], 'raw_paired_success': r['paired_success'],
            'pmi_close_margin': pm1, 'pmi_far_margin': pm2,
            'pmi_close_correct': pm1 > 0, 'pmi_far_correct': pm2 > 0,
            'pmi_accuracy': ((pm1 > 0) + (pm2 > 0)) / 2,
            'pmi_paired_success': pm1 > 0 and pm2 > 0,
            'exact_ties': int(pm1 == 0) + int(pm2 == 0),
        })
    write_csv(out / 'item_scores.csv', scored)
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in scored))
    groups = defaultdict(list)
    for r in scored:
        groups[(r['dataset'], r['group'])].append(r)
    summaries = []
    for (dataset, group), rows in groups.items():
        n = len(rows)
        summaries.append({
            'dataset': dataset, 'group': group, 'probes': n,
            'raw_accuracy': sum(r['raw_accuracy'] for r in rows) / n,
            'pmi_accuracy': sum(r['pmi_accuracy'] for r in rows) / n,
            'raw_paired_success': sum(r['raw_paired_success'] for r in rows) / n,
            'pmi_paired_success': sum(r['pmi_paired_success'] for r in rows) / n,
            'pmi_close_accuracy': sum(r['pmi_close_correct'] for r in rows) / n,
            'pmi_far_accuracy': sum(r['pmi_far_correct'] for r in rows) / n,
            'wrong_to_right': sum(not r[f'raw_{c}_correct'] and r[f'pmi_{c}_correct'] for r in rows for c in ['close', 'far']),
            'right_to_wrong': sum(r[f'raw_{c}_correct'] and not r[f'pmi_{c}_correct'] for r in rows for c in ['close', 'far']),
            'exact_ties': sum(r['exact_ties'] for r in rows),
        })
    write_csv(out / 'summary.csv', summaries)
    summary = {
        'model': model_path, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'dtype': 'float32', 'score_reduction': 'mean', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'bos_token_id': resolve_bos_token_id(tokenizer),
        'baseline_convention': 'Repository EWoK PMI convention: target text as written after BOS; no descriptive context, no added leading space, no appended EOS.',
        'formula': 'mean_logp(T|C) - mean_logp(T|BOS)',
        'note': 'Both terms include all target tokens, including punctuation. Mean-token PMI-style calibration; no tuned scale. Raw mean scores remain available.',
        'source_runs': sources, 'unique_targets': len(targets), 'groups': summaries,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for r in summaries:
        print(r['dataset'], r['group'], 'raw', round(r['raw_accuracy']*100,2), 'PMI', round(r['pmi_accuracy']*100,2), 'paired', round(r['pmi_paired_success']*100,2), flush=True)


if __name__ == '__main__':
    main()
