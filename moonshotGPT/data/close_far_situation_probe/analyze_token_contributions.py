#!/usr/bin/env python3
"""Analyze saved scores with and without punctuation; no model inference."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'runs/research/bos_aligned_proto'
DATASETS = {
    'definition': RUNS / 'close_far_definition_probe/qwen3_359m_step19500_v1',
    'situations': RUNS / 'close_far_situation_probe/qwen3_359m_step19500_v1',
    'context_diagnostics': RUNS / 'close_far_situation_probe/qwen3_359m_step19500_context_diagnostics_v1',
}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    model = json.loads((DATASETS['definition'] / 'summary.json').read_text())['model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    judgments, decompositions = [], []
    for dataset, folder in DATASETS.items():
        records = [json.loads(line) for line in (folder / 'item_scores.jsonl').read_text().splitlines()]
        records = [r for r in records if r['score_reduction'] == 'mean']
        lookup = {}
        for line in (folder / 'token_scores.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = (r['probe_id'], r['combination']) if dataset == 'definition' else (r['context'], r['target'])
            lookup[key] = r['target_token_log_probs']
        for probe in records:
            group = probe['direction'] if dataset == 'definition' else probe['condition_id']
            positive = probe.get('target_negation_pattern_id', 'positive') == 'positive'
            for c in [1, 2]:
                context = probe[f'Context{c}']
                entries = []
                for t in [1, 2]:
                    target = probe[f'Target{t}']
                    key = (probe['probe_id'], f'S{c}{t}') if dataset == 'definition' else (context, target)
                    scores = lookup[key]
                    prefix_ids = tokenizer.encode(context, add_special_tokens=False)
                    joined_ids = tokenizer.encode(context + ' ' + target, add_special_tokens=False)
                    assert joined_ids[:len(prefix_ids)] == prefix_ids
                    ids = joined_ids[len(prefix_ids):]
                    assert len(ids) == len(scores) and tokenizer.decode([ids[-1]]) == '.'
                    full_mean = sum(scores) / len(scores)
                    assert abs(full_mean - probe[f'S{c}{t}']) < 2e-6
                    words = [tokenizer.decode([token]).strip() for token in ids]
                    candidates = [i for i, word in enumerate(words) if word in {'close', 'far', 'small', 'large', 'short', 'long'}]
                    adjective_index = candidates[0] if positive and len(candidates) == 1 else None
                    entries.append({
                        'target': target, 'tokens': tokenizer.convert_ids_to_tokens(ids),
                        'log_probs': scores, 'full_mean': probe[f'S{c}{t}'],
                        'no_period_mean': sum(scores[:-1]) / (len(scores) - 1),
                        'period_logp': scores[-1],
                        'adjective_index': adjective_index,
                        'adjective_logp': scores[adjective_index] if adjective_index is not None else None,
                        'following_words_sum': sum(scores[adjective_index + 1:-1]) if adjective_index is not None else None,
                    })
                a, b = entries
                sign = 1 if c == 1 else -1
                full_margin = sign * (a['full_mean'] - b['full_mean'])
                no_period_margin = sign * (a['no_period_mean'] - b['no_period_mean'])
                adj_margin = sign * (a['adjective_logp'] - b['adjective_logp']) if positive else None
                row = {
                    'dataset': dataset, 'group': group, 'probe_id': probe['probe_id'],
                    'context_slot': c, 'context': context, 'target1': a['target'], 'target2': b['target'],
                    'full_mean_margin': full_margin, 'no_period_mean_margin': no_period_margin,
                    'full_correct': full_margin > 0, 'no_period_correct': no_period_margin > 0,
                    'no_period_tie': no_period_margin == 0,
                    'changed_preference_without_period': (full_margin > 0) != (no_period_margin > 0),
                    'adjective_margin': adj_margin,
                    'adjective_correct': adj_margin > 0 if adj_margin is not None else None,
                    'period_T1_minus_T2': a['period_logp'] - b['period_logp'],
                    'period_favors_correct_target': sign * (a['period_logp'] - b['period_logp']) > 0,
                    'T1_adjective_logp': a['adjective_logp'], 'T2_adjective_logp': b['adjective_logp'],
                    'T1_following_words_sum': a['following_words_sum'], 'T2_following_words_sum': b['following_words_sum'],
                    'T1_period_logp': a['period_logp'], 'T2_period_logp': b['period_logp'],
                }
                judgments.append(row)
                decompositions.append({'dataset': dataset, 'group': group, 'probe_id': probe['probe_id'], 'context_slot': c, 'context': context, 'targets': entries})
    assert len(judgments) == 1536
    write_csv(out / 'judgments.csv', judgments)
    (out / 'token_breakdowns.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decompositions))
    summaries = []
    groups = defaultdict(list)
    for r in judgments:
        groups[(r['dataset'], r['group'])].append(r)
    for (dataset, group), rows in groups.items():
        n = len(rows)
        probes = defaultdict(list)
        for r in rows:
            probes[r['probe_id']].append(r)
        adj = [r for r in rows if r['adjective_correct'] is not None]
        summaries.append({
            'dataset': dataset, 'group': group, 'judgments': n,
            'full_mean_accuracy': sum(r['full_correct'] for r in rows) / n,
            'no_period_mean_accuracy': sum(r['no_period_correct'] for r in rows) / n,
            'full_paired_success': sum(all(r['full_correct'] for r in p) for p in probes.values()) / len(probes),
            'no_period_paired_success': sum(all(r['no_period_correct'] for r in p) for p in probes.values()) / len(probes),
            'wrong_to_right_without_period': sum(not r['full_correct'] and r['no_period_correct'] for r in rows),
            'right_to_wrong_without_period': sum(r['full_correct'] and not r['no_period_correct'] for r in rows),
            'adjective_only_accuracy': sum(r['adjective_correct'] for r in adj) / len(adj) if adj else None,
            'period_favors_T1_fraction': sum(r['period_T1_minus_T2'] > 0 for r in rows) / n,
            'mean_period_T1_minus_T2': sum(r['period_T1_minus_T2'] for r in rows) / n,
        })
    write_csv(out / 'summary.csv', summaries)
    (out / 'summary.json').write_text(json.dumps({'model': model, 'primary_metric': 'original full-target mean log likelihood', 'diagnostic_metrics': ['mean excluding final period', 'adjective-only log probability for non-negated targets'], 'source_runs': {k: str(v) for k, v in DATASETS.items()}, 'no_model_inference': True, 'groups': summaries}, indent=2) + '\n')
    for row in summaries:
        print(row['dataset'], row['group'], 'full', round(row['full_mean_accuracy']*100,2), 'no period', round(row['no_period_mean_accuracy']*100,2), 'adjective', None if row['adjective_only_accuracy'] is None else round(row['adjective_only_accuracy']*100,2), 'fixed/broken', row['wrong_to_right_without_period'], row['right_to_wrong_without_period'])


if __name__ == '__main__':
    main()
