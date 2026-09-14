#!/usr/bin/env python3
"""Check whether matching contexts improve target scores over saved BOS baselines."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PMI_ROOT = REPO / 'runs/research/bos_aligned_proto/close_far_pmi_mean_v1'


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    return {
        'judgments': n,
        'correct_target_improves_count': sum(r['correct_boost'] > 0 for r in rows),
        'incorrect_target_improves_count': sum(r['incorrect_boost'] > 0 for r in rows),
        'correct_target_improves_fraction': sum(r['correct_boost'] > 0 for r in rows) / n,
        'incorrect_target_improves_fraction': sum(r['incorrect_boost'] > 0 for r in rows) / n,
        'both_targets_improve_count': sum(r['correct_boost'] > 0 and r['incorrect_boost'] > 0 for r in rows),
        'only_correct_target_improves_count': sum(r['correct_boost'] > 0 and r['incorrect_boost'] <= 0 for r in rows),
        'mean_correct_boost': sum(r['correct_boost'] for r in rows) / n,
        'mean_incorrect_boost': sum(r['incorrect_boost'] for r in rows) / n,
        'minimum_correct_boost': min(r['correct_boost'] for r in rows),
        'minimum_incorrect_boost': min(r['incorrect_boost'] for r in rows),
        'correct_boost_larger_fraction': sum(r['correct_boost'] > r['incorrect_boost'] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    manifest = json.loads((PMI_ROOT / 'summary.json').read_text())
    lookup = {}
    for dataset, info in manifest['source_runs'].items():
        path = Path(info['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['score_reduction'] == 'mean':
                lookup[(dataset, row['probe_id'])] = row
    judgments = []
    for line in (PMI_ROOT / 'item_scores.jsonl').read_text().splitlines():
        pmi = json.loads(line)
        raw = lookup[(pmi['dataset'], pmi['probe_id'])]
        for c in [1, 2]:
            correct, incorrect = c, 3 - c
            correct_score, incorrect_score = raw[f'S{c}{correct}'], raw[f'S{c}{incorrect}']
            correct_base, incorrect_base = pmi[f'B{correct}_mean'], pmi[f'B{incorrect}_mean']
            correct_boost, incorrect_boost = correct_score - correct_base, incorrect_score - incorrect_base
            pmi_margin = pmi['pmi_close_margin' if c == 1 else 'pmi_far_margin']
            assert abs(correct_boost - incorrect_boost - pmi_margin) < 1e-10
            judgments.append({
                'dataset': pmi['dataset'], 'group': pmi['group'], 'probe_id': pmi['probe_id'],
                'context_slot': c, 'context': raw[f'Context{c}'],
                'correct_target': raw[f'Target{correct}'], 'incorrect_target': raw[f'Target{incorrect}'],
                'correct_baseline_mean': correct_base, 'correct_context_mean': correct_score,
                'correct_boost': correct_boost,
                'incorrect_baseline_mean': incorrect_base, 'incorrect_context_mean': incorrect_score,
                'incorrect_boost': incorrect_boost,
            })
    assert len(judgments) == 1536
    groups = defaultdict(list)
    for r in judgments:
        groups[(r['dataset'], r['group'])].append(r)
    summaries = [{'dataset': d, 'group': g, **summarize(rows)} for (d, g), rows in groups.items()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'judgments.csv', judgments)
    write_csv(out / 'summary.csv', summaries)
    summary = {
        'model': manifest['model'], 'score_reduction': 'mean',
        'test': 'mean_logp(correct target | matching context) > mean_logp(correct target | BOS)',
        'baseline_convention': manifest['baseline_convention'],
        'overall': summarize(judgments), 'groups': summaries,
        'no_model_inference': True,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('OVERALL', json.dumps(summary['overall']))
    for r in summaries:
        print(r['dataset'], r['group'], 'n', r['judgments'], 'correct+', r['correct_target_improves_count'], 'incorrect+', r['incorrect_target_improves_count'], 'min correct', round(r['minimum_correct_boost'],4))
    failures = [r for r in judgments if r['correct_boost'] <= 0]
    (out / 'non_improving_correct_targets.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in failures))
    print('EXAMPLES', json.dumps(failures[:3], indent=2))


if __name__ == '__main__':
    main()
