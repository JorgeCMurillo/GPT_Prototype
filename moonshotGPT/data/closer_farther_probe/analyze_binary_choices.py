#!/usr/bin/env python3
"""Re-score the three requested binary contrasts from saved mean likelihoods."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.spatial_bias_report.generate import generate_one as generate_bias_table

CONTRASTS = {
    'closer_vs_farther': ('Target1', 'Target2'),
    'closer_vs_unchanged': ('Target1', 'Target3'),
    'farther_vs_unchanged': ('Target2', 'Target3'),
}
LABELS = {'Target1': 'closer', 'Target2': 'farther', 'Target3': 'unchanged'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    counts = Counter(r['predicted_outcome'] for r in rows)
    return {
        'n_contexts': len(rows), 'n_correct': sum(r['correct'] for r in rows),
        'accuracy': sum(r['correct'] for r in rows) / len(rows),
        'mean_gold_margin': sum(r['gold_margin'] for r in rows) / len(rows),
        **{f'predicted_{label}': counts[label] for label in [*LABELS.values(), 'tie']},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    source, out = Path(args.source_dir), Path(args.out_dir)
    source_summary = json.loads((source / 'summary.json').read_text())
    path = source / 'item_scores.jsonl'
    items = [json.loads(s) for s in path.read_text().splitlines()]
    assert source_summary['score_reduction'] == 'mean'
    assert len(items) == 7776 and all(r['score_reduction'] == 'mean' for r in items)
    methods = ['raw'] + (['pmi'] if 'pmi' in source_summary['overall'] else [])
    rows = []
    for contrast, options in CONTRASTS.items():
        eligible = [r for r in items if r['correct_target'] in options]
        assert len(eligible) == 5184
        assert Counter(r['correct_target'] for r in eligible) == {k: 2592 for k in options}
        for item in eligible:
            gold = item['correct_target']
            other = next(k for k in options if k != gold)
            for method in methods:
                scores = {k: item[f'{method}_{k}_mean'] for k in options}
                margin = scores[gold] - scores[other]
                winner = gold if margin > 0 else other if margin < 0 else 'tie'
                rows.append({
                    'contrast': contrast, 'method': method,
                    **{k: item[k] for k in ['probe_id', 'matched_group_id', 'length_match_id', 'condition_id',
                        'length_band', 'template_id', 'evidence_type', 'outcome', 'name_id', 'object_id',
                        'numeric_case_id', 'unit_id', 'Context']},
                    'description_family': 'explicit' if item['condition_id'] == 'explicit_distance' else 'situation',
                    'option_a': options[0], 'option_b': options[1],
                    'target_a': item[options[0]], 'target_b': item[options[1]],
                    'score_a_mean': scores[options[0]], 'score_b_mean': scores[options[1]],
                    'correct_target': gold, 'predicted_target': winner,
                    'predicted_outcome': LABELS.get(winner, winner), 'gold_margin': margin, 'correct': margin > 0,
                })
    grouped = []
    for factors in [(), ('length_band',), ('description_family',), ('description_family', 'length_band'),
                    ('condition_id', 'length_band'), ('outcome',), ('outcome', 'length_band'),
                    ('unit_id',), ('name_id',), ('object_id',), ('numeric_case_id',)]:
        groups = defaultdict(list)
        for row in rows:
            groups[(row['contrast'], row['method'], *(row[k] for k in factors))].append(row)
        for (contrast, method, *key), group in groups.items():
            grouped.append({'contrast': contrast, 'method': method, 'grouping': '+'.join(factors) or 'overall',
                'group': '|'.join(key) or 'all', **summarize(group)})
    pair_groups = defaultdict(list)
    for row in rows:
        pair_groups[row['contrast'], row['method'], row['matched_group_id'], row['description_family']].append(row)
    pairs = []
    for (contrast, method, group_id, family), group in pair_groups.items():
        assert len(group) == 2 and len({r['correct_target'] for r in group}) == 2
        a, b = sorted(group, key=lambda r: r['correct_target'])
        pairs.append({'contrast': contrast, 'method': method, 'matched_group_id': group_id,
            'description_family': family, 'length_band': a['length_band'],
            'probe_a': a['probe_id'], 'probe_b': b['probe_id'],
            'both_correct': a['correct'] and b['correct'],
            'same_prediction': a['predicted_target'] == b['predicted_target'],
            'context_a_correct': a['correct'], 'context_b_correct': b['correct']})
    pair_summary = []
    for contrast in CONTRASTS:
        for method in methods:
            group = [r for r in pairs if r['contrast'] == contrast and r['method'] == method]
            assert len(group) == 2592
            pair_summary.append({'contrast': contrast, 'method': method, 'n_pairs': len(group),
                'n_both_correct': sum(r['both_correct'] for r in group),
                'both_correct_fraction': sum(r['both_correct'] for r in group) / len(group),
                'same_prediction_fraction': sum(r['same_prediction'] for r in group) / len(group)})
    summary = {
        'model': source_summary['model'], 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_dir': str(source.resolve()), 'source_item_scores_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_probe_sha256': source_summary['probe_sha256'], 'score_reduction': 'mean', 'primary_method': 'raw',
        'conditional_scoring_reused': True, 'new_model_inference': False,
        'eligibility': 'Keep only contexts whose gold target belongs to the selected binary contrast; omit third-outcome contexts.',
        'n_unique_source_contexts': len(items), 'n_binary_judgments_per_method': len(rows) // len(methods),
        'note': 'Each original context occurs in two contrasts. Variants are repeated measurements, not independent scenarios.',
        'groups': grouped, 'matched_pair_summary': pair_summary,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'item_scores.csv', rows)
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_csv(out / 'grouped_scores.csv', grouped)
    write_csv(out / 'matched_pair_scores.csv', pairs)
    write_csv(out / 'matched_pair_summary.csv', pair_summary)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Separate binary distance comparisons', '',
        'Qwen3 359M, 19.5k checkpoint. Primary scoring: mean log likelihood over each full target, including punctuation.',
        'Each comparison includes 5,184 contexts, balanced 2,592 per correct answer. Contexts with the excluded third outcome are omitted.',
        'These scores reuse the saved model likelihoods; no new GPU inference was needed. Binary random-choice accuracy is 50%.', '',
        '| Comparison | Compact | Standard | Expanded | Overall | Both matched contexts correct |',
        '|---|---:|---:|---:|---:|---:|']
    def accuracy(contrast, method, grouping, key):
        return next(r['accuracy'] for r in grouped if (r['contrast'], r['method'], r['grouping'], r['group']) == (contrast, method, grouping, key))
    for contrast in CONTRASTS:
        vals = [accuracy(contrast, 'raw', 'length_band', band) for band in ['compact', 'standard', 'expanded']]
        vals.append(accuracy(contrast, 'raw', 'overall', 'all'))
        vals.append(next(r['both_correct_fraction'] for r in pair_summary if r['contrast'] == contrast and r['method'] == 'raw'))
        report.append('| ' + contrast + ' | ' + ' | '.join(f'{v:.2%}' for v in vals) + ' |')
    report += ['', 'Matched pairs contain opposite correct outcomes with the same numeric/entity/unit/length assignment and description family. Movement and orientation form the situation pairs for contrasts involving unchanged distance.', '',
        '| Comparison | Explicit contexts | Situation contexts |', '|---|---:|---:|']
    for contrast in CONTRASTS:
        report.append('| ' + contrast + ' | ' + ' | '.join(f"{accuracy(contrast, 'raw', 'description_family', family):.2%}" for family in ['explicit', 'situation']) + ' |')
    report += ['', 'The explicit and situation families are each balanced within a contrast. Individual movement or orientation subsets are not balanced for contrasts involving unchanged distance.',
        'Length comparisons also vary syntax and reference wording. The original targets have different structures and lengths. These results alone do not isolate conceptual knowledge from target-form preferences.']
    if 'pmi' in methods:
        report += ['', '## Secondary PMI comparison', '', '| Comparison | Accuracy |', '|---|---:|']
        for contrast in CONTRASTS:
            report.append(f"| {contrast} | {accuracy(contrast, 'pmi', 'overall', 'all'):.2%} |")
        report += ['', 'PMI reuses the previous mean-token target-only BOS baseline, with no added leading space. Raw mean likelihood remains primary.']
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    generate_bias_table(out)
    print('\n'.join(report))
    print('\nRaw outcome breakdown:')
    for r in grouped:
        if r['method'] == 'raw' and r['grouping'] == 'outcome':
            print(r)


if __name__ == '__main__':
    main()
