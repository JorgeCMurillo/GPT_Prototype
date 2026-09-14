#!/usr/bin/env python3
"""Generate reproducible numerical distance and matched movement diagnostics."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = '/SSD2/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/qwen3_liger_muon_steps19500_ws6_seed42_restart_20260816_120352/runs/qwen3_liger_muon/babygpt_fineweb_stream_mbs6_T1024_qwen3_d1024_h16_L24_tok516096_efftok516096_ws6_gas14_seed42_steps19500/ckpt_final_step0019500'


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER, help='Local tokenizer for token counts; defaults to the Qwen3 19.5k checkpoint.')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    path = ROOT / 'components.json'
    config = json.loads(path.read_text())
    rng = random.Random(config['random_seed'])
    starts = rng.sample(range(config['initial_distance_min'], config['initial_distance_max'] + 1), config['numeric_cases'])
    numbers = []
    for i, start in enumerate(starts):
        maximum = min(start - 1, config['maximum_final_distance'] - start, config['maximum_movement'])
        movement = rng.randint(1, maximum)
        numbers.append({
            'id': f'num_{i:02d}', 'initial_distance': start, 'movement_distance': movement,
            'closer_final_distance': start - movement,
            'farther_final_distance': start + movement,
            'unchanged_final_distance': start,
        })
    rows, matches = [], []
    for unit, name, obj, num in product(config['units'], config['names'], config['objects'], numbers):
        numeric_entity_id = f"{name['id']}__{obj['id']}__{num['id']}"
        base_id = numeric_entity_id if unit['id'] == 'unit_metres' else numeric_entity_id + '__' + unit['id']
        target_fields = {outcome['target_key']: outcome['target_template'].format(name=name['text'], object=obj['text']) for outcome in config['outcomes']}
        target_keys = {o['id']: o['target_key'] for o in config['outcomes']}
        indexed = {}
        for condition in config['conditions']:
            for template, outcome in product(condition['templates'], condition['outcomes']):
                initial, final = num['initial_distance'], num[outcome + '_final_distance']
                # Orientation-only descriptions must not be repeated for each unused movement magnitude.
                direction = 'toward' if outcome == 'closer' else 'away from' if outcome == 'farther' else None
                band = template['length_band']
                original_id = f"{base_id}__{condition['id']}__{outcome}"
                variant_suffix = '' if band == 'standard' else '__' + band
                context = template['text'].format(
                    name=name['text'], object=obj['text'], initial=initial, final=final,
                    movement=num['movement_distance'], direction=direction,
                    initial_unit=unit['singular'] if initial == 1 else unit['plural'],
                    final_unit=unit['singular'] if final == 1 else unit['plural'],
                    movement_unit=unit['singular'] if num['movement_distance'] == 1 else unit['plural'],
                )
                row = {
                    'probe_id': original_id + variant_suffix,
                    'probe_version': config['version'], 'probe_type': config['probe_name'],
                    'matched_group_id': base_id + variant_suffix, 'condition_id': condition['id'],
                    'length_match_id': original_id, 'template_id': template['id'],
                    'length_band': band, 'evidence_type': condition['evidence_type'],
                    'outcome': outcome, 'correct_target': target_keys[outcome],
                    'name_id': name['id'], 'name': name['text'],
                    'object_id': obj['id'], 'object': obj['text'],
                    'numeric_case_id': num['id'], 'unit': unit['unit'],
                    'unit_id': unit['id'], 'unit_notation': unit['notation'],
                    'unit_match_id': f"{numeric_entity_id}__{condition['id']}__{outcome}" + variant_suffix,
                    'initial_distance': initial, 'final_distance': final,
                    'signed_distance_change': final - initial,
                    'movement_distance': num['movement_distance'] if outcome != 'unchanged' else 0,
                    'object_stationary': True, 'passes_object': False,
                    'reference_order_id': 'person_relative_to_object',
                    'wording_id': 'closer_farther_same_distance',
                    'Context': context, **target_fields,
                }
                for field in ['Context', 'Target1', 'Target2', 'Target3']:
                    row[field + '_word_count'] = len(re.findall(r'\b\w+\b', row[field]))
                    row[field + '_char_count'] = len(row[field])
                    row[field + '_token_count'] = len(tokenizer.encode(row[field], add_special_tokens=False))
                    if field != 'Context':
                        # Match the evaluator's context-space-target token boundary.
                        joined = tokenizer.encode(context + ' ' + row[field], add_special_tokens=False)
                        prefix = tokenizer.encode(context, add_special_tokens=False)
                        assert joined[:len(prefix)] == prefix
                        row[field + '_conditional_token_count'] = len(joined) - len(prefix)
                row['probe_row_index'] = len(rows)
                rows.append(row)
                indexed[(condition['id'], outcome, band)] = row
        for outcome, band in product(target_keys, ['compact', 'standard', 'expanded']):
            explicit = indexed[('explicit_distance', outcome, band)]
            situation = indexed[('orientation_control' if outcome == 'unchanged' else 'movement_description', outcome, band)]
            assert all(explicit[k] == situation[k] for k in ['initial_distance', 'final_distance', 'correct_target', 'Target1', 'Target2', 'Target3'])
            matches.append({
                'matched_group_id': explicit['matched_group_id'], 'outcome': outcome, 'unit_id': unit['id'],
                'length_band': band,
                'explicit_probe_id': explicit['probe_id'], 'situation_probe_id': situation['probe_id'],
                'initial_distance': explicit['initial_distance'], 'final_distance': explicit['final_distance'],
                'match_type': 'same_entities_endpoints_and_targets',
            })
    assert len(numbers) == 12 and len({n['initial_distance'] for n in numbers}) == 12
    assert all(0 < n['closer_final_distance'] < n['initial_distance'] < n['farther_final_distance'] <= 99 for n in numbers)
    assert len(rows) == 7776 and len(matches) == 3888
    assert len({r['probe_id'] for r in rows}) == 7776
    assert len({r['Context'] for r in rows}) == 7776
    assert Counter(r['outcome'] for r in rows) == {'closer': 2592, 'farther': 2592, 'unchanged': 2592}
    assert set(Counter(r['unit_id'] for r in rows).values()) == {2592}
    assert Counter(r['length_band'] for r in rows) == {'compact': 2592, 'standard': 2592, 'expanded': 2592}
    for band in ['compact', 'standard', 'expanded']:
        assert Counter(r['outcome'] for r in rows if r['length_band'] == band) == {'closer': 864, 'farther': 864, 'unchanged': 864}
    for row in rows:
        delta = row['signed_distance_change']
        assert row['outcome'] == ('closer' if delta < 0 else 'farther' if delta > 0 else 'unchanged')
        assert '{' not in row['Context'] and 'None' not in row['Context']
        if row['outcome'] == 'closer':
            assert row['movement_distance'] < row['initial_distance']
        if row['condition_id'] == 'orientation_control':
            assert row['initial_distance'] == row['final_distance'] and row['movement_distance'] == 0
    assert set(Counter(r['name_id'] for r in rows).values()) == {1944}
    assert set(Counter(r['object_id'] for r in rows).values()) == {2592}
    length_groups = defaultdict(dict)
    for row in rows:
        length_groups[row['length_match_id']][row['length_band']] = row
    length_matches = []
    for match_id, group in length_groups.items():
        assert set(group) == {'compact', 'standard', 'expanded'}
        reference = group['standard']
        for count in ['Context_word_count', 'Context_token_count']:
            assert group['compact'][count] < reference[count] < group['expanded'][count]
        for band in ['compact', 'expanded']:
            variant = group[band]
            assert all(reference[k] == variant[k] for k in [
                'initial_distance', 'final_distance', 'movement_distance', 'unit_id',
                'name_id', 'object_id', 'correct_target', 'Target1', 'Target2', 'Target3',
                'condition_id', 'evidence_type', 'outcome',
            ])
            length_matches.append({
                'length_match_id': match_id, 'reference_probe_id': reference['probe_id'],
                'variant_probe_id': variant['probe_id'], 'reference_length_band': 'standard',
                'variant_length_band': band, 'condition_id': reference['condition_id'],
                'evidence_type': reference['evidence_type'],
                'match_type': 'same_facts_targets_and_evidence_type_varied_wording_and_length',
            })
    assert len(length_groups) == 2592 and len(length_matches) == 5184
    metre_rows = {r['unit_match_id']: r for r in rows if r['unit_id'] == 'unit_metres'}
    unit_matches = []
    unit_lookup = {u['id']: u for u in config['units']}
    for row in rows:
        if row['unit_id'] == 'unit_metres':
            continue
        reference = metre_rows[row['unit_match_id']]
        unit = unit_lookup[row['unit_id']]
        normalized = re.sub(
            r'(\d+) (?:' + re.escape(unit['singular']) + '|' + re.escape(unit['plural']) + r')\b',
            lambda match: match[1] + (' metre' if int(match[1]) == 1 else ' metres'),
            row['Context'],
        )
        assert normalized == reference['Context']
        assert all(row[k] == reference[k] for k in ['initial_distance', 'final_distance', 'correct_target', 'Target1', 'Target2', 'Target3'])
        unit_matches.append({'unit_match_id': row['unit_match_id'], 'reference_probe_id': reference['probe_id'], 'variant_probe_id': row['probe_id'], 'reference_unit_id': reference['unit_id'], 'variant_unit_id': row['unit_id'], 'match_type': 'same_numbers_and_outcome_not_converted_physical_distance'})
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    (out / 'probes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_csv(out / 'probes.csv', rows)
    write_csv(out / 'matched_examples.csv', matches)
    write_csv(out / 'unit_matches.csv', unit_matches)
    write_csv(out / 'length_matches.csv', length_matches)
    write_csv(out / 'numeric_cases.csv', numbers)
    for key in ['names', 'objects', 'outcomes', 'conditions', 'units']:
        write_csv(out / (key + '.csv'), config[key])
    write_csv(out / 'templates.csv', [
        {'condition_id': condition['id'], 'evidence_type': condition['evidence_type'], **template}
        for condition in config['conditions'] for template in condition['templates']
    ])
    length_summary = []
    for condition, band in product(config['conditions'], ['compact', 'standard', 'expanded']):
        subset = [r for r in rows if r['condition_id'] == condition['id'] and r['length_band'] == band]
        length_summary.append({
            'condition_id': condition['id'], 'length_band': band, 'context_items': len(subset),
            **{f'{kind}_{stat}': fn(r[f'Context_{kind}_count'] for r in subset)
               for kind in ['word', 'token'] for stat, fn in [('min', min), ('max', max)]},
        })
    write_csv(out / 'length_summary.csv', length_summary)
    review = [r for r in rows if r['name_id'] == 'name_0' and r['object_id'] == 'obj_cone']
    write_csv(out / 'review_examples.csv', review)
    manifest = {
        'probe_name': config['probe_name'], 'version': config['version'],
        'random_seed': config['random_seed'], 'numeric_cases': len(numbers),
        'base_entity_numeric_assignments_per_unit': 144,
        'base_entity_numeric_unit_assignments': 432, 'context_items': len(rows),
        'targets_per_context': 3, 'conditional_scores_required': 3 * len(rows),
        'by_condition': dict(Counter(r['condition_id'] for r in rows)),
        'by_outcome': dict(Counter(r['outcome'] for r in rows)),
        'by_unit': dict(Counter(r['unit_id'] for r in rows)),
        'by_length_band': dict(Counter(r['length_band'] for r in rows)),
        'by_evidence_type': dict(Counter(r['evidence_type'] for r in rows)),
        'context_templates': sum(len(c['templates']) for c in config['conditions']),
        'length_match_groups': len(length_groups), 'length_matches': len(length_matches),
        'tokenizer_source': args.tokenizer,
        'token_counts_include_special_tokens': False,
        'standalone_token_counts_and_conditional_target_counts': True,
        'unit_matches': len(unit_matches),
        'matched_comparisons': len(matches), 'default_score_reduction': 'mean',
        'components_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'structural_and_arithmetic_validation': 'passed', 'model_evaluation_performed': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    print('Numerical cases:', numbers)


if __name__ == '__main__':
    main()
