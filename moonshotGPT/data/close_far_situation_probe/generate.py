#!/usr/bin/env python3
"""Generate a balanced core and one-control-at-a-time situation variants."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = ('Context1', 'Context2', 'Target1', 'Target2')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def sentence_start(text):
    return text[0].upper() + text[1:]


def main():
    config_path = ROOT / 'components.json'
    config = json.loads(config_path.read_text())
    objects = {x['id']: x['text'] for x in config['objects']}
    phrases = {x['id']: x for x in config['phrase_pairs']}
    patterns = {x['id']: x for x in config['target_patterns']}
    rows = []
    for scenario in config['scenarios']:
        person = scenario['entity_type'] == 'person_object'
        anchors = config['names'] if person else [{'id': k, 'text': objects[k]} for k in scenario['anchor_object_ids']]
        for anchor, object_id in product(anchors, scenario['object_ids']):
            object_text = objects[object_id]
            reference = anchor['text'] if person else 'the ' + anchor['text']
            values = {'name': anchor['text'] if person else None, 'anchor_object': anchor['text'] if not person else None, 'object': object_text, 'reference': reference}
            base_id = f"{scenario['id']}__{anchor['id']}__{object_id}"
            for condition in config['conditions']:
                phrase = phrases[condition['phrase_pair_id']]
                location = scenario[phrase['id']] if phrase['source'] == 'scenario' else phrase
                setup = scenario['setup'].format(**values)
                c1 = setup + ' ' + location['close'].format(**values)
                c2 = setup + ' ' + location['far'].format(**values)
                subject, target_reference = 'the ' + object_text, reference
                if condition['reference_order_id'] == 'anchor_first':
                    subject, target_reference = target_reference, subject
                target_values = {'subject': sentence_start(subject), 'reference': target_reference}
                pattern = patterns[condition['target_pattern_id']]
                row = {
                    'probe_id': base_id + '__' + condition['id'],
                    'probe_version': config['version'], 'probe_type': 'close_far_situation',
                    'matched_group_id': base_id, 'condition_id': condition['id'],
                    'control': condition['control'], 'scenario_id': scenario['id'],
                    'scenario_family': scenario['family'], 'setting_id': scenario['id'],
                    'setting': scenario['setting'], 'entity_type': scenario['entity_type'],
                    'name_id': anchor['id'] if person else None,
                    'name': anchor['text'] if person else None,
                    'object_id': object_id, 'object': object_text,
                    'anchor_object_id': anchor['id'] if not person else None,
                    'anchor_object': anchor['text'] if not person else None,
                    'phrase_pair_id': phrase['id'],
                    'location_phrase_pair_id': scenario['id'] + '__' + phrase['id'],
                    'evidence_type': phrase['evidence_type'],
                    'semantic_match': phrase['semantic_match'],
                    'reference_order_id': condition['reference_order_id'],
                    'target_negation_pattern_id': condition['target_pattern_id'],
                    'sentence_structure_id': 'anchor_setup_then_object_location',
                    'distance_standard_id': 'ordinary_physical_separation',
                    'correct_target_for_context1': 'Target1',
                    'correct_target_for_context2': 'Target2',
                    'Domain': 'spatial-relations', 'ConceptA': 'close', 'ConceptB': 'far',
                    'ContextType': 'direct' if phrase['evidence_type'] == 'proximity_word' else 'indirect',
                    'ContextDiff': 'location phrase contrast',
                    'TargetDiff': 'concept swap' if condition['target_pattern_id'] in ('positive', 'both_negated') else 'negation',
                    'Context1': c1, 'Context2': c2,
                    'Target1': pattern['close'].format(**target_values),
                    'Target2': pattern['far'].format(**target_values),
                }
                for field in TEXT_FIELDS:
                    row[field + '_word_count'] = len(re.findall(r"\b\w+\b", row[field]))
                    row[field + '_char_count'] = len(row[field])
                rows.append(row)
    for index, row in enumerate(rows):
        row['probe_row_index'] = index
    groups = defaultdict(list)
    for row in rows:
        groups[row['matched_group_id']].append(row)
    assert len(groups) == 52 and len(rows) == 468
    assert len({r['probe_id'] for r in rows}) == len(rows)
    assert len({tuple(r[k] for k in TEXT_FIELDS) for r in rows}) == len(rows)
    comparisons = []
    for group_id, group in groups.items():
        assert len(group) == 9
        baseline = next(r for r in group if r['condition_id'] == 'baseline')
        for row in group:
            for field in TEXT_FIELDS:
                assert row[field].endswith('.') and '{' not in row[field] and 'None' not in row[field]
            assert row['Context1'] != row['Context2'] and row['Target1'] != row['Target2']
            if row['control'] in ('reference_reversal', 'target_negation'):
                assert (row['Context1'], row['Context2']) == (baseline['Context1'], baseline['Context2'])
            if row['control'] in ('location_phrasing', 'evidence_wording'):
                assert (row['Target1'], row['Target2']) == (baseline['Target1'], baseline['Target2'])
            if row['control'] == 'reference_reversal':
                obj = 'the ' + row['object']
                anchor = row['name'] if row['name'] else 'the ' + row['anchor_object']
                assert row['Target1'] == f'{sentence_start(anchor)} is close to {obj}.'
                assert row['Target2'] == f'{sentence_start(anchor)} is far from {obj}.'
            if row['control'] == 'target_negation':
                assert row['Context1'] == baseline['Context1']
                if row['condition_id'] == 'negate_close':
                    assert row['Target1'] == baseline['Target1']
                    assert row['Target2'] == baseline['Target1'].replace(' is close to ', ' is not close to ')
                elif row['condition_id'] == 'negate_far':
                    assert row['Target1'] == baseline['Target2'].replace(' is far from ', ' is not far from ')
                    assert row['Target2'] == baseline['Target2']
                else:
                    assert row['Target1'] == baseline['Target2'].replace(' is far from ', ' is not far from ')
                    assert row['Target2'] == baseline['Target1'].replace(' is close to ', ' is not close to ')
            if row['condition_id'] != 'baseline':
                comparisons.append({'matched_group_id': group_id, 'baseline_probe_id': baseline['probe_id'], 'variant_probe_id': row['probe_id'], 'control': row['control'], 'condition_id': row['condition_id'], 'match_type': 'identical_contexts' if row['control'] in ('reference_reversal', 'target_negation') else 'category_matched_contexts'})
    # Every person appears equally often with each eligible object in every scene.
    for scenario in config['scenarios']:
        subset = [r for r in rows if r['scenario_id'] == scenario['id']]
        expected = 72 if scenario['entity_type'] == 'person_object' else 36
        assert len(subset) == expected
        assert set(Counter(r['object_id'] for r in subset).values()) == {expected // 2}
        assert set(Counter(r['condition_id'] for r in subset).values()) == {expected // 9}
        if scenario['entity_type'] == 'person_object':
            assert set(Counter(r['name_id'] for r in subset).values()) == {18}
    # Explicit substitution links allow matched name/object diagnostics as well.
    substitution_matches = []
    for axis, fixed in [
        ('name_id', ['scenario_id', 'object_id', 'condition_id']),
        ('object_id', ['scenario_id', 'name_id', 'anchor_object_id', 'condition_id']),
        ('anchor_object_id', ['scenario_id', 'object_id', 'condition_id']),
    ]:
        subsets = defaultdict(list)
        for row in rows:
            if row[axis] is not None:
                subsets[tuple(row[k] for k in fixed)].append(row)
        for key, subset in subsets.items():
            ordered = sorted(subset, key=lambda r: r[axis])
            for variant in ordered[1:]:
                substitution_matches.append({'component': axis, 'reference_probe_id': ordered[0]['probe_id'], 'variant_probe_id': variant['probe_id'], 'reference_value': ordered[0][axis], 'variant_value': variant[axis]})
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    (out / 'probes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_csv(out / 'probes.csv', rows)
    write_csv(out / 'control_matches.csv', comparisons)
    write_csv(out / 'substitution_matches.csv', substitution_matches)
    for key in ('names', 'objects', 'phrase_pairs', 'reference_orders', 'target_patterns', 'conditions', 'scenarios'):
        write_csv(out / (key + '.csv'), config[key])
    representatives = {}
    for row in rows:
        representatives.setdefault((row['scenario_id'], row['condition_id']), row)
    write_csv(out / 'review_examples.csv', list(representatives.values()))
    manifest = {
        'probe_name': config['probe_name'], 'version': config['version'],
        'components_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'paired_probes': len(rows), 'context_level_judgments': 2 * len(rows),
        'conditional_likelihood_combinations': 4 * len(rows),
        'base_entity_assignments': len(groups), 'scenario_templates': len(config['scenarios']),
        'by_entity_type': dict(Counter(r['entity_type'] for r in rows)),
        'by_condition': dict(Counter(r['condition_id'] for r in rows)),
        'by_evidence_type': dict(Counter(r['evidence_type'] for r in rows)),
        'control_matches': len(comparisons), 'substitution_matches': len(substitution_matches),
        'default_score_reduction': 'mean', 'model_evaluation_performed': False,
        'structural_validation': 'passed', 'independent_human_semantic_review': 'pending',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
