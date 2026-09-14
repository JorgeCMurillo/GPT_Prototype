#!/usr/bin/env python3
"""Create matched context-expansion and body-reference diagnostics."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = ('Context1', 'Context2', 'Target1', 'Target2')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config_path = ROOT / 'components.json'
    config = json.loads(config_path.read_text())
    original_path = ROOT.parent / 'generated/probes.jsonl'
    source_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    original_config = json.loads((ROOT.parent / 'components.json').read_text())
    original_scenes = {s['id']: s for s in original_config['scenarios']}
    scenes = {s['id']: s for s in config['scenarios']}
    baselines = [r for r in original if r['condition_id'] == 'baseline']
    assert len(baselines) == 52
    rows = []
    for base in baselines:
        scene = scenes[base['scenario_id']]
        values = {key: base[key] for key in ('name', 'object', 'anchor_object')}
        reference_forms = ['placement', 'person_direct'] if scene['body_reference'] else ['placement']
        for reference_form in reference_forms:
            close_template = config['person_direct_close'] if reference_form == 'person_direct' else scene['close']
            close_location, far_location = close_template.format(**values), scene['far'].format(**values)
            for form in config['context_forms']:
                prefix = ''
                if form['id'] == 'setting':
                    prefix = scene['setting_prefix'].format(**values)
                elif form['id'] == 'endpoint':
                    prefix = original_scenes[scene['id']]['setup'].format(**values)
                row = {k: base[k] for k in ('matched_group_id', 'scenario_id', 'scenario_family', 'setting_id', 'setting', 'entity_type', 'name_id', 'name', 'object_id', 'object', 'anchor_object_id', 'anchor_object', 'Domain', 'ConceptA', 'ConceptB', 'Target1', 'Target2', 'reference_order_id', 'target_negation_pattern_id', 'distance_standard_id')}
                row.update({
                    'probe_id': f"context_diagnostic__{base['matched_group_id']}__{reference_form}__{form['id']}",
                    'probe_version': config['version'], 'probe_type': config['probe_name'],
                    'original_baseline_probe_id': base['probe_id'],
                    'context_form_id': form['id'], 'reference_form_id': reference_form,
                    'condition_id': reference_form + '__' + form['id'],
                    'body_comparison_available': bool(scene['body_reference']),
                    'body_reference_type': scene['body_reference_type'] if reference_form == 'placement' else None,
                    'body_reference': scene['body_reference'] if reference_form == 'placement' else None,
                    'evidence_type': 'physical_arrangement' if reference_form == 'placement' else 'proximity_word',
                    'ContextType': 'indirect' if reference_form == 'placement' else 'mixed',
                    'ContextDiff': 'location phrase contrast', 'TargetDiff': 'concept swap',
                    'prefix_text': prefix, 'close_location_sentence': close_location, 'far_location_sentence': far_location,
                    'Context1': (prefix + ' ' if prefix else '') + close_location,
                    'Context2': (prefix + ' ' if prefix else '') + far_location,
                    'correct_target_for_context1': 'Target1', 'correct_target_for_context2': 'Target2',
                    'close_evidence_type': 'proximity_word' if reference_form == 'person_direct' else 'physical_arrangement',
                    'far_evidence_type': 'physical_arrangement',
                })
                for field in TEXT_FIELDS:
                    row[field + '_word_count'] = len(re.findall(r'\b\w+\b', row[field]))
                    row[field + '_char_count'] = len(row[field])
                row['probe_row_index'] = len(rows)
                rows.append(row)
    assert len(rows) == 228 and len({r['probe_id'] for r in rows}) == 228
    assert len({tuple(r[k] for k in TEXT_FIELDS) for r in rows}) == 228
    assert Counter(r['context_form_id'] for r in rows) == {'compact': 76, 'setting': 76, 'endpoint': 76}
    for row in rows:
        for field in TEXT_FIELDS:
            assert '{' not in row[field] and 'None' not in row[field] and row[field].endswith('.')
    indexed = {(r['matched_group_id'], r['reference_form_id'], r['context_form_id']): r for r in rows}
    matches = []
    for row in rows:
        compact = indexed[(row['matched_group_id'], row['reference_form_id'], 'compact')]
        assert row['Target1'] == compact['Target1'] and row['Target2'] == compact['Target2']
        assert row['close_location_sentence'] == compact['Context1']
        assert row['far_location_sentence'] == compact['Context2']
        if row['context_form_id'] != 'compact':
            assert row['Context1'] == row['prefix_text'] + ' ' + compact['Context1']
            assert row['Context2'] == row['prefix_text'] + ' ' + compact['Context2']
            matches.append({'control': 'context_expansion', 'reference_probe_id': compact['probe_id'], 'variant_probe_id': row['probe_id'], 'context_form_id': row['context_form_id'], 'match_type': 'identical_location_sentences_and_targets'})
        if row['reference_form_id'] == 'person_direct':
            placement = indexed[(row['matched_group_id'], 'placement', row['context_form_id'])]
            assert row['Context2'] == placement['Context2']
            assert row['Target1'] == placement['Target1'] and row['Target2'] == placement['Target2']
            assert placement['close_location_sentence'].replace("'s " + placement['body_reference'], '') == row['close_location_sentence']
            matches.append({'control': 'body_reference', 'reference_probe_id': placement['probe_id'], 'variant_probe_id': row['probe_id'], 'context_form_id': row['context_form_id'], 'match_type': 'identical_far_context_and_targets'})
    # Check full name/object balance within each scenario and diagnostic condition.
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scenario_id'], row['condition_id'])].append(row)
    for group in groups.values():
        assert set(Counter(r['object_id'] for r in group).values()) == {len(group) // 2}
        if group[0]['entity_type'] == 'person_object':
            assert len(group) == 8 and set(Counter(r['name_id'] for r in group).values()) == {2}
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    (out / 'probes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_csv(out / 'probes.csv', rows)
    write_csv(out / 'diagnostic_matches.csv', matches)
    write_csv(out / 'original_matches.csv', [{'probe_id': r['probe_id'], 'original_baseline_probe_id': r['original_baseline_probe_id'], 'comparison_note': 'Original wording may differ; use diagnostic_matches for controlled effects.'} for r in rows])
    for key in ('context_forms', 'reference_forms', 'scenarios'):
        write_csv(out / (key + '.csv'), config[key])
    write_csv(out / 'names.csv', original_config['names'])
    write_csv(out / 'objects.csv', original_config['objects'])
    examples = {}
    for row in rows:
        examples.setdefault((row['scenario_id'], row['condition_id']), row)
    write_csv(out / 'review_examples.csv', list(examples.values()))
    manifest = {
        'probe_name': config['probe_name'], 'version': config['version'],
        'paired_probes': len(rows), 'context_level_judgments': 2 * len(rows),
        'base_entity_assignments': len(baselines),
        'by_context_form': dict(Counter(r['context_form_id'] for r in rows)),
        'by_reference_form': dict(Counter(r['reference_form_id'] for r in rows)),
        'diagnostic_matches': len(matches),
        'original_probe_sha256': source_hash,
        'components_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'default_score_reduction': 'mean', 'structural_validation': 'passed',
        'model_evaluation_performed': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == source_hash
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
