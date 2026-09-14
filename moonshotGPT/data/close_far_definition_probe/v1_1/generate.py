#!/usr/bin/env python3
"""Add explicit naming probes matched to all 18 original reverse probes."""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = ('Context1', 'Context2', 'Target1', 'Target2')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(path, rows):
    path.with_suffix('.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_csv(path.with_suffix('.csv'), rows)


def main():
    source = ROOT.parent / 'generated/probes.jsonl'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = [json.loads(line) for line in source.read_text().splitlines()]
    config = json.loads((ROOT / 'naming_contexts.json').read_text())
    assert len(original) == 72
    naming_by_definition = {c['matched_definition_structure_id']: c for c in config['naming_contexts']}
    new, matches = [], []
    for old in original:
        if old['direction'] != 'definition_to_word':
            continue
        context = naming_by_definition[old['definition_structure_id']]
        row = dict(old)
        row.update({
            'probe_id': f"definition_to_word__{context['id']}__{old['definition_variant_id']}",
            'probe_version': config['version'],
            'task_form': config['task_form'],
            'naming_context_id': context['id'],
            'context_structure_label': context['structure_label'],
            'label_target_id': config['label_target_id'],
            'matched_original_probe_id': old['probe_id'],
            'Context1': context['template'].format(modifier=old['modifier'], adjective=old['close_adjective']),
            'Context2': context['template'].format(modifier=old['modifier'], adjective=old['far_adjective']),
            'Target1': config['targets']['close'],
            'Target2': config['targets']['far'],
            'probe_row_index': len(original) + len(new),
        })
        for field in TEXT_FIELDS:
            row[f'{field}_word_count'] = len(re.findall(r'\b\w+\b', row[field]))
            row[f'{field}_char_count'] = len(row[field])
        assert row['Context1_word_count'] == row['Context2_word_count']
        assert 9 <= row['Context1_word_count'] <= 15
        assert row['Context1'].replace(old['close_adjective'], '<adjective>') == row['Context2'].replace(old['far_adjective'], '<adjective>')
        new.append(row)
        matches.append({
            'definition_variant_id': old['definition_variant_id'],
            'original_probe_id': old['probe_id'], 'naming_probe_id': row['probe_id'],
            'definition_structure_id': old['definition_structure_id'],
            'naming_context_id': context['id'],
            'modifier_id': old['modifier_id'], 'adjective_pair_id': old['adjective_pair_id'],
            'original_context_word_count': old['Context1_word_count'],
            'naming_context_word_count': row['Context1_word_count'],
        })
    assert len(new) == 18
    combined = original + new
    assert len({row['probe_id'] for row in combined}) == 90
    assert len({tuple(row[field] for field in TEXT_FIELDS) for row in combined}) == 90
    assert combined[:72] == original
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    write_dataset(out / 'probes', combined)
    write_dataset(out / 'naming_probes', new)
    write_csv(out / 'naming_contexts.csv', config['naming_contexts'])
    write_csv(out / 'naming_matches.csv', matches)
    manifest = {
        'version': config['version'], 'original_probes': 72, 'new_naming_probes': 18,
        'total_probes': 90, 'total_individual_judgments': 180,
        'original_probe_sha256': source_hash,
        'naming_context_word_counts': {row['naming_context_id']: row['Context1_word_count'] for row in new},
        'validation': 'passed', 'new_probes_evaluated': False,
        'note': 'probe_row_index refers to combined probes.jsonl; naming_probes.jsonl retains those indices.'
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
