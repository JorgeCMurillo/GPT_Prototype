#!/usr/bin/env python3
"""Short cardinal relation updates with matched wording and persistence cues."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AXES = {'north_south': ('north', 'south'), 'east_west': ('east', 'west')}
CASES = {
    'static': ('static_placement', 'control'),
    'target_crosses': ('target_crosses', 'event'),
    'reference_crosses': ('reference_crosses', 'event'),
    'toward_without_crossing': ('target_moves_without_crossing', 'event'),
    'away': ('target_moves_without_crossing', 'event'),
    'both_equal': ('both_move', 'event'),
    'swap': ('both_move', 'event'),
    'observer_turn': ('observer_turn', 'observer_turn'),
}
EXPLICIT = {
    'static': 'Neither moves.',
    'target_crosses': 'B stays still.',
    'reference_crosses': 'A stays still.',
    'toward_without_crossing': 'B stays still.',
    'away': 'B stays still.',
    'observer_turn': 'A stays still.',
}


def states(case, side):
    """Representative ordinal geometry, not coordinates stated in the text."""
    assert case in CASES and side in (-1, 1)
    a, b = 2 * side, 0
    if case == 'target_crosses': end = (-2 * side, b)
    elif case == 'reference_crosses': end = (a, 4 * side)
    elif case == 'toward_without_crossing': end = (side, b)
    elif case == 'away': end = (3 * side, b)
    elif case == 'both_equal': end = (a - side, b - side)
    elif case == 'swap': end = (b, a)
    else: end = (a, b)
    return (a, b), end


def variants(case):
    verbs = ('moves', 'heads') if case in (
        'target_crosses', 'reference_crosses', 'toward_without_crossing', 'away', 'both_equal'
    ) else ('not_applicable',)
    preps = ('toward', 'to') if case in ('target_crosses', 'reference_crosses') else (
        'toward' if case == 'toward_without_crossing' else 'not_applicable',)
    base = [(verb, prep, 'implicit' if case in EXPLICIT else 'not_applicable')
            for verb in verbs for prep in preps]
    if case in EXPLICIT:
        base.append((verbs[0], preps[0], 'explicit'))
    return base


def render(case, initial, opposite, verb, prep, persistence):
    text = f'A is {initial} of B.'
    action = ''
    if case in ('target_crosses', 'reference_crosses'):
        mover, other = ('A', 'B') if case == 'target_crosses' else ('B', 'A')
        action = f'{mover} {verb} straight {prep} {other} and continues past {other}.'
    elif case == 'toward_without_crossing':
        action = f'A {verb} toward B but stops before reaching B.'
    elif case == 'away':
        action = f'A {verb} farther {initial}.'
    elif case == 'both_equal':
        plural = {'moves': 'move', 'heads': 'head'}[verb]
        action = f'Both {plural} {opposite} for the same distance.'
    elif case == 'swap': action = 'A and B swap positions.'
    elif case == 'observer_turn': action = 'B turns around without moving.'
    parts = [text, action, EXPLICIT[case] if persistence == 'explicit' else '', 'Final positions:']
    return ' '.join(part for part in parts if part)


def make_row(axis, case, verb, prep, persistence):
    positive, negative = AXES[axis]
    family, block = CASES[case]
    row = {
        'probe_id': '__'.join((axis, case, verb, prep, persistence)),
        'probe_version': '1.0', 'probe_family': block,
        'axis': axis, 'case_id': case, 'scene_cluster_id': f'{axis}__{case}',
        'event_family': family, 'event_subtype': case,
        'evidence_format': 'direct_initial_relation', 'length_band': 'minimal',
        'reference_frame': 'absolute_cardinal', 'geometry_precision': 'representative_ordinal',
        'initial_relation_explicit': True, 'final_relation_explicit': case == 'static',
        'movement_verb': verb, 'preposition': prep, 'persistence_cue': persistence,
        'requires_implicit_persistence': persistence == 'implicit',
        'answer_bridge': 'Final positions:',
        'Target1': f'A is {positive} of B.', 'Target2': f'A is {negative} of B.',
        'target1_relation_word': positive, 'target2_relation_word': negative,
    }
    for i, side in ((1, -1), (2, 1)):
        initial = negative if side == -1 else positive
        opposite = positive if side == -1 else negative
        start, end = states(case, side)
        gold = positive if end[0] > end[1] else negative
        row[f'Context{i}'] = render(case, initial, opposite, verb, prep, persistence)
        row[f'context{i}_initial_relation'] = initial
        row[f'context{i}_gold_relation'] = gold
        row[f'context{i}_order_reversed'] = initial != gold
        row[f'correct_target_for_context{i}'] = 'Target1' if gold == positive else 'Target2'
        for j, entity in enumerate(('target', 'reference')):
            row[f'context{i}_{entity}_start'] = start[j]
            row[f'context{i}_{entity}_end'] = end[j]
    for field in ('Context1', 'Context2', 'Target1', 'Target2'):
        row[field + '_word_count'] = len(row[field].split())
    return row


def build():
    rows = [make_row(axis, case, *v) for axis in AXES for case in CASES for v in variants(case)]
    lookup = {(r['axis'], r['case_id'], r['movement_verb'], r['preposition'], r['persistence_cue']): r
              for r in rows}
    links = []
    for key, row in lookup.items():
        axis, case, verb, prep, cue = key
        candidates = []
        if verb == 'moves': candidates.append(('movement_verb', (axis, case, 'heads', prep, cue)))
        if prep == 'toward': candidates.append(('preposition', (axis, case, verb, 'to', cue)))
        if cue == 'implicit': candidates.append(('persistence_cue', (axis, case, verb, prep, 'explicit')))
        if axis == 'north_south': candidates.append(('axis', ('east_west', case, verb, prep, cue)))
        if case == 'static':
            candidates.append(('observer_turn_invariance', (axis, 'observer_turn', verb, prep, cue)))
        for factor, other_key in candidates:
            if other_key not in lookup: continue
            other = lookup[other_key]
            if factor != 'axis':
                assert all(row[f'Target{i}'] == other[f'Target{i}'] for i in (1, 2))
            assert all(row[f'correct_target_for_context{i}'] == other[f'correct_target_for_context{i}'] for i in (1, 2))
            links.append({'control': factor, 'base_probe_id': row['probe_id'], 'variant_probe_id': other['probe_id']})
    assert len(rows) == len(lookup) == 46
    return rows, links


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'generated')
    args = parser.parse_args()
    rows, links = build()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'probes.csv', rows)
    write_csv(out / 'variant_matches.csv', links)
    (out / 'probes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = ['# Minimal relation-first cardinal scenes', '',
              'Gold fields are authoritative: C1 does not always support Target1.', '',
              'Implicit variants assume unmentioned positions persist. Coordinates are representative, not stated distances.', '']
    for row in rows:
        report.extend([f"## {row['probe_id']}", ''])
        for field in ('Context1', 'Context2', 'Target1', 'Target2', 'correct_target_for_context1', 'correct_target_for_context2'):
            report.extend([f'{field}: {row[field]}', ''])
    (out / 'review_examples.md').write_text('\n'.join(report))
    lengths = [r[f'Context{i}_word_count'] for r in rows for i in (1, 2)]
    manifest = {
        'probe_name': 'cardinal_minimal_initial_relations', 'version': '1.0',
        'paired_rows': len(rows), 'context_judgments': 2 * len(rows),
        'cases_per_axis': len(CASES), 'by_axis': dict(Counter(r['axis'] for r in rows)),
        'by_block': dict(Counter(r['probe_family'] for r in rows)),
        'by_persistence_cue': dict(Counter(r['persistence_cue'] for r in rows)),
        'variant_matches': len(links), 'mean_context_words': sum(lengths) / len(lengths),
        'min_context_words': min(lengths), 'max_context_words': max(lengths),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluation_status': 'not_evaluated',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
