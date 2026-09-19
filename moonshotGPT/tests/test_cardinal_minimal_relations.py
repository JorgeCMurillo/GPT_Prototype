"""Check short relation updates independently of the renderer's answer logic."""
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'data/cardinal_situation_probe/minimal_relations'
spec = importlib.util.spec_from_file_location('cardinal_minimal', ROOT / 'generate.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture(scope='module')
def dataset():
    return generator.build()


def test_coverage_and_balance(dataset):
    rows, _ = dataset
    assert len(rows) == len({r['probe_id'] for r in rows}) == 46
    assert Counter(r['axis'] for r in rows) == {'north_south': 23, 'east_west': 23}
    assert Counter(r['probe_family'] for r in rows) == {'event': 38, 'control': 4, 'observer_turn': 4}
    assert len({r['case_id'] for r in rows}) == 8
    assert sum(r['persistence_cue'] == 'explicit' for r in rows) == 12


def test_text_and_geometry_imply_gold(dataset):
    rows, _ = dataset
    inverse = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
    for r in rows:
        for i in (1, 2):
            text = r[f'Context{i}']
            initial = text.split('.')[0].split()[2]
            crosses = 'continues past' in text or 'swap positions' in text
            expected = inverse[initial] if crosses else initial
            answer = r[r[f'correct_target_for_context{i}']]
            assert answer == f'A is {expected} of B.'
            assert r[f'context{i}_gold_relation'] == expected
            a0, a1 = (r[f'context{i}_target_{t}'] for t in ('start', 'end'))
            b0, b1 = (r[f'context{i}_reference_{t}'] for t in ('start', 'end'))
            positive = 'north' if r['axis'] == 'north_south' else 'east'
            assert (a1 > b1) == (expected == positive)
            if r['case_id'] == 'toward_without_crossing':
                assert 0 < abs(a1 - b1) < abs(a0 - b0) and b0 == b1
                assert 'stops before reaching B' in text
            if r['case_id'] == 'away':
                assert abs(a1 - b1) > abs(a0 - b0) and b0 == b1
            if r['case_id'] == 'both_equal':
                assert a1 - a0 == b1 - b0 != 0
                assert 'for the same distance' in text
            if r['case_id'] == 'swap': assert (a1, b1) == (b0, a0)
            if r['case_id'] in ('static', 'observer_turn'): assert (a1, b1) == (a0, b0)
            if r['case_id'] == 'target_crosses': assert b0 == b1
            if r['case_id'] == 'reference_crosses': assert a0 == a1
            assert text.endswith('Final positions:')
            assert r[f'Context{i}_word_count'] == len(text.split()) <= 25
            assert 'map' not in text and 'row' not in text and 'column' not in text


def test_minimal_pairs_and_surface_links(dataset):
    rows, links = dataset
    by_id = {r['probe_id']: r for r in rows}
    assert {l['control'] for l in links} == {
        'axis', 'movement_verb', 'preposition', 'persistence_cue', 'observer_turn_invariance'}
    for link in links:
        a, b = (by_id[link[k]] for k in ('base_probe_id', 'variant_probe_id'))
        factor = link['control']
        for i in (1, 2):
            assert a[f'correct_target_for_context{i}'] == b[f'correct_target_for_context{i}']
            for entity in ('target', 'reference'):
                for endpoint in ('start', 'end'):
                    key = f'context{i}_{entity}_{endpoint}'
                    assert a[key] == b[key]
            if factor != 'axis': assert a[f'Target{i}'] == b[f'Target{i}']
            text, other = a[f'Context{i}'], b[f'Context{i}']
            if factor == 'movement_verb':
                assert text.replace(' moves ', ' heads ').replace(' move ', ' head ') == other
            elif factor == 'preposition':
                assert text.replace(' toward ', ' to ') == other
                assert a['case_id'] in ('target_crosses', 'reference_crosses')
                assert 'continues past' in other
            elif factor == 'persistence_cue':
                assert not a['persistence_cue'] == b['persistence_cue']
                prefix = text.removesuffix('Final positions:')
                assert other == prefix + generator.EXPLICIT[a['case_id']] + ' Final positions:'
            elif factor == 'observer_turn_invariance':
                assert a['case_id'] == 'static' and b['case_id'] == 'observer_turn'


def test_implicit_and_explicit_assumptions(dataset):
    rows, _ = dataset
    for r in rows:
        assert r['requires_implicit_persistence'] == (r['persistence_cue'] == 'implicit')
        if r['persistence_cue'] == 'explicit':
            for i in (1, 2): assert generator.EXPLICIT[r['case_id']] in r[f'Context{i}']
        else:
            for i in (1, 2):
                assert 'stays still' not in r[f'Context{i}'] and 'Neither moves' not in r[f'Context{i}']
        if r['case_id'] in ('both_equal', 'swap'):
            assert r['persistence_cue'] == 'not_applicable'


def test_generated_files(dataset):
    rows, links = dataset
    assert [json.loads(line) for line in (ROOT / 'generated/probes.jsonl').read_text().splitlines()] == rows
    with (ROOT / 'generated/variant_matches.csv').open() as stream:
        assert list(csv.DictReader(stream)) == links


def test_invalid_case_and_side():
    with pytest.raises(AssertionError): generator.states('unknown', 1)
    with pytest.raises(AssertionError): generator.states('static', 0)
