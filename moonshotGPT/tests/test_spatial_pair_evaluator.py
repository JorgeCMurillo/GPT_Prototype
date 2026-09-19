import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_spatial_pairs.py'
spec = importlib.util.spec_from_file_location('spatial_pair_eval', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_explicit_gold_same_answer_and_ties():
    row = {'Context1': 'c1', 'Context2': 'c2', 'Target1': 'a', 'Target2': 'b',
           'correct_target_for_context1': 'Target2', 'correct_target_for_context2': 'Target2'}
    scores = {(c, t): {'mean': -2 if t == 'a' else -1, 'sum': -2} for c in ('c1', 'c2') for t in ('a', 'b')}
    result = module.decisions(row, scores)
    assert result['accuracy'] == 1 and result['both_correct']
    assert result['sum_accuracy'] == 0 and result['sum_ties'] == 2
    row['correct_target_for_context2'] = 'Target1'
    assert module.decisions(row, scores)['accuracy'] == .5
    row['correct_target_for_context1'] = 'invalid'
    with pytest.raises(ValueError): module.decisions(row, scores)


def test_family_format_case_balancing():
    def row(family, fmt, case, value):
        return dict(event_family=family, evidence_format=fmt, case_id=case,
                    **{k: value for k in module.METRICS})
    rows = [row('static', 'numeric', 'a', 1)] * 10 + [row('static', 'numeric', 'b', 0),
            row('static', 'named', 'c', 0), row('cross', 'numeric', 'd', 1)]
    result = module.balanced(rows)
    assert result['families']['static']['mean_accuracy'] == .25
    assert result['mean_accuracy'] == .625
    assert module.balanced([]) is None


def test_both_match_schemas():
    simple = {'control': 'axis', 'base_probe_id': 'a', 'variant_probe_id': 'b'}
    assert module.normalize_link(simple) == simple
    front = {'match_type': 'facing', 'probe_id_a': 'a', 'probe_id_b': 'b', 'expected_gold_change': 'True'}
    result = module.normalize_link(front)
    assert result['base_probe_id'] == 'a' and result['variant_probe_id'] == 'b'
    assert result['control'] == 'facing' and result['expected_gold_change'] == 'True'
