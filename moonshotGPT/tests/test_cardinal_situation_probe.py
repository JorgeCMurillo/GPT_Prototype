"""Independent checks of cardinal scene labels and paired event semantics."""
import importlib.util
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]/'data/cardinal_situation_probe'
spec=importlib.util.spec_from_file_location('cardinal_situations',ROOT/'generate.py')
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture(scope='module')
def dataset():
    config=json.loads((ROOT/'components.json').read_text())
    rows,links=generator.build(config)
    return config,rows,links


def test_decode_endpoints_and_answers(dataset):
    _,rows,_=dataset
    for row in rows:
        if row['probe_family']!='event': continue
        names=row['locations_negative_to_positive'].split(' | ')
        for i in (1,2):
            decoded={}
            for entity in ('target','reference'):
                for point in ('start','end'):
                    label=row[f'context{i}_{entity}_{point}_label']
                    n=int(label.split()[-1])-1 if row['evidence_format']=='numeric' else names.index(label.removeprefix('the '))
                    if row['numeric_label_order']=='decreasing': n=row['position_count']-1-n
                    assert n==row[f'context{i}_{entity}_{point}']
                    assert label in row[f'Context{i}']
                    if point=='end': decoded[entity]=n
            assert decoded['target']!=decoded['reference']
            positive=decoded['target']>decoded['reference']
            expected=('north' if positive else 'south') if row['axis']=='north_south' else ('east' if positive else 'west')
            if row['target_entity_order']=='target_first':
                assert row[f'Target{i}']==f'Marker A is {expected} of marker B.'
            else:
                inverse={'north':'south','south':'north','east':'west','west':'east'}[expected]
                assert row[f'Target{i}']==f'Marker B is {inverse} of marker A.'
            assert row[f'Context{i}'].endswith(row['answer_bridge'])


def test_no_crossing_direction_held_fixed(dataset):
    _,rows,_=dataset
    subset=[r for r in rows if r['event_family']=='target_moves_without_crossing']
    assert len(subset)==288
    for r in subset:
        assert r['context1_target_motion']==r['context2_target_motion']!='still'
        for i in (1,2):
            assert not r[f'context{i}_order_reversed']
            assert r[f'context{i}_reference_motion']=='still'


def test_names_balanced_across_axes_and_lengths(dataset):
    config,rows,links=dataset
    ranks=defaultdict(list)
    for r in rows:
        if r['evidence_format']=='named_locations' and r['length_band']=='compact':
            names=r['locations_negative_to_positive'].split(' | ')
            for i,name in enumerate(names): ranks[name].append(i/(len(names)-1))
    assert set(ranks)==set(config['locations'])
    for values in ranks.values(): assert sum(values)/len(values)==pytest.approx(.5)
    lookup={r['probe_id']:r for r in rows}
    for link in links:
        if link['control']=='length_band':
            a,b=lookup[link['base_probe_id']],lookup[link['variant_probe_id']]
            assert a['locations_negative_to_positive']==b['locations_negative_to_positive']
            assert a['Target1']==b['Target1'] and a['Target2']==b['Target2']


def test_reject_duplicate_orientation_and_invalid_endpoints(dataset):
    config,_,_=dataset
    cases=deepcopy(config['cases'])
    duplicate=deepcopy(cases[2]);duplicate['positive'],duplicate['negative']=duplicate['negative'],duplicate['positive']
    with pytest.raises(AssertionError): generator.validate_cases(cases+[duplicate])
    cases=deepcopy(config['cases']);cases[2]['positive']['target'][1]=1
    with pytest.raises(AssertionError): generator.validate_cases(cases)


def test_entity_order_variants(dataset):
    _,rows,links=dataset
    lookup={r['probe_id']:r for r in rows}
    for r in rows:
        for i in (1,2):
            text=r[f'Context{i}']
            if r['probe_family']=='event':
                assert (text.index('A ')<text.index('B '))==(r['context_entity_order']=='target_first')
            else:
                truth=r[f'context{i}_gold_relation']
                if r['context_entity_order']=='reference_first':
                    truth={'north':'south','south':'north','east':'west','west':'east'}[truth]
                if r['event_family']=='screen_to_cardinal':
                    truth={'north':'above','south':'below','east':'to the right of','west':'to the left of'}[truth]
                assert truth in text.split('After the scene')[0]
    for kind in ('context_entity_order','target_entity_order'):
        matched=[l for l in links if l['control']==kind]
        assert len(matched)==len(rows)//2
        for link in matched:
            a,b=lookup[link['base_probe_id']],lookup[link['variant_probe_id']]
            fields=('Context1','Context2') if kind=='target_entity_order' else ('Target1','Target2')
            assert all(a[f]==b[f] for f in fields)


def test_list_and_numbering_reversals(dataset):
    _,rows,links=dataset
    lookup={r['probe_id']:r for r in rows}
    for r in rows:
        if r['probe_family']!='event': continue
        negative,positive=('south','north') if r['axis']=='north_south' else ('west','east')
        expected=list(range(1,r['position_count']+1)) if r['evidence_format']=='numeric' else r['locations_negative_to_positive'].split(' | ')
        if r['numeric_label_order']=='decreasing': expected.reverse()
        if r['location_list_order']=='positive_to_negative':
            expected.reverse();negative,positive=positive,negative
        for i in (1,2):
            text=r[f'Context{i}'].split('. ')[1]
            rendered=', '.join(map(str,expected)) if r['evidence_format']=='numeric' else ', '.join('the '+x for x in expected)
            assert rendered in text
            assert f'from {negative} to {positive}' in text.lower()
    for link in links:
        if link['control'] not in ('location_list_order','numeric_label_order'): continue
        a,b=lookup[link['base_probe_id']],lookup[link['variant_probe_id']]
        for i in (1,2):
            assert a[f'Target{i}']==b[f'Target{i}']
            for e in ('target','reference'):
                for point in ('start','end'):
                    k=f'context{i}_{e}_{point}'
                    assert a[k]==b[k]
                    if link['control']=='location_list_order': assert a[k+'_label']==b[k+'_label']
