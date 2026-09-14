#!/usr/bin/env python3
"""Generate non-numeric binary movement probes and links to numeric variants."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    config_path = ROOT/'components.json'
    config = json.loads(config_path.read_text())
    parent_path = ROOT/config['parent_components']
    parent = json.loads(parent_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    objects = parent['objects'] + config['additional_objects']
    assert len({o['id'] for o in objects}) == len(objects)
    def add_counts(row):
        for field in ['Context', 'Target1', 'Target2']:
            row[field+'_word_count'] = len(re.findall(r'\b\w+\b', row[field]))
            row[field+'_char_count'] = len(row[field])
            row[field+'_token_count'] = len(tokenizer.encode(row[field], add_special_tokens=False))
            if field != 'Context':
                prefix = tokenizer.encode(row['Context'], add_special_tokens=False)
                joined = tokenizer.encode(row['Context']+' '+row[field], add_special_tokens=False)
                assert joined[:len(prefix)] == prefix
                row[field+'_conditional_token_count'] = len(joined)-len(prefix)
    rows, pairs = [], []
    for name, obj, template in product(parent['names'], objects, config['templates']):
        base = f"non_numeric_v1_1__{name['id']}__{obj['id']}"
        pair_id = base + '__' + template['length_band']
        targets = {o['target_key']:o['target_template'].format(name=name['text'],object=obj['text'])
                   for o in parent['outcomes'] if o['id'] in ['closer','farther']}
        pair_rows = []
        for direction in config['directions']:
            context = template['text'].format(name=name['text'],object=obj['text'],direction=direction['text'],
                initial_distance_phrase=config['unspecified_distance_phrase'], movement_distance_phrase=config['unspecified_distance_phrase'])
            assert not re.search(r'\d|\b(?:metres?|ft|inches?)\b',context)
            assert not re.search(r'\b(?:closer|farther|further)\b',context)
            row = {
                'probe_id': pair_id+'__'+direction['outcome'],
                'probe_type':config['probe_name'],'probe_version':config['version'],
                'matched_group_id':pair_id,'length_match_id':base+'__'+direction['outcome'],
                'condition_id':'non_numeric_movement','template_id':template['id'],
                'length_band':template['length_band'],'evidence_type':config['evidence_type'],
                'numeric_information':config['numeric_information'],
                'trajectory_constraint_id':config['trajectory_constraint_id'],
                'object_stationary':True,'passes_object':False,
                'reference_stationarity':'explicit','no_overshoot_status':'explicit',
                'interpretation_basis':'explicit_trajectory_constraints',
                'template_family':'aligned_distance_phrases',
                'name_id':name['id'],'name':name['text'],'object_id':obj['id'],'object':obj['text'],
                'outcome':direction['outcome'],'direction':direction['text'],'correct_target':direction['correct_target'],
                'Context':context,**targets,
            }
            add_counts(row)
            row['probe_row_index']=len(rows)
            rows.append(row)
            pair_rows.append(row)
        a,b=pair_rows
        assert a['Context'].replace('toward','away from')==b['Context']
        pairs.append({'pair_id':pair_id,'length_band':template['length_band'],'template_id':template['id'],
            'name_id':name['id'],'object_id':obj['id'],'evidence_type':config['evidence_type'],
            'closer_probe_id':a['probe_id'],'farther_probe_id':b['probe_id'],
            'Context1':a['Context'],'Context2':b['Context'],**targets})
    assert len(rows)==120 and len(pairs)==60
    assert len({r['Context'] for r in rows})==120 and len({r['probe_id'] for r in rows})==120
    assert Counter(r['outcome'] for r in rows)=={'closer':60,'farther':60}
    assert Counter(r['length_band'] for r in rows)=={'compact':40,'standard':40,'expanded':40}
    assert set(Counter(r['object_id'] for r in rows).values())=={24}
    lengths=defaultdict(dict)
    for row in rows: lengths[row['length_match_id']][row['length_band']]=row
    length_matches=[]
    for key,group in lengths.items():
        for field in ['Context_word_count','Context_token_count']:
            assert group['compact'][field]<group['standard'][field]<group['expanded'][field]
        for band in ['compact','expanded']:
            assert all(group[band][k]==group['standard'][k] for k in ['Target1','Target2','outcome','name_id','object_id'])
            length_matches.append({'length_match_id':key,'reference_probe_id':group['standard']['probe_id'],
                'variant_probe_id':group[band]['probe_id'],'variant_length_band':band})
    numeric_path=ROOT.parent/'generated/probes.jsonl'
    original=[json.loads(s) for s in numeric_path.read_text().splitlines()]
    original_lookup={(r['name_id'],r['object_id'],r['length_band'],r['outcome'],r['numeric_case_id'],r['unit_id']):r
                     for r in original if r['condition_id']=='movement_description'}
    cases=list(csv.DictReader((ROOT.parent/'generated/numeric_cases.csv').open()))
    templates={t['id']:t for t in config['templates']}
    links,numeric=[],[]
    for variant,case,unit in product(rows,cases,parent['units']):
        initial,movement,final = int(case['initial_distance']),int(case['movement_distance']),int(case[variant['outcome']+'_final_distance'])
        assert 0 < movement < initial and (final==initial-movement if variant['outcome']=='closer' else final==initial+movement)
        phrase=lambda value: str(value)+' '+(unit['singular'] if value==1 else unit['plural'])
        context=templates[variant['template_id']]['text'].format(name=variant['name'],object=variant['object'],direction=variant['direction'],
            initial_distance_phrase=phrase(initial),movement_distance_phrase=phrase(movement))
        normalized=re.sub(r'\b\d+ (?:metres?|ft|inches|inch)\b',config['unspecified_distance_phrase'],context)
        assert normalized==variant['Context']
        key=(variant['name_id'],variant['object_id'],variant['length_band'],variant['outcome'],case['id'],unit['id'])
        old=original_lookup.get(key)
        if old:
            assert context==old['Context'][:-1]+' without reaching or passing it.'
            assert all(old[k]==variant[k] for k in ['Target1','Target2','correct_target'])
        row={**variant,'probe_id':variant['probe_id'].replace('non_numeric_v1_1','numeric_aligned_v1_1')+'__'+case['id']+'__'+unit['id'],
            'condition_id':'numeric_aligned_movement','numeric_information':'specific',
            'matched_group_id':variant['matched_group_id'].replace('non_numeric_v1_1','numeric_aligned_v1_1')+'__'+case['id']+'__'+unit['id'],
            'length_match_id':variant['length_match_id'].replace('non_numeric_v1_1','numeric_aligned_v1_1')+'__'+case['id']+'__'+unit['id'],
            'numeric_case_id':case['id'],'unit_id':unit['id'],'initial_distance':initial,'movement_distance':movement,'final_distance':final,
            'Context':context,'probe_row_index':len(numeric)}
        add_counts(row)
        numeric.append(row)
        links.append({'numeric_probe_id':row['probe_id'],'non_numeric_probe_id':variant['probe_id'],
            'original_numeric_probe_id':old['probe_id'] if old else '',
            'numeric_case_id':case['id'],'unit_id':unit['id'],
            'length_band':row['length_band'],'outcome':row['outcome'],
            'match_type':'only_distance_phrases_differ_specific_values_vs_some_distance'})
    assert len(links)==4320 and set(Counter(r['non_numeric_probe_id'] for r in links).values())=={36}
    assert len({r['Context'] for r in numeric})==len({r['probe_id'] for r in numeric})==4320
    numeric_groups=defaultdict(dict)
    for row in numeric: numeric_groups[row['matched_group_id']][row['outcome']]=row
    numeric_pairs=[]
    for key,group in numeric_groups.items():
        a,b=group['closer'],group['farther']
        assert a['Context'].replace('toward','away from')==b['Context']
        numeric_pairs.append({'pair_id':key,'length_band':a['length_band'],'name_id':a['name_id'],'object_id':a['object_id'],
            'numeric_case_id':a['numeric_case_id'],'unit_id':a['unit_id'],
            'closer_probe_id':a['probe_id'],'farther_probe_id':b['probe_id'],
            'Context1':a['Context'],'Context2':b['Context'],'Target1':a['Target1'],'Target2':a['Target2']})
    assert len(numeric_pairs)==2160
    # Minimal forms have no distance slots. Generate each only once per entity/direction,
    # not once per unused numeric case or unit.
    aligned_rows=list(rows)
    aligned_lookup={(r['name_id'],r['object_id'],r['outcome']):r for r in aligned_rows if r['length_band']=='compact'}
    minimal_rows=[]
    minimal_lookup={}
    for name,obj,template in product(parent['names'],objects,config['minimal_templates']):
        pair_id=f"non_numeric_v1_2__{name['id']}__{obj['id']}__{template['id']}"
        pair_rows=[]
        for direction in config['directions']:
            reference=aligned_lookup[name['id'],obj['id'],direction['outcome']]
            context=template['text'].format(name=name['text'],object=obj['text'],direction=direction['text'])
            row={**reference,'probe_id':pair_id+'__'+direction['outcome'],
                'matched_group_id':pair_id,'length_match_id':'',
                'condition_id':template['id'],'template_id':template['id'],
                'length_band':template['length_band'],'template_family':'minimal_movement',
                'reference_stationarity':template['reference_stationarity'],
                'object_stationary':True if template['reference_stationarity']=='explicit' else None,
                'passes_object':None,'no_overshoot_status':'assumed',
                'trajectory_constraint_id':'not_stated',
                'interpretation_basis':'ordinary_movement_reading_with_unstated_assumptions',
                'Context':context,'probe_row_index':len(rows)}
            add_counts(row)
            assert not re.search(r'\d|\b(?:closer|farther|further)\b',context)
            rows.append(row)
            minimal_rows.append(row)
            pair_rows.append(row)
            minimal_lookup[name['id'],obj['id'],direction['outcome'],template['id']]=row
        a,b=pair_rows
        assert a['Context'].replace('toward','away from')==b['Context']
        pairs.append({'pair_id':pair_id,'length_band':template['length_band'],'template_id':template['id'],
            'name_id':name['id'],'object_id':obj['id'],'evidence_type':config['evidence_type'],
            'closer_probe_id':a['probe_id'],'farther_probe_id':b['probe_id'],
            'Context1':a['Context'],'Context2':b['Context'],'Target1':a['Target1'],'Target2':a['Target2']})
    controls=[]
    for row in minimal_rows:
        ref=aligned_lookup[row['name_id'],row['object_id'],row['outcome']]
        assert all(ref[k]==row[k] for k in ['Target1','Target2','correct_target'])
        controls.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':row['probe_id'],
            'control':'aligned_compact_to_minimal','variant_template_id':row['template_id'],
            'match_type':'same_entities_direction_targets_varied_wording_and_explicitness'})
        if row['template_id']=='minimal_stationary':
            ref=minimal_lookup[row['name_id'],row['object_id'],row['outcome'],'minimal_unqualified']
            assert row['Context'].replace('the stationary ','the ')==ref['Context']
            assert row['Context_word_count']==ref['Context_word_count']+1
            controls.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':row['probe_id'],
                'control':'add_stationary','variant_template_id':row['template_id'],
                'match_type':'only_add_stationary_to_reference_object'})
        if row['template_id']=='reference_clause_stationary':
            ref=minimal_lookup[row['name_id'],row['object_id'],row['outcome'],'minimal_stationary']
            controls.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':row['probe_id'],
                'control':'stationary_adjective_vs_relative_clause','variant_template_id':row['template_id'],
                'match_type':'same_stationarity_word_varied_sentence_structure'})
        if row['template_id']=='reference_clause_still':
            ref=minimal_lookup[row['name_id'],row['object_id'],row['outcome'],'reference_clause_stationary']
            assert row['Context'].replace('remained still.','remained stationary.')==ref['Context']
            assert row['Context_word_count']==ref['Context_word_count']
            controls.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':row['probe_id'],
                'control':'stationary_vs_still','variant_template_id':row['template_id'],
                'match_type':'only_swap_stationary_and_still'})
    assert len(minimal_rows)==160 and len(controls)==280
    assert len(rows)==280 and len(pairs)==140
    assert len({r['probe_id'] for r in rows})==len({r['Context'] for r in rows})==280
    assert Counter(r['outcome'] for r in rows)=={'closer':140,'farther':140}
    assert set(Counter(r['length_band'] for r in rows).values())=={40}
    assert set(Counter(r['object_id'] for r in rows).values())=={56}
    assert set(Counter(r['name_id'] for r in rows).values())=={70}
    out=ROOT/'generated'
    out.mkdir(exist_ok=True)
    for file,contents in [('probes',rows),('pairs',pairs),('length_matches',length_matches),('numeric_matches',links),
                          ('minimal_control_matches',controls),
                          ('numeric_aligned_probes',numeric),('numeric_aligned_pairs',numeric_pairs),
                          ('templates',config['templates']+config['minimal_templates']),('names',parent['names']),('objects',objects)]:
        write_csv(out/(file+'.csv'),contents)
    for file,contents in [('probes',rows),('pairs',pairs),('numeric_aligned_probes',numeric),('numeric_aligned_pairs',numeric_pairs)]:
        (out/(file+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in contents))
    write_csv(out/'review_examples.csv',[r for r in rows if r['name_id']=='name_0' and r['object_id']=='obj_cone'])
    by_id={r['probe_id']:r for r in numeric+rows}
    write_csv(out/'numeric_comparison_review.csv',[
        {'numeric_context':by_id[r['numeric_probe_id']]['Context'],'non_numeric_context':by_id[r['non_numeric_probe_id']]['Context'],
         'length_band':r['length_band'],'outcome':r['outcome']}
        for r in links if r['numeric_case_id']=='num_11' and r['unit_id']=='unit_ft'
        and by_id[r['non_numeric_probe_id']]['name_id']=='name_0' and by_id[r['non_numeric_probe_id']]['object_id']=='obj_cone'])
    manifest={
        'probe_name':config['probe_name'],'version':config['version'],'context_items':len(rows),
        'matched_pairs':len(pairs),'targets_per_context':2,'conditional_scores_required':len(rows)*2,
        'by_outcome':dict(Counter(r['outcome'] for r in rows)),
        'by_length_band':dict(Counter(r['length_band'] for r in rows)),
        'context_templates':len(config['templates'])+len(config['minimal_templates']),
        'aligned_non_numeric_contexts':len(aligned_rows),'minimal_contexts':len(minimal_rows),
        'minimal_control_matches':len(controls),
        'length_matches':len(length_matches),'numeric_matches':len(links),
        'numeric_aligned_context_items':len(numeric),'numeric_aligned_pairs':len(numeric_pairs),
        'numeric_aligned_conditional_scores_required':len(numeric)*2,
        'tokenizer_source':args.tokenizer,'default_score_reduction':'mean','model_evaluation_performed':False,
        'components_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'parent_components_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        'numeric_source_sha256':hashlib.sha256(numeric_path.read_bytes()).hexdigest(),
        'validation':'passed',
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    for row in rows:
        if row['name_id']=='name_0' and row['object_id']=='obj_cone':
            print(row['length_band'],row['outcome'],row['Context'],row['Context_word_count'],row['Context_token_count'])


if __name__=='__main__':
    main()
