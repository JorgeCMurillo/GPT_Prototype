#!/usr/bin/env python3
"""Generate matched evidence conditions with target reference order crossed."""
import argparse
import hashlib
import json
import re
import sys
from itertools import product
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',default=DEFAULT_TOKENIZER)
    args=parser.parse_args()
    config=json.loads((ROOT/'components.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    rows=[]
    for name,obj,evidence,order in product(config['names'],config['objects'],config['evidence_conditions'],config['target_orders']):
        fmt={'name':name['text'],'object':obj['text']}
        base=name['id']+'__'+obj['id']
        row={'probe_id':base+'__'+evidence['id']+'__'+order['id'],
            'probe_version':config['version'],'evidence_match_id':base+'__'+order['id'],
            'reference_match_id':base+'__'+evidence['id'],
            'name_id':name['id'],'object_id':obj['id'],'condition_id':evidence['id'],
            'evidence_type':evidence['evidence_type'],'target_order':order['id'],
            'context_entity_order':evidence['context_entity_order'],
            'context_structure':evidence['context_structure'],
            'reverses_context_entity_order':order['id']!=evidence['context_entity_order'],
            'Context1':evidence['close'].format(**fmt),'Context2':evidence['far'].format(**fmt),
            'Target1':order['close'].format(**fmt),'Target2':order['far'].format(**fmt),
            'correct_target_repeats_context':evidence['id']=='direct_label' and order['id']=='object_first'}
        for k in ['Context1','Context2','Target1','Target2']:
            row[k+'_word_count']=len(re.findall(r'\b\w+\b',row[k]))
            row[k+'_token_count']=len(tokenizer.encode(row[k],add_special_tokens=False))
        rows.append(row)
    assert len(rows)==200 and len({r['probe_id'] for r in rows})==200
    assert set(Counter(r['condition_id'] for r in rows).values())=={40}
    evidence_groups,reference_groups=defaultdict(dict),defaultdict(dict)
    for r in rows:
        evidence_groups[r['evidence_match_id']][r['condition_id']]=r
        reference_groups[r['reference_match_id']][r['target_order']]=r
    matches=[]
    for key,g in evidence_groups.items():
        assert len(g)==5
        ref=g['direct_label']
        for condition in ['proximity_paraphrase','physical_placement']:
            variant=g[condition]
            assert all(ref[k]==variant[k] for k in ['Target1','Target2','name_id','object_id','target_order'])
            matches.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':variant['probe_id'],
                'control':'evidence','variant_condition':condition,'target_order':ref['target_order']})
        ref=g['physical_placement']
        for condition in ['physical_placement_fronted','physical_placement_person_subject']:
            variant=g[condition]
            assert all(ref[k]==variant[k] for k in ['Target1','Target2','name_id','object_id','target_order'])
            if condition=='physical_placement_fronted':
                for field in ['Context1','Context2']:
                    assert Counter(re.findall(r'\b\w+\b',ref[field].lower()))==Counter(re.findall(r'\b\w+\b',variant[field].lower()))
            matches.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':variant['probe_id'],
                'control':'context_order','variant_condition':condition,'target_order':ref['target_order']})
    for key,g in reference_groups.items():
        ref,variant=g['person_first'],g['object_first']
        assert all(ref[k]==variant[k] for k in ['Context1','Context2'])
        matches.append({'reference_probe_id':ref['probe_id'],'variant_probe_id':variant['probe_id'],
            'control':'target_reference_order','variant_condition':ref['condition_id'],'target_order':'object_first'})
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    write_csv(out/'probes.csv',rows)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_csv(out/'matches.csv',matches)
    write_csv(out/'review_examples.csv',[r for r in rows if r['name_id']=='name_0' and r['object_id']=='obj_ball'])
    for k in ['names','objects','evidence_conditions','target_orders']:write_csv(out/(k+'.csv'),config[k])
    manifest={'probe_name':config['probe_name'],'version':config['version'],'matched_pairs':len(rows),
        'context_judgments':len(rows)*2,'unique_context_texts':len({r[k] for r in rows for k in ['Context1','Context2']}),
        'conditional_scores_required':len(rows)*4,'matches':len(matches),'default_score_reduction':'mean',
        'tokenizer':args.tokenizer,'components_sha256':hashlib.sha256((ROOT/'components.json').read_bytes()).hexdigest(),'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
