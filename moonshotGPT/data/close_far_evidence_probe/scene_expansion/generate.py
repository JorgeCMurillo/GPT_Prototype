#!/usr/bin/env python3
"""Expand spatial scenes, retaining matched evidence and reference-order controls."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter,defaultdict
from itertools import product
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',default=DEFAULT_TOKENIZER)
    args=parser.parse_args()
    cfg_path=ROOT/'components.json';cfg=json.loads(cfg_path.read_text())
    parent_path=ROOT/cfg['parent_components'];parent=json.loads(parent_path.read_text())
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    objects={o['id']:o for o in cfg['objects']}
    rows=[]
    def add_row(name,obj,order,scene_id,role,evidence,condition,structure,entity_order,c1,c2,assumption,setting):
        fmt={'name':name['text'],'object':obj['text']}
        base=name['id']+'__'+obj['id']
        row={'probe_id':scene_id+'__'+base+'__'+condition+'__'+order['id'],
            'probe_version':cfg['version'],'scene_id':scene_id,'scene_role':role,
            'name_id':name['id'],'object_id':obj['id'],'condition_id':condition,
            'context_structure':structure,'context_entity_order':entity_order,'target_order':order['id'],
            'reverses_context_entity_order':entity_order!=order['id'],
            'evidence_type':evidence,'setting':setting,'assumption':assumption,
            'include_in_primary_scene_summary':role=='main',
            'gold_requires_unstated_worn_shoes':role=='ambiguity_control',
            'Context1':c1,'Context2':c2,'Target1':order['close'].format(**fmt),'Target2':order['far'].format(**fmt)}
        for field in ['Context1','Context2','Target1','Target2']:
            row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
            row[field+'_token_count']=len(tokenizer.encode(row[field],add_special_tokens=False))
        rows.append(row)
        return row
    placements={}
    for scene in cfg['scenes']:
        assert len(scene['object_ids'])==4
        for name,obj_id,context_order,target_order in product(parent['names'],scene['object_ids'],cfg['context_orders'],parent['target_orders']):
            obj=objects[obj_id];fmt={'name':name['text'],'object':obj['text']}
            near=scene['close_location'].format(**fmt)
            far=scene['far_location']+' '+name['text']
            if context_order['id']=='object_subject':
                c1=f"The {obj['text']} is {near}.";c2=f"The {obj['text']} is {far}."
            elif context_order['id']=='fronted_location':
                c1=near[0].upper()+near[1:]+f" is the {obj['text']}."
                c2=far[0].upper()+far[1:]+f" is the {obj['text']}."
            else:
                c1=scene['person_close'].format(**fmt)
                c2=f"{name['text']} is {scene['far_location']} the {obj['text']}."
            assert not re.search(r'\b(?:close|far|closer|farther)\b',c1+' '+c2)
            row=add_row(name,obj,target_order,scene['id'],scene['role'],scene['evidence_type'],context_order['condition_id'],context_order['id'],context_order['entity_order'],c1,c2,scene['assumption'],scene['setting'])
            placements[scene['id'],name['id'],obj_id,context_order['id'],target_order['id']]=row
    assert len(rows)==864
    assert sum(r['include_in_primary_scene_summary'] for r in rows)==672
    # Shared direct/paraphrase controls are generated once per entity and target order.
    baselines={}
    for name,obj,order,evidence in product(parent['names'],cfg['objects'],parent['target_orders'],[e for e in parent['evidence_conditions'] if e['id'] in ['direct_label','proximity_paraphrase']]):
        fmt={'name':name['text'],'object':obj['text']}
        row=add_row(name,obj,order,'shared_control','shared_control',evidence['evidence_type'],evidence['id'],'object_subject','object_first',evidence['close'].format(**fmt),evidence['far'].format(**fmt),'Direct or paraphrased relation; shared across eligible scenes.','unspecified')
        baselines[name['id'],obj['id'],order['id'],evidence['id']]=row
    assert len(rows)==1024 and len({r['probe_id'] for r in rows})==1024
    matches=[]
    def match(a,b,control):
        assert all(a[k]==b[k] for k in ['name_id','object_id'])
        if control!='target_order':assert all(a[k]==b[k] for k in ['Target1','Target2'])
        matches.append({'reference_probe_id':a['probe_id'],'variant_probe_id':b['probe_id'],
            'control':control,'scene_id':b['scene_id'],'context_structure':b['context_structure'],'target_order':b['target_order']})
    for key,row in placements.items():
        scene,name,obj,structure,order=key
        for evidence in ['direct_label','proximity_paraphrase']:
            match(baselines[name,obj,order,evidence],row,'evidence_from_'+evidence)
        if structure!='object_subject':
            ref=placements[scene,name,obj,'object_subject',order]
            if structure=='fronted_location':
                for field in ['Context1','Context2']:
                    assert Counter(re.findall(r'\b\w+\b',ref[field].lower()))==Counter(re.findall(r'\b\w+\b',row[field].lower()))
            match(ref,row,'context_order')
        if order=='object_first':
            ref=placements[scene,name,obj,structure,'person_first']
            assert all(ref[k]==row[k] for k in ['Context1','Context2'])
            match(ref,row,'target_order')
        if scene in ['between_owned_shoes','between_worn_shoes']:
            ref=placements['between_feet',name,obj,structure,order]
            assert ref['Context2']==row['Context2']
            match(ref,row,'feet_vs_'+scene)
            if scene=='between_worn_shoes':
                owned=placements['between_owned_shoes',name,obj,structure,order]
                assert owned['Context2']==row['Context2']
                match(owned,row,'make_shoes_wearing_explicit')
    assert len(matches)==3024
    counts=Counter((r['scene_id'],r['context_structure'],r['target_order']) for r in rows if r['scene_role']!='shared_control')
    assert set(counts.values())=={16}
    for r in rows:
        assert '{' not in r['Context1'] and '{' not in r['Context2']
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    write_csv(out/'probes.csv',rows)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_csv(out/'matches.csv',matches)
    for key in ['objects','context_orders','scenes']:write_csv(out/(key+'.csv'),cfg[key])
    write_csv(out/'names.csv',parent['names']);write_csv(out/'target_orders.csv',parent['target_orders'])
    scene_lookup={s['id']:s for s in cfg['scenes']}
    review=[r for r in rows if r['name_id']=='name_0' and r['scene_id'] in scene_lookup and r['object_id']==scene_lookup[r['scene_id']]['object_ids'][0]]
    write_csv(out/'review_examples.csv',review)
    write_csv(out/'scene_overview.csv',[{'scene_id':s['id'],'role':s['role'],'evidence_type':s['evidence_type'],
        'close_context':next(r['Context1'] for r in review if r['scene_id']==s['id'] and r['context_structure']=='object_subject'),
        'far_context':next(r['Context2'] for r in review if r['scene_id']==s['id'] and r['context_structure']=='object_subject'),
        'eligible_objects':','.join(objects[k]['text'] for k in s['object_ids'])} for s in cfg['scenes']])
    manifest={'probe_name':cfg['probe_name'],'version':cfg['version'],'matched_pairs':len(rows),'context_judgments':2*len(rows),
        'primary_scene_count':7,'primary_placement_pairs':672,'primary_placement_context_judgments':1344,
        'primary_pairs_per_context_and_target_order':112,'primary_judgments_per_context_and_target_order':224,
        'shoe_control_pairs':192,'shared_control_pairs':160,'matches':len(matches),
        'unique_context_texts':len({r[k] for r in rows for k in ['Context1','Context2']}),
        'unique_conditional_sequences':len({(r[f'Context{c}'],r[f'Target{t}']) for r in rows for c in [1,2] for t in [1,2]}),
        'default_score_reduction':'mean','tokenizer_source':args.tokenizer,'model_evaluation_performed':False,
        'components_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),'parent_components_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
