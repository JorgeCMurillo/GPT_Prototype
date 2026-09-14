#!/usr/bin/env python3
"""Keep close contexts and targets fixed while replacing only far evidence."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',default=DEFAULT_TOKENIZER)
    args=parser.parse_args()
    config_path=ROOT/'components.json';cfg=json.loads(config_path.read_text())
    source_path=ROOT/cfg['source_probes']
    source=[json.loads(s) for s in source_path.read_text().splitlines()]
    source=[r for r in source if r['scene_role']!='shared_control']
    assert len(source)==864
    tables=source_path.parent
    names={r['id']:r['text'] for r in csv.DictReader((tables/'names.csv').open())}
    objects={r['id']:r['text'] for r in csv.DictReader((tables/'objects.csv').open())}
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    rows=[];matches=[];lexical_matches=[]
    lookup={}
    for old in source:
        base={**old,'source_probe_id':old['probe_id'],'source_probe_version':old['probe_version'],
            'probe_version':cfg['version'],'probe_type':cfg['probe_name'],
            'far_wording_match_id':old['probe_id'],'close_evidence_type':old['evidence_type'],
            'close_context_structure':old['context_structure'],
            'far_context_entity_order':old['context_entity_order']}
        variants=[{'id':cfg['source_far_variant_id'],'evidence_type':'large_space_separation','context':old['Context2']}]
        for v in cfg['far_variants']:
            variants.append({**v,'context':v['templates'][old['context_structure']].format(name=names[old['name_id']],object=objects[old['object_id']])})
        for variant in variants:
            row={**base,'probe_id':old['probe_id'] if variant['id']==cfg['source_far_variant_id'] else old['probe_id']+'__far_'+variant['id'],
                'far_variant_id':variant['id'],'far_evidence_type':variant['evidence_type'],
                'Context2':variant['context'],
                'far_setting_explicit':variant['id']==cfg['source_far_variant_id'],
                'far_target_repeats_context':variant['context']==old['Target2']}
            # "setting" is inherited source-scene metadata, not a claim it is stated in every far variant.
            if variant['id']=='large_distance':
                row['far_context_structure']='distance_subject' if old['context_structure']=='fronted_location' else 'coordinated_entities'
            else:
                row['far_context_structure']=old['context_structure']
            for field in ['Context1','Context2','Target1','Target2']:
                row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
                row[field+'_token_count']=len(tokenizer.encode(row[field],add_special_tokens=False))
            assert all(row[k]==old[k] for k in ['Context1','Target1','Target2','name_id','object_id','scene_id','target_order'])
            c2=row['Context2'].lower()
            assert (c2.index(names[row['name_id']].lower())<c2.index(objects[row['object_id']]))==(row['far_context_entity_order']=='person_first')
            rows.append(row);lookup[old['probe_id'],variant['id']]=row
            if variant['id']!=cfg['source_far_variant_id']:
                matches.append({'reference_probe_id':old['probe_id'],'variant_probe_id':row['probe_id'],
                    'control':'far_evidence_only','far_variant_id':variant['id'],'scene_id':row['scene_id'],
                    'scene_role':row['scene_role'],'context_structure':row['context_structure'],'target_order':row['target_order']})
        direct=lookup[old['probe_id'],'direct_far'];synonym=lookup[old['probe_id'],'synonym_distant']
        assert re.sub(r'\bfar\b','distant',direct['Context2'],flags=re.I).lower()==synonym['Context2'].lower()
        assert direct['Context2_word_count']==synonym['Context2_word_count']
        lexical_matches.append({'reference_probe_id':direct['probe_id'],'variant_probe_id':synonym['probe_id'],
            'control':'far_to_distant_only','scene_id':old['scene_id'],'context_structure':old['context_structure'],'target_order':old['target_order']})
    assert len(rows)==3456 and len({r['probe_id'] for r in rows})==3456
    assert len(matches)==2592 and len(lexical_matches)==864
    assert set(Counter(r['far_variant_id'] for r in rows).values())=={864}
    assert set(Counter(r['far_variant_id'] for r in rows if r['include_in_primary_scene_summary']).values())=={672}
    group=defaultdict(list)
    for r in rows:group[r['far_wording_match_id']].append(r)
    for variants in group.values():
        assert len(variants)==4
        assert len({(r['Context1'],r['Target1'],r['Target2']) for r in variants})==1
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    write_csv(out/'probes.csv',rows)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_csv(out/'matches.csv',matches)
    write_csv(out/'lexical_matches.csv',lexical_matches)
    write_csv(out/'far_variants.csv',cfg['far_variants'])
    write_csv(out/'review_examples.csv',[r for r in rows if r['scene_id']=='between_feet' and r['name_id']=='name_0' and r['object_id']=='obj_ball'])
    manifest={'probe_name':cfg['probe_name'],'version':cfg['version'],'source_placement_pairs':len(source),
        'matched_pairs':len(rows),'context_judgments':2*len(rows),'new_pairs':len(matches),'retained_baseline_pairs':len(source),
        'main_scene_pairs_per_far_variant':672,'shoe_control_pairs_per_far_variant':192,
        'far_variants_including_baseline':4,'far_wording_matches':len(matches),'lexical_matches':len(lexical_matches),
        'unique_context_texts':len({r[k] for r in rows for k in ['Context1','Context2']}),
        'unique_conditional_sequences':len({(r[f'Context{c}'],r[f'Target{t}']) for r in rows for c in [1,2] for t in [1,2]}),
        'default_score_reduction':'mean','tokenizer_source':args.tokenizer,
        'source_probes_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'components_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'new_variants_evaluated':False,'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
