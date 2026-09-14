#!/usr/bin/env python3
"""Generate close-only substitutions and deduplicated symmetric wording pairs."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',default=DEFAULT_TOKENIZER)
    args=parser.parse_args()
    cfg_path=ROOT/'components.json';cfg=json.loads(cfg_path.read_text())
    source_path=ROOT/cfg['source_probes']
    source=[json.loads(s) for s in source_path.read_text().splitlines()]
    source=[r for r in source if r['include_in_primary_scene_summary']]
    assert len(source)==672
    names={r['id']:r['text'] for r in csv.DictReader((source_path.parent/'names.csv').open())}
    objects={r['id']:r['text'] for r in csv.DictReader((source_path.parent/'objects.csv').open())}
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    rows=[];matches=[];paired={};close_only={}
    def finish(row):
        for field in ['Context1','Context2','Target1','Target2']:
            row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
            row[field+'_token_count']=len(tokenizer.encode(row[field],add_special_tokens=False))
        row['close_target_repeats_context']=row['Context1']==row['Target1']
        row['far_target_repeats_context']=row['Context2']==row['Target2']
        rows.append(row)
        return row
    def link(a,b,control,family):
        assert all(a[k]==b[k] for k in ['Target1','Target2','name_id','object_id','target_order'])
        matches.append({'reference_probe_id':a['probe_id'],'variant_probe_id':b['probe_id'],
            'control':control,'wording_family_id':family,'source_scene_id':a.get('source_scene_id',''),
            'context_structure':a['context_structure'],'target_order':a['target_order']})
    for old in source:
        common={**old,'source_probe_id':old['probe_id'],'source_probe_version':old['probe_version'],
            'probe_version':cfg['version'],'probe_type':cfg['probe_name'],'source_scene_id':old['scene_id'],
            'source_assumption':old['assumption'],'source_close_evidence_type':old['evidence_type'],
            'close_evidence_type':old['evidence_type'],'far_evidence_type':'large_space_separation',
            'close_context_structure':old['context_structure'],'far_context_structure':old['context_structure']}
        base=finish({**common,'mode':'original_baseline','wording_family_id':'original_placement','adjective_pair_id':'none'})
        fmt={'name':names[old['name_id']],'object':objects[old['object_id']]}
        for family in cfg['families']:
            c1=family['close'][old['context_structure']].format(**fmt)
            c2=family['far'][old['context_structure']].format(**fmt)
            structure=('distance_subject' if old['context_structure']=='fronted_location' else 'coordinated_entities') if family['evidence_type']=='distance_magnitude_phrase' else old['context_structure']
            a=finish({**common,'probe_id':old['probe_id']+'__close_'+family['id'],
                'mode':'close_only','wording_family_id':family['id'],'adjective_pair_id':family['adjective_pair_id'],
                'Context1':c1,'close_evidence_type':family['evidence_type'],'close_context_structure':structure,
                'assumption':'Close is stated or paraphrased directly; source placement wording is replaced.'})
            assert a['Context2']==base['Context2']
            link(base,a,'close_evidence_only',family['id'])
            close_only[old['probe_id'],family['id']]=a
            key=(old['name_id'],old['object_id'],old['context_structure'],old['target_order'],family['id'])
            if key not in paired:
                b=finish({**a,'probe_id':'paired__'+'__'.join(key),'mode':'symmetric_pair',
                    'Context2':c2,'far_evidence_type':family['evidence_type'],'far_context_structure':structure,
                    'scene_id':'generic_wording','scene_role':'wording_control','source_scene_id':'',
                    'source_probe_id':'','source_probe_version':'','source_assumption':'','source_close_evidence_type':'',
                    'setting':'unspecified','assumption':'Both distance relations are stated or paraphrased; no source placement or setting is specified.',
                    'include_in_primary_scene_summary':False})
                paired[key]=b
            b=paired[key]
            assert b['Context1']==a['Context1'] and b['Context2']==c2
            link(a,b,'far_evidence_only_given_close',family['id'])
    for old in source:
        a=close_only[old['probe_id'],'distance_short_long'];b=close_only[old['probe_id'],'distance_small_large']
        assert a['Context1'].replace('short','small')==b['Context1'] and a['Context2']==b['Context2']
        link(a,b,'close_adjective_short_vs_small','short_long_vs_small_large')
    for key,a in paired.items():
        if key[-1]!='distance_short_long':continue
        b=paired[key[:-1]+('distance_small_large',)]
        assert a['Context1'].replace('short','small')==b['Context1']
        assert a['Context2'].replace('long','large')==b['Context2']
        link(a,b,'adjective_pair_short_long_vs_small_large','short_long_vs_small_large')
    assert len(rows)==4320 and len({r['probe_id'] for r in rows})==4320
    assert len(paired)==960 and len(close_only)==2688 and len(matches)==6288
    assert Counter(r['mode'] for r in rows)=={'original_baseline':672,'close_only':2688,'symmetric_pair':960}
    for family in cfg['families']:
        assert sum(r['mode']=='close_only' and r['wording_family_id']==family['id'] for r in rows)==672
        assert sum(r['mode']=='symmetric_pair' and r['wording_family_id']==family['id'] for r in rows)==240
    assert len({(r['Context1'],r['Context2'],r['Target1'],r['Target2']) for r in paired.values()})==960
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    write_csv(out/'probes.csv',rows);write_csv(out/'matches.csv',matches)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_csv(out/'families.csv',cfg['families'])
    write_csv(out/'review_examples.csv',[r for r in rows if r['name_id']=='name_0' and r['object_id']=='obj_ball' and (r['source_scene_id']=='between_feet' or r['mode']=='symmetric_pair')])
    write_csv(out/'paired_overview.csv',[r for r in paired.values() if r['name_id']=='name_0' and r['object_id']=='obj_ball' and r['context_structure']=='fronted_location' and r['target_order']=='person_first'])
    manifest={'probe_name':cfg['probe_name'],'version':cfg['version'],'matched_pairs':len(rows),'context_judgments':2*len(rows),
        'baseline_pairs':672,'close_only_pairs_per_family':672,'symmetric_pairs_per_family':240,
        'wording_families':4,'modes':dict(Counter(r['mode'] for r in rows)),'matches':len(matches),
        'unique_conditional_sequences':len({(r[f'Context{c}'],r[f'Target{t}']) for r in rows for c in [1,2] for t in [1,2]}),
        'source_probes_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'components_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),'tokenizer_source':args.tokenizer,
        'default_score_reduction':'mean','new_forms_evaluated':False,'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
