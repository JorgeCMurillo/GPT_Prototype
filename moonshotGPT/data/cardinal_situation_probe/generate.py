#!/usr/bin/env python3
"""Matched cardinal event scenes from validated endpoints, not inferred heuristics."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sign(x):
    assert x != 0
    return 1 if x > 0 else -1


def validate_cases(cases):
    signatures=set()
    for case in cases:
        # Opposing orientations belong to the same pair, regardless of ordering.
        signature=tuple(sorted(tuple(case[truth][e][i] for e in ('target','reference') for i in (0,1)) for truth in ('positive','negative')))
        assert signature not in signatures
        signatures.add(signature)
        for truth,expected in (('positive',1),('negative',-1)):
            a,b=case[truth]['target'],case[truth]['reference']
            assert all(type(n) is int and 0<=n<case['positions'] for n in a+b)
            assert sign(a[1]-b[1])==expected
            crossing=sign(a[0]-b[0])!=expected
            family=case['family']
            if family=='static_placement': assert a[0]==a[1] and b[0]==b[1]
            elif family=='target_crosses': assert a[0]!=a[1] and b[0]==b[1] and crossing
            elif family=='reference_crosses': assert a[0]==a[1] and b[0]!=b[1] and crossing
            elif family=='target_moves_without_crossing': assert a[0]!=a[1] and b[0]==b[1] and not crossing
            else:
                assert family=='both_move' and a[0]!=a[1] and b[0]!=b[1]
                assert crossing==(case['subtype']=='swap_two_positions')
        if case['family']=='target_moves_without_crossing':
            assert sign(case['positive']['target'][1]-case['positive']['target'][0])==sign(case['negative']['target'][1]-case['negative']['target'][0])


def assignment(config,case_index,axis_index):
    offset=sum(c['positions'] for c in config['cases'][:case_index])
    case=config['cases'][case_index]
    names=[config['locations'][(offset+i)%len(config['locations'])] for i in range(case['positions'])]
    # Across the two axes each name occurs at mirrored ordinal ranks.
    return names if axis_index==0 else names[::-1]


def position_label(n,axis,fmt,names,numbering='increasing'):
    number=len(names)-n if numbering=='decreasing' else n+1
    return f"{axis['place']} {number}" if fmt=='numeric' else 'the '+names[n]


def event_context(config,case,axis,fmt,length,names,truth,context_order='target_first',list_order='negative_to_positive',numbering='increasing'):
    labels=[position_label(i,axis,fmt,names,numbering) for i in range(case['positions'])]
    if fmt=='numeric': labels=[s.split()[-1] for s in labels]
    first,last=axis['negative'],axis['positive']
    if list_order=='positive_to_negative':
        labels.reverse();first,last=last,first
    order=', '.join(labels)
    if fmt=='numeric':
        anchors=f"The {axis['place']}s are ordered {order} from {first} to {last}."
        alignment=f"Both markers stay in the same {axis['alignment']}."
    else:
        anchors=f"From {first} to {last}, the locations are {order}."
        alignment=f"These locations lie along one {axis['negative']}–{axis['positive']} line."
    intro=config['frame']+' '+anchors+' '+alignment
    if length=='standard': intro+=' The locations stay fixed.'
    elif length=='expanded': intro+=' These positions remain fixed throughout the scene, and the map does not rotate.'
    clauses=[]
    sequence=('target','reference') if context_order=='target_first' else ('reference','target')
    for entity in sequence:
        a,b=case[truth][entity]
        kind='static' if case['family']=='static_placement' else 'still' if a==b else 'moving'
        clauses.append(config['entity_templates'][length][kind].format(entity=config['entities'][entity],start=position_label(a,axis,fmt,names,numbering),end=position_label(b,axis,fmt,names,numbering)))
    return ' '.join([intro,*clauses,config['answer_bridge']])


def control_context(config,axis,kind,length,truth,context_order='target_first'):
    subject,reference='A','B'
    if context_order=='reference_first':
        subject,reference='B','A'
        truth='negative' if truth=='positive' else 'positive'
    direction=axis[truth]
    relation=f'{direction} of' if kind=='direct_relation' else axis['screen_'+truth]
    sentence={
        'compact':f'{subject} is {relation} {reference}.',
        'standard':f'Marker {subject} is positioned {relation} marker {reference}. Both markers stay still.',
        'expanded':f'Marker {subject} is positioned {relation} marker {reference}. Both markers remain at their initial positions throughout the scene; neither marker moves.',
    }[length]
    return config['frame']+' '+sentence+' '+config['answer_bridge']


def make_row(config,axis,case,fmt,length,names,context_order='target_first',target_order='target_first',list_order='negative_to_positive',numbering='increasing'):
    control=case is None
    family=fmt if control else case['family']
    row={'probe_id':'__'.join((axis['id'],family if control else case['id'],fmt,length)),
        'probe_version':config['version'],'probe_family':'control' if control else 'event',
        'axis':axis['id'],'case_id':family if control else case['id'],'event_family':family,
        'event_subtype':'' if control else case['subtype'],'evidence_format':fmt,'length_band':length,
        'template_id':'__'.join((family,fmt,length)), 'reference_frame':'north_up_east_right_map',
        'target_entity':'A','reference_entity':'B','context_entity_order':context_order,'target_entity_order':target_order,
        'last_described_entity':'reference' if context_order=='target_first' else 'target','position_count':0 if control else case['positions'],
        'locations_negative_to_positive':' | '.join(names) if fmt=='named_locations' else '',
        'answer_bridge':config['answer_bridge'],
        'location_list_order':'not_applicable' if control else list_order,
        'numeric_label_order':numbering if fmt=='numeric' else 'not_applicable',
        'correct_target_for_context1':'Target1','correct_target_for_context2':'Target2'}
    if (context_order,target_order)!=('target_first','target_first'):
        row['probe_id']+='__'+context_order+'__'+target_order
    if not control and list_order!='negative_to_positive': row['probe_id']+='__list_'+list_order
    if fmt=='numeric' and numbering!='increasing': row['probe_id']+='__labels_'+numbering
    for i,truth in enumerate(('positive','negative'),1):
        row[f'Context{i}']=control_context(config,axis,fmt,length,truth,context_order) if control else event_context(config,case,axis,fmt,length,names,truth,context_order,list_order,numbering)
        answer_truth=truth if target_order=='target_first' else 'negative' if truth=='positive' else 'positive'
        subject,reference=('A','B') if target_order=='target_first' else ('B','A')
        row[f'Target{i}']=f"Marker {subject} is {axis[answer_truth]} of marker {reference}."
        row[f'target{i}_relation_word']=axis[answer_truth]
        row[f'context{i}_gold_relation']=axis[truth]
        for entity in ('target','reference'):
            points=None if control else case[truth][entity]
            for j,point in enumerate(('start','end')):
                row[f'context{i}_{entity}_{point}']='' if control else points[j]
                row[f'context{i}_{entity}_{point}_label']='' if control else position_label(points[j],axis,fmt,names,numbering)
            row[f'context{i}_{entity}_motion']='' if control else 'still' if points[0]==points[1] else axis['positive'] if points[1]>points[0] else axis['negative']
        if control:
            row[f'context{i}_initial_relation']=''
            row[f'context{i}_order_reversed']=''
        else:
            a,b=case[truth]['target'],case[truth]['reference']
            row[f'context{i}_initial_relation']=axis['positive'] if a[0]>b[0] else axis['negative']
            row[f'context{i}_order_reversed']=sign(a[0]-b[0])!=sign(a[1]-b[1])
    for field in ('Context1','Context2','Target1','Target2'):
        row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
    return row


def build(config):
    validate_cases(config['cases'])
    rows=[]
    for ai,axis in enumerate(config['axes']):
        for ci,case in enumerate(config['cases']):
            names=assignment(config,ci,ai)
            for fmt in config['formats']:
                for length in config['lengths']:
                    for co in config['context_entity_orders']:
                        for to in config['target_entity_orders']:
                            for order in config['location_list_orders']:
                                for numbering in config['numeric_label_orders'] if fmt=='numeric' else ['increasing']:
                                    rows.append(make_row(config,axis,case,fmt,length,names,co,to,order,numbering))
        for kind in config['control_types']:
            for length in config['lengths']:
                for co in config['context_entity_orders']:
                    for to in config['target_entity_orders']:
                        rows.append(make_row(config,axis,None,kind,length,[],co,to))
    assert len(rows)==1344 and len({r['probe_id'] for r in rows})==1344
    lookup={(r['axis'],r['case_id'],r['evidence_format'],r['length_band'],r['context_entity_order'],r['target_entity_order'],r['location_list_order'],r['numeric_label_order']):r for r in rows}
    links=[]
    for row in rows:
        axis,case,fmt,length=(row[k] for k in ('axis','case_id','evidence_format','length_band'))
        co,to=row['context_entity_order'],row['target_entity_order']
        order,numbering=row['location_list_order'],row['numeric_label_order']
        variants=[]
        if length=='compact':
            variants.extend(('length_band',(axis,case,fmt,v,co,to)) for v in ('standard','expanded'))
        if fmt=='numeric' and numbering=='increasing': variants.append(('evidence_format',(axis,case,'named_locations',length,co,to)))
        if axis=='north_south': variants.append(('axis',('east_west',case,fmt,length,co,to)))
        if co=='target_first': variants.append(('context_entity_order',(axis,case,fmt,length,'reference_first',to)))
        if to=='target_first': variants.append(('target_entity_order',(axis,case,fmt,length,co,'reference_first')))
        variants=[(control,key+(order,'not_applicable' if control=='evidence_format' else numbering)) for control,key in variants]
        if order=='negative_to_positive': variants.append(('location_list_order',(axis,case,fmt,length,co,to,'positive_to_negative',numbering)))
        if numbering=='increasing': variants.append(('numeric_label_order',(axis,case,fmt,length,co,to,order,'decreasing')))
        for control,key in variants:
            other=lookup[key]
            if control not in ('axis','target_entity_order'): assert (row['Target1'],row['Target2'])==(other['Target1'],other['Target2'])
            if control=='target_entity_order':
                assert (row['Context1'],row['Context2'])==(other['Context1'],other['Context2'])
                assert row['target1_relation_word']!=other['target1_relation_word']
            if control=='length_band':
                assert row['locations_negative_to_positive']==other['locations_negative_to_positive']
                assert all(row[f'Context{i}_word_count']<other[f'Context{i}_word_count'] for i in (1,2))
            for i in (1,2):
                for entity in ('target','reference'):
                    for point in ('start','end'):
                        k=f'context{i}_{entity}_{point}'
                        assert row[k]==other[k]
            links.append({'control':control,'base_probe_id':row['probe_id'],'variant_probe_id':other['probe_id']})
    return rows,links


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'generated')
    args=parser.parse_args()
    source=ROOT/'components.json';config=json.loads(source.read_text())
    rows,links=build(config)
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'probes.csv',rows);write_csv(out/'variant_matches.csv',links)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'case_catalog.json').write_text(json.dumps(config['cases'],indent=2)+'\n')
    report=['# Cardinal event and simple-control examples','',
        'C1 supports T1; C2 supports T2. Score both candidates after each context. Positions are final positions.','']
    for row in rows:
        report.extend([f"## {row['probe_id']}",'',*[f"{field}: {row[field]}\n" for field in ('Context1','Target1','Context2','Target2')]])
    (out/'review_examples.md').write_text('\n'.join(report))
    manifest={'probe_name':config['probe_name'],'version':config['version'],'paired_rows':len(rows),
        'event_pairs':1296,'control_pairs':48,'binary_judgments':2688,'conditional_likelihoods':5376,
        'numeric_event_pairs':864,'named_event_pairs':432,
        'location_list_orders':2,'numeric_label_orders':2,
        'context_entity_orders':2,'target_entity_orders':2,
        'underlying_matched_cases':9,'axes':2,'entities_per_scene':2,'location_pool_size':12,
        'template_combinations':len({r['template_id'] for r in rows}), 'variant_matches':len(links),
        'by_event_family':dict(Counter(r['event_family'] for r in rows)),
        'components_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
