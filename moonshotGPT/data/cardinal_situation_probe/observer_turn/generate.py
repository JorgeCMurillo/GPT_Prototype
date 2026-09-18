#!/usr/bin/env python3
"""Matched observer turns: cardinal invariance versus person-relative changes."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VECTORS={'north':(0,1),'east':(1,0),'south':(0,-1),'west':(-1,0)}
LENGTHS=('compact','standard','expanded')


def relative_side(facing,position):
    fx,fy=VECTORS[facing]
    x,y=position
    # Observer's right vector is clockwise from forward.
    lateral=fy*x-fx*y
    assert lateral!=0 and fx*x+fy*y==0
    return 'right' if lateral>0 else 'left'


def context(facing,direction,length,turn):
    if length=='compact':
        intro=f'Observer B faces {facing}. Object A is {direction} of B.'
        action='B turns halfway around in place.' if turn else 'B keeps facing the same direction.'
        end='Neither changes location.'
    elif length=='standard':
        intro=f'Observer B initially faces {facing}. Object A is positioned {direction} of B.'
        action='B makes a half-turn in place, ending facing the opposite direction.' if turn else 'B does not turn and keeps facing the initial direction.'
        end='Object A stays fixed, and B remains at the same location.'
    else:
        intro=f'At the start of the scene, observer B faces {facing}. Object A occupies a fixed position {direction} of B.'
        action='B turns halfway around without leaving that location, ending facing the opposite direction.' if turn else 'B remains at that location without turning, continuing to face the initial direction.'
        end='Throughout the scene, object A stays at its initial position. There is no change in either location.'
    return ' '.join((intro,action,end,'At the end, to summarize the relationship:'))


def build():
    rows=[]
    for facing,vector in VECTORS.items():
        final=next(d for d,v in VECTORS.items() if v==(-vector[0],-vector[1]))
        for direction,position in VECTORS.items():
            if sum(a*b for a,b in zip(vector,position))!=0: continue
            axis='north_south' if position[0]==0 else 'east_west'
            pair=('north','south') if axis=='north_south' else ('east','west')
            scene=f'face_{facing}__object_{direction}'
            for length in LENGTHS:
                for query in ('cardinal','observer_relative'):
                    answers=pair if query=='cardinal' else ('left','right')
                    gold=(direction,direction) if query=='cardinal' else (relative_side(facing,position),relative_side(final,position))
                    row={'probe_id':f'{scene}__{length}__{query}','scene_id':scene,'length_band':length,
                         'query_frame':query,'axis':axis,'event_family':'observer_turn',
                         'initial_facing':facing,'context1_final_facing':facing,'context2_final_facing':final,
                         'context1_action':'no_turn','context2_action':'half_turn',
                         'observer_start_position':[0,0],'observer_end_position':[0,0],
                         'object_start_position':list(position),'object_end_position':list(position),
                         'object_cardinal_relation':direction,'gold_relation_context1':gold[0],
                         'gold_relation_context2':gold[1],
                         'expected_relation_change':query=='observer_relative',
                         'Context1':context(facing,direction,length,False),
                         'Context2':context(facing,direction,length,True)}
                    for i,answer in enumerate(answers,1):
                        row[f'Target{i}']=(f'Object A is {answer} of observer B.' if query=='cardinal'
                                          else f"Object A is on observer B's {answer} side.")
                    for i,g in enumerate(gold,1):row[f'correct_target_for_context{i}']='Target'+str(answers.index(g)+1)
                    rows.append(row)
    for row in rows:
        row.update(description_structure='original',object_description='object_A')
    for base in list(rows):
        row=dict(base)
        row.update(probe_id=base['probe_id']+'__fixed_flag',description_structure='fixed_flag',object_description='flag_A')
        facing,direction,length=row['initial_facing'],row['object_cardinal_relation'],row['length_band']
        for i,turn in ((1,False),(2,True)):
            intro=f'B faces {facing}. Flag A stands {direction} of B.'
            action='B turns to face the opposite direction.' if turn else f'B continues looking {facing}.'
            ending='B and flag A remain in their original locations.'
            if length=='standard':
                intro=f'At the start, observer B faces {facing}. Flag A stands at a fixed location {direction} of B.'
                ending='Observer B and flag A remain in their original locations throughout the scene.'
            elif length=='expanded':
                intro=f'At the start of the scene, observer B faces {facing}. Flag A stands at a fixed location {direction} of B.'
                ending='Observer B stays at the initial location. Flag A also stays where it started, so neither position changes during the scene.'
            row[f'Context{i}']=' '.join((intro,action,ending,'At the end, to summarize the relationship:'))
            row[f'Target{i}']=row[f'Target{i}'].replace('Object A','Flag A')
        rows.append(row)
    assert len(rows)==96 and len({r['probe_id'] for r in rows})==96
    lookup={(r['scene_id'],r['length_band'],r['query_frame'],r['description_structure']):r for r in rows}
    links=[]
    for row in rows:
        candidates=[]
        if row['query_frame']=='cardinal':candidates.append(('query_frame',(row['scene_id'],row['length_band'],'observer_relative')))
        if row['length_band']=='compact':
            candidates.extend(('length_band',(row['scene_id'],length,row['query_frame'])) for length in LENGTHS[1:])
        candidates=[(control,key+(row['description_structure'],)) for control,key in candidates]
        if row['description_structure']=='original':
            candidates.append(('description_structure',(row['scene_id'],row['length_band'],row['query_frame'],'fixed_flag')))
        for control,key in candidates:
            other=lookup[key]
            if control=='query_frame':
                assert all(row[f'Context{i}']==other[f'Context{i}'] for i in (1,2))
            elif control=='length_band': assert all(row[f'Target{i}']==other[f'Target{i}'] for i in (1,2))
            else: assert all(row[f'correct_target_for_context{i}']==other[f'correct_target_for_context{i}'] for i in (1,2))
            links.append({'control':control,'base_probe_id':row['probe_id'],'variant_probe_id':other['probe_id']})
    return rows,links


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    rows,links=build()
    out=ROOT/'generated';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'probes.csv',rows);write_csv(out/'variant_matches.csv',links)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=['# Observer-turn matched scenes','',
            'C1: no turn. C2: half-turn. Cardinal answers stay the same; observer-relative answers change. Gold target fields are authoritative.','']
    for r in rows:
        report.extend([f"## {r['probe_id']}",'',*[f'{k}: {r[k]}\n' for k in ('Context1','Context2','Target1','Target2','correct_target_for_context1','correct_target_for_context2')]])
    (out/'review_examples.md').write_text('\n'.join(report))
    manifest={'version':'1.1','scene_geometries':8,'wording_lengths':3,'query_frames':2,'description_structures':2,
              'paired_rows':96,'binary_judgments':192,'conditional_likelihoods':384,
              'variant_matches':len(links),'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
