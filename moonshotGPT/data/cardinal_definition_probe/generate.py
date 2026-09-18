#!/usr/bin/env python3
"""Small matched cardinal definition inventory, with separate map-position controls."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(config):
    rows = []
    for axis in config['axes']:
        for block, templates in (
            ('direction_definition', config['definition_templates']),
            ('opposite_direction', config['opposite_templates']),
            ('relative_map_position', config['position_templates']),
        ):
            for template in templates:
                row = {'probe_id': axis['id']+'__'+template['id'], 'probe_version':config['version'],
                    'probe_family':block, 'axis':axis['id'], 'template_id':template['id'],
                    'mapping_direction':template.get('mapping','word_to_opposite'),
                    'wording_family':template.get('wording_family', 'relative_position' if block=='relative_map_position' else 'opposite'),
                    'wording_id':template['wording_id'],
                    'reference_frame':config['frame_id'] if block!='opposite_direction' else 'cardinal_directions',
                    'reference_surface':'none' if block=='opposite_direction' else 'map',
                    'context_entity_order':'A_then_B' if block=='relative_map_position' else '',
                    'target_entity_order':'A_relative_to_B' if block=='relative_map_position' else '',
                    'correct_target_for_context1':'Target1','correct_target_for_context2':'Target2'}
                for i in (0,1):
                    values={'frame':config['frame_text'], 'direction':axis['directions'][i],
                        'edge':axis['edge_phrases'][i], 'movement':axis['movement_words'][i],
                        'position':axis['position_phrases'][i]}
                    row[f'Context{i+1}']=template['context'].format(**values)
                    row[f'Target{i+1}']=(axis['directions'][1-i]+'.' if block=='opposite_direction'
                                        else template['target'].format(**values))
                    row[f'context{i+1}_direction']=axis['directions'][i]
                    row[f'gold{i+1}_cardinal_direction']=axis['directions'][1-i] if block=='opposite_direction' else axis['directions'][i]
                    row[f'context{i+1}_screen_axis']=axis['screen_axis'] if block!='opposite_direction' else ''
                    row[f'context{i+1}_screen_sign']=axis['screen_signs'][i] if block!='opposite_direction' else ''
                for field in ('Context1','Context2','Target1','Target2'):
                    assert '{' not in row[field] and '}' not in row[field]
                    row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
                assert row['Context1']!=row['Context2'] and row['Target1']!=row['Target2']
                rows.append(row)
    for original in list(rows):
        if original['probe_family']!='relative_map_position':
            continue
        row=dict(original)
        row['probe_id']+='__globe'
        row['reference_surface']='globe'
        row['reference_frame']=config['globe_frame_id']
        for i in (1,2):
            field=f'Context{i}'
            row[field]=row[field].replace(config['frame_text'],config['globe_frame_text']).replace('the map','the globe')
            row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
            assert row[f'Target{i}']==original[f'Target{i}']
        rows.append(row)
    extra_links=[]
    for row in rows:
        row.update(preposition='not_applicable',design_extension='original',preposition_match_id='')
    for axis in config['axes']:
        base=next(r for r in rows if r['axis']==axis['id'] and r['template_id']=='edge_forward_plain')
        edges=('the upper edge','the lower edge') if axis['id']=='north_south' else ('the right-hand edge','the left-hand edge')
        for mapping in ('word_to_meaning','meaning_to_word'):
            for wording in ('traveling','going'):
                matched=[]
                for prep in ('toward','to'):
                    row=dict(base)
                    tid=f'edge_alt_{mapping}_{wording}_{prep}'
                    row.update(probe_id=axis['id']+'__'+tid,template_id=tid,mapping_direction=mapping,
                        wording_family='alternate_edge',wording_id=wording,preposition=prep,
                        design_extension='edge_preposition',preposition_match_id=f"{axis['id']}__{mapping}__{wording}")
                    for i,(direction,edge) in enumerate(zip(axis['directions'],edges),1):
                        stem=wording.capitalize()
                        if mapping=='word_to_meaning':
                            row[f'Context{i}']=f"{config['frame_text']} {stem} {direction} takes you {prep}"
                            row[f'Target{i}']=edge+'.'
                        else:
                            row[f'Context{i}']=f"{config['frame_text']} {stem} {prep} {edge} means going"
                            row[f'Target{i}']=direction+'.'
                    rows.append(row);matched.append(row)
                extra_links.append({'control':'preposition','base_probe_id':matched[0]['probe_id'],
                    'variant_probe_id':matched[1]['probe_id'],'axis':axis['id'],'probe_family':'direction_definition'})
        base=next(r for r in rows if r['axis']==axis['id'] and r['template_id']=='opposite_plain')
        route_templates={
            'backward':'A straight route leads {direction}. Following the same route backward means traveling',
            'return':'A straight route leads {direction} from the start to the destination. Returning along that route means traveling',
            'retrace':'You travel {direction} along a straight path. Retracing that path to where you started means traveling',
            'reverse_route':'The outward leg of a straight journey goes {direction}. The return leg along the same path goes'}
        for wording,template in route_templates.items():
            row=dict(base);tid='opposite_route_'+wording
            row.update(probe_id=axis['id']+'__'+tid,template_id=tid,wording_id=wording,
                wording_family='route_reversal',design_extension='route_reversal')
            for i,direction in enumerate(axis['directions'],1):row[f'Context{i}']=template.format(direction=direction)
            rows.append(row)
    for base in list(rows):
        if base['probe_family']!='relative_map_position':continue
        row=dict(base)
        row.update(probe_id=base['probe_id']+'__reciprocal_named',template_id=base['template_id']+'__reciprocal_named',
            context_entity_order='cabin_then_lake',target_entity_order='lake_relative_to_cabin',
            design_extension='reciprocal_named_entities')
        axis=next(a for a in config['axes'] if a['id']==row['axis'])
        frame=config['frame_text'] if row['reference_surface']=='map' else config['globe_frame_text']
        for i in (1,2):
            direction=axis['directions'][i-1];inverse=axis['directions'][2-i]
            pos=axis['position_phrases'][i-1];inverse_pos=axis['position_phrases'][2-i]
            if row['mapping_direction']=='word_to_meaning':
                clause=f'The cabin is {direction} of the lake.' if row['wording_id']=='plain' else f'The cabin lies {direction} of the lake.'
                row[f'Context{i}']=f'{frame} {clause} The lake appears'
                row[f'Target{i}']=f'{inverse_pos} the cabin.'
            else:
                clause=f'The cabin appears {pos} the lake.' if row['wording_id']=='plain' else f'The cabin is drawn {pos} the lake.'
                row[f'Context{i}']=f'{frame} {clause} The lake is'
                row[f'Target{i}']=f'{inverse} of the cabin.'
            row[f'gold{i}_cardinal_direction']=inverse
        rows.append(row)
        extra_links.append({'control':'reciprocal_named_entities','base_probe_id':base['probe_id'],
            'variant_probe_id':row['probe_id'],'axis':row['axis'],'probe_family':row['probe_family']})
    for row in rows:
        for field in ('Context1','Context2','Target1','Target2'):
            row[field+'_word_count']=len(re.findall(r'\b\w+\b',row[field]))
    assert len(rows)==80 and len({r['probe_id'] for r in rows})==80
    assert Counter(r['probe_family'] for r in rows)=={'direction_definition':32,'opposite_direction':16,'relative_map_position':32}
    links=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['axis']!=b['axis'] or a['probe_family']!=b['probe_family']:
                continue
            if a['design_extension']!=b['design_extension'] or a['preposition']!=b['preposition']:continue
            control=None
            if a['reference_surface']!=b['reference_surface']:
                if a['template_id']!=b['template_id']:
                    continue
                control='reference_surface'
            elif a['wording_family']==b['wording_family'] and a['mapping_direction']==b['mapping_direction'] and a['wording_id']!=b['wording_id']:
                control='wording'
            elif a['wording_family']==b['wording_family'] and a['wording_id']==b['wording_id'] and a['mapping_direction']!=b['mapping_direction']:
                control='mapping_direction'
            elif a['wording_family']!=b['wording_family'] and a['wording_id']==b['wording_id'] and a['mapping_direction']==b['mapping_direction']:
                control='expression_family'
            if control:
                if control in ('wording','reference_surface'):
                    assert (a['Target1'],a['Target2'])==(b['Target1'],b['Target2'])
                links.append({'control':control,'base_probe_id':a['probe_id'],'variant_probe_id':b['probe_id'],
                              'axis':a['axis'],'probe_family':a['probe_family']})
    return rows,links+extra_links


def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'generated')
    args=parser.parse_args()
    source=ROOT/'components.json'
    config=json.loads(source.read_text())
    rows,links=build(config)
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'probes.csv',rows);csv_write(out/'variant_matches.csv',links)
    (out/'probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=['# Cardinal definition and map-position examples','',
        'Each pair has two opposing contexts and the same two candidate continuations. C1 supports T1; C2 supports T2. Append a space before a continuation.','']
    for r in rows:
        report.extend([f"## {r['probe_id']}",'',f"Block: {r['probe_family']}; mapping: {r['mapping_direction']}",'',
            f"C1: {r['Context1']}",'',f"T1: {r['Target1']}",'',f"C2: {r['Context2']}",'',f"T2: {r['Target2']}",''])
    (out/'review_examples.md').write_text('\n'.join(report))
    manifest={'probe_name':config['probe_name'],'version':config['version'],'paired_rows':len(rows),
        'binary_judgments':2*len(rows),'conditional_likelihoods':4*len(rows),
        'by_block':dict(Counter(r['probe_family'] for r in rows)),
        'by_axis':dict(Counter(r['axis'] for r in rows)), 'variant_matches':len(links),
        'by_reference_surface':dict(Counter(r['reference_surface'] for r in rows)),
        'components_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'validation':'passed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
