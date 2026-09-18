import importlib.util
from pathlib import Path
from collections import Counter

path=Path(__file__).resolve().parents[1]/'data/cardinal_situation_probe/observer_turn/generate.py'
spec=importlib.util.spec_from_file_location('observer_turn',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_physical_relations_and_gold():
    rows,_=module.build()
    right_of={'north':'east','east':'south','south':'west','west':'north'}
    for row in rows:
        assert row['observer_start_position']==row['observer_end_position']
        assert row['object_start_position']==row['object_end_position']
        for i in (1,2):
            facing=row[f'context{i}_final_facing']
            expected=row['object_cardinal_relation'] if row['query_frame']=='cardinal' else 'right' if right_of[facing]==row['object_cardinal_relation'] else 'left'
            assert row[f'gold_relation_context{i}']==expected
            assert expected in row[row[f'correct_target_for_context{i}']]
        same=row['correct_target_for_context1']==row['correct_target_for_context2']
        assert same==(row['query_frame']=='cardinal')
    for frame in ('cardinal','observer_relative'):
        for i in (1,2):
            counts=Counter(r[f'correct_target_for_context{i}'] for r in rows if r['query_frame']==frame)
            assert counts=={'Target1':24,'Target2':24}


def test_frame_matches_hold_context_identical():
    rows,links=module.build();lookup={r['probe_id']:r for r in rows}
    frame_links=[l for l in links if l['control']=='query_frame']
    assert len(frame_links)==48
    for link in frame_links:
        a,b=[lookup[link[k]] for k in ('base_probe_id','variant_probe_id')]
        assert a['Context1']==b['Context1'] and a['Context2']==b['Context2']
        assert not a['expected_relation_change'] and b['expected_relation_change']
