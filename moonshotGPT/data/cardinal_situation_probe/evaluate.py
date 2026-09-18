#!/usr/bin/env python3
"""Evaluate cardinal scenes with raw likelihood and balanced family/format summaries."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from evaluation.ewok import per_token_conditional_log_likelihood


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(rows):
    return {'pairs':len(rows),'accuracy':mean(r['accuracy'] for r in rows),
            'both_correct':mean(r['both_correct'] for r in rows),
            'positive_word_choice':sum(r['positive_word_choices'] for r in rows)/(2*len(rows)),
            'ties':sum(r['ties'] for r in rows)}


def balanced(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['event_family'],r['evidence_format']].append(r)
    families=defaultdict(list)
    for (family,fmt),rs in groups.items():families[family].append(summarize(rs))
    return {k:mean(mean(g[k] for g in gs) for gs in families.values())
            for k in ('accuracy','both_correct','positive_word_choice')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=12)
    args=parser.parse_args()
    probes=[json.loads(l) for l in (ROOT/'generated/probes.jsonl').read_text().splitlines()]
    assert len(probes)==json.loads((ROOT/'generated/manifest.json').read_text())['paired_rows']
    assert torch.cuda.is_available()
    out=args.out_dir;out.mkdir(parents=True,exist_ok=False)
    for file in ('probes.jsonl','manifest.json','variant_matches.csv'):
        (out/('input_'+file)).write_bytes((ROOT/'generated'/file).read_bytes())
    (out/'input_components.json').write_bytes((ROOT/'components.json').read_bytes())
    (out/'evaluator.py').write_bytes(Path(__file__).read_bytes())
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
    model.config.pad_token_id=tokenizer.pad_token_id;model.config.use_cache=False
    keys=list(dict.fromkeys((r[f'Context{c}'],r[f'Target{t}']) for r in probes for c in (1,2) for t in (1,2)))
    for context,target in keys:
        prefix=tokenizer.encode(context,add_special_tokens=False)
        joint=tokenizer.encode(context+' '+target,add_special_tokens=False)
        assert joint[:len(prefix)]==prefix and len(joint)>len(prefix)
    scores={}
    with torch.inference_mode(),(out/'token_scores.jsonl').open('w') as stream:
        for start in range(0,len(keys),192):
            batch=keys[start:start+192]
            values=per_token_conditional_log_likelihood(model,tokenizer,[c for c,t in batch],[t for c,t in batch],device='cuda',batch_size=args.batch_size)
            assert len(values)==len(batch)
            for (c,t),value in zip(batch,values):
                value=value.detach().float().cpu();assert len(value) and torch.isfinite(value).all()
                scores[c,t]=float(value.mean())
                stream.write(json.dumps({'context':c,'target':t,'token_log_probs':value.tolist()})+'\n')
            stream.flush();print(f'Scored {len(scores)}/{len(keys)} sequences',flush=True)
    items=[]
    for r in probes:
        row=dict(r);correct=[];positive=0;ties=0
        for c in (1,2):
            for t in (1,2):row[f'S{c}{t}']=scores[r[f'Context{c}'],r[f'Target{t}']]
            s1,s2=row[f'S{c}1'],row[f'S{c}2']
            pred=0 if s1==s2 else 1 if s1>s2 else 2
            row[f'prediction_context{c}']='tie' if not pred else 'Target'+str(pred)
            row[f'correct_context{c}']=row[f'prediction_context{c}']==r[f'correct_target_for_context{c}']
            correct.append(row[f'correct_context{c}']);ties+=not pred
            word='tie' if not pred else r[f'target{pred}_relation_word']
            positive+=word in ('north','east')
        row.update(accuracy=sum(correct)/2,both_correct=all(correct),positive_word_choices=positive,ties=ties)
        items.append(row)
    write_csv(out/'item_scores.csv',items)
    (out/'item_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    grouped=[]
    for factors in [('axis','event_family'),('axis','event_family','evidence_format'),
        ('axis','evidence_format','location_list_order','numeric_label_order'),
        ('axis','event_family','context_entity_order','target_entity_order'),('axis','event_family','length_band')]:
        buckets=defaultdict(list)
        for r in items:buckets[tuple(r[f] for f in factors)].append(r)
        for key,rs in buckets.items():grouped.append({'grouping':'+'.join(factors),'group':'|'.join(key),**summarize(rs)})
    write_csv(out/'grouped_scores.csv',grouped)
    lookup={r['probe_id']:r for r in items}
    with (out/'input_variant_matches.csv').open() as f:links=list(csv.DictReader(f))
    matched=[]
    for link in links:
        a,b=lookup[link['base_probe_id']],lookup[link['variant_probe_id']]
        matched.append({**link,'base_accuracy':a['accuracy'],'variant_accuracy':b['accuracy'],
            'prediction_changes':sum(a[f'prediction_context{c}']!=b[f'prediction_context{c}'] for c in (1,2)),
            'both_variants_fully_correct':a['both_correct'] and b['both_correct']})
    write_csv(out/'matched_changes.csv',matched)
    results={}
    for axis in ('north_south','east_west'):
        rs=[r for r in items if r['axis']==axis];events=[r for r in rs if r['probe_family']=='event']
        results[axis]={'balanced_events':balanced(events),'pooled_events':summarize(events),
            'families':{family:balanced([r for r in events if r['event_family']==family]) for family in sorted({r['event_family'] for r in events})},
            'controls':{family:summarize([r for r in rs if r['event_family']==family]) for family in ('direct_relation','screen_to_cardinal')}}
    # Validate persisted token means independently of the scoring lookup.
    reconstructed={}
    for line in (out/'token_scores.jsonl').read_text().splitlines():
        r=json.loads(line);reconstructed[r['context'],r['target']]=float(torch.tensor(r['token_log_probs'],dtype=torch.float32).mean())
    for r in items:
        for c in (1,2):
            a,b=[reconstructed[r[f'Context{c}'],r[f'Target{t}']] for t in (1,2)]
            assert a==r[f'S{c}1'] and b==r[f'S{c}2']
            gold=r[f'correct_target_for_context{c}']
            assert r[f'correct_context{c}']==((a>b) if gold=='Target1' else (b>a))
    summary={'model':str(Path(args.model).resolve()),'created_utc':datetime.now(timezone.utc).isoformat(),
        'score':'raw mean full-target conditional log likelihood, including punctuation; no PMI; ties incorrect',
        'pairs':len(items),'unique_sequences':len(keys),'results':results,
        'balancing':'equal weight to evidence formats within each event family, then equal weight to five families',
        'probe_sha256':hashlib.sha256((out/'input_probes.jsonl').read_bytes()).hexdigest(),
        'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float32','attention':'eager',
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'validation':'All persisted token means and gold choices reconstructed successfully.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Cardinal situation evaluation','',summary['score'],'',summary['balancing'],
        '', '| Axis | Family | Accuracy | Both contexts correct | North/east word chosen |','|---|---|---:|---:|---:|']
    for axis,data in results.items():
        for family,value in {**data['families'],'ALL EVENTS':data['balanced_events'],**data['controls']}.items():
            report.append(f"| {axis} | {family} | {value['accuracy']:.2%} | {value['both_correct']:.2%} | {value['positive_word_choice']:.2%} |")
    report+=['','The bridge is conditioning text, not scored target text. Numeric variants outnumber named variants, hence the balanced headline scores. Controls are separate. Repeated variants are not independent scenes. Observer-turn and definition datasets are not included.', '',summary['validation']]
    (out/'report.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)


if __name__=='__main__':main()
