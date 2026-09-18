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
    sources={'definitions':ROOT.parent/'cardinal_definition_probe',
             'observer_turn':ROOT/'observer_turn'}
    assert torch.cuda.is_available()
    out=args.out_dir;out.mkdir(parents=True,exist_ok=False)
    probes=[]
    for dataset,source in sources.items():
        subset=[json.loads(l) for l in (source/'generated/probes.jsonl').read_text().splitlines()]
        assert len(subset)==json.loads((source/'generated/manifest.json').read_text())['paired_rows']
        for row in subset:
            probes.append({**row,'dataset':dataset})
        snapshot=out/dataset;snapshot.mkdir()
        for name in ('probes.jsonl','manifest.json','variant_matches.csv'):
            (snapshot/('input_'+name)).write_bytes((source/'generated'/name).read_bytes())
        (snapshot/'generator.py').write_bytes((source/'generate.py').read_bytes())
        if (source/'components.json').exists():
            (snapshot/'input_components.json').write_bytes((source/'components.json').read_bytes())
    (out/'input_probes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in probes))
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
        row=dict(r);correct=[];first_choices=0;ties=0
        for c in (1,2):
            for t in (1,2):row[f'S{c}{t}']=scores[r[f'Context{c}'],r[f'Target{t}']]
            s1,s2=row[f'S{c}1'],row[f'S{c}2']
            pred=0 if s1==s2 else 1 if s1>s2 else 2
            row[f'prediction_context{c}']='tie' if not pred else 'Target'+str(pred)
            row[f'correct_context{c}']=row[f'prediction_context{c}']==r[f'correct_target_for_context{c}']
            correct.append(row[f'correct_context{c}']);ties+=not pred
            first_choices+=pred==1
        row.update(accuracy=sum(correct)/2,both_correct=all(correct),first_choices=first_choices,ties=ties,
                   predictions_same=row['prediction_context1']==row['prediction_context2'] and not ties)
        items.append(row)
    fields=list(dict.fromkeys(k for r in items for k in r))
    write_csv(out/'item_scores.csv',[{k:r.get(k,'') for k in fields} for r in items])
    (out/'item_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    groups=[]
    factors_by_dataset={
        'definitions':[('axis','probe_family'),('axis','probe_family','mapping_direction'),
                       ('axis','probe_family','reference_surface'),('axis','wording_family','wording_id'),
                       ('axis','probe_family','design_extension'),('axis','mapping_direction','preposition')],
        'observer_turn':[('axis','query_frame'),('axis','query_frame','length_band'),
                         ('query_frame','initial_facing'),('axis','query_frame','description_structure')]}
    for dataset,factor_sets in factors_by_dataset.items():
        subset=[r for r in items if r['dataset']==dataset]
        for factors in factor_sets:
            buckets=defaultdict(list)
            for row in subset:buckets[tuple(row[f] for f in factors)].append(row)
            for key,rs in buckets.items():
                groups.append({'dataset':dataset,'grouping':'+'.join(factors),'group':'|'.join(key),
                    'pairs':len(rs),'accuracy':mean(r['accuracy'] for r in rs),
                    'both_correct':mean(r['both_correct'] for r in rs),
                    'context1_accuracy':mean(r['correct_context1'] for r in rs),
                    'context2_accuracy':mean(r['correct_context2'] for r in rs),
                    'first_answer_fraction':sum(r['first_choices'] for r in rs)/(2*len(rs)),
                    'predictions_same_fraction':mean(r['predictions_same'] for r in rs),
                    'ties':sum(r['ties'] for r in rs)})
    write_csv(out/'grouped_scores.csv',groups)
    lookup={r['probe_id']:r for r in items}
    matches=[]
    for dataset in sources:
        with (out/dataset/'input_variant_matches.csv').open() as f: links=list(csv.DictReader(f))
        for link in links:
            a,b=lookup[link['base_probe_id']],lookup[link['variant_probe_id']]
            matches.append({'dataset':dataset,**link,
                'base_accuracy':a['accuracy'],'variant_accuracy':b['accuracy'],
                'both_variants_fully_correct':a['both_correct'] and b['both_correct'],
                'base_predictions_same':a['predictions_same'],'variant_predictions_same':b['predictions_same']})
    # Link tables have different schemas, so normalize columns before CSV output.
    fields=list(dict.fromkeys(k for r in matches for k in r))
    write_csv(out/'matched_changes.csv',[{k:r.get(k,'') for k in fields} for r in matches])
    reconstructed={}
    for line in (out/'token_scores.jsonl').read_text().splitlines():
        r=json.loads(line)
        reconstructed[r['context'],r['target']]=float(torch.tensor(r['token_log_probs'],dtype=torch.float32).mean())
    for r in items:
        for c in (1,2):
            a,b=[reconstructed[r[f'Context{c}'],r[f'Target{t}']] for t in (1,2)]
            assert a==r[f'S{c}1'] and b==r[f'S{c}2']
            assert r[f'correct_context{c}']==((a>b) if r[f'correct_target_for_context{c}']=='Target1' else (b>a))
    summary={'model':str(Path(args.model).resolve()),'created_utc':datetime.now(timezone.utc).isoformat(),
        'score':'raw mean full-target conditional log likelihood; no PMI; ties incorrect; explicit per-context gold labels',
        'pairs':len(items),'unique_sequences':len(keys),'groups':groups,
        'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float32','attention':'eager',
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'probe_sha256':hashlib.sha256((out/'input_probes.jsonl').read_bytes()).hexdigest(),
        'validation':'All persisted token means and per-context gold choices independently reconstructed.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Cardinal definitions and observer turns','',summary['score'],'',
            '| Dataset | Axis / block | Pairs | Accuracy | Both contexts correct |',
            '|---|---|---:|---:|---:|']
    for g in groups:
        if g['grouping'] in ('axis+probe_family','axis+query_frame'):
            report.append(f"| {g['dataset']} | {g['group'].replace('|', ' / ')} | {g['pairs']} | {g['accuracy']:.2%} | {g['both_correct']:.2%} |")
    report+=['','Observer-turn rows compare no turn (C1) with a half-turn (C2). Cardinal gold stays unchanged; observer-relative gold switches. Both switch directions and both cardinal answers are balanced. Prediction stability by itself is not correct invariance.',
             'Relative-position definition rows include both map and globe variants, separately broken down in grouped_scores.csv. These small sets reuse wording and concepts and are not independent scene samples.',
             '',summary['validation']]
    (out/'report.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)


if __name__=='__main__':main()
