#!/usr/bin/env python3
"""Evaluate nonnumeric and aligned numeric binary probes with shared EWoK scoring."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]))
from evaluation.ewok import per_token_conditional_log_likelihood, per_token_unconditional_log_likelihood, resolve_bos_token_id
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv
from data.spatial_bias_report.generate import generate_one as generate_bias_table


def summarize(rows):
    n=len(rows)
    result={'n_pairs':n,'n_contexts':2*n}
    for method in ['raw','pmi','context']:
        result.update({
            method+'_accuracy':sum(r[method+'_a_correct']+r[method+'_b_correct'] for r in rows)/(2*n),
            method+'_closer_accuracy':sum(r[method+'_a_correct'] for r in rows)/n,
            method+'_farther_accuracy':sum(r[method+'_b_correct'] for r in rows)/n,
            method+'_both_correct':sum(r[method+'_a_correct'] and r[method+'_b_correct'] for r in rows)/n,
            method+'_exact_ties':sum((r[method+'_a_margin']==0)+(r[method+'_b_margin']==0) for r in rows),
        })
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=DEFAULT_TOKENIZER)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    assert torch.cuda.is_available(),'CUDA required.'
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    for path in [ROOT/'components.json',*sorted((ROOT/'generated').glob('*.jsonl')),*sorted((ROOT/'generated').glob('*.csv')),ROOT/'generated/manifest.json']:
        (out/('input_'+path.name)).write_bytes(path.read_bytes())
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    records=[]
    pairs=[]
    for dataset,prefix in [('non_numeric',''),('numeric_aligned','numeric_aligned_')]:
        records.extend({**json.loads(s),'dataset':dataset} for s in (out/f'input_{prefix}probes.jsonl').read_text().splitlines())
        pairs.extend({**json.loads(s),'dataset':dataset} for s in (out/f'input_{prefix}pairs.jsonl').read_text().splitlines())
    manifest=json.loads((out/'input_manifest.json').read_text())
    expected_contexts=manifest['context_items']+manifest['numeric_aligned_context_items']
    expected_pairs=manifest['matched_pairs']+manifest['numeric_aligned_pairs']
    assert len(records)==expected_contexts and len(pairs)==expected_pairs
    assert len({r['probe_id'] for r in records})==expected_contexts
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
    model.config.use_cache=False
    model.config.pad_token_id=tokenizer.pad_token_id
    sequences=list(dict.fromkeys((r['Context'],r[k]) for r in records for k in ['Target1','Target2']))
    ids={}
    for c,t in sequences:
        prefix=tokenizer.encode(c,add_special_tokens=False)
        joined=tokenizer.encode(c+' '+t,add_special_tokens=False)
        assert joined[:len(prefix)]==prefix
        ids[c,t]=joined[len(prefix):]
    print(f'Loaded model on {torch.cuda.get_device_name(0)}; scoring {len(sequences)} sequences.',flush=True)
    lookup={}
    with torch.inference_mode(),(out/'token_scores.jsonl').open('w') as stream:
        for start in range(0,len(sequences),256):
            chunk=sequences[start:start+256]
            values=per_token_conditional_log_likelihood(model,tokenizer,[p[0] for p in chunk],[p[1] for p in chunk],device='cuda',batch_size=args.batch_size)
            assert len(values)==len(chunk)
            for pair,scores in zip(chunk,values):
                scores=scores.detach().float().cpu()
                assert len(scores)==len(ids[pair]) and len(scores)>0 and torch.isfinite(scores).all()
                lookup[pair]=float(scores.mean())
                stream.write(json.dumps({'context':pair[0],'target':pair[1],'target_token_ids':ids[pair],'target_token_log_probs':scores.tolist()})+'\n')
            stream.flush()
            print(f'Scored {min(start+256,len(sequences))}/{len(sequences)}.',flush=True)
    targets=list(dict.fromkeys(r[k] for r in records for k in ['Target1','Target2']))
    priors={}
    with torch.inference_mode(),(out/'target_priors.jsonl').open('w') as stream:
        values=per_token_unconditional_log_likelihood(model,tokenizer,targets,device='cuda',batch_size=args.batch_size)
        for target,scores in zip(targets,values):
            scores=scores.detach().float().cpu()
            token_ids=tokenizer.encode(target,add_special_tokens=False)
            assert len(scores)==len(token_ids) and torch.isfinite(scores).all()
            priors[target]=float(scores.mean())
            stream.write(json.dumps({'target':target,'mean_log_likelihood':priors[target],'token_ids':token_ids,'token_log_probs':scores.tolist()})+'\n')
    results=[]
    for row in records:
        scored={**row,'score_reduction':'mean'}
        for k in ['Target1','Target2']:
            assert len(ids[row['Context'],row[k]])==row[k+'_conditional_token_count']
            scored['raw_'+k+'_mean']=lookup[row['Context'],row[k]]
            scored['pmi_'+k+'_mean']=lookup[row['Context'],row[k]]-priors[row[k]]
        for method in ['raw','pmi']:
            m=scored[method+'_Target1_mean']-scored[method+'_Target2_mean']
            pred='Target1' if m>0 else 'Target2' if m<0 else 'tie'
            scored[method+'_prediction']=pred
            scored[method+'_correct']=pred==row['correct_target']
        results.append(scored)
    by_id={r['probe_id']:r for r in results}
    pair_scores=[]
    for pair in pairs:
        a,b=by_id[pair['closer_probe_id']],by_id[pair['farther_probe_id']]
        assert a['correct_target']=='Target1' and b['correct_target']=='Target2'
        assert all(a[k]==b[k] for k in ['Target1','Target2','name_id','object_id','length_band'])
        scored={**pair,'template_id':a['template_id'],'template_family':a['template_family']}
        for method in ['raw','pmi']:
            ma=a[method+'_Target1_mean']-a[method+'_Target2_mean']
            mb=b[method+'_Target2_mean']-b[method+'_Target1_mean']
            scored.update({method+'_a_margin':ma,method+'_b_margin':mb,method+'_a_correct':ma>0,method+'_b_correct':mb>0})
        ka=a['raw_Target1_mean']-b['raw_Target1_mean']
        kb=b['raw_Target2_mean']-a['raw_Target2_mean']
        assert abs(ka-(a['pmi_Target1_mean']-b['pmi_Target1_mean']))<1e-10
        assert abs(kb-(b['pmi_Target2_mean']-a['pmi_Target2_mean']))<1e-10
        assert abs(ka+kb-scored['raw_a_margin']-scored['raw_b_margin'])<1e-10
        scored.update({'context_a_margin':ka,'context_b_margin':kb,'context_a_correct':ka>0,'context_b_correct':kb>0})
        pair_scores.append(scored)
    grouped=[]
    for factors in [(),('length_band',),('template_family',),('object_id',),('name_id',),('length_band','object_id')]:
        groups=defaultdict(list)
        for row in pair_scores:groups[(row['dataset'],*(row[k] for k in factors))].append(row)
        for (dataset,*key),group in groups.items():
            grouped.append({'dataset':dataset,'grouping':'+'.join(factors) or 'overall','group':'|'.join(key) or 'all',**summarize(group)})
    write_csv(out/'item_scores.csv',results)
    (out/'item_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    write_csv(out/'pair_scores.csv',pair_scores)
    (out/'pair_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pair_scores))
    write_csv(out/'grouped_scores.csv',grouped)
    for filename,ref_key,var_key in [('numeric_matches','numeric_probe_id','non_numeric_probe_id'),('minimal_control_matches','reference_probe_id','variant_probe_id'),('length_matches','reference_probe_id','variant_probe_id')]:
        comparisons=[]
        for match in csv.DictReader((out/f'input_{filename}.csv').open()):
            ref,var=by_id[match[ref_key]],by_id[match[var_key]]
            assert ref['correct_target']==var['correct_target']
            for method in ['raw','pmi']:
                comparisons.append({**match,'method':method,'reference_correct':ref[method+'_correct'],
                    'variant_correct':var[method+'_correct'],
                    'same_prediction':ref[method+'_prediction']==var[method+'_prediction'],
                    'wrong_to_right':not ref[method+'_correct'] and var[method+'_correct'],
                    'right_to_wrong':ref[method+'_correct'] and not var[method+'_correct']})
        write_csv(out/(filename+'_scores.csv'),comparisons)
    summary={
        'model':str(Path(args.model).resolve()),'created_utc':datetime.now(timezone.utc).isoformat(),
        'dataset_version':json.loads((out/'input_manifest.json').read_text())['version'],
        'source_sha256':sources,'score_reduction':'mean','primary_method':'raw_completion_choice',
        'dtype':'float32','attention_implementation':'eager','batch_size':args.batch_size,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_name':torch.cuda.get_device_name(0),
        'torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'bos_token_id':resolve_bos_token_id(tokenizer),'unique_conditional_sequences':len(sequences),
        'unique_target_priors':len(priors),'groups':grouped,'pmi_context_invariance_verified':True,
        'pmi_convention':'mean logp(T|C) - mean logp(T|BOS); baseline text as written, no added leading space, no appended EOS.',
        'ties':'Strict positive margin required; exact ties count as incorrect.',
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Qwen3 closer/farther: nonnumeric and matched numeric results','',
        'Qwen3 359M, 19.5k checkpoint. All metrics use mean full-target token log likelihood, including punctuation.',
        'Choice holds context fixed and compares targets. Context sensitivity holds target fixed and compares the paired contexts. PMI adjusts choice using target-only means; it leaves context sensitivity unchanged.', '',
        '| Dataset | Form | Contexts | Choice | PMI choice | Context sensitivity | Both contexts correct (choice) | Both targets correct (context sensitivity) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in grouped:
        if r['grouping'] in ['overall','length_band']:
            report.append(f"| {r['dataset']} | {r['group']} | {r['n_contexts']} | "+' | '.join(f"{r[k]:.2%}" for k in ['raw_accuracy','pmi_accuracy','context_accuracy','raw_both_correct','context_both_correct'])+' |')
    report+=['','Each form is balanced between closer and farther. Both accuracy metrics have a 50% random-choice baseline per individual judgment.',
        'Compare the numeric aligned forms only with their matching compact/standard/expanded nonnumeric forms, not with the full five-form nonnumeric average.',
        'Each of the 120 aligned nonnumeric contexts has 36 numeric counterparts. Repeated contexts and entity/template variants are not independent observations.',
        'Minimal forms use ordinary movement assumptions; the unqualified form does not state that the object is stationary. Minimal forms leave endpoints unstated. Aligned forms make the trajectory constraints explicit.',
        'Input snapshots, individual scores, per-token scores, priors, matches, and diagnostic groupings are saved alongside this report.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    generate_bias_table(out)
    print('\n'.join(report),flush=True)


if __name__=='__main__':main()
