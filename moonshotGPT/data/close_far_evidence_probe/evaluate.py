#!/usr/bin/env python3
"""Evaluate close/far evidence contrasts using mean EWoK likelihoods."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from evaluation.ewok import per_token_conditional_log_likelihood,per_token_unconditional_log_likelihood,resolve_bos_token_id
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER,write_csv
from data.spatial_bias_report.generate import generate_one as generate_bias_table


def summarize(rows):
    n=len(rows)
    out={'n_pairs':n,'n_judgments':2*n}
    for m in ['raw','pmi','context']:
        out.update({m+'_accuracy':sum(r[m+'_close_correct']+r[m+'_far_correct'] for r in rows)/(2*n),
            m+'_close_accuracy':sum(r[m+'_close_correct'] for r in rows)/n,
            m+'_far_accuracy':sum(r[m+'_far_correct'] for r in rows)/n,
            m+'_both_correct':sum(r[m+'_close_correct'] and r[m+'_far_correct'] for r in rows)/n,
            m+'_exact_ties':sum((r[m+'_close_margin']==0)+(r[m+'_far_margin']==0) for r in rows)})
    out['raw_close_choice_fraction']=sum((r['raw_close_margin']>0)+(r['raw_far_margin']<0) for r in rows)/(2*n)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=DEFAULT_TOKENIZER)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--dataset-root',type=Path,default=ROOT)
    args=parser.parse_args()
    dataset_root=args.dataset_root.resolve()
    assert torch.cuda.is_available()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    sources={}
    for path in [dataset_root/'components.json',*sorted((dataset_root/'generated').glob('*'))]:
        (out/('input_'+path.name)).write_bytes(path.read_bytes())
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    probes=[json.loads(s) for s in (out/'input_probes.jsonl').read_text().splitlines()]
    manifest=json.loads((out/'input_manifest.json').read_text())
    assert len(probes)==manifest['matched_pairs']
    expanded='scene_id' in probes[0]
    far_wording='far_variant_id' in probes[0]
    wording_pairs='wording_family_id' in probes[0]
    matched_scenes=manifest.get('probe_name')=='close_far_matched_scenes'
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
    model.config.use_cache=False;model.config.pad_token_id=tokenizer.pad_token_id
    seq=list(dict.fromkeys((r[f'Context{c}'],r[f'Target{t}']) for r in probes for c in [1,2] for t in [1,2]))
    lookup={}
    print(f'Scoring {len(seq)} sequences on {torch.cuda.get_device_name(0)}.',flush=True)
    with torch.inference_mode(),(out/'token_scores.jsonl').open('w') as stream:
        for start in range(0,len(seq),128):
            chunk=seq[start:start+128]
            scores=per_token_conditional_log_likelihood(model,tokenizer,[p[0] for p in chunk],[p[1] for p in chunk],device='cuda',batch_size=args.batch_size)
            for (context,target),values in zip(chunk,scores):
                prefix=tokenizer.encode(context,add_special_tokens=False)
                joined=tokenizer.encode(context+' '+target,add_special_tokens=False)
                assert joined[:len(prefix)]==prefix
                ids=joined[len(prefix):]
                values=values.detach().float().cpu()
                assert len(values)==len(ids) and torch.isfinite(values).all()
                lookup[context,target]=float(values.mean())
                stream.write(json.dumps({'context':context,'target':target,'token_ids':ids,'token_log_probs':values.tolist()})+'\n')
            print(f'Scored {min(start+128,len(seq))}/{len(seq)}.',flush=True)
    targets=list(dict.fromkeys(r[k] for r in probes for k in ['Target1','Target2']))
    priors={}
    with torch.inference_mode(),(out/'target_priors.jsonl').open('w') as stream:
        scores=per_token_unconditional_log_likelihood(model,tokenizer,targets,device='cuda',batch_size=args.batch_size)
        for target,values in zip(targets,scores):
            values=values.detach().float().cpu()
            ids=tokenizer.encode(target,add_special_tokens=False)
            assert len(values)==len(ids) and torch.isfinite(values).all()
            priors[target]=float(values.mean())
            stream.write(json.dumps({'target':target,'mean_log_likelihood':priors[target],'token_ids':ids,'token_log_probs':values.tolist()})+'\n')
    results=[]
    for probe in probes:
        r={**probe,'score_reduction':'mean'}
        for c in [1,2]:
            for t in [1,2]:r[f'S{c}{t}']=lookup[probe[f'Context{c}'],probe[f'Target{t}']]
        r.update({'B1':priors[probe['Target1']],'B2':priors[probe['Target2']]})
        margins={'raw':(r['S11']-r['S12'],r['S22']-r['S21']),
            'pmi':(r['S11']-r['B1']-r['S12']+r['B2'],r['S22']-r['B2']-r['S21']+r['B1']),
            'context':(r['S11']-r['S21'],r['S22']-r['S12'])}
        assert abs(sum(margins['raw'])-sum(margins['context']))<1e-10
        for method,(a,b) in margins.items():
            r.update({method+'_close_margin':a,method+'_far_margin':b,method+'_close_correct':a>0,method+'_far_correct':b>0})
        results.append(r)
    grouped=[]
    factors_list=[(),('condition_id',),('target_order',),('condition_id','target_order'),('condition_id','target_order','object_id'),('condition_id','target_order','name_id')]
    scopes={'all':results}
    if expanded and not matched_scenes:
        factors_list += [('scene_id',),('scene_role',),('scene_id','target_order'),('scene_id','context_structure','target_order')]
        scopes['main_scenes']=[r for r in results if r['include_in_primary_scene_summary']]
        expected_main=(manifest['baseline_pairs']+manifest['close_only_pairs_per_family']*manifest['wording_families']) if wording_pairs else (manifest['main_scene_pairs_per_far_variant']*manifest['far_variants_including_baseline'] if far_wording else manifest['primary_placement_pairs'])
        assert len(scopes['main_scenes'])==expected_main
        assert not any(r['gold_requires_unstated_worn_shoes'] for r in scopes['main_scenes'])
    if far_wording:
        factors_list += [('far_variant_id',),('far_variant_id','target_order'),
            ('far_variant_id','context_structure','target_order'),('far_variant_id','scene_id'),
            ('far_variant_id','far_target_repeats_context')]
    if wording_pairs:
        factors_list=[('mode','wording_family_id'),('mode','wording_family_id','target_order'),
            ('mode','wording_family_id','context_structure','target_order'),
            ('mode','wording_family_id','object_id'),('mode','wording_family_id','name_id'),
            ('mode','wording_family_id','close_target_repeats_context','far_target_repeats_context')]
        scopes={'all':results}
    if matched_scenes:
        factors_list += [('setting_id',),('condition_id','setting_id'),
            ('condition_id','context_entity_order','target_order')]
    for scope,subset in scopes.items():
        for factors in factors_list:
            groups=defaultdict(list)
            for r in subset:groups[tuple(r[k] for k in factors)].append(r)
            for key,rows in groups.items():grouped.append({'scope':scope,'grouping':'+'.join(factors) or 'overall','group':'|'.join(str(v) for v in key) or 'all',**summarize(rows)})
    by_id={r['probe_id']:r for r in results}
    matches=[]
    match_records=list(csv.DictReader((out/'input_matches.csv').open()))
    if far_wording:
        match_records += list(csv.DictReader((out/'input_lexical_matches.csv').open()))
    for match in match_records:
        a,b=by_id[match['reference_probe_id']],by_id[match['variant_probe_id']]
        if far_wording:
            assert all(a[k]==b[k] for k in ['Context1','Target1','Target2','S11','S12'])
        if wording_pairs:
            assert all(a[k]==b[k] for k in ['Target1','Target2'])
            if match['control'] in ['close_evidence_only','close_adjective_short_vs_small']:
                assert all(a[k]==b[k] for k in ['Context2','S21','S22'])
            elif match['control']=='far_evidence_only_given_close':
                assert all(a[k]==b[k] for k in ['Context1','S11','S12'])
        if expanded and (match['control'].startswith('feet_vs_') or match['control']=='make_shoes_wearing_explicit'):
            assert all(a[k]==b[k] for k in ['Context2','Target1','Target2','S21','S22'])
        r={**match}
        for m in ['raw','pmi','context']:
            for label in ['close','far']:
                r[f'{m}_{label}_fixed']=not a[f'{m}_{label}_correct'] and b[f'{m}_{label}_correct']
                r[f'{m}_{label}_broken']=a[f'{m}_{label}_correct'] and not b[f'{m}_{label}_correct']
        matches.append(r)
    write_csv(out/'item_scores.csv',results)
    (out/'item_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    write_csv(out/'grouped_scores.csv',grouped);write_csv(out/'matched_scores.csv',matches)
    summary={'model':str(Path(args.model).resolve()),'created_utc':datetime.now(timezone.utc).isoformat(),
        'score_reduction':'mean','dtype':'float32','attention_implementation':'eager','batch_size':args.batch_size,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_name':torch.cuda.get_device_name(0),
        'torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'bos_token_id':resolve_bos_token_id(tokenizer),'source_sha256':sources,
        'conditional_sequences':len(seq),'unique_targets':len(targets),'groups':grouped,
        'dataset_root':str(dataset_root),'dataset_version':manifest['version'],
        'primary_summary':summarize(scopes.get('main_scenes',results)),
        'primary_scope':'main_scenes' if expanded else 'all',
        'pmi_convention':'mean logp(T|C) - mean logp(T|BOS); baseline as written, no leading space or appended EOS.',
        'ties':'Strict positive margins count as correct; exact ties are incorrect.'}
    if far_wording:
        variants=list(dict.fromkeys(r['far_variant_id'] for r in results))
        summary['primary_summary']={v:summarize([r for r in scopes['main_scenes'] if r['far_variant_id']==v]) for v in variants}
        summary['primary_scope']='main_scenes_separately_by_far_variant'
        summary['unchanged_close_context_scores_verified']=True
        summary['new_comparison_note']='Generic far contexts recur across scenes; pooled repetitions are not independent observations.'
    if wording_pairs:
        summary['primary_summary']={r['group']: {k:v for k,v in r.items() if k not in ['scope','grouping','group']} for r in grouped if r['grouping']=='mode+wording_family_id'}
        summary['primary_scope']='separate_mode_and_wording_family'
        summary['matched_fixed_context_scores_verified']=True
    if matched_scenes:
        summary['primary_scope']='all_conditions_reported_separately'
        summary['condition_summaries']={r['group']: {k:v for k,v in r.items() if k not in ['scope','grouping','group']} for r in grouped if r['grouping']=='condition_id'}
        summary['distance_order_summaries']={r['group']: {k:v for k,v in r.items() if k not in ['scope','grouping','group']} for r in grouped if r['grouping']=='condition_id+context_entity_order+target_order' and r['group'].startswith('distance_phrase')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Close/far: matched evidence test','',
        'Qwen3 359M at 19.5k steps. Primary scoring is mean full-target token log likelihood, including punctuation.',
        'Each evidence × target-order condition has 20 matched pairs / 40 balanced context judgments. Names and objects are crossed; identical target sentences are used across all context conditions.', '',
        '| Evidence | Target order | Choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context | Close-choice frequency |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in grouped:
        if r['grouping']=='condition_id+target_order':
            evidence,order=r['group'].split('|')
            report.append('| '+evidence+' | '+order+' | '+' | '.join(f'{r[k]:.2%}' for k in ['raw_accuracy','pmi_accuracy','context_accuracy','raw_both_correct','context_both_correct','raw_close_choice_fraction'])+' |')
    report += ['', 'The original contexts mention the object first. Person-first targets reverse that order; object-first direct-label targets repeat the context verbatim. New fronted-location and person-subject placement contexts mention the person first and are scored under both target orders. Fronted-location contexts preserve the original words and word counts up to capitalization, while changing syntax and token order; person-subject contexts also change grammatical roles and pronouns.',
        'Physical placement uses between the person\'s shoes versus across the full length of a long gym. It assumes the shoes are worn. Evidence conditions vary in length, wording, and specificity, so this comparison cannot attribute changes solely to inference difficulty.',
        'Both accuracy metrics have a 50% random-choice baseline per judgment. Target-only PMI subtraction cancels in fixed-target context sensitivity.',
        'The same contexts recur across the two target orders; entity and template variants are not independent scenarios. This is a controlled diagnostic, not a replacement for the full EWoK benchmark.']
    if expanded and not far_wording and not wording_pairs:
        report=['# Close/far: expanded scene evaluation','',
            'Qwen3 359M at 19.5k steps. All metrics use mean full-target token likelihood, including punctuation.',
            'Primary results include only seven main scenes: 672 pairs / 1,344 balanced judgments. Owned/worn-shoes controls and shared direct/paraphrase controls are reported separately. Main scenes contribute equally.', '',
            '| Context structure | Target order | Judgments | Choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        metrics=['raw_accuracy','pmi_accuracy','context_accuracy','raw_both_correct','context_both_correct']
        for r in grouped:
            if r['scope']=='main_scenes' and r['grouping']=='condition_id+target_order':
                condition,order=r['group'].split('|')
                report.append('| '+condition+' | '+order+' | '+str(r['n_judgments'])+' | '+' | '.join(f'{r[k]:.2%}' for k in metrics)+' |')
        primary=summary['primary_summary']
        report += ['',f"Main-scene overall: choice {primary['raw_accuracy']:.2%}, PMI choice {primary['pmi_accuracy']:.2%}, context sensitivity {primary['context_accuracy']:.2%}.",'',
            '| Main scene | Judgments | Choice | PMI choice | Context sensitivity |',
            '|---|---:|---:|---:|---:|']
        for r in grouped:
            if r['scope']=='main_scenes' and r['grouping']=='scene_id':
                report.append('| '+r['group']+' | '+str(r['n_judgments'])+' | '+' | '.join(f'{r[k]:.2%}' for k in metrics[:3])+' |')
        report += ['', '## Feet and footwear comparison', '',
            'The following averages use all three context structures and both target orders. Each condition has 96 pairs / 192 judgments. Far contexts and targets are identical within each matched comparison; only the close description changes.', '',
            '| Close description | Choice | Close-context accuracy | Far-context accuracy | PMI choice | Context sensitivity |',
            '|---|---:|---:|---:|---:|---:|']
        for r in grouped:
            if r['scope']=='all' and r['grouping']=='scene_id' and r['group'] in ['between_feet','between_owned_shoes','between_worn_shoes']:
                report.append('| '+r['group']+' | '+' | '.join(f'{r[k]:.2%}' for k in ['raw_accuracy','raw_close_accuracy','raw_far_accuracy','pmi_accuracy','context_accuracy'])+' |')
        report += ['', '## Shared lexical controls', '',
            '| Condition | Target order | Choice | PMI choice | Context sensitivity |', '|---|---|---:|---:|---:|']
        for r in grouped:
            if r['scope']=='all' and r['grouping']=='condition_id+target_order' and r['group'].split('|')[0] in ['direct_label','proximity_paraphrase']:
                condition,order=r['group'].split('|')
                report.append('| '+condition+' | '+order+' | '+' | '.join(f'{r[k]:.2%}' for k in metrics[:3])+' |')
        report += ['',
            'Choice compares targets within a context; context sensitivity compares the same target across its matched close/far contexts. A fixed target-only PMI baseline cancels from context sensitivity. Exact ties count as incorrect.',
            'Feet directly locates the object relative to the person. Owned shoes only imply proximity under an unstated worn-shoes assumption; that condition is excluded from the primary score. Explicitly worn shoes states the anchor, but also changes sentence structure and length.',
            'Context fronting preserves words and word count, while the person-subject form also changes grammar/pronouns. Near arrangements and far settings vary together across scenes. Object eligibility differs by scene, and variants are repeated measurements rather than independent scenarios.',
            'Do not count shared lexical controls repeatedly through match links. Identical far-context scores in footwear comparisons were checked. Input snapshots, per-token scores, grouped summaries, and matched corrections/regressions are saved with this report.']
    if far_wording:
        report=['# Close/far: far-description comparison','',
            'Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation.',
            'Each primary row uses the same seven main scenes: 672 pairs / 1,344 context judgments per far variant. Footwear controls are excluded from primary rows. The close contexts and targets are identical across far variants.', '',
            '| Far description | Far-context accuracy | Overall choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for variant,s in summary['primary_summary'].items():
            report.append('| '+variant+' | '+' | '.join(f'{s[k]:.2%}' for k in ['raw_far_accuracy','raw_accuracy','pmi_accuracy','context_accuracy','raw_both_correct','context_both_correct'])+' |')
        report += ['', '## Far accuracy by context and target order', '',
            '| Far description | Close-context structure | Target order | Far accuracy | PMI far accuracy | Overall choice | Context sensitivity |',
            '|---|---|---|---:|---:|---:|---:|']
        for r in grouped:
            if r['scope']=='main_scenes' and r['grouping']=='far_variant_id+context_structure+target_order':
                variant,structure,order=r['group'].split('|')
                report.append('| '+variant+' | '+structure+' | '+order+' | '+' | '.join(f'{r[k]:.2%}' for k in ['raw_far_accuracy','pmi_far_accuracy','raw_accuracy','context_accuracy'])+' |')
        report += ['', '## Repetition control', '',
            'Some direct-label far contexts repeat the correct target verbatim. These rows are labeled; synonym and magnitude conditions do not repeat the target.', '',
            '| Far description | Far target repeats context | Far accuracy | Overall choice |',
            '|---|---|---:|---:|']
        for r in grouped:
            if r['scope']=='main_scenes' and r['grouping']=='far_variant_id+far_target_repeats_context':
                variant,repeat=r['group'].split('|')
                report.append(f"| {variant} | {repeat} | {r['raw_far_accuracy']:.2%} | {r['raw_accuracy']:.2%} |")
        report += ['',
            'Far/distant differs by one word within each context order. Large-distance phrasing also changes syntax; new far descriptions do not specify the original setting. This compares kinds of evidence, not identical geometric descriptions.',
            'All baseline/variant and direct/synonym matches were checked for identical close-context scores and targets. Consequently raw/PMI close-context choice accuracy cannot change across variants, while either context-sensitivity judgment can change.',
            'Generic far contexts recur across scenes, and each close context appears in all four versions. Do not interpret repeated judgments as independent scenes or pool all variants into the primary result.',
            'Both metrics use strict positive margins, with ties incorrect. PMI is a separately labeled mean-token target-only adjustment and cancels in fixed-target context sensitivity. Per-token scores, input snapshots, grouped results (including shoe controls), and matched changes are saved.']
    if wording_pairs:
        report=['# Close/far: matched wording evaluation','',
            'Mean full-target token log likelihood, including punctuation. Raw target choice is primary; PMI subtracts the mean target-only BOS likelihood. Context sensitivity compares the same target across the two contexts.', '',
            '| Mode | Wording family | Pairs | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy | Both correct: choice | Both correct: context |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label,s in summary['primary_summary'].items():
            mode,family=label.split('|')
            report.append('| '+mode+' | '+family+' | '+str(s['n_pairs'])+' | '+' | '.join(f'{s[k]:.2%}' for k in ['raw_accuracy','pmi_accuracy','context_accuracy','raw_close_accuracy','raw_far_accuracy','raw_both_correct','context_both_correct'])+' |')
        report += ['', '## Symmetric pairs by context and target order', '',
            '| Family | Context structure | Target order | Choice | PMI choice | Context sensitivity |',
            '|---|---|---|---:|---:|---:|']
        for r in grouped:
            if r['grouping']=='mode+wording_family_id+context_structure+target_order' and r['group'].startswith('symmetric_pair|'):
                _,family,structure,order=r['group'].split('|')
                report.append('| '+family+' | '+structure+' | '+order+' | '+' | '.join(f'{r[k]:.2%}' for k in ['raw_accuracy','pmi_accuracy','context_accuracy'])+' |')
        report += ['',
            'Symmetric wording pairs are deduplicated across source scenes: 240 pairs / 480 judgments per family. Close-only substitutions retain the baseline far context and targets: 672 pairs per family. Modes have different weighting and must not be pooled or compared as an isolated intervention without using their match links.',
            'Repeated entities and templates are not independent scenes. Direct labels can repeat targets; repetition flags are included in grouped results. Distance phrasing changes syntax and length. These conditions test explicit or paraphrased distance descriptions, not physical-placement inference.',
            'Matched unchanged-context scores and targets were verified. Exact ties count as incorrect. Target-only PMI cancels in context sensitivity. Input snapshots, token scores, priors, and matched changes are saved.']
    if matched_scenes:
        report=['# Close/far: matched-scene evaluation','',
            'Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation. Each condition has 384 pairs / 768 balanced context judgments.', '',
            '| Evidence condition | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy | Both correct: choice | Both correct: context |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition,s in summary['condition_summaries'].items():
            report.append('| '+condition+' | '+' | '.join(f'{s[k]:.2%}' for k in ['raw_accuracy','pmi_accuracy','context_accuracy','raw_close_accuracy','raw_far_accuracy','raw_both_correct','context_both_correct'])+' |')
        report += ['', '## Distance phrase order by target order', '',
            '| Context order | Target order | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |',
            '|---|---|---:|---:|---:|---:|---:|']
        for label,s in summary['distance_order_summaries'].items():
            condition,context_order,target_order=label.split('|')
            report.append('| '+context_order+' | '+target_order+' | '+' | '.join(f'{s[k]:.2%}' for k in ['raw_accuracy','pmi_accuracy','context_accuracy','raw_close_accuracy','raw_far_accuracy'])+' |')
        report += ['',
            'The two distance contexts have the same word multiset apart from order and capitalization. Both target orders are evaluated under each context order. Direct-label and endpoint-placement contexts differ in syntax and length from the distance phrases.',
            'Settings, names, objects, target orders, and templates are crossed repeated measurements rather than independent scenes. Choice compares the two targets within a context; context sensitivity compares one target across close and far contexts. Target-only PMI cancels from context sensitivity. Exact ties count as incorrect.']
    (out/'report.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)
    generate_bias_table(out)


if __name__=='__main__':main()
