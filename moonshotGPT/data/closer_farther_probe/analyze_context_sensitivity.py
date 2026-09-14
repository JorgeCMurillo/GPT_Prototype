#!/usr/bin/env python3
"""Compare each fixed target across its matched correct and incorrect contexts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS = {'Target1': 'closer', 'Target2': 'farther', 'Target3': 'unchanged'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    return {
        'n_pairs': n, 'n_target_judgments': 2*n,
        'n_correct': sum(r['a_correct'] + r['b_correct'] for r in rows),
        'context_sensitivity_accuracy': sum(r['a_correct'] + r['b_correct'] for r in rows)/(2*n),
        'target_a_accuracy': sum(r['a_correct'] for r in rows)/n,
        'target_b_accuracy': sum(r['b_correct'] for r in rows)/n,
        'both_correct_count': sum(r['both_correct'] for r in rows),
        'both_correct_fraction': sum(r['both_correct'] for r in rows)/n,
        'both_wrong_fraction': sum(r['a_margin'] < 0 and r['b_margin'] < 0 for r in rows)/n,
        'same_context_preferred_fraction': sum(r['a_margin']*r['b_margin'] < 0 for r in rows)/n,
        'exact_ties': sum((r['a_margin'] == 0) + (r['b_margin'] == 0) for r in rows),
        'mean_a_margin': sum(r['a_margin'] for r in rows)/n,
        'mean_b_margin': sum(r['b_margin'] for r in rows)/n,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True)
    parser.add_argument('--binary-dir',required=True)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    source,binary,out=Path(args.source_dir),Path(args.binary_dir),Path(args.out_dir)
    source_path=source/'item_scores.jsonl'
    items={r['probe_id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    pair_path=binary/'matched_pair_scores.csv'
    matches=[r for r in csv.DictReader(pair_path.open()) if r['method']=='raw']
    rows=[]
    for match in matches:
        a,b=items[match['probe_a']],items[match['probe_b']]
        ta,tb=a['correct_target'],b['correct_target']
        assert ta!=tb
        for k in ['name_id','object_id','numeric_case_id','unit_id','length_band','matched_group_id']:
            assert a[k]==b[k]
        assert a[ta]==b[ta] and a[tb]==b[tb]
        s11,s12=a[f'raw_{ta}_mean'],a[f'raw_{tb}_mean']
        s21,s22=b[f'raw_{ta}_mean'],b[f'raw_{tb}_mean']
        ka,kb=s11-s21,s22-s12
        assert abs(ka+kb-((s11-s12)+(s22-s21)))<1e-10
        if f'pmi_{ta}_mean' in a:
            assert abs(ka-(a[f'pmi_{ta}_mean']-b[f'pmi_{ta}_mean']))<1e-10
            assert abs(kb-(b[f'pmi_{tb}_mean']-a[f'pmi_{tb}_mean']))<1e-10
        rows.append({
            **{k:match[k] for k in ['contrast','matched_group_id','description_family','length_band','probe_a','probe_b']},
            **{k:a[k] for k in ['name_id','object_id','numeric_case_id','unit_id']},
            'target_a_label':LABELS[ta],'target_b_label':LABELS[tb],
            'ContextA':a['Context'],'ContextB':b['Context'],'TargetA':a[ta],'TargetB':a[tb],
            'S_AA':s11,'S_AB':s12,'S_BA':s21,'S_BB':s22,
            'a_margin':ka,'b_margin':kb,'a_correct':ka>0,'b_correct':kb>0,
            'both_correct':ka>0 and kb>0,
        })
    assert len(rows)==7776
    grouped=[]
    for factors in [(),('length_band',),('description_family',),('description_family','length_band'),
                    ('unit_id',),('name_id',),('object_id',),('numeric_case_id',)]:
        groups=defaultdict(list)
        for row in rows:
            groups[(row['contrast'],*(row[k] for k in factors))].append(row)
        for (contrast,*key),group in groups.items():
            grouped.append({'contrast':contrast,'grouping':'+'.join(factors) or 'overall',
                'group':'|'.join(key) or 'all','target_a_label':group[0]['target_a_label'],
                'target_b_label':group[0]['target_b_label'],**summarize(group)})
    source_summary=json.loads((source/'summary.json').read_text())
    summary={
        'model':source_summary['model'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'score_reduction':'mean','metric':'EWoK-style fixed-target context sensitivity',
        'formula_a':'mean_logp(TargetA|ContextA) - mean_logp(TargetA|ContextB)',
        'formula_b':'mean_logp(TargetB|ContextB) - mean_logp(TargetB|ContextA)',
        'correct_rule':'Margin strictly greater than zero; exact ties count as incorrect.',
        'pmi_invariance_verified':True,
        'source_item_scores_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'source_matches_sha256':hashlib.sha256(pair_path.read_bytes()).hexdigest(),
        'source_dir':str(source.resolve()),'binary_dir':str(binary.resolve()),
        'new_model_inference':False,'groups':grouped,
    }
    report=['# Fixed-target context sensitivity','',
        'Qwen3 359M at 19.5k steps; mean full-target log likelihood, reusing the saved scores.',
        'For each target, compare its likelihood in the matching context with its likelihood in the opposite-outcome context. Each contrast has 2,592 matched pairs and 5,184 target judgments. Accuracy averages the two target judgments; both-correct requires success on both.', '',
        '| Contrast | Compact | Standard | Expanded | Overall | Both targets correct |',
        '|---|---:|---:|---:|---:|---:|']
    def get(contrast,grouping,key):
        return next(r for r in grouped if (r['contrast'],r['grouping'],r['group'])==(contrast,grouping,key))
    contrasts=list(dict.fromkeys(r['contrast'] for r in rows))
    for contrast in contrasts:
        vals=[get(contrast,'length_band',band)['context_sensitivity_accuracy'] for band in ['compact','standard','expanded']]
        overall=get(contrast,'overall','all')
        vals += [overall['context_sensitivity_accuracy'],overall['both_correct_fraction']]
        report.append('| '+contrast+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    report += ['', '| Contrast | Description family | Accuracy | Target A accuracy | Target B accuracy | Both correct |',
        '|---|---|---:|---:|---:|---:|']
    for contrast in contrasts:
        for family in ['explicit','situation']:
            r=get(contrast,'description_family',family)
            report.append('| '+contrast+' | '+family+' | '+' | '.join(f'{r[k]:.2%}' for k in ['context_sensitivity_accuracy','target_a_accuracy','target_b_accuracy','both_correct_fraction'])+' |')
    report += ['', 'Target A/B follow the contrast name: closer/farther, closer/unchanged, farther/unchanged.',
        'Subtracting a fixed target-only PMI baseline cancels when comparing that same target across contexts. This invariance was verified for every margin.',
        'A correct result here can coexist with wrong target choice within a context. Requiring both target judgments to be correct helps reveal whether both targets merely prefer the same context.',
        'Situation comparisons involving unchanged distance match movement descriptions against orientation descriptions, so they also change evidence type and sentence structure. Their scores do not isolate distance reasoning alone. Explicit comparisons hold the template fixed while varying the endpoint.',
        'The same scenes, targets, and contexts recur across contrasts and wording variants; these judgments are not independent samples. Exact ties count as incorrect.']
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'pair_scores.csv',rows)
    (out/'pair_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_csv(out/'grouped_scores.csv',grouped)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print('Overall details:')
    for r in grouped:
        if r['grouping']=='overall': print(r)


if __name__=='__main__':
    main()
