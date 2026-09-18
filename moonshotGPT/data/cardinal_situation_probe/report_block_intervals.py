#!/usr/bin/env python3
"""Descriptive Wilson intervals for saved block judgments, with dependence caveat."""
import argparse
import csv
import json
from math import sqrt
from pathlib import Path


def wilson(correct,total):
    z=1.959963984540054
    p=correct/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return center-half,center+half


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args();out=args.results
    items=[json.loads(l) for l in (out/'item_scores.jsonl').read_text().splitlines()]
    records=[]
    for axis in ('north_south','east_west'):
        for block in ('direction_definition','opposite_direction','relative_map_position','cardinal','observer_relative'):
            rows=[r for r in items if r['axis']==axis and r.get('probe_family',r.get('query_frame'))==block]
            n=2*len(rows);k=sum(r[f'correct_context{c}'] for r in rows for c in (1,2))
            lo,hi=wilson(k,n)
            records.append({'axis':axis,'block':block,'pairs':len(rows),'judgments':n,'correct':k,
                'accuracy':k/n,'wilson_95_low':lo,'wilson_95_high':hi,
                'both_correct_pairs':sum(r['both_correct'] for r in rows)})
    with (out/'confidence_intervals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=['# Expanded block scores and reference intervals','',
        '95% Wilson intervals treating individual judgments as independent. The paired contexts and reused scene/template variants are dependent, so these intervals are not calibrated estimates of generalization to new scene families. No adjustment for multiple comparisons.', '',
        '| Axis | Block | Correct / judgments | Accuracy | 95% Wilson interval |',
        '|---|---|---:|---:|---:|']
    for r in records:
        report.append(f"| {r['axis']} | {r['block']} | {r['correct']}/{r['judgments']} | {r['accuracy']:.2%} | {r['wilson_95_low']:.1%}–{r['wilson_95_high']:.1%} |")
    (out/'confidence_intervals.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    for axis in ('north_south','east_west'):
        for prep in ('toward','to'):
            rs=[r for r in items if r['axis']==axis and r.get('preposition')==prep]
            print(axis,prep,sum(r['accuracy'] for r in rs)/len(rs),len(rs),'pairs')


if __name__=='__main__':main()
