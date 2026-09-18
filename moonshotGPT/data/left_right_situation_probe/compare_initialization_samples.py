#!/usr/bin/env python3
"""Compare nested seed samples without conflating sampling noise with bias."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('extended', type=Path)
    args = parser.parse_args()
    paths = [args.baseline, args.extended]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
    counts = [len(m['seeds']) for m in manifests]
    samples = [np.load(p/'vocabulary_probability_moments.npz') for p in paths]
    assert manifests[1]['seeds'][:counts[0]] == manifests[0]['seeds']
    rows = []
    for arch in ('qwen3','llama'):
        before = [json.loads(l) for l in (paths[0]/f'{arch}_seeds.jsonl').read_text().splitlines()]
        after = [json.loads(l) for l in (paths[1]/f'{arch}_seeds.jsonl').read_text().splitlines()]
        assert after[:len(before)] == before
        assert len(after)==counts[1]*len(manifests[1]['contexts'])
        for context in manifests[0]['contexts']:
            row = {'architecture':arch,'context':context}
            for count,data in zip(counts,samples):
                probs=data[f'{arch}_{context}_mean']
                assert np.isfinite(probs).all() and abs(probs.sum()-1)<1e-10
                ratio=len(probs)*probs
                row[str(count)]={'p05':float(np.quantile(ratio,.05)), 'median':float(np.median(ratio)),
                    'p95':float(np.quantile(ratio,.95)), 'std_across_token_means':float(ratio.std()),
                    'tv_from_uniform':float(np.abs(ratio-1).mean()/2)}
            row['observed_std_ratio'] = row[str(counts[1])]['std_across_token_means']/row[str(counts[0])]['std_across_token_means']
            rows.append(row)
    expected=float(np.sqrt(counts[0]/counts[1]))
    output={'baseline_seeds':counts[0],'extended_seeds':counts[1],
            'sampling_noise_std_ratio_reference':expected,
            'validation':'Original seed records unchanged; all average distributions finite and normalized.', 'comparisons':rows}
    (args.extended/'sample_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
    for arch,ax in zip(('qwen3','llama'),axes):
        for count,data,color in zip(counts,samples,('#8996a5','#2677bd')):
            arr=data[f'{arch}_bos_only_mean'];ratios=arr*len(arr)
            bins=np.linspace(.4,2.1,86)
            assert ratios.min()>=bins[0] and ratios.max()<=bins[-1]
            ax.hist(ratios,bins=bins,weights=np.full(len(arr),1/len(arr)),histtype='step',linewidth=2,color=color,label=f'{count} seeds')
        ax.axvline(1,color='#222',linestyle='--',linewidth=1,label='Uniform token probability')
        ax.set_title('Qwen3' if arch=='qwen3' else 'Llama',fontsize=14)
        ax.set_xlabel('Seed-averaged token probability ÷ uniform probability')
        ax.set_xlim(.4,2.1)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9)
    axes[0].set_ylabel('Share of vocabulary tokens')
    fig.suptitle(f'Does the average move closer to uniform? {counts[0]} → {counts[1]} seeds\nBOS-only input · same original seeds plus ten new seeds',fontsize=15)
    fig.text(.5,.015,'Average each token across seeds first, then plot across tokens. Narrowing is expected as sampling noise decreases.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.88))
    for ext in ('png','pdf'):
        fig.savefig(args.extended/f'20_vs_30_seeds.{ext}',dpi=180)
    plt.close(fig)
    report=['# Initialization sample extension','',f'Extended {counts[0]} to {counts[1]} seeds, preserving all original samples.', '',
        '| Model (BOS only) | 20-seed central 90% | 30-seed central 90% | Spread ratio (30 / 20) |',
        '|---|---:|---:|---:|']
    for row in rows:
        if row['context']=='bos_only':
            a,b=row[str(counts[0])],row[str(counts[1])]
            report.append(f"| {row['architecture']} | {a['p05']:.3f}–{a['p95']:.3f} × uniform | {b['p05']:.3f}–{b['p95']:.3f} × uniform | {row['observed_std_ratio']:.3f} |")
    report.extend(['',f'For independent seed noise around equal expectations, standard deviation scales approximately as 1/sqrt(n), predicting a spread ratio of {expected:.3f}. This comparison is descriptive; the samples overlap and are not independent experiments. It does not prove equal expectations for every token.', '',output['validation']])
    (args.extended/'sample_comparison.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__=='__main__':
    main()
