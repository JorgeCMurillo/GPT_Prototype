#!/usr/bin/env python3
"""Visualize per-token probabilities averaged over initialization seeds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    arrays = np.load(root / 'vocabulary_probability_moments.npz')
    manifest = json.loads((root / 'manifest.json').read_text())
    n = len(manifest['seeds'])
    contexts = list(manifest['contexts'])
    labels = ['BOS only', 'BOS + “ball”', 'BOS + “left”', 'BOS + “right”',
              '“The objects are”', 'Left-scene prefix', 'Right-scene prefix', 'Summary-bridge prefix']
    colors = {'qwen3': '#2677bd', 'llama': '#c56829'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1.15]})
    table = []
    for col, arch in enumerate(colors):
        color = colors[arch]
        arr = arrays[f'{arch}_bos_only_mean']
        vocab = len(arr)
        ratios = vocab * arr
        ax = axes[0, col]
        bins = np.linspace(0.35, 2.25, 77)
        assert ratios.min() >= bins[0] and ratios.max() <= bins[-1]
        ax.hist(ratios, bins=bins, weights=np.full(vocab, 1/vocab), color=color, alpha=.85)
        ax.axvline(1, color='#222222', linestyle='--', linewidth=1.5, label='Uniform: every token at 1')
        lo, med, hi = np.quantile(ratios, [.05, .5, .95])
        ax.set(title=f'{"Qwen3" if arch == "qwen3" else "Llama"} · BOS-only input',
               xlabel='Seed-averaged token probability ÷ uniform probability',
               ylabel='Share of vocabulary tokens', xlim=(.35, 2.25), ylim=(0,.09))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(frameon=False, fontsize=10)
        ax.text(.97,.75,f'Central 90% of tokens:\n{lo:.2f}–{hi:.2f} × uniform\nMedian: {med:.2f} ×',
                transform=ax.transAxes,ha='right',va='top',fontsize=10)
        ax.grid(axis='y', alpha=.15)
        ax = axes[1, col]
        for i, (context, label) in enumerate(zip(contexts, labels)):
            ratios = vocab * arrays[f'{arch}_{context}_mean']
            low, median, high = np.quantile(ratios, [.05,.5,.95])
            ax.plot([low, high], [i,i], color=color, linewidth=5, solid_capstyle='round', alpha=.65)
            ax.plot(median, i, 'o', color=color, markersize=7)
            table.append({'architecture':arch,'context':context,'vocabulary_size':vocab,'seeds':n,
                'probability_ratio_p05':float(low),'probability_ratio_median':float(median),
                'probability_ratio_p95':float(high),'mean_over_vocabulary':float(ratios.mean())})
        ax.axvline(1, color='#222222', linestyle='--', linewidth=1.5)
        ax.set(yticks=range(len(contexts)), yticklabels=labels, xlim=(.65,1.4),
               xlabel='Seed-averaged token probability ÷ uniform probability',
               title='Across inputs: central 90% of vocabulary tokens')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
    fig.suptitle(f'How uniform is the average untrained model?\n{n} random initializations per architecture · 50,257 vocabulary tokens',
                 fontsize=18, fontweight='bold', y=.985)
    fig.text(.5,.025,
        'Each token is averaged across seeds FIRST; plots then show the spread across vocabulary tokens.\n'
        '1 × uniform = 0.00199% probability. Bars below are token percentiles, not confidence intervals.\n'
        'Remaining spread includes finite-seed noise; it does not by itself establish a systematic token bias.',
        ha='center',va='bottom',fontsize=10,color='#444444')
    fig.tight_layout(rect=(0,.115,1,.93), h_pad=2.0, w_pad=2.5)
    fig.savefig(root/'average_initialization_distributions.png', dpi=180)
    fig.savefig(root/'average_initialization_distributions.pdf')
    plt.close(fig)
    (root/'plot_quantiles.json').write_text(json.dumps(table,indent=2)+'\n')
    print(json.dumps(table[:1]+table[8:9],indent=2))


if __name__=='__main__':
    main()
