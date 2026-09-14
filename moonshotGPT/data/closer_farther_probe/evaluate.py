#!/usr/bin/env python3
"""Three-way closer/farther evaluation; full-target mean likelihood is primary."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from evaluation.ewok import per_token_conditional_log_likelihood, per_token_unconditional_log_likelihood, resolve_bos_token_id
from data.closer_farther_probe.generate import DEFAULT_TOKENIZER, write_csv

TARGETS = ['Target1', 'Target2', 'Target3']
LABELS = dict(zip(TARGETS, ['closer', 'farther', 'unchanged']))


def decision(values, gold):
    best = max(values.values())
    winners = [k for k, v in values.items() if v == best]
    prediction = winners[0] if len(winners) == 1 else 'tie'
    margin = values[gold] - max(v for k, v in values.items() if k != gold)
    return prediction, margin, margin > 0


def summarize(rows, method):
    counts = Counter(r[f'{method}_prediction'] for r in rows)
    return {
        'n_contexts': len(rows), 'n_correct': sum(r[f'{method}_correct'] for r in rows),
        'accuracy': sum(r[f'{method}_correct'] for r in rows) / len(rows),
        'mean_gold_margin': sum(r[f'{method}_gold_margin'] for r in rows) / len(rows),
        **{f'predicted_{LABELS[k]}': counts[k] for k in TARGETS}, 'ties': counts['tie'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_TOKENIZER)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--include-pmi', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'generated/probes.jsonl'
    probes = [json.loads(line) for line in path.read_text().splitlines()]
    manifest = json.loads((ROOT / 'generated/manifest.json').read_text())
    assert len(probes) == manifest['context_items'] == len({r['probe_id'] for r in probes})
    assert torch.cuda.is_available(), 'CUDA is required for this evaluation.'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Snapshot the exact dataset and component tables used by this run.
    for name in ['probes.jsonl', 'manifest.json', 'matched_examples.csv', 'length_matches.csv', 'unit_matches.csv']:
        (out / ('input_' + name)).write_bytes((ROOT / 'generated' / name).read_bytes())
    (out / 'input_components.json').write_bytes((ROOT / 'components.json').read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager',
    ).to('cuda').eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    pairs = list(dict.fromkeys((r['Context'], r[k]) for r in probes for k in TARGETS))
    token_ids = {}
    for context, target in pairs:
        prefix = tokenizer.encode(context, add_special_tokens=False)
        joined = tokenizer.encode(context + ' ' + target, add_special_tokens=False)
        assert joined[:len(prefix)] == prefix and len(joined) > len(prefix)
        token_ids[context, target] = joined[len(prefix):]
    print(f'Loaded {args.model}; scoring {len(pairs)} conditional sequences on {torch.cuda.get_device_name(0)}.', flush=True)
    lookup = {}
    with torch.inference_mode(), (out / 'token_scores.jsonl').open('w') as stream:
        for start in range(0, len(pairs), 384):
            chunk = pairs[start:start + 384]
            values = per_token_conditional_log_likelihood(
                model, tokenizer, [p[0] for p in chunk], [p[1] for p in chunk],
                device='cuda', batch_size=args.batch_size,
            )
            assert len(values) == len(chunk)
            for pair, scores in zip(chunk, values):
                scores = scores.detach().float().cpu()
                assert len(scores) == len(token_ids[pair]) and torch.isfinite(scores).all()
                lookup[pair] = float(scores.mean())
                stream.write(json.dumps({'context': pair[0], 'target': pair[1],
                    'target_token_ids': token_ids[pair], 'target_token_log_probs': scores.tolist()}) + '\n')
            stream.flush()
            print(f'Scored {min(start + 384, len(pairs))}/{len(pairs)} conditional sequences.', flush=True)
    priors = {}
    if args.include_pmi:
        targets = list(dict.fromkeys(r[k] for r in probes for k in TARGETS))
        with torch.inference_mode():
            values = per_token_unconditional_log_likelihood(model, tokenizer, targets, device='cuda', batch_size=args.batch_size)
        prior_rows = []
        for target, scores in zip(targets, values):
            scores = scores.detach().float().cpu()
            ids = tokenizer.encode(target, add_special_tokens=False)
            assert len(scores) == len(ids) and torch.isfinite(scores).all()
            priors[target] = float(scores.mean())
            prior_rows.append({'target': target, 'mean_log_likelihood': priors[target],
                'token_ids': ids, 'token_log_probs': scores.tolist()})
        (out / 'target_priors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prior_rows))
    methods = ['raw'] + (['pmi'] if args.include_pmi else [])
    results = []
    for probe in probes:
        row = {**probe, 'score_reduction': 'mean'}
        raw = {k: lookup[probe['Context'], probe[k]] for k in TARGETS}
        for k in TARGETS:
            assert len(token_ids[probe['Context'], probe[k]]) == probe[k + '_conditional_token_count']
        for method in methods:
            values = raw if method == 'raw' else {k: raw[k] - priors[probe[k]] for k in TARGETS}
            pred, margin, correct = decision(values, probe['correct_target'])
            row.update({f'{method}_{k}_mean': v for k, v in values.items()})
            row.update({f'{method}_prediction': pred, f'{method}_gold_margin': margin, f'{method}_correct': correct})
        results.append(row)
    write_csv(out / 'item_scores.csv', results)
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    groupings = [(), ('condition_id',), ('length_band',), ('outcome',), ('unit_id',), ('evidence_type',),
        ('template_id',), ('name_id',), ('object_id',), ('numeric_case_id',),
        ('condition_id', 'length_band'), ('condition_id', 'length_band', 'outcome'),
        ('condition_id', 'unit_id'), ('length_band', 'outcome')]
    grouped, confusion = [], []
    for factors in groupings:
        groups = defaultdict(list)
        for row in results:
            groups[tuple(row[k] for k in factors)].append(row)
        for key, group in groups.items():
            for method in methods:
                descriptor = {'method': method, 'grouping': '+'.join(factors) or 'overall', 'group': '|'.join(key) or 'all'}
                grouped.append({**descriptor, **summarize(group, method)})
                if factors in [(), ('condition_id',), ('condition_id', 'length_band')]:
                    counts = Counter((r['outcome'], r[f'{method}_prediction']) for r in group)
                    for gold in LABELS.values():
                        for pred in TARGETS + ['tie']:
                            confusion.append({**descriptor, 'gold': gold, 'predicted': LABELS.get(pred, pred), 'count': counts[gold, pred]})
    write_csv(out / 'grouped_scores.csv', grouped)
    write_csv(out / 'confusion_matrices.csv', confusion)
    by_id = {r['probe_id']: r for r in results}
    match_summary = []
    for source in ['length_matches', 'unit_matches', 'matched_examples']:
        matches = list(csv.DictReader((out / f'input_{source}.csv').open()))
        comparisons = []
        for match in matches:
            a, b = ('explicit_probe_id', 'situation_probe_id') if source == 'matched_examples' else ('reference_probe_id', 'variant_probe_id')
            reference, variant = by_id[match[a]], by_id[match[b]]
            assert reference['correct_target'] == variant['correct_target']
            for method in methods:
                comparisons.append({**match, 'method': method,
                    'reference_correct': reference[f'{method}_correct'], 'variant_correct': variant[f'{method}_correct'],
                    'both_correct': reference[f'{method}_correct'] and variant[f'{method}_correct'],
                    'same_prediction': reference[f'{method}_prediction'] == variant[f'{method}_prediction'],
                    'wrong_to_right': not reference[f'{method}_correct'] and variant[f'{method}_correct'],
                    'right_to_wrong': reference[f'{method}_correct'] and not variant[f'{method}_correct']})
        write_csv(out / f'{source}_scores.csv', comparisons)
        for method in methods:
            group = [r for r in comparisons if r['method'] == method]
            match_summary.append({'match_source': source, 'method': method, 'n_matches': len(group),
                **{key: sum(r[key] for r in group) / len(group) for key in ['reference_correct', 'variant_correct', 'both_correct', 'same_prediction']},
                **{key: sum(r[key] for r in group) for key in ['wrong_to_right', 'right_to_wrong']}})
    write_csv(out / 'match_summary.csv', match_summary)
    consistency = []
    for kind in ['wording_lengths', 'explicit_outcomes', 'situation_outcomes']:
        groups = defaultdict(list)
        for row in results:
            if kind == 'wording_lengths':
                key = row['length_match_id']
            else:
                if (row['condition_id'] == 'explicit_distance') != (kind == 'explicit_outcomes'):
                    continue
                key = row['matched_group_id']
            groups[key].append(row)
        for key, group in groups.items():
            assert len(group) == 3
            for method in methods:
                consistency.append({'kind': kind, 'group_id': key, 'method': method,
                    'n_correct': sum(r[f'{method}_correct'] for r in group),
                    'all_correct': all(r[f'{method}_correct'] for r in group),
                    'same_prediction': len({r[f'{method}_prediction'] for r in group}) == 1})
    write_csv(out / 'consistency.csv', consistency)
    summary = {
        'model': str(Path(args.model).resolve()), 'dataset_version': manifest['version'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'probe_sha256': hashlib.sha256((out / 'input_probes.jsonl').read_bytes()).hexdigest(),
        'components_sha256': hashlib.sha256((out / 'input_components.json').read_bytes()).hexdigest(),
        'primary_method': 'raw', 'score_reduction': 'mean', 'dtype': 'float32', 'attention_implementation': 'eager',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device_name': torch.cuda.get_device_name(0),
        'batch_size': args.batch_size, 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'bos_token_id': resolve_bos_token_id(tokenizer), 'unique_conditional_sequences': len(pairs),
        'unique_target_priors': len(priors), 'ties': 'Exact maximum ties count as incorrect; no arbitrary slot tie-break.',
        'pmi_convention': 'Secondary: mean logp(T|C) minus mean logp(T|BOS); target-only text as written, no added leading space, no appended EOS.',
        'overall': {m: summarize(results, m) for m in methods}, 'groups': grouped, 'match_summary': match_summary,
        'consistency': [{
            'kind': kind, 'method': method, 'n_groups': len(subset),
            'all_correct_fraction': sum(r['all_correct'] for r in subset) / len(subset),
            'same_prediction_fraction': sum(r['same_prediction'] for r in subset) / len(subset),
        } for kind in ['wording_lengths', 'explicit_outcomes', 'situation_outcomes'] for method in methods
          for subset in [[r for r in consistency if r['kind'] == kind and r['method'] == method]]],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Qwen3 closer/farther evaluation', '', f"Dataset v{manifest['version']}: {len(results):,} contexts, three targets per context.",
        'Primary scoring: full-target mean token log likelihood, including punctuation. Uniform three-way chance: 33.33%.', '',
        '| Scenario | Length | N | Raw accuracy | PMI accuracy (secondary) |', '|---|---|---:|---:|---:|']
    for row in grouped:
        if row['method'] == 'raw' and row['grouping'] == 'condition_id+length_band':
            other = next((r for r in grouped if r['method'] == 'pmi' and r['grouping'] == row['grouping'] and r['group'] == row['group']), None)
            condition, band = row['group'].split('|')
            pmi = f"{other['accuracy']:.2%}" if other else 'not computed'
            report.append(f"| {condition} | {band} | {row['n_contexts']} | {row['accuracy']:.2%} | {pmi} |")
    report += ['', f"Overall raw accuracy: {summary['overall']['raw']['accuracy']:.2%}.", '',
        'Lengths also differ in syntax, punctuation, and reference wording; these are matched wording-and-length comparisons.',
        'The three conditions have different outcome coverage. Inspect per-outcome results and confusion matrices before comparing pooled scores.',
        'Rows reuse numeric cases, entities, and templates; they are not independent scenarios.',
        'PMI uses the previous repository convention and remains secondary. Per-token scores, all item scores, matches, and input snapshots are saved alongside this report.']
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(summary['overall'], indent=2), flush=True)
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
