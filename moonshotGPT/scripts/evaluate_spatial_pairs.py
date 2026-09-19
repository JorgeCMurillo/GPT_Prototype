#!/usr/bin/env python3
"""Evaluate explicit-gold spatial pairs, including minimal cardinal and front/behind."""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def decisions(row, scores):
    result = dict(row)
    for reduction in ('mean', 'sum'):
        correct = []
        ties = 0
        for c in (1, 2):
            gold = row[f'correct_target_for_context{c}']
            if gold not in ('Target1', 'Target2'):
                raise ValueError(f'Invalid gold: {gold}')
            values = [scores[row[f'Context{c}'], row[f'Target{t}']][reduction] for t in (1, 2)]
            if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite score')
            for t, value in enumerate(values, 1): result[f'{reduction}_S{c}{t}'] = value
            prediction = 'tie' if values[0] == values[1] else 'Target1' if values[0] > values[1] else 'Target2'
            result[f'{reduction}_prediction_context{c}'] = prediction
            result[f'{reduction}_correct_context{c}'] = prediction == gold
            correct.append(prediction == gold)
            ties += prediction == 'tie'
        result[f'{reduction}_accuracy'] = mean(correct)
        result[f'{reduction}_both_correct'] = all(correct)
        result[f'{reduction}_ties'] = ties
    result.update(accuracy=result['mean_accuracy'], both_correct=result['mean_both_correct'], score_reduction='mean')
    return result


METRICS = ('mean_accuracy', 'mean_both_correct', 'sum_accuracy', 'sum_both_correct')


def summary(rows):
    if not rows: return None
    return {'pairs': len(rows), **{k: mean(r[k] for r in rows) for k in METRICS},
            'mean_ties': sum(r['mean_ties'] for r in rows), 'sum_ties': sum(r['sum_ties'] for r in rows)}


def balanced(rows):
    """Variants -> case -> evidence format -> family; handle incomplete format/case crossings."""
    if not rows: return None
    cells = defaultdict(list)
    for r in rows: cells[r['event_family'], r['evidence_format'], r['case_id']].append(r)
    formats = defaultdict(list)
    for (family, fmt, case), rs in cells.items():
        formats[family, fmt].append({k: mean(r[k] for r in rs) for k in METRICS})
    families = defaultdict(list)
    for (family, fmt), cases in formats.items():
        families[family].append({k: mean(c[k] for c in cases) for k in METRICS})
    by_family = {f: {k: mean(v[k] for v in values) for k in METRICS} for f, values in families.items()}
    return {'pairs': len(rows), 'families': by_family,
            **{k: mean(f[k] for f in by_family.values()) for k in METRICS}}


def report_groups(items, dataset):
    groups = {}
    if dataset == 'minimal_cardinal':
        for axis in ('north_south', 'east_west'):
            axis_rows = [r for r in items if r['axis'] == axis]
            for label, predicate in (
                ('main_movement', lambda r: r['probe_family'] == 'event' and r['persistence_cue'] != 'explicit'),
                ('explicit_movement', lambda r: r['probe_family'] == 'event' and r['persistence_cue'] == 'explicit'),
                ('static_controls', lambda r: r['probe_family'] == 'control'),
                ('observer_turns', lambda r: r['probe_family'] == 'observer_turn'),
            ):
                groups[f'{axis}/{label}'] = balanced([r for r in axis_rows if predicate(r)])
    else:
        for block in ('event', 'control', 'observer_turn'):
            groups[block] = balanced([r for r in items if r['probe_family'] == block])
    return groups


def normalize_link(link):
    if 'base_probe_id' in link:
        return dict(link)
    return {**link, 'control': link['match_type'],
            'base_probe_id': link['probe_id_a'], 'variant_probe_id': link['probe_id_b']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, choices=('minimal_cardinal', 'front_behind'))
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=12)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.ewok import per_token_conditional_log_likelihood
    source = ROOT / 'data' / ('cardinal_situation_probe/minimal_relations' if args.dataset == 'minimal_cardinal'
                              else 'front_behind_situation_probe')
    probes = [json.loads(l) for l in (source / 'generated/probes.jsonl').read_text().splitlines()]
    manifest = json.loads((source / 'generated/manifest.json').read_text())
    assert len(probes) == manifest['paired_rows']
    assert len({r['probe_id'] for r in probes}) == len(probes)
    assert torch.cuda.is_available()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for name in ('probes.jsonl', 'manifest.json', 'variant_matches.csv'):
        (out / ('input_' + name)).write_bytes((source / 'generated' / name).read_bytes())
    for name in ('generate.py', 'components.json'):
        if (source / name).exists(): (out / ('input_' + name)).write_bytes((source / name).read_bytes())
    (out / 'evaluator.py').write_bytes(Path(__file__).read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    keys = list(dict.fromkeys((r[f'Context{c}'], r[f'Target{t}']) for r in probes for c in (1, 2) for t in (1, 2)))
    for context, target in keys:
        prefix = tokenizer.encode(context, add_special_tokens=False)
        joint = tokenizer.encode(context + ' ' + target, add_special_tokens=False)
        assert joint[:len(prefix)] == prefix and len(joint) > len(prefix)
    scores = {}
    with torch.inference_mode(), (out / 'token_scores.jsonl').open('w') as stream:
        for start in range(0, len(keys), 192):
            batch = keys[start:start + 192]
            values = per_token_conditional_log_likelihood(model, tokenizer,
                [c for c, t in batch], [t for c, t in batch], device='cuda', batch_size=args.batch_size)
            assert len(values) == len(batch)
            for (c, t), value in zip(batch, values):
                value = value.detach().float().cpu()
                assert len(value) and torch.isfinite(value).all()
                scores[c, t] = {'mean': float(value.mean()), 'sum': float(value.sum()), 'count': len(value)}
                stream.write(json.dumps({'context': c, 'target': t, 'token_log_probs': value.tolist()}) + '\n')
            stream.flush()
            print(f'Scored {len(scores)}/{len(keys)} sequences', flush=True)
    finalize(args, probes, scores, out)


def finalize(args, probes, scores, out):
    """Build reports from complete token scores; also usable for CPU-only recovery."""
    import torch
    import transformers
    keys = {(r[f'Context{c}'], r[f'Target{t}']) for r in probes for c in (1, 2) for t in (1, 2)}
    assert keys == set(scores)
    (out / 'report_evaluator.py').write_bytes(Path(__file__).read_bytes())
    items = [decisions(row, scores) for row in probes]
    (out / 'item_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items))
    # Independently reconstruct persisted token reductions and explicit-gold decisions.
    reconstructed = {}
    for line in (out / 'token_scores.jsonl').read_text().splitlines():
        r = json.loads(line)
        values = torch.tensor(r['token_log_probs'], dtype=torch.float32)
        reconstructed[r['context'], r['target']] = {'mean': float(values.mean()), 'sum': float(values.sum())}
    assert len(reconstructed) == len(keys)
    assert [decisions(r, reconstructed) for r in probes] == items
    import csv
    with (out / 'input_variant_matches.csv').open() as stream: links = list(csv.DictReader(stream))
    lookup = {r['probe_id']: r for r in items}
    matches = []
    for link in links:
        link = normalize_link(link)
        a, b = lookup[link['base_probe_id']], lookup[link['variant_probe_id']]
        matches.append({**link, 'base_accuracy': a['accuracy'], 'variant_accuracy': b['accuracy'],
                        'choice_changes': sum(a[f'mean_prediction_context{i}'] != b[f'mean_prediction_context{i}'] for i in (1, 2))})
    (out / 'matched_changes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in matches))
    grouped = {}
    for field in ('event_subtype', 'evidence_format', 'persistence_cue', 'movement_verb', 'preposition'):
        if field not in items[0]: continue
        for value in sorted({r[field] for r in items}):
            grouped[f'{field}/{value}'] = summary([r for r in items if r[field] == value])
    result = {'model': str(Path(args.model).resolve()), 'dataset': args.dataset,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'score_reduction': 'mean',
              'scoring': 'raw mean full-target conditional token likelihood; no PMI; ties incorrect; sum is secondary',
              'dtype': 'float32', 'attention': 'eager', 'torch': torch.__version__,
              'transformers': transformers.__version__, 'pairs': len(items), 'unique_sequences': len(keys),
              'probe_sha256': hashlib.sha256((out / 'input_probes.jsonl').read_bytes()).hexdigest(),
              'groups': report_groups(items, args.dataset), 'pooled': summary(items), 'diagnostics': grouped,
              'balancing': 'equal family, then evidence format, then case; variants averaged within cases',
              'uncertainty_note': 'Few hand-built cases; wording variants are not independent samples. No binomial confidence intervals reported.',
              'validation': 'All saved token means, sums and explicit-gold decisions reconstructed.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [f'# {args.dataset}: spatial pair evaluation', '', result['scoring'], '', result['balancing'], '',
             '| Block | Pairs | Mean accuracy | Both contexts correct | Sum accuracy |',
             '|---|---:|---:|---:|---:|']
    for label, group in result['groups'].items():
        if group:
            lines.append(f"| {label} | {group['pairs']} | {group['mean_accuracy']:.2%} | {group['mean_both_correct']:.2%} | {group['sum_accuracy']:.2%} |")
    lines += ['', '## Event families', '', '| Block | Family | Mean accuracy | Both correct |', '|---|---|---:|---:|']
    for label, group in result['groups'].items():
        if group:
            for family, score in group['families'].items():
                lines.append(f"| {label} | {family} | {score['mean_accuracy']:.2%} | {score['mean_both_correct']:.2%} |")
    lines += ['', result['uncertainty_note']]
    if args.dataset == 'minimal_cardinal':
        lines += ['', 'Explicit persistence uses only a matched baseline subset; use matched_changes.jsonl for controlled wording/cue comparisons, not unmatched group differences.']
    lines += ['', result['validation']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__': main()
