#!/usr/bin/env python3
"""Monte Carlo next-token distributions from untrained checkpoint configurations."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t as student_t
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--extend-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path('/SSD2/tokenPred/moonshotGPT/runs/research/bos_aligned_proto')
    configs = {
        'qwen3': root / 'qwen3_liger_muon_steps19500_ws6_seed42_restart_20260816_120352/runs/qwen3_liger_muon/babygpt_fineweb_stream_mbs6_T1024_qwen3_d1024_h16_L24_tok516096_efftok516096_ws6_gas14_seed42_steps19500/ckpt_final_step0019500',
        'llama': root / 'llama_liger_muon_steps19500_ws6_seed42_queued_after_restart_20260816_120352/runs/llama_liger_muon/babygpt_fineweb_stream_mbs6_T1024_llama_d1024_h16_L24_tok516096_efftok516096_ws6_gas14_seed42_steps19500/ckpt_final_step0019500',
    }
    tokenizer = AutoTokenizer.from_pretrained(configs['qwen3'], local_files_only=True)
    bos = tokenizer.bos_token_id
    texts = {'bos_only': '', 'single_ball': ' ball', 'single_left': ' left', 'single_right': ' right',
        'neutral': 'The objects are',
        'left_scene': 'The ball is to the left of the cone. The ball is to the',
        'right_scene': 'The ball is to the right of the cone. The ball is to the',
        'summary_bridge': 'The ball is to the left of the cone. To summarize the positions: The ball is to the'}
    candidates = {w: tokenizer.encode(' '+w, add_special_tokens=False) for w in ('left','right','above','below','north','south','east','west')}
    assert all(len(ids)==1 for ids in candidates.values())
    contexts = {k:[bos]+tokenizer.encode(t,add_special_tokens=False) for k,t in texts.items()}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'seeds':list(range(args.seeds)), 'configs':{k:str(v) for k,v in configs.items()},
        'contexts':{k:{'text':texts[k], 'token_ids':v} for k,v in contexts.items()},
        'candidate_token_ids':candidates, 'torch':torch.__version__, 'transformers':transformers.__version__,
        'dtype':'float32','attention':'sdpa', 'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'method':'Fresh AutoModelForCausalLM.from_config for every seed; no checkpoint weights loaded; no training.'}
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]
    arrays={}
    start_seed = 0
    previous = None
    if args.extend_from:
        old_manifest = json.loads((args.extend_from/'manifest.json').read_text())
        start_seed = len(old_manifest['seeds'])
        assert old_manifest['seeds'] == list(range(start_seed))
        assert args.seeds > start_seed >= 2
        for key in ('configs', 'contexts', 'candidate_token_ids', 'torch', 'transformers', 'dtype', 'attention'):
            assert old_manifest[key] == manifest[key], key
        previous = np.load(args.extend_from/'vocabulary_probability_moments.npz')
        manifest['extended_from'] = str(args.extend_from.resolve())
        manifest['reused_seeds'] = list(range(start_seed))
        (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for arch,path in configs.items():
        config=AutoConfig.from_pretrained(path,local_files_only=True)
        (args.out_dir/f'{arch}_config.json').write_text(config.to_json_string())
        vocab=config.vocab_size
        sums={k:np.zeros(vocab,dtype=np.float64) for k in contexts}
        squares={k:np.zeros(vocab,dtype=np.float64) for k in contexts}
        if previous is not None:
            old_config = json.loads((args.extend_from/f'{arch}_config.json').read_text())
            new_config = json.loads(config.to_json_string())
            assert old_config == new_config
            for name in contexts:
                avg = previous[f'{arch}_{name}_mean']
                sd = previous[f'{arch}_{name}_std']
                sums[name] = avg * start_seed
                squares[name] = sd**2 * (start_seed-1) + start_seed * avg**2
        with (args.out_dir/f'{arch}_seeds.jsonl').open('w') as stream:
            if previous is not None:
                old_records = [json.loads(line) for line in (args.extend_from/f'{arch}_seeds.jsonl').read_text().splitlines()]
                assert len(old_records) == start_seed * len(contexts)
                assert {(r['seed'],r['context']) for r in old_records} == {(s,c) for s in range(start_seed) for c in contexts}
                records.extend(old_records)
                stream.write(''.join(json.dumps(r)+'\n' for r in old_records))
            for seed in range(start_seed, args.seeds):
                torch.manual_seed(seed)
                model=AutoModelForCausalLM.from_config(config,attn_implementation='sdpa',torch_dtype=torch.float32).to('cuda').eval()
                assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
                with torch.inference_mode():
                    for name,ids in contexts.items():
                        x=torch.tensor([ids],device='cuda')
                        logits=model(input_ids=x,use_cache=False).logits[0,-1].double()
                        logp=logits.log_softmax(-1); probs=logp.exp()
                        assert torch.isfinite(probs).all() and abs(probs.sum().item()-1)<1e-10
                        entropy=-(probs*logp).sum().item()
                        row={'architecture':arch,'seed':seed,'context':name,'vocab_size':vocab,
                            'entropy_nats':entropy,'kl_to_uniform_nats':math.log(vocab)-entropy,
                            'total_variation_from_uniform':(probs-1/vocab).abs().sum().item()/2,
                            'max_probability':probs.max().item(),'max_token_id':probs.argmax().item(),
                            'last_input_token_ratio':probs[ids[-1]].item()*vocab,
                            'bos_ratio':probs[bos].item()*vocab,
                            **{w+'_ratio':probs[token[0]].item()*vocab for w,token in candidates.items()}}
                        records.append(row);stream.write(json.dumps(row)+'\n')
                        p=probs.cpu().numpy();sums[name]+=p;squares[name]+=p*p
                stream.flush()
                del model
                gc.collect();torch.cuda.empty_cache()
                print(f'{arch}: seed {seed+1}/{args.seeds} complete',flush=True)
        for name in contexts:
            arrays[f'{arch}_{name}_mean']=sums[name]/args.seeds
            arrays[f'{arch}_{name}_std']=np.sqrt(np.maximum(0,(squares[name]-sums[name]**2/args.seeds)/(args.seeds-1)))
    np.savez_compressed(args.out_dir/'vocabulary_probability_moments.npz',**arrays)
    groups=[]
    keys=['kl_to_uniform_nats','total_variation_from_uniform','max_probability','last_input_token_ratio','bos_ratio']+[w+'_ratio' for w in candidates]
    for arch in configs:
        for name in contexts:
            rs=[r for r in records if r['architecture']==arch and r['context']==name]
            group={'architecture':arch,'context':name,'n_seeds':len(rs)}
            for key in keys:
                vals=[r[key] for r in rs];avg=mean(vals);sd=stdev(vals)
                width = float(student_t.ppf(.975,len(vals)-1))*sd/math.sqrt(len(vals))
                group[key]={'mean':avg,'std_across_seeds':sd,'approx_95ci_mean':[avg-width,avg+width]}
            avg=arrays[f'{arch}_{name}_mean']
            group['finite_seed_mean_distribution_tv']=float(np.abs(avg-1/len(avg)).sum()/2)
            groups.append(group)
    (args.out_dir/'summary.json').write_text(json.dumps(groups,indent=2)+'\n')
    report=['# Random initialization distributions','',
        f'{args.seeds} seeds per architecture, fresh weights from the trained runs’ configurations. Each row averages next-token distributions for one fixed context. No pretrained weights loaded.',
        '', 'Uniform probability is 1/50,257 = 0.0000198977. Ratio columns multiply probability by vocabulary size; uniform is 1. KL and total variation are zero for uniform predictions.', '',
        '| Architecture | Context | Mean KL to uniform (nats) | Mean total variation | Mean maximum probability | Last-input token ratio | Left ratio | Right ratio |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        report.append('| '+g['architecture']+' | '+g['context']+' | '+' | '.join(f"{g[k]['mean']:.6g}" for k in ['kl_to_uniform_nats','total_variation_from_uniform','max_probability','last_input_token_ratio','left_ratio','right_ratio'])+' |')
    report+=['',f'summary.json includes across-seed standard deviations and approximate 95% t intervals for means ({args.seeds} seeds). These can be unstable for skewed probabilities and are not simultaneous confidence intervals. Finite-seed averaged distributions retain sampling noise; deviation from uniform in that average alone does not establish a population asymmetry.',
        '', 'This estimates the initialization procedure in the installed library with matched architectural settings, not the exact original training-start RNG state. Token ratios refer to leading-space single tokens. This is a next-token diagnostic, not full-answer probe accuracy.']
    (args.out_dir/'report.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    main()
