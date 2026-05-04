# Spatial Relations Synthetic Fine-Tuning

Generate balanced synthetic spatial-relations causal-LM text:

```bash
python research/bos_aligned_proto/spatial_synth/generate_spatial_relations_csv.py \
  --n 10000 \
  --seed 42 \
  --difficulty mixed \
  --out runs/research/bos_aligned_proto/spatial_synth/spatial_relations_synth.csv
```

The generator mixes the original explicit spatial statements with implicit
families for vertical inverse roles, turn-based reference-frame updates, and
pass-by/front-behind changes. It also includes a dedicated
`implicit_pass_through` family for straight-line crossing to the far side of an
object, with kept-going, crossed/overshot, landmark, direction-specific,
before/after, inverse-wording, and everyday-scene phrasings. Each row includes
`template_family` so those families can be filtered or ablated later. Rows also
include `context` and `completion`, where `completion` is the final sentence
used by optional completion-only loss masking.

Fine-tune one or more checkpoints with the default GPT-2 tokenizer and EWoK
BabyLM completion full-mean evaluation:

```bash
python research/bos_aligned_proto/spatial_synth/train_spatial_relations_causal_lm.py \
  --data runs/research/bos_aligned_proto/spatial_synth/spatial_relations_synth.csv \
  --checkpoints gpt2-medium \
  --learning-rates 5e-5 1e-5 \
  --epochs 3 \
  --epoch-eval 0.5 \
  --difficulty mixed \
  --loss-mode full \
  --ewok-variant full
```

Use `--loss-mode completion` to train only on the final completion sentence
while still feeding the full scenario as context. The wrapper scripts expose the
same switch through `LOSS_MODE=completion`.

Use `--loss-mode mixed --mixed-full-loss-ratio 0.7` for a sequence-level mix:
70% of training examples use full loss, and 30% use completion-only loss. The
fast-eval sweep wrapper
`run_8k_12k_16k_fast_eval_mixed_loss.sh` sets this up directly.

Use `--loss-mode weighted --completion-loss-ratio 0.7` for the older token
weighting behavior: 70% of each example's loss mass on the completion sentence,
and 30% on the context/setup tokens.

Each run writes `step_metrics.json`, `ewok_items.jsonl`, selected train/val
rows, per-run plots, and multi-run summary plots when more than one checkpoint
or learning rate is provided. Final model saving is off by default; pass
`--save-final` when you want to keep the fine-tuned checkpoint.

## Current Hypothesis Notes

The v8-v12 fast-eval one-epoch sweeps branch from v6 to test one targeted
intervention at a time:

- `v8`: reciprocal close/far
- `v9`: low-weight generic distance contrast
- `v10`: direct turn-around left/right side flips
- `v11`: cardinal direction guardrails
- `v12`: symmetric-vs-inverse relation-type contrasts

Margin results so far suggest `v8` is the strongest data-content intervention:
it improved overall spatial margin relative to v6 and made close/far much less
negative. `v10` locally improved turn-around left/right but did not improve
overall spatial margin. `v7` and `v9` suggest generic distance/reachability
examples are less useful than compact reciprocal close/far examples. `v11`
mostly failed as a cardinal-direction guardrail, and `v12` had mixed but noisy
effects. The working takeaway is that close/far needs symmetry/reciprocity more
than generic distance exposure, while many remaining losses likely come from
training dynamics and distribution narrowing rather than a single missing fact.

The v13-v15 sweeps keep the v8 reciprocal close/far base and test whether
template order helps the turn-based left/right facts transfer. Run them with:

```bash
bash research/bos_aligned_proto/spatial_synth/run_v13_to_v15_fast_eval_full_loss_1epoch.sh
```

- `v13`: v8 + `implicit_turn_lr_order_variants`. This keeps reciprocal
  close/far from v8, then adds examples where a fixed object starts in front or
  behind an actor, the actor turns left/right, and the text states the new
  left/right side. The same spatial fact is written in several orders:
  premise-first, turn-first, answer-first, and compact contrast.
- `v14`: v8 + `implicit_turn_around_lr`. This keeps reciprocal close/far from
  v8, then adds direct 180-degree turn examples where an object on the actor's
  left becomes on the actor's right, or vice versa, without either entity moving.
- `v15`: v8 + `implicit_turn_around_lr` + `implicit_turn_lr_order_variants`.
  This tests whether the two left/right interventions are complementary:
  direct turn-around flips plus order-varied left/right turns from front/back.

The order-variant hypothesis is that the model may learn "turn first, then
recover the old position" as a surface pattern without learning the composition
both ways. These examples therefore test whether changing the sentence order
helps the model learn facts like "in front + pivot right -> left-hand side" and
"pivot right + in front -> left-hand side" as the same underlying relation.

## EWoK Answer-Exposure Probe

`generate_ewok_answer_exposure_csv.py` builds a deliberately leaky
memorization dataset from the correct EWoK BabyLM pairs:

- `C_1,T_1`: `Context1` followed by `Target1`
- `C_2,T_2`: `Context2` followed by `Target2`

The wrapper below fine-tunes the 8k/12k/16k checkpoints on those correct
answers, uses completion-only loss by default, evaluates every quarter epoch,
and writes the same 4x3 domain plots plus the spatial-relations main plot:

```bash
bash research/bos_aligned_proto/spatial_synth/run_ewok_answer_exposure_8k_12k_16k.sh
```

This is not meant as a fair EWoK result. It asks whether the model can memorize
the answer pairs at all, how quickly margins move once the correct completions
are supervised, and whether spatial-relations behaves differently from the
other domains. Use `EWOK_VARIANT=full` for the full filtered set or
`DOMAINS="spatial-relations"` to train only on spatial answer pairs.

## v14 Batch/Data Sweeps

`run_v16_to_v18_v14_batch_data_sweep.sh` keeps the v14 data recipe fixed and
tests whether optimization scale or dataset size explains some of the v14
behavior:

- `v16`: v14, `N=10000`, effective-batch sweep `8/16/32`
- `v17`: v14, `N=15000`, effective batch `32`
- `v18`: v14, `N=15000`, effective-batch sweep `8/16/32`

Effective batch is `PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS`. The wrapper keeps
`GRAD_ACCUM_STEPS=8` and uses per-device batches `1/2/4` for effective batches
`8/16/32`, so the old batch-8 condition remains directly represented.

```bash
bash research/bos_aligned_proto/spatial_synth/run_v16_to_v18_v14_batch_data_sweep.sh
```

## v19-v21 Left/Right Paired Contrasts

`run_v19_to_v21_left_right_paired_batch_sweep.sh` keeps v14 as the base and
adds matched left/right contrast pairs. These examples deliberately hold most
of the scene fixed while changing exactly one variable:

- same world position, opposite facing direction
- same front/behind landmark, left turn versus right turn
- same pair, subject/reference role inverted

The versions differ only in how often this new family is sampled:

- `v19`: v14 + paired left/right contrasts at normal weight
- `v20`: v14 + paired left/right contrasts at medium weight
- `v21`: v14 + paired left/right contrasts at high weight

Each version sweeps effective batches `8/16/32` by default:

```bash
bash research/bos_aligned_proto/spatial_synth/run_v19_to_v21_left_right_paired_batch_sweep.sh
```
