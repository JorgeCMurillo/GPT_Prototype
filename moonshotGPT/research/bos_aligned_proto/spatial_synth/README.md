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

## Synthetic Spatial Eval Mode

Use `--synthetic-spatial-eval three_tier` to add a held-out EWoK-style
synthetic spatial eval at the same cadence as `--epoch-eval`. This diagnostic
does not replace EWoK; it checks whether the model can answer our own spatial
formats and whether failures look like concept failure or format transfer
failure.

The default smoke eval uses `--synthetic-spatial-eval-n-per-tier 300`, with
balanced items for `in_format`, `paraphrase`, and `composition` tiers. The
concept buckets are left/right, front/behind, north/south, east/west,
above/below, and close/far. It writes:

- `synthetic_spatial_eval_dataset.jsonl`: the fixed held-out eval rows
- `synthetic_spatial_items.jsonl`: per-item scores at each eval step
- extra plots for synthetic tier accuracy/margin, concept accuracy/margin, and
  the synthetic-vs-EWoK spatial transfer gap

For the bash runners:

```bash
SYNTHETIC_SPATIAL_EVAL=three_tier \
SYNTHETIC_SPATIAL_EVAL_N_PER_TIER=300 \
bash research/bos_aligned_proto/spatial_synth/run_8k_12k_16k_fast_eval.sh
```

Interpretation guide: high `in_format` with low EWoK points toward a transfer
problem; low `in_format` and low EWoK points toward the synthetic curriculum
itself; high `in_format` but low `paraphrase` suggests template memorization.

## Current Hypothesis Notes

### Ablation Ledger

Unless otherwise noted, completed rows summarize the fast EWoK one-epoch sweeps
averaged over the 8k/12k/16k checkpoints and learning rates `4e-5`/`8e-5`.
The shorthand `spatial acc/margin` refers to EWoK BabyLM completion full-mean
spatial-relations accuracy and signed margin at the final eval.

| Version | Hypothesis / change | Status | Observed result | Working conclusion |
| --- | --- | --- | --- | --- |
| `v6` | Reference recipe with left/right contrast plus turn-left/right from front/back examples. | Done | Spatial acc/margin `0.548`/`0.037`; spatial still drops from the starting checkpoints. | Useful comparison point, but not enough to stabilize EWoK spatial transfer. |
| `v7` | Add generic implicit distance/reachability contrasts. | Done | Spatial acc/margin `0.522`/`0.014`, worse than v6 on spatial. | Broad distance examples seem too noisy or too weakly targeted. |
| `v8` | Add compact reciprocal close/far examples. | Done | Spatial acc/margin `0.549`/`0.058`; strongest margin among v6-v12. | Best data-content intervention so far; close/far benefits more from explicit reciprocity than generic distance exposure. |
| `v9` | Same generic distance family as v7, but lower weight. | Done | Spatial acc/margin `0.522`/`0.033`, still below v8. | Lowering the generic-distance weight did not fix the issue. |
| `v10` | Add direct turn-around left/right side flips. | Done | Spatial acc/margin `0.515`/`0.030`; local turn-around gains did not translate overall. | Helpful conceptually, but not enough as a standalone branch from v6. |
| `v11` | Add cardinal direction guardrails. | Done | Spatial acc/margin `0.513`/`0.022`, weakest of v8-v12. | Cardinal guardrails mostly failed as an EWoK spatial intervention. |
| `v12` | Add symmetric-vs-inverse relation-type contrasts. | Done | Spatial acc/margin `0.545`/`0.036`, mixed and close to v6. | Reasonable idea, but no clear improvement over v8. |
| `v13` | v8 + order variants for turn-left/right from front/back. | Done | Spatial acc/margin `0.541`/`0.045`. | Order variation alone did not beat v8; may still be useful as a diagnostic. |
| `v14` | v8 + direct turn-around left/right flips. | Done | Spatial acc/margin `0.540`/`0.041`. | Chosen as a simple keeper/base because it adds a clean left/right idea without adding much complexity. |
| `v15` | v8 + both order variants and turn-around flips. | Done | Spatial acc/margin `0.542`/`0.041`. | Combining v13 and v14 did not clearly help; extra complexity was not rewarded. |
| `v16` | v14 with `N=10000`, effective-batch sweep `8/16/32`. | Done | Best spatial accuracy came from effective batch 16 (`0.549`), while effective batch 8 had stronger spatial margin than 16/32. | Batch size changes matter, but they do not solve the transfer problem by themselves. |
| `v17` | v14 with `N=15000`, effective batch 32. | Pending / not run | No completed result folder found. | Dataset-size hypothesis still untested. |
| `v18` | v14 with `N=15000`, effective-batch sweep `8/16/32`. | Pending / not run | No completed result folder found. | Dataset-size plus batch-size interaction still untested. |
| `v19` | v14 + matched left/right paired contrasts at normal weight. | Done | Best setting was effective batch 8: spatial acc/margin `0.553`/`0.047`; modestly improved left/right and cardinal slices, but close/far got worse. | Keep as current base: it is the best recent tradeoff, though left/right remains far below baseline. |
| `v20` | v14 + matched left/right paired contrasts at medium weight. | Pending / not run | No completed result folder found. | Tests whether stronger left/right density helps or overfits. |
| `v21` | v14 + matched left/right paired contrasts at high weight. | Pending / not run | No completed result folder found. | Tests the upper limit of paired left/right emphasis. |

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
