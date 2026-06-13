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
evaluation:

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

## Contrastive Spatial Sets

`generate_spatial_relations_csv.py` can also emit contrastive triplets for
mixed NTP plus representation learning:

```bash
python research/bos_aligned_proto/spatial_synth/generate_spatial_relations_csv.py \
  --contrastive \
  --n 1000 \
  --seed 42 \
  --out runs/research/bos_aligned_proto/spatial_synth/spatial_relations_contrastive_n1000_seed42.csv
```

In contrastive mode, `--n` is the number of contrast sets, not the number of
rows. Each set writes three rows:

- `anchor`: a plausible scene text
- `positive`: an equivalent or semi-equivalent plausible scene text
- `hard_negative`: a near-miss text with the wrong final relation

Rows include `contrast_set_id`, `equiv_class_id`, `contrast_role`,
`contrast_family`, `negative_type`, `is_plausible`, `ntp_weight`, and
`contrast_weight`. Hard negatives are marked `is_plausible=0` and
`ntp_weight=0`, so the causal-LM objective does not train the model to continue
with implausible scene descriptions. They are used only by the contrastive
objective.

Current contrastive families:

- `rotated_latent_transition`: same abstract egocentric transition under a
  rotation of the world axes.
- `role_viewpoint_reciprocal`: same physical arrangement from reciprocal
  observer/reference viewpoints, such as "the object is to Ava's left" paired
  with "Ava is to the object's right."
- `vertical_reciprocal_equivalence`: same above/below arrangement from
  reciprocal subject/reference views, such as "Ava is above the box" paired with
  "the box is below Ava."
- `vertical_motion_same_latent`: same vertical motion event rendered as direct
  and inverse final relations, such as "Ava moved higher, so Ava is above the
  box" paired with "the box is below Ava."
- `pass_by_same_latent`: same forward pass-by scene rendered with two
  paraphrases, plus a wrong final-relation negative.
- `pass_through_forward_same_latent`: same forward pass-through scene rendered
  with two paraphrases, plus a wrong final-relation negative.

Two less-common mirror families are optional because they are geometrically
useful but less natural than ordinary forward walking past a landmark:

```bash
python research/bos_aligned_proto/spatial_synth/generate_spatial_relations_csv.py \
  --contrastive \
  --include-backing-past \
  --include-pass-through-mirrors \
  --n 1000 \
  --seed 42 \
  --out runs/research/bos_aligned_proto/spatial_synth/spatial_relations_contrastive_mirrors_n1000_seed42.csv
```

- `--include-backing-past` adds `pass_by_backward_same_latent`: the object starts
  behind the agent, the agent backs past it without turning, and the object ends
  in front.
- `--include-pass-through-mirrors` adds:
  - `pass_through_backward_same_latent`: the agent backs through/past the
    object's position.
  - `pass_through_carried_object_same_latent`: someone carries the object from
    behind the agent to in front of the agent.

For training, enable an auxiliary contrastive objective with
`--contrastive-loss-weight`. The default is
`--contrastive-objective representation`: the trainer mean-pools final-layer
hidden states and applies cosine triplet margin loss:

```text
L_repr = max(0, margin + sim(anchor, hard_negative) - sim(anchor, positive))
```

An alternative is `--contrastive-objective rank`, which directly ranks the
positive and negative completions under the same context:

```text
s+ = length_normalized_log P_theta(T+ | C)
s- = length_normalized_log P_theta(T- | C)
L_rank = softplus((s- - s+) / tau)
```

Use `--rank-temperature` for `tau`. The `both` objective averages `L_repr` and
`L_rank` before applying `--contrastive-loss-weight`.

The total training loss is:

```text
causal_lm_loss + contrastive_loss_weight * selected_contrastive_loss
```

Example:

```bash
python research/bos_aligned_proto/spatial_synth/train_spatial_relations_causal_lm.py \
  --data runs/research/bos_aligned_proto/spatial_synth/spatial_relations_contrastive_n1000_seed42.csv \
  --checkpoints gpt2-medium \
  --learning-rates 1e-5 \
  --epochs 3 \
  --loss-mode full \
  --contrastive-loss-weight 0.1 \
  --contrastive-objective rank \
  --rank-temperature 1.0 \
  --contrastive-margin 0.2
```

Useful first sweeps are `--contrastive-loss-weight 0.03/0.1/0.3` and
`--contrastive-margin 0.1/0.2/0.4` for representation loss, and
`--rank-temperature 0.3/1.0/3.0` for rank loss. Group-aware train/val splitting
keeps each anchor/positive/negative triplet together.

For natural-text dose sweeps, build mixed CSVs with
`generate_natural_synth_mix_csv.py`. When the synthetic CSV contains
`contrast_set_id`, the mixer samples whole contrast sets together, so an
anchor/positive/hard-negative triplet is never split across the synthetic dose.
This can overshoot the requested synthetic token budget by up to one contrast
set, and the manifest records the selected group, role, and family counts.

For the contrastive dose-sweep runs, use EWoK context sensitivity as the primary
metric:

```bash
python research/bos_aligned_proto/spatial_synth/train_spatial_relations_causal_lm.py \
  --data runs/research/bos_aligned_proto/spatial_synth/natural_contrastive_mix_10pct.csv \
  --checkpoints /path/to/checkpoint_16000 \
  --learning-rates 1e-5 \
  --epochs 1 \
  --loss-mode full \
  --contrastive-loss-weight 0.03 \
  --contrastive-objective rank \
  --rank-temperature 1.0 \
  --primary-ewok-metric context
```

## EWoK Metric Conventions

By default, this project reports **EWoK BabyLM completion full-mean** scores
with `score_reduction=mean`. Pass `--primary-ewok-metric context` to make EWoK
context sensitivity drive the console logs and primary training plots instead.
Both methods are still saved in `step_metrics.json`.

For each EWoK item, the completion-choice scorer compares both directions of
the paired completion task:

```text
m1 = logp(Target1 | Context1) - logp(Target2 | Context1)
m2 = logp(Target2 | Context2) - logp(Target1 | Context2)
m  = 0.5 * (m1 + m2)
```

When completion-choice is the primary metric, the reported EWoK accuracy in the
training plots is the **combined completion accuracy**:

```text
combined_acc = 0.5 * (1[m1 > 0] + 1[m2 > 0])
```

So an item can contribute `1.0`, `0.5`, or `0.0`. This is pair-aware, but it is
not strict both-right accuracy. Strict both-right would require `m1 > 0` and
`m2 > 0` simultaneously, and is usually lower.

Naming conventions in the saved files:

- `eval_primary_full_mean`: the selected primary metric for plots, either
  completion-choice by default or context sensitivity with
  `--primary-ewok-metric context`.
- `eval_primary_margin_stats_mean`: margin stats for the selected primary
  metric.
- `eval_babylm_completion_choice_official_mean`: only the `m1` side.
- `eval_babylm_completion_choice_full_mean`: the `(m1_acc, m2_acc)` pair; plots
  average the two sides into combined accuracy when completion is primary.
- `eval_babylm_completion_choice_margin_stats_mean`: includes `mean_signed_m`,
  the mean of `m = 0.5 * (m1 + m2)`.
- `eval_context_sensitivity_full_mean`: EWoK context-sensitivity scores.
- `eval_context_sensitivity_margin_stats_mean`: context-sensitivity margin
  stats.
- `ewok_items.jsonl`: per-item rows include `correct_official`,
  `correct_symmetric`, `correct_combined`, `margin_official_m1`,
  `margin_symmetric_m2`, and `margin_combined`.

Do not compare an official-side number directly to a combined/full-mean number.
For example, a left/right slice can look like `0.75` or higher on one side while
the combined paired score is much lower if the model only gets one direction of
the pair right. Older analysis tables may also use concept-mapped unique EWoK
items, while quick row-level slices can use a different denominator; always
check the slice definition and metric field before comparing numbers.

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

## Natural-Mix and GPT-5.2 Cardinal Takeaways

`v19` is the broad algorithmic spatial curriculum used as the strongest
hand-coded base before the natural-mix tests. It combines v14 with matched
left/right contrast pairs, role/reference inversions, turn contrasts, movement
updates, and other spatial families.

`v22` is the natural/synthetic dose sweep built from v19. It mixes FineWeb-style
natural text with v19 synthetic rows at `0%`, `5%`, `10%`, `20%`, and `40%`
synthetic token ratios. The main lesson was dose sensitivity: small doses were
least harmful, while larger doses often increased margins but reduced EWoK
spatial accuracy.

At `lr=4e-5`, the 8k checkpoint did not benefit much from v22. Natural-only
went `0.590 -> 0.575` spatial accuracy, `5%` synthetic mostly preserved
accuracy (`0.590 -> 0.585`), and `20-40%` synthetic hurt (`0.535` and `0.525`
final spatial accuracy). The 12k checkpoint was more receptive: `10%` synthetic
improved the cardinal slices (`north/south: 0.607 -> 0.714`,
`east/west: 0.583 -> 0.708`) but hurt `left/right` (`0.682 -> 0.515`). This
suggested that broad algorithmic synthetic data can help targeted concepts, but
with substantial collateral damage.

The GPT-5.2 cardinal mix was a narrower follow-up. It used about `9.8%`
GPT-5.2-generated cardinal text mixed with natural data, targeting only
`north/south` and `east/west`, and trained best at `lr=1e-5`.

| Checkpoint | Spatial accuracy | Spatial margin | Main concept effect |
| --- | --- | --- | --- |
| 8k | `0.590 -> 0.685` | `+0.052 -> +0.141` | `north/south: 0.464 -> 0.786`, `east/west: 0.583 -> 0.833`; `left/right` also held up (`0.621 -> 0.667`). |
| 12k | `0.625 -> 0.660` | `+0.064 -> +0.153` | `north/south: 0.607 -> 0.786`, `east/west: 0.583 -> 0.833`; `left/right` roughly stable with a small drop (`0.682 -> 0.667`). |
| 16k | `0.625 -> 0.690` | `+0.068 -> +0.164` | `north/south: 0.429 -> 0.786`, `east/west: 0.542 -> 0.833`; `left/right` dropped (`0.697 -> 0.621`). |

Higher GPT-5.2 cardinal learning rates (`4e-5`, `8e-5`) pushed cardinal margins
much higher, but they also damaged unrelated spatial concepts more strongly.
The current interpretation is that synthetic data works better as targeted
concept repair than as broad replacement training: narrow scope, low synthetic
dose, natural-data mixing, and low learning rate produced the cleanest transfer
so far.
