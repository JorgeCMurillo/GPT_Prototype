# Report: Top-1k Priority Windows vs Matched Random Control Across 8k, 12k, and 16k Checkpoints

## Executive summary

We tested a simple but important hypothesis: if we continue pretrain the model on a small amount of text that is unusually rich in entity persistence and relations, will the model improve on EWoK variable-swap problems, which are intended to probe role binding?

The answer is: **yes, but only selectively**.

- The clearest positive result appears at the **12k checkpoint**.
- The **8k checkpoint** shows only weak role-binding gains.
- The **16k checkpoint** still shows some variable-swap gains, but they come with a clearer degradation in overall EWoK accuracy.

So the current evidence suggests that the filtering metrics are not random and are not useless. They do appear to enrich for text that improves the target behavior. However, the gains are checkpoint-sensitive, learning-rate-sensitive, and they occur inside a small-data continued-pretraining setup that can also degrade the broader model.

In other words: the experiment gives evidence that the selected text is directionally relevant to role binding, but it does **not** yet show that this intervention is a clean absolute improvement to the whole model.

## What we were trying to test

The central scientific question was:

> Can we mine a subset of natural text whose discourse structure nudges a model toward better role binding?

Here, "role binding" means roughly: can the model keep track of **who did what to whom**, and update its prediction correctly when names or roles are swapped?

EWoK variable-swap problems are a reasonable proxy for this. They require the model to use relational structure rather than only lexical co-occurrence. If a model improves on variable-swap items, that is at least suggestive evidence that it is getting better at tracking role assignments in context.

More concretely, the experiment tested three nested hypotheses:

1. **Text-selection hypothesis**
   Windows with strong discourse-tracking signals such as persistent entities, recurring entity pairs, and relation-dense sentences should be more useful for role binding than generic text.

2. **Ranking hypothesis**
   A targeted ranking inside that useful region should work better than treating all "good enough" windows as equally informative.

3. **Checkpoint interaction hypothesis**
   The same text intervention may help more at some checkpoints than others, because the model's internal representation of entities and relations changes over training.

## What was the treated data?

The treated set was not "random FineWeb." It came from a mining pipeline over `100,000` candidate windows from FineWeb-Edu.

For the current mined run at `/tmp/fineweb10b_seed42_mined_gte`:

- Total candidate windows: `100,000`
- Eligible windows: `99,819`
- Positive-pool windows: `3,308`
- Random-control-pool windows: `3,308`

The actual CPT intervention used only **1,000 treated windows**, so the experiment is best understood as a **small-dose, highly selective** continued-pretraining test.

## What does the positive pool mean?

The positive pool is a **binary filter**. It is intentionally broad. A window enters the positive pool if it passes a set of threshold rules that say, in effect, "this looks plausibly rich enough in discourse structure to be worth considering."

In the current miner, a window had to first be eligible:

- at least `3` sentences
- at least `2` unique entities
- at least `96` text tokens

Then positive-pool membership required all of the following:

- `entity_persistence` at or above the eligible-set `75th` percentile
- `entity_recurrence` at or above the eligible-set `60th` percentile
- `relation_density` at or above the eligible-set `75th` percentile
- `repeated_3gram_ratio` at or below the eligible-set `50th` percentile
- `duplicate_sentence_fraction` at or below the eligible-set `75th` percentile
- `unique_entity_count >= 4`

For this run, those empirical thresholds were:

- `entity_persistence >= 0.15`
- `entity_recurrence >= 0.125`
- `relation_density >= 0.2286`
- `repeated_3gram_ratio <= 0.0260`
- `duplicate_sentence_fraction <= 0.0`
- `unique_entity_count >= 4`

This is important: **positive-pool membership is not a ranking**. It says "keep this candidate in the broad set of potentially promising windows." It does not say that every positive example is equally good for the downstream hypothesis.

## What does the priority score mean?

The priority score is a **continuous ranking function**. It is the mechanism we used to decide which positive windows to spend the small CPT budget on.

The current formula is:

`z(entity_persistence) + z(entity_recurrence) + z(relation_density) + 0.75*z(adjacent_entity_overlap) + 0.75*z(pair_recurrence) + 0.75*z(top_pair_sentence_share) + 0.25*z(sentence_count) + 0.10*z(unique_entity_count) - 0.50*z(entity_churn) - 0.50*z(bos_contamination_penalty) - 0.75*z(repeated_3gram_ratio) - 0.50*z(duplicate_sentence_fraction) - 0.75*z(effective_cast_size_overflow)`

Intuitively:

- It rewards windows where entities persist over multiple sentences.
- It rewards windows where the same entity pairs recur, which is a rough proxy for stable relational structure.
- It rewards adjacent-sentence continuity, so topic-jumpy passages are downgraded.
- It penalizes repetition, duplicate sentences, stray BOS artifacts, and windows with an overly diffuse cast.

So the priority score is more targeted than the positive pool. The positive pool says "this is in the right region." The priority score says "inside that region, this is especially concentrated in the signals we think matter for role binding."

## Why use top-1k priority instead of all positives?

Because the positive pool is still heterogeneous.

Out of `3,308` positive windows, we only trained on `1,000`. That means we were forced to choose a subset. Using all positives equally would have been equivalent to saying that every positive window is equally informative, which is exactly what we were uncertain about.

The top-1k priority subset was:

- `1000 / 3308 = 30.2%` of the positive pool
- minimum priority score in top-1k: `6.95`
- mean priority score in top-1k: `9.88`
- mean priority score over all positives: `5.89`

So the intervention was not "positive pool vs control." It was more precise:

> top 1,000 windows from within the positive pool, ranked by a role-binding-oriented priority score, versus a matched random control set under the same CPT budget.

This targeted ranking was chosen because the budget was small. If we only train on 1,000 windows, we want those 1,000 windows to be the densest possible concentration of the behavior we care about.

That design choice was also motivated by earlier pilot comparisons. In an earlier 3-seed 12k run, a **random 1,000 positives** improved variable swap less than the **top-1k priority** subset, especially for ContextDiff variable swap. That does not prove the ranking is optimal, but it does support the idea that ranking inside the positive pool matters.

## Experimental setup

The main report below refers to the completed **6-seed multi-checkpoint sweep** in:

`/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/outputs/fineweb10b_seed42_top1000_multi_ckpt_ablation`

Setup:

- Base checkpoints: `8k`, `12k`, `16k`
- Learning rates: `2e-5`, `4e-5`, `8e-5`
- Seeds: `1, 2, 6, 17, 71, 82`
- Epochs: `3`
- Treated dataset: top-1k priority windows from the positive pool
- Control dataset: matched random control windows

The most interpretable numbers below are **final EWoK `eval2_acc` accuracies**. I also checked the margin summaries in the ablation outputs; they tell the same qualitative story.

## How to interpret treated vs control

There are two different comparisons in this experiment, and they answer different questions:

1. **Treated vs control**
   This asks whether the selected text is better than a matched alternative under the same CPT budget. This is the main comparison for the text-selection hypothesis.

2. **Treated or control vs base checkpoint**
   This asks whether the entire continued-pretraining intervention helps or hurts the base model in absolute terms.

These are not the same question.

It is possible for:

- treated to beat control on variable swap
- while both treated and control are somewhat worse than the base model overall

That would mean the selected text is useful **relative to control**, but the small-data CPT intervention itself may still be too strong or too narrow.

## Results by checkpoint

### 8k checkpoint

At `8k`, the top-1k priority intervention looks mildly beneficial overall, but only weakly beneficial on the specific role-binding target.

| LR | Overall treated | Overall control | TargetDiff var swap treated | TargetDiff var swap control | ContextDiff var swap treated | ContextDiff var swap control |
|---|---:|---:|---:|---:|---:|---:|
| `2e-5` | `0.5443` | `0.5402` | `0.5103` | `0.5169` | `0.5008` | `0.5095` |
| `4e-5` | `0.5477` | `0.5361` | `0.5169` | `0.5151` | `0.5182` | `0.5134` |
| `8e-5` | `0.5508` | `0.5406` | `0.5242` | `0.5217` | `0.5253` | `0.5221` |

Interpretation:

- Overall accuracy is slightly better for treated than control.
- Variable-swap gains are small.
- At `2e-5`, variable swap is actually slightly worse than control.
- At `4e-5` and `8e-5`, variable swap is slightly better than control, but not by much.

So `8k` does not look like a strong success for the role-binding hypothesis. It looks more like a mild generic improvement regime with only weak specialization on the target behavior.

### 12k checkpoint

At `12k`, the picture changes substantially. This is the checkpoint where the role-binding hypothesis is best supported.

| LR | Overall treated | Overall control | TargetDiff var swap treated | TargetDiff var swap control | ContextDiff var swap treated | ContextDiff var swap control |
|---|---:|---:|---:|---:|---:|---:|
| `2e-5` | `0.5543` | `0.5546` | `0.5231` | `0.5180` | `0.5150` | `0.5063` |
| `4e-5` | `0.5463` | `0.5523` | `0.5224` | `0.5184` | `0.5186` | `0.5150` |
| `8e-5` | `0.5484` | `0.5501` | `0.5352` | `0.5070` | `0.5288` | `0.5083` |

Interpretation:

- The overall EWoK score is roughly flat to slightly worse for treated than control.
- But the variable-swap gains are clear, especially at `8e-5`.
- At `8e-5`, treated beats control by:
  - `+2.83` points on TargetDiff variable swap
  - `+2.05` points on ContextDiff variable swap

That is the cleanest evidence in this experiment that the selected text is doing what it was supposed to do.

The 6-seed confidence intervals strengthen this interpretation. At `12k`, `8e-5`:

- TargetDiff variable swap lift: `+2.83` points, 95% CI `[+1.64, +4.01]`
- ContextDiff variable swap lift: `+2.05` points, 95% CI `[+0.26, +3.84]`

So by the time we aggregate across six seeds, the `12k` role-binding effect no longer looks like a single lucky run. It still looks checkpoint-specific and somewhat narrow, but it is much harder to dismiss as noise.

### 16k checkpoint

At `16k`, the selected text still helps on some variable-swap measures, but the broader model looks less happy with the intervention.

| LR | Overall treated | Overall control | TargetDiff var swap treated | TargetDiff var swap control | ContextDiff var swap treated | ContextDiff var swap control |
|---|---:|---:|---:|---:|---:|---:|
| `2e-5` | `0.5551` | `0.5667` | `0.5385` | `0.5407` | `0.5320` | `0.5458` |
| `4e-5` | `0.5526` | `0.5600` | `0.5385` | `0.5206` | `0.5391` | `0.5308` |
| `8e-5` | `0.5547` | `0.5637` | `0.5323` | `0.5114` | `0.5276` | `0.5197` |

Interpretation:

- Overall EWoK accuracy is worse than control at all three learning rates.
- TargetDiff variable swap still improves meaningfully at `4e-5` and `8e-5`.
- ContextDiff variable swap improves at `4e-5` and `8e-5`, but the effect is smaller and less stable than at `12k`.

At `16k`, `8e-5`:

- TargetDiff variable swap lift: `+2.09` points, 95% CI `[+1.05, +3.13]`
- ContextDiff variable swap lift: `+0.79` points, 95% CI `[-0.45, +2.03]`

So `16k` is not a null result. But it is a more ambiguous one than `12k`. The text still appears to help the target behavior, especially TargetDiff variable swap, but it does so in a regime where overall accuracy degrades more clearly.

## Comparison across checkpoints

The most important pattern is:

- `8k`: weak role-binding lift
- `12k`: strongest and cleanest role-binding lift
- `16k`: still some lift, but with a stronger overall degradation cost

This suggests that the usefulness of the selected text depends on the model's state.

One plausible interpretation is that around `12k` the model is plastic enough to benefit from this targeted discourse signal, but not yet so specialized that the small-data CPT dose simply drags it off-distribution. By `16k`, the intervention may still move the model in the intended direction on the targeted behavior, but the cost of moving it at all is higher.

## What we noticed across learning rates

The learning-rate pattern is also informative.

### At 8k

- `2e-5` is too weak or poorly aligned to produce a robust variable-swap gain.
- `4e-5` and `8e-5` are slightly better, but the effects remain small.

### At 12k

- `2e-5` already gives some variable-swap improvement.
- `4e-5` is mixed overall but still positive on variable swap.
- `8e-5` gives the strongest role-binding effect.

### At 16k

- `2e-5` looks poor.
- `4e-5` is arguably the best compromise if one cares about both some variable-swap lift and not making the broader model too unstable.
- `8e-5` gives the strongest TargetDiff variable-swap lift, but not the best overall behavior.

So the learning-rate story is not simply "lower is safer" or "higher is better." The effect depends on checkpoint and objective:

- if the objective is maximum variable-swap lift, `12k` with `8e-5` looks best
- if the objective is cleaner overall behavior, high LR is less attractive

## Why do both treated and control sometimes degrade relative to the base checkpoint?

This is one of the most important limitations in the interpretation.

In several settings, especially later checkpoints, both treated and control appear worse than the base model on broader EWoK accuracy. That means the intervention is not just changing *which* text the model sees. It is also applying a **small, narrow CPT dose** that can itself move the model away from a good checkpoint.

Several plausible mechanisms could explain this:

1. **Small-data overfitting**
   The intervention uses only `1,000` windows. Over `3` epochs, the model sees the same narrow slice repeatedly. Even the control arm can drift because this is not a large, distribution-preserving training set.

2. **Distribution narrowing**
   Both treated and control are much narrower than the original pretraining distribution. The model may specialize to those windows at the expense of broader EWoK competence.

3. **Checkpoint-dependent plasticity**
   A checkpoint at `12k` may be in a regime where this intervention adds useful structure. A checkpoint at `16k` may be more brittle or more specialized already, so extra CPT causes more collateral damage.

4. **Learning-rate interaction with narrow data**
   Even a moderate LR can be too strong when the dataset is tiny and revisited multiple times. In that regime, the question is not only "is the LR large?" but "is the LR too large for this much repeated exposure on this narrow distribution?"

5. **Objective mismatch**
   EWoK variable swap is only one slice of EWoK. If the selected text is specifically good for entity-role tracking but weakly connected to other EWoK domains, it can improve the target behavior while harming overall accuracy.

## What this experiment does and does not show

### What it does show

- The text-selection metrics are not arbitrary. They are capable of producing **checkpoint- and LR-dependent gains on variable swap**, especially at `12k`.
- The priority score seems more useful than treating all positives as interchangeable, which is exactly what we hoped for when we made the ranking more targeted.
- The best current evidence is that **top-1k priority windows enrich for text that helps role binding relative to matched control**.

### What it does not yet show

- It does not show that the intervention improves the model **absolutely** in a clean way.
- It does not show that the current priority score is the best possible ranking.
- It does not show that the effect scales to larger budgets or more natural training regimes.

So this should be treated as a **targeted causal probe of text selection**, not yet as a production recipe for improving the model overall.

## Why the positive pool alone was not enough

The positive pool was a good first stage, but it was deliberately broad. That was useful because it avoided prematurely hard-coding too much theory into the miner.

However, once the actual CPT budget was fixed at only `1,000` windows, a broad binary filter was not enough. We needed a way to concentrate the budget on the subset of positives that most strongly expressed the hypothesized ingredients of role binding:

- stable entity persistence
- recurring relations
- adjacent-sentence continuity
- repeated entity pairs
- low repetition and low topic jumpiness

That is why the priority score mattered. It was not a replacement for the positive pool. It was a **second-stage concentration mechanism**.

## Robustness relative to earlier pilot runs

Earlier 3-seed pilot runs had already hinted at the same story:

- `12k` looked best
- `16k` showed some lift with more collateral degradation
- `8k` looked weaker

The completed 6-seed sweep did not reverse that story. If anything, it made it clearer.

That is useful because it means the main conclusion does not depend on a single short sweep.

## Bottom line

The current evidence supports the following statement:

> Ranking positive-pool windows by a discourse-targeted priority score produces a treated set that improves EWoK variable swap relative to matched random control, with the clearest effect at the 12k checkpoint and the strongest gains at higher learning rates, especially 8e-5.

But the evidence also supports a second statement:

> This improvement occurs inside a small-data CPT regime that can degrade broader EWoK performance, so the intervention currently looks more like a useful role-binding probe than a globally safe model-improvement recipe.

That is a meaningful result. It says the mining metrics are doing something real, but that the next round of experiments should probably vary **dataset size**, **dose**, and **early stopping**, not just checkpoint and learning rate.

## Appendix: earlier 3-seed pilot results

Before the completed 6-seed multi-checkpoint run, we also ran shorter 3-seed pilots. Those were not the final evidence base for this report, but they are useful because they already pointed in the same direction.

At `12k`, `8e-5`, 3 seeds:

- overall accuracy: treated `0.5485`, control `0.5511`
- TargetDiff variable swap: treated `0.5382`, control `0.5088`, delta `+2.94` points
- ContextDiff variable swap: treated `0.5348`, control `0.5055`, delta `+2.92` points

At `16k`, `8e-5`, 3 seeds:

- overall accuracy: treated `0.5497`, control `0.5644`
- TargetDiff variable swap: treated `0.5294`, control `0.5140`, delta `+1.54` points
- ContextDiff variable swap: treated `0.5237`, control `0.5182`, delta `+0.55` points

So even before the 6-seed run, the broad pattern was already visible:

- `12k` looked like the strongest checkpoint for the intervention
- `16k` still had some target-behavior lift, but with a less favorable broader tradeoff

The 6-seed sweep should therefore be viewed as a robustness extension of that earlier pilot story, not as a completely different result.
