# Discourse Tracking Methods

This note makes the current `discourse_tracking` subpackage explicit in
methods-style language.

## Goal

The package is trying to find training spans that are more likely to pressure a
model to keep entity identity and role assignments stable across context.

That is narrower than:

- finding spans with many named entities;
- finding spans with many facts;
- finding generally interesting text.

The intended target is text where mistakes about "who is who" or "who did what
to whom" would matter across multiple sentences.

## Unit Of Analysis

The unit is a decoded training span, usually one exact training example window.

Candidates can come from either:

- a checkpoint-local attribution export such as `row_summary_stepXXXXXXXX.csv`;
- a dataset-native sampler that draws exact windows from a corpus such as
  `fineweb_edu_10B`.

Each candidate is identified by the same stable metadata:

- `candidate_id`
- `candidate_kind`
- `shard_path`
- `local_example_idx`
- `token_offset_start`
- `token_offset_end`

## Package Pseudocode

```text
input:
    candidate windows
    +
    dataset/tokenizer access

for each candidate:
    decode exact text
    compute simple discourse features
    write one feature row

over all candidates:
    mark eligibility
    compute quantile thresholds on the eligible subset
    assign pool labels by fixed rules
    compute priority_score for ranking within pools

optional:
    embed only the positive pool
    cluster it
    keep clusters whose mean feature values still look strong

output:
    candidate_features.csv
    candidate_text.jsonl
    positive/random/negative pool files
    summary.json
```

## Feature Computation

For each decoded span, the package computes simple span-level features.

Core discourse features:

- `entity_persistence`
  Maximum fraction of sentences covered by any one tracked entity.
- `mean_entity_persistence`
  Mean sentence coverage over tracked entities.
- `entity_recurrence`
  Fraction of tracked entities that reappear in at least two sentences.
- `entity_churn`
  Rate at which entities are first introduced after sentence 1.
- `relation_density`
  `relation_count / sentence_count`.
- `effective_cast_size`
  Effective number of discourse-active entities from sentence-coverage
  concentration. Higher values mean a more diffuse cast.
- `sentence_count`
- `token_count_text`

Guardrail features:

- `duplicate_sentence_fraction`
- `repeated_3gram_ratio`

Backend behavior:

- `spaCy` mode uses sentence segmentation, NER, and dependency-based relation
  counting.
- regex fallback uses capitalized-name heuristics plus a lightweight
  relation-bearing sentence proxy.

## Eligibility

Pool construction happens only after an eligibility check.

A candidate is eligible if:

- `sentence_count >= min_sentences`
- `unique_entity_count >= min_entities`
- `token_count_text >= min_text_tokens`

With current defaults, that means:

- `min_sentences = 3`
- `min_entities = 2`
- `min_text_tokens = 96`

## What Constitutes The Positive Pool

The positive pool is not "top K by score."

Instead, it is the set of eligible candidates that satisfy a fixed set of
threshold rules computed from the eligible sample itself.

Current positive-pool rule:

- `entity_persistence >= q75(entity_persistence)`
- `entity_recurrence >= q60(entity_recurrence)`
- `relation_density >= q75(relation_density)`
- `repeated_3gram_ratio <= q50(repeated_3gram_ratio)`
- `duplicate_sentence_fraction <= q75(duplicate_sentence_fraction)`
- `unique_entity_count >= positive_unique_entities_min`

where:

- `positive_unique_entities_min = max(min_entities, min(4, ceil(q50(unique_entity_count))))`

So the positive pool is:

- long enough to matter;
- multi-entity enough to matter;
- high on persistence, recurrence, and relation structure;
- low on obvious junk repetition.

This means a candidate can have a relatively low `priority_score` and still be
in the positive pool if it passes these rules.

## Other Pools

### Negative Low-Binding

Eligible, not positive, and:

- `entity_persistence <= q25(entity_persistence)`
- `relation_density <= q40(relation_density)`
- `repeated_3gram_ratio <= q90(repeated_3gram_ratio)`

This is meant to capture text with weak discourse-binding pressure rather than
mere repetition.

### Negative Repetition

Eligible, not positive, and:

- `repeated_3gram_ratio >= q90(repeated_3gram_ratio)`
- `relation_density <= q75(relation_density)`

This is meant to capture repetition-heavy text that may look superficially
structured but is less likely to teach useful discourse tracking.

### Random Control

The random control pool is a deterministic sample from the eligible remainder
after positive and negative-control assignment.

By default its size matches the positive pool size.

## Priority Score

`priority_score` is a ranking heuristic used after pool construction.

It is not the definition of the positive pool.

Current formula:

```text
z(entity_persistence)
+ z(entity_recurrence)
+ z(relation_density)
+ 0.25*z(sentence_count)
+ 0.10*z(unique_entity_count)
- 0.50*z(entity_churn)
- 0.75*z(repeated_3gram_ratio)
- 0.50*z(duplicate_sentence_fraction)
- 0.75*z(effective_cast_size_overflow)
```

where:

- `effective_cast_size_overflow = max(0, effective_cast_size - 6)`

What it is used for:

- ranking examples within a pool;
- selecting fallback positives if the hard positive rule returns no members;
- ordering clusters later by mean quality.

Why it happens after pool assignment:

- hard rules preserve interpretability;
- the score gives a finer ordering inside those interpretable buckets.

## Positive Fallback Rule

If the strict positive rule produces zero rows, the package falls back to a
small top slice of a filtered eligible pool.

Fallback filter:

- `relation_density >= q40(relation_density)`
- `repeated_3gram_ratio <= q90(repeated_3gram_ratio)`
- `unique_entity_count >= min_entities`

Fallback selection:

- sort by `priority_score` descending;
- take the top `min(64, ceil(0.10 * filtered_pool_size))`.

## Optional Clustering

Clustering is optional and only runs inside the positive pool.

Workflow:

1. Build embeddings for positive-pool texts.
2. Run `KMeans`.
3. Summarize each cluster by mean feature values.
4. Keep only clusters that still satisfy fixed feature-based selection rules.

Current cluster-selection rules require:

- cluster size at least `min_cluster_size`;
- mean persistence at or above the positive-pool median;
- mean recurrence at or above the positive-pool median;
- mean relation density at or above the positive-pool median;
- mean repetition features at or below the positive-pool medians.

So clustering is used as an optional refinement step, not as the first-pass
definition of relevance.

## Materialization And Intervention

Once a treated/control contrast looks promising, the package materializes exact
training-example datasets for continued-pretraining.

Typical downstream comparison:

- `positive` vs `random_control`
- or `selected_cluster` vs `random_control`

These matched datasets can then be passed to the existing TrackStar CPT
ablation runner and evaluated on:

- EWoK variable-swap behavior;
- margins;
- tie rates.

## Practical Reading Of The Package

The package should be understood as:

- an interpretable first-pass miner;
- a way to build auditable treated and control pools;
- a bridge from text features to short continued-training interventions.

It should not be understood as:

- a final semantic judge of role-binding quality;
- a coreference-aware discourse model;
- a replacement for attribution analysis.
