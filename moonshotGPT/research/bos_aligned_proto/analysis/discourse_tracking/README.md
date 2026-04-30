# Discourse Tracking

This subpackage is the feature-first mining layer for the question:

Which training spans actually pressure the model to keep entity identity and
role assignments stable across context, in the same general regime that matters
for EWoK variable-swap behavior?

The workflow is intentionally simple and interpretable:

1. Decode a candidate set of training examples.
2. Score span-level discourse features.
3. Build fixed-rule positive, random-control, and negative-control pools.
4. Optionally cluster only inside the promising pool.
5. Materialize exact treated/control datasets for short continued-pretraining.

It is meant to complement the maintained attribution pipeline, not replace it.
Attribution tells you which exposed rows looked helpful for a checkpoint.
This package helps you ask a different question first:

Does the text itself look like discourse that should teach identity tracking?

For an explicit methods-style description of the current package behavior, see
[METHODS.md](./METHODS.md).

## Selector Hypotheses

Each selector defines one treated/control intervention hypothesis. The primary
readout is whether short continued-pretraining on the selected snippets improves
EWoK behavior, especially variable-swap margins and accuracy, more than the
selector's matched control snippets.

| Selector | Hypothesis | Current status |
| --- | --- | --- |
| `relation_role` | Directional relation prose should improve asymmetric role-binding: the model should better preserve who acted on whom, who held which role, and which entity belongs in which argument position. | Primary selector for variable-swap and asymmetric role-binding. |
| `entity_persistence` | Text where the same entity or entity pair remains active across nearby sentences should improve discourse identity tracking, giving the model more practice maintaining referents across context. | Broad identity-continuity baseline; useful but less specific than relation-role binding. |
| `attribute_rich` | Entity-linked descriptive text should improve entity-property binding: the model should better keep attributes attached to the correct object or agent. | Secondary binding baseline; intentionally distinct from directional relation binding. |
| `state_update` | Spans with explicit state changes should help the model track how entities, institutions, objects, or processes change over local discourse. | Useful but broad; current recipe is not yet a pure persistent-entity state-update selector. |
| `internal_state` | Spans with explicit goals, intentions, desires, preferences, or mental-state complements should improve agent-property style tracking, especially where EWoK requires binding an internal state to the right agent. | Primary-ish selector for agent-property / belief-desire-intent behavior; stronger anchors now downweight weak-only perception/affect text. |
| `mixed_structural` | Snippets combining multiple discourse pressures should test whether broad multi-signal discourse complexity improves EWoK more than a matched control. | Secondary stress test; interpretable as "mixed discourse pressure" rather than one clean mechanism. |
| `role_alternation` | Recurrent entity pairs with changed or reversed roles should be a sharper test of distinguishing `A acted on B` from `B acted on A`. | Deferred follow-up; current pool is sparse/noisy and should not be a primary intervention yet. |

## Code Layout

The sentence-snippet miner is split by responsibility:

- `sentence_windows.py`
  Sentence splitting and contiguous sentence-window generation.
- `snippet_features.py`
  Text/span feature extraction, including discourse cues and layout-noise
  guardrails.
- `selector_recipes.py`
  Selector gates, scalar ranking scores, sort keys, and non-overlap helpers.
- `pool_selection.py`
  DataFrame-level treated/control assignment and optional clustering.
- `sentence_snippets.py`
  Compatibility exports for older imports.

## Feature Set

The current first-pass features are:

- `entity_persistence`
  The max fraction of sentences covered by one tracked entity.
- `entity_churn`
  How often new entities are introduced after sentence 1.
- `entity_recurrence`
  Fraction of tracked entities that reappear in at least two sentences.
- `adjacent_entity_overlap`
  Mean entity-set overlap between neighboring sentences, used to reward local
  cast continuity rather than topic-jumpy windows.
- `pair_recurrence`
  Fraction of co-mentioned entity pairs that recur across at least two
  sentences, used to reward repeated pairwise tracking pressure.
- `top_pair_sentence_share`
  The fraction of sentences covered by the single most persistent entity pair,
  used to reward spans where one pair really drives the discourse.
- `bos_contamination_penalty`
  A binary penalty that turns on when an internal `[BOS]` marker appears in the
  decoded text, used to downweight stitched or topic-reset windows.
- `relation_density`
  Relation edges per sentence.
- `effective_cast_size`
  Effective number of discourse-active entities from sentence-coverage
  concentration, used to penalize very diffuse casts.
- `discourse_length`
  Represented by `sentence_count` and `token_count_text`.
- repetition guardrails
  `duplicate_sentence_fraction` and `repeated_3gram_ratio`, to separate useful
  discourse from junk repetition.

These are deliberately interpretable and auditable. The idea is to start with
high-signal rules before adding semantic clustering.

## spaCy vs Fallback

Preferred path:

- `spaCy`
  Use it for sentence segmentation, NER, and dependency-based relation counts.

Fallback path:

- regex
  When spaCy or its English model is unavailable, the scripts still run using
  capitalized-name heuristics plus a lightweight verb-cue relation proxy.

That makes the workflow runnable now, even in environments where `spacy` is
not installed.

Recommended install if you want the stronger backend:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Embeddings

Recommended semantic option:

- `sentence-transformers`
  Good when you want unsupervised semantic structure inside the already
  promising pool.

Zero-extra-dependency fallback:

- `scikit-learn`
  `TF-IDF -> TruncatedSVD -> KMeans`

That fallback is already enough for:

- splitting dialogue/relationship-heavy text from templatic repetition;
- keeping cluster selection rule-based;
- avoiding a dependency cliff before the feature rules are working.

If you want semantic embeddings later:

```bash
pip install sentence-transformers
```

## Suggested Workflow

### 1. Start from an interpretable candidate universe

The easiest input is an existing attribution export such as
`row_summary_stepXXXXXXXX.csv`, because it already contains:

- `candidate_id`
- `candidate_kind`
- shard metadata
- token offsets

That means the mining outputs can be materialized directly into CPT datasets.

There is also a dataset-native option when you want discovery rather than
checkpoint-local attribution. The sampler draws exact training examples with a
fixed seed and writes the same candidate schema that the miner already accepts.

Example:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.sample_dataset_candidates \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --output_csv /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/fineweb10b_seed42_candidates.csv \
  --candidate_kind stream_window \
  --seq_len 1024 \
  --num_samples 5000 \
  --seed 42
```

This sampler is shard-aware and reproducible:

- it counts exact training examples per shard;
- samples global example ids uniformly without replacement;
- maps them back to shard/local offsets;
- records the fixed seed in the summary JSON.

### 2. Mine feature-based pools

Example:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.mine_candidate_pools \
  --candidate_csv /home/jorge/tokenPred/moonshotGPT/experiments/<run>/analysis/attribution/<attrib_run>/row_summary_step00016000.csv \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --checkpoint_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run>/ckpt_periodic_step0016000 \
  --output_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/step16000_regex \
  --parser_backend auto \
  --embedding_backend tfidf_svd \
  --n_process 4
```

Main outputs:

- `candidate_features.csv`
- `positive_pool.csv`
- `random_control_pool.csv`
- `negative_low_binding_pool.csv`
- `negative_repetition_pool.csv`
- `summary.json`

Optional clustering outputs:

- `cluster_assignments.csv`
- `cluster_summary.csv`
- `selected_cluster_pool.csv`

`--n_process` parallelizes the feature-scoring stage:

- spaCy mode uses `nlp.pipe(..., n_process=N)`;
- regex mode uses a Python process pool.

Decoding now supports a sparse path as well:

- `--decode_strategy auto`
  Default. Uses sparse decoding.
- `--decode_strategy sparse`
  Decodes directly from candidate `shard_path` and token offsets, or directly
  from packed-index `candidate_id`, without first building a full example
  manifest over the whole dataset.
- `--decode_strategy manifest`
  Keeps the older full-manifest decode path for verification or comparison.

## Fixed Rules, Not Vibes

Pool membership is driven by written thresholds from the candidate sample:

- positive pool:
  high `entity_persistence`, high `entity_recurrence`, high `relation_density`,
  low repetition
- negative low-binding pool:
  low persistence and low relation density
- negative repetition pool:
  high repeated n-grams with weak relation structure
- random control pool:
  deterministic sample from the eligible remainder

The positive pool is not "top K by score." It is the subset of eligible windows
that satisfy the written threshold rule. `priority_score` is used afterward to
rank examples within those interpretable buckets and to support fallback
selection if the strict positive rule returns zero rows.

The current `priority_score` now includes both local continuity and pairwise
recurrence, plus a reward for dominant-pair coverage and a penalty for internal
document resets:

```text
z(entity_persistence) + z(entity_recurrence) + z(relation_density) +
0.75*z(adjacent_entity_overlap) + 0.75*z(pair_recurrence) +
0.75*z(top_pair_sentence_share) +
0.25*z(sentence_count) + 0.10*z(unique_entity_count) -
0.50*z(entity_churn) - 0.50*z(bos_contamination_penalty) -
0.75*z(repeated_3gram_ratio) -
0.50*z(duplicate_sentence_fraction) -
0.75*z(effective_cast_size_overflow)
```

The intuition behind the two newer terms is:

- `top_pair_sentence_share`
  Push the score toward spans where one entity pair keeps reappearing across
  many sentences, rather than spans that merely mention many pairs once.
- `bos_contamination_penalty`
  Push the score away from windows that decode across internal BOS boundaries,
  since those are more likely to reflect topic jumps or stitched documents than
  true discourse continuity.

Cluster selection is also fixed-rule:

- only cluster the positive pool;
- select clusters whose mean feature values stay above the positive-pool
  medians on persistence, recurrence, relation density, adjacent continuity,
  and pair recurrence;
- reject clusters whose mean repetition features are above the positive-pool
  medians.

## Why Score And Clusters Can Disagree

`priority_score` is an example-level ranking. Cluster selection is a
cluster-level filter. Those are intentionally different.

In a real `40k` `fineweb_edu_10B` + `gte-small` run:

- only about `59%` of the top `100` priority-scored positives survived cluster
  selection;
- `CID 4277117` scored `11.49` but was rejected because it lived in a cluster
  whose average profile missed the cluster-level selection rule;
- `CID 9523153` scored only `0.75` but was kept because it belonged to a
  cluster whose average profile cleared the cluster-level rule.

That example is useful because it shows:

- high `priority_score` means one window looks individually strong;
- selected-cluster membership means the window belongs to a broader subtype
  that looks strong on average.

This also exposes a current limitation. In the same run, one selected cluster
looked like literary/mythic narrative with tight character interaction
(`Mandalore`, `Kratos`, `Othello`, `Hamlet`), while another selected cluster
still contained entity-dense Asia/history exposition (`Lao`, `China`, `Japan`,
`India`). So the current miner is better than a raw entity-count heuristic, but
it is still not a perfect role-binding detector.

## 3. Materialize CPT pools

Once you like a treated/control contrast, materialize exact training-example
datasets:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.materialize_pools \
  --features_csv /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/step16000_regex/candidate_features.csv \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --output_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/step16000_positive_vs_random \
  --treated_pool positive \
  --control_pool random_control \
  --num_treated 1024
```

By default, if `num_treated` is smaller than the treated pool, the materializer
keeps the highest-`priority_score` rows. To draw a reproducible random treated
subset instead, add:

```bash
  --selection_score random \
  --selection_seed 42
```

You can also materialize a cluster ablation batch from `cluster_assignments.csv`.
This writes one matched-pool directory per listed cluster, plus a balanced
mixed-cluster condition that tries to split `--cluster_mix_num_treated` evenly
across the requested clusters and redistributes any leftover quota if some
clusters are too small:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.materialize_pools \
  --features_csv /tmp/fineweb10b_seed42_mined_gte/candidate_features.csv \
  --cluster_assignments_csv /tmp/fineweb10b_seed42_mined_gte/cluster_assignments.csv \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --output_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/outputs/fineweb10b_seed42_clusters_2_4_5 \
  --treated_pool positive \
  --control_pool random_control \
  --cluster_ids 2,4,5 \
  --cluster_mix_num_treated 1000
```

That batch root will contain:

- `cluster_2/`
- `cluster_4/`
- `cluster_5/`
- `cluster_mix/`
- `summary.json`

This writes:

- `treated_dataset/`
- `control_dataset/`
- `pairings.csv`
- `summary.json`

The datasets are exact row-packed views of the selected training windows, so
you can point the existing TrackStar CPT ablation runner at them.

## 4. Run Short Interventions

Then use the maintained TrackStar ablation runner:

```bash
python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_cpt_ablation \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run>/ckpt_periodic_step0016000 \
  --matched_pool_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/step16000_positive_vs_random \
  --output_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/discourse_tracking/outputs/step16000_positive_vs_random_ablation \
  --learning_rates 1e-5,2e-5 \
  --seeds 1,2 \
  --num_epochs 1
```

Then compare:

- variable-swap accuracy;
- margin changes;
- tie rates;
- treated-minus-control deltas.

## Streaming Top-K Sentence Snippet Mining

For larger selector ablations, use the streaming sentence-snippet miner instead
of the eager `mine_sentence_snippets.py` path. It streams parent candidates
twice:

1. pass 1 keeps bounded top-K frontiers per selector;
2. pass 2 samples selector-specific controls while excluding snippets that
   overlap treated intervals from the same parent candidate.

Example for a first 100k-parent-window run:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.mine_sentence_snippets_streaming \
  --candidate_csv /path/to/candidates.csv \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --checkpoint_dir /path/to/base_checkpoint \
  --output_dir /path/to/selector_top10k_streaming \
  --max_candidates 100000 \
  --top_k 10000 \
  --selectors all \
  --min_sentences 3 \
  --max_sentences 6 \
  --max_snippets_per_parent_per_selector 3 \
  --length_balance proportional \
  --parser_backend regex \
  --rerank_backend spacy \
  --frontier_multiplier 3
```

The default control matching is approximate and selector-specific:

- same parent-shard basename;
- same snippet sentence count;
- same `token_count_text` bucket.

The miner also writes human inspection artifacts:

- `review/<selector>_topN.md`
- `review/<selector>_stratified.md`
- `review/<selector>_control_sampleN.md`
- matching JSONL previews under `previews/`
- compact diagnostics under `diagnostics/`

The output `snippet_features.csv` and `snippet_text.jsonl` remain compatible
with `materialize_sentence_snippet_pools.py`; the materializer now prefers
`is_control_<selector>` when present and falls back to the legacy shared
`is_random_control_pool` column otherwise.

## State Update Selector Rationale

The `state_update` selector is meant to test whether training on spans with
explicit entity-state changes improves variable-swap behavior. The target text
is not just "many change verbs"; it is text where an entity, group, institution,
object, or process changes state across a short local context.

Examples of the intended pressure include:

- historical/institutional transitions, such as an organization moving,
  expanding, merging, closing, or changing role;
- narrative state transitions, such as a person falling out of favor, becoming
  responsible for something, or being displaced by another actor;
- process/mechanism transitions, such as a biological, physical, or procedural
  system changing over time.

The early recipe over-relied on `change_verb_density` and
`same_entity_event_chain_count`. That produced some good examples, but the top
ranks also exposed two important failure modes:

- bibliography/citation blocks can repeat author names across citation
  sentences and contain phrases like "changes in weight", which inflated the
  same-entity chain signal without creating a useful local discourse update;
- outline, slide, and list-heavy text can contain many verbs like "increase",
  "change", and "reduce" while mostly testing formatting or topical density
  rather than entity-state tracking.

The current `state_update` score is therefore still simple, but more guarded:

```text
0.85 * min(change_verb_density, 5.0) +
0.65 * min(same_entity_event_chain_count, 3.0) +
0.35 * temporal_marker_density +
0.35 * result_state_pattern_count -
2.10 * noise_penalty -
1.50 * bibliography_noise_score
```

The gate also requires:

- the snippet to pass the general validity, repetition, and layout-noise checks;
- `change_verb_density > 0`;
- `state_update_score > 0`;
- `bibliography_noise_score <= 0.85`;
- at least one stronger state-update anchor:
  `same_entity_event_chain_count > 0`, `result_state_pattern_count > 0`, or
  `temporal_marker_count > 0`.

The cap on `same_entity_event_chain_count` is especially important. It keeps one
pathological source, such as a bibliography with many repeated names, from
dominating the ranking just because the same title-cased strings recur. The cap
does not remove that signal; it just says that beyond three chained entities, the
extra count is no longer strong evidence of better state-update pressure.

The bibliography and inline-list features are candidate-quality guardrails, not
outcome-tuned filters. They were added after inspecting top-ranked examples from
the cached selector previews, before running any EWoK intervention comparison
with those examples. On the `100k` parent-window cache, the conservative
`state_update` rerank changed the pool but did not make it sparse:

- gate-passing cached snippets went from `45,322` to `33,802`;
- locally non-overlapping snippets went from `39,058` to `29,242`;
- the top-10k still filled completely;
- top-10k overlap with the previous recipe was `8,766 / 10,000`;
- in the top-200 preview, bibliography-like examples went from `4` to `0`,
  inline-bullet examples went from `2` to `1`, outline-like examples went from
  `10` to `5`, and `layout_noise_score > 0.5` went from `10` to `6`.

Known costs of the change:

- some legitimate scientific or biomedical process text can be demoted if it is
  citation-heavy;
- some useful structured educational text can be demoted if it is formatted like
  slides or notes;
- simple state changes with only a change verb and no temporal/result/entity
  anchor can be excluded;
- cached reranking cannot recover sentence windows that were never written into
  the original streaming cache.

If future results are sensitive to `state_update`, these guardrails should be
reported as prespecified text-quality filters for aligning the selector with the
hypothesis, not as model-performance filters.

A remaining limitation is that the current recipe still tests broad
state/change discourse, not a pure persistent-entity state-update construct. It
can rank etymology, definition, survey, or process-description text when the
change-word signal is strong, even if the snippet does not require tracking one
stable discourse object through multiple updates. A future follow-up selector
could explicitly hybridize `state_update` with `entity_persistence`, rewarding
state-change cues only when the same entity, institution, person, object, or
process remains active across the snippet. That would test the sharper
hypothesis that variable-swap improvements come from persistent entities whose
states change, rather than from change/process language in general.

## Relation Role Selector Rationale

The `relation_role` selector is the main asymmetric role-binding selector. It is
meant to find snippets where multiple entities are connected by directional
relations: who helped whom, who defeated whom, who led which group, who founded
or commanded what, who replaced whom, and similar role-bearing events. This is
the closest selector to the variable-swap hypothesis that models fail when they
do not bind the right entity to the right role.

`directed_relation_count` is a heuristic count of relation edges. In each
sentence, the extractor looks for at least two entity-like spans plus a relation
cue verb, then creates ordered entity-pair edges. The cue list includes social
relations such as `asked`, `gave`, `helped`, `led`, `met`, `told`, and
`warned`, plus institutional/history verbs such as `founded`, `commanded`,
`appointed`, `recruited`, `trained`, `attacked`, `captured`, `arrested`,
`supported`, `opposed`, `replaced`, `succeeded`, `joined`, `created`, and
`built`.

The first relation-role recipe ranked too heavily by raw
`directed_relation_count`. That let tables, references, product/catalog text,
and name-dense lists dominate the very top because a single formatted block
could create tens or hundreds of shallow entity-pair edges. The score now caps
the relation-count contribution:

```text
2.0 * min(directed_relation_count, 16.0) +
3.0 * two_entity_relation_sentence_fraction +
0.25 * min(relation_density, 6.0) -
2.0 * noise_penalty
```

The cap preserves the high-count signal for normal prose while preventing
extreme formatted examples from receiving arbitrary extra credit. On the `100k`
parent-window cache, the cap plus cue expansion kept the selector non-sparse and
substantially changed the pool:

- gate-passing cached snippets went from `23,524` to `34,294`;
- locally non-overlapping snippets went from `18,901` to `28,170`;
- the top-10k still filled completely;
- top-10k overlap with the previous recipe was `6,512 / 10,000`;
- in the top-200 preview, pipe-table examples went from `21` to `1`, inline
  bullets went from `5` to `0`, reference/catalog-like examples went from `8`
  to `2`, and `layout_noise_score > 0.5` went from `43` to `2`.

Known costs of the change:

- the cue expansion makes the selector broader, especially for historical and
  institutional prose;
- some high-count but genuinely relational examples no longer outrank lower
  count examples once the cap is reached;
- the regex fallback still approximates direction from entity order and cue
  co-occurrence, so it can misread long or syntactically complex sentences;
- remaining table/catalog artifacts may still require a future explicit
  table-or-catalog noise score if they reappear in larger runs.

## Role Alternation Status

`role_alternation` is intended to be the sharper asymmetric stress test: snippets
where the same entity pair recurs with changed or reversed roles. In principle,
this is very close to the variable-swap hypothesis because it asks whether the
model can distinguish `A acted on B` from `B acted on A`, rather than merely
tracking that `A` and `B` co-occur.

In the current `100k` parent-window cache, this selector is both sparse and
noisy:

- gate-passing cached snippets: `6,132`;
- locally non-overlapping snippets: `4,074`;
- requested top-k: `10,000`, so the treated pool does not fill;
- exact stratified controls are limited, requiring substantial fallback
  matching;
- top previews include some genuinely asymmetric examples, but also
  concordance/index pages, tables, medical/process exposition, and other
  repeated-titlecase artifacts.

For now, treat `role_alternation` as a later follow-up rather than a primary
intervention condition. It likely needs stricter table/reference filtering,
caps on repeated-pair counts, and perhaps a definition built on top of the
cleaner capped `relation_role` features before it can cleanly test the intended
hypothesis.

## Attribute Rich Selector Status

`attribute_rich` is meant to test entity-property binding: whether training on
snippets where entities are described by properties improves the model's ability
to keep attributes attached to the correct discourse object. This is a different
binding hypothesis from `relation_role`: it is about which entity is hot, cold,
large, small, opaque, open, closed, heavy, light, etc., rather than who acted on
whom.

The initial recipe leaned strongly on raw attribute-word density and property
word count. That found many genuine descriptive snippets, but also ranked some
entity-free adjective density, recipes, symptom lists, and catalog/spec text.
The current recipe lightly shifts the score toward entity-linked attributes:

```text
0.85 * min(attribute_density, 5.0) +
0.65 * min(entity_attribute_edge_count, 8.0) +
0.55 * min(property_word_count, 8.0) +
0.45 * min(unique_attribute_count, 6.0) -
2.25 * noise_penalty -
1.0 * bibliography_noise_score -
0.25 * min(inline_list_glyph_count, 4.0) -
0.80 * table_catalog_symptom_noise_score
```

The gate now also requires at least one detected entity. On the `100k`
parent-window cache, this kept the selector non-sparse while moving the top
preview toward entity-linked attributes:

- gate-passing cached snippets went from `41,454` to `38,913`;
- locally non-overlapping snippets went from `34,704` to `32,566`;
- the top-10k still filled completely;
- top-10k overlap with the previous recipe was `6,878 / 10,000`;
- in the top-200 preview, snippets with zero detected entities went from `19`
  to `0`, snippets with zero entity-attribute edges went from `9` to `0`, and
  median `entity_attribute_edge_count` went from `7` to `14.5`.
- relative to the entity-linked rerank before the mild table/catalog/symptom
  penalty, top-200 table/list-ish examples went from `17` to `11`, catalog-term
  examples went from `19` to `13`, and keyword-stuff-ish examples went from
  `26` to `22`.

Known costs and remaining issues:

- exact stratified controls changed slightly, from `9,847` in the original
  recipe to `9,747` in this recipe;
- property-word count is less dominant, so some common-noun descriptive process
  text may move down if it lacks detected titlecase entities;
- recipe, symptom-list, product/spec, and catalog-like snippets can still rank
  highly when they contain many entity-like tokens plus property words;
- the table/catalog/symptom-list penalty is intentionally mild, so it demotes
  but does not exclude such examples.

## Cached Sentence Snippet Reranking

For selector iteration, use the cached reranker instead of rerunning the full
streaming miner. It consumes an existing `snippet_features.csv` plus
`snippet_text.jsonl`, applies the current selector gates/scores, rebuilds
non-overlapping treated/control pools, and writes the same review and
diagnostic artifacts.

Example:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.rerank_sentence_snippet_cache \
  --input_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/outputs/discourse_tracking_selector_top10k_100k_seed42/ckpt_periodic_step0016000 \
  --output_dir /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/outputs/discourse_tracking_selector_top10k_100k_seed42_cache_rerank_v2/ckpt_periodic_step0016000 \
  --selectors all \
  --top_k 10000 \
  --num_control_snippets 10000 \
  --max_snippets_per_parent_per_selector 3 \
  --length_balance proportional
```

By default, `--parser_backend cache` reuses the cached scalar features and only
recomputes layout-noise features from the saved text. Use `--parser_backend
regex` or `--parser_backend spacy` when feature extraction logic itself changed
and you want to recompute all snippet features from cached text.

The limitation is deliberate: cached reranking can only select from snippets
that were written into the original cache. It cannot recover sentence windows
discarded by the original streaming frontier. Once the recipes look right, run
the streaming miner again for the final full candidate universe.

## Batch Sentence Snippet Pool Materialization

Once the selector outputs live in separate tuned rerank directories, use the
batch materializer to pack all selected treated/control snippet pools with one
command. The output root is intentionally a directory of selector child
conditions, which lets `run_cpt_ablation.py` auto-discover them.

The manifest can be a simple selector-to-directory mapping:

```json
{
  "relation_role": "/path/to/relation_role_rerank/ckpt_periodic_step0016000",
  "internal_state": "/path/to/internal_state_rerank/ckpt_periodic_step0016000",
  "attribute_rich": "/path/to/attribute_rich_rerank/ckpt_periodic_step0016000",
  "state_update": "/path/to/state_update_rerank/ckpt_periodic_step0016000",
  "entity_persistence": "/path/to/entity_persistence_rerank/ckpt_periodic_step0016000",
  "mixed_structural": "/path/to/mixed_structural_rerank/ckpt_periodic_step0016000"
}
```

Each directory must contain `snippet_features.csv` and `snippet_text.jsonl`.
For unusual layouts, a selector entry can instead provide explicit
`snippet_features_csv` and `snippet_text_jsonl` fields.

Example:

```bash
python -m research.bos_aligned_proto.analysis.discourse_tracking.materialize_sentence_snippet_pools_batch \
  --manifest /path/to/selector_manifest.json \
  --checkpoint_dir /path/to/base_checkpoint_for_tokenizer \
  --output_dir /path/to/selector_cpt_pools \
  --num_treated_snippets 10000 \
  --num_control_snippets 10000
```

The command writes one child directory per selector:

- `relation_role/`
- `internal_state/`
- `attribute_rich/`
- `state_update/`
- `entity_persistence/`
- `mixed_structural/`
- `batch_summary.json`

## Practical Recommendations

- Start with the feature rules first. They are easier to debug than embeddings.
- Keep the first interventions short and matched.
- Compare `positive vs random` before `positive vs negative`.
- Use embeddings only inside the positive pool, not across the whole universe.
- Treat `sentence-transformers` as optional refinement, not the first filter.

That ordering makes it much easier to tell whether any gain came from discourse
tracking rather than from uncontrolled text quality differences.
