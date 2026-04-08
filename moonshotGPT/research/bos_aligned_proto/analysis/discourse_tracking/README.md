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

## Feature Set

The current first-pass features are:

- `entity_persistence`
  The max fraction of sentences covered by one tracked entity.
- `entity_churn`
  How often new entities are introduced after sentence 1.
- `entity_recurrence`
  Fraction of tracked entities that reappear in at least two sentences.
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

Cluster selection is also fixed-rule:

- only cluster the positive pool;
- select clusters whose mean feature values stay above the positive-pool
  medians on persistence, recurrence, and relation density;
- reject clusters whose mean repetition features are above the positive-pool
  medians.

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

## Practical Recommendations

- Start with the feature rules first. They are easier to debug than embeddings.
- Keep the first interventions short and matched.
- Compare `positive vs random` before `positive vs negative`.
- Use embeddings only inside the positive pool, not across the whole universe.
- Treat `sentence-transformers` as optional refinement, not the first filter.

That ordering makes it much easier to tell whether any gain came from discourse
tracking rather than from uncontrolled text quality differences.
