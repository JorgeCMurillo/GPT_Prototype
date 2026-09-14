# Far-context wording controls

This extension adds the three requested far descriptions to the expanded placement scenes:

| Variant | Example far context |
|---|---|
| Retained baseline | The ball is across the full length of a long gym from Maya. |
| Direct label | The ball is far from Maya. |
| Synonym | The ball is distant from Maya. |
| Distance magnitude | The ball and Maya are separated by a large distance. |

For that example, every version uses exactly the same close context, "The ball is between Maya's feet," and the same target alternatives. The trailing periods are included in the generated sentences and scoring convention.

## Reference and syntax controls

The far descriptions preserve the original entity mention order for each context structure:

| Original context structure | Direct label | Synonym | Distance phrase |
|---|---|---|---|
| Object subject | The ball is far from Maya. | The ball is distant from Maya. | The ball and Maya are separated by a large distance. |
| Fronted location | Far from Maya is the ball. | Distant from Maya is the ball. | A large distance separates Maya and the ball. |
| Person subject | Maya is far from the ball. | Maya is distant from the ball. | Maya and the ball are separated by a large distance. |

The far/distant comparison changes only one word (with initial capitalization where needed), preserving word count. The magnitude phrase preserves mention order but uses different syntax; `far_context_structure` records that distinction. `context_structure` and `close_context_structure` identify the inherited close-context structure. `far_context_entity_order` records first entity mention.

`far_setting_explicit` is false for the new variants: the source scene's setting is retained as metadata but no longer stated in those far sentences. These are category-matched far descriptions, not claims that the new wording specifies the same exact geometry. In some direct-label conditions the correct far target repeats the context; `far_target_repeats_context` identifies this easy control.

## Coverage and counts

All seven main scenes and the two shoe controls are included, preserving the source names, eligible objects, three context structures, and two target orders. Shared direct/paraphrase pairs from the parent dataset are not duplicated here, because they would change the close context too.

- 864 original placement pairs retained as across-space baselines.
- 2,592 new pairs: three far variants for each original pair.
- 3,456 paired rows total across four far-wording conditions.
- Each condition has 672 main-scene pairs and 192 footwear-control pairs.
- 2,592 baseline-to-variant links, plus 864 direct-far-to-distant links.

The same close context appears in all four versions, and generic far descriptions recur across scene types. These repetitions do not increase the number of independent scenes. Deduplicate context/target strings during evaluation, and report results by `far_variant_id` rather than presenting a pooled accuracy as the main result.

The ownership-only shoes condition retains its ambiguity label and exclusion from main-scene results. Use `include_in_primary_scene_summary` to select the seven main scenes separately within each far variant.

## Measurement and status

The new variants have been **evaluated on Qwen3 359M at 19.5k steps**. See the [saved report](../../../../runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_far_wording_v1/report.md). The generated manifest describes generation status; evaluation metadata and snapshots are stored in the run directory. The original across-space baselines already have saved results in the parent expansion run. They preserve source probe IDs; `source_probe_id` and source hashes support exact matching to those scores. Variant IDs append the far-wording label.

Keep mean full-target log likelihood as default. Report far-context accuracy, close-context accuracy, full binary choice accuracy, PMI choice, fixed-target context sensitivity, and both-side success per far variant and context/target order. With targets and close contexts fixed, conditional target-choice scores for close contexts must be identical across variants. Context sensitivity can still change for either target because its contrasting far context changes.

This test distinguishes performance on the original large-space wording from direct and paraphrased farness evidence. A higher score for distant or far alone would not establish better inference from spatial positions.

The parent evaluator supports far-variant grouping and this dataset's main/control accounting. Its `primary_summary` contains one summary per far variant, rather than a pooled main score. It also reports far-target repetition and asserts that close-context scores are identical within every match.

## Files and regeneration

- `components.json`: three variant types with templates by source context structure.
- `generated/probes.csv` / `.jsonl`: all 3,456 pairs.
- `generated/matches.csv`: 2,592 baseline-to-variant comparisons.
- `generated/lexical_matches.csv`: 864 one-word far/distant comparisons.
- `generated/review_examples.csv`: 24 Maya/ball/feet pairs, covering four far descriptions × three context structures × two target orders.
- `generated/far_variants.csv` and `manifest.json`: lookup table and provenance/validation metadata.

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/scene_expansion/far_wording/generate.py
```

Generation loads the existing local tokenizer on CPU; it does not load model weights or modify prior datasets or evaluation outputs.

Evaluate on a free GPU, with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/evaluate.py \
  --dataset-root data/close_far_evidence_probe/scene_expansion/far_wording \
  --out-dir runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_far_wording_v1 \
  --batch-size 16
```

The evaluator saves input snapshots, per-token scores, target-only priors, all pair scores, grouped summaries, and scores for both baseline and one-word synonym matches.
