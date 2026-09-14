# Close/far scene expansion

The separate [far-wording extension](far_wording/README.md) adds direct far labels, distant synonyms, and large-distance phrases while preserving each close context and both targets. It retains this set's across-space versions as baselines. The added far variants have been evaluated on Qwen3 19.5k; their README links the report.

This extension adds **seven main spatial scene types**, each instantiated with four names and four eligible objects. It preserves the three context structures (object subject, location first, person subject) and two target reference orders from the previous probe. It also includes shared direct-label/paraphrase controls and separate footwear ambiguity controls.

## Main scenes

| Scene | Example close context | Example far context |
|---|---|---|
| Between feet | The ball is between Maya's feet. | The ball is across the full length of a long gym from Maya. |
| Held in a hand | The ball is in Maya's hand. | The ball is across the full width of a large courtyard from Maya. |
| On a lap | The book is on Maya's lap. | The book is across the full depth of a large auditorium from Maya. |
| Touching an elbow | The ball is against Maya's elbow. | The ball is across the full length of a large workshop from Maya. |
| Neighboring seat | The bag is on the seat immediately beside Maya. | The bag is across the full width of a large waiting room from Maya. |
| Shared small floor tile | The ball is on the small floor tile under Maya's feet. | The ball is across the full length of a large hall from Maya. |
| Pocket of a worn jacket | The key is inside the pocket of the jacket Maya is wearing. | The key is across the full length of a long hallway from Maya. |

Names are Maya, Jesse, Li, and Omar. Ten objects are available, with four eligible per scene: ball, bag, book, cone, helmet, cup, key, coin, marble, and pebble. The pocket scene uses only key, coin, marble, and pebble. Eligibility is a component table, not unrestricted substitution. Because objects are not fully crossed with all scenes, scene effects and object effects are not completely separable; use shared-object subsets where possible.

`evidence_type`, `scene_id`, `scene_role`, `setting`, and `assumption` record what varies. The neighboring-seat scene contains the explicit proximity cue beside, unlike several other arrangements. Far settings and near relations vary together between scenes, so a scene effect does not isolate either one. Large-space far descriptions remain structurally similar; this expands relation and setting coverage without claiming exhaustive spatial coverage.

## Feet versus shoes

Ownership does not establish that footwear is being worn. The main set therefore uses **feet**, with two separate matched controls:

- Owned shoes: "The ball is between Maya's shoes."
- Explicitly worn shoes: "The ball is between the shoes Maya is wearing."

These controls use the same four objects, names, target sentences, far contexts, and context orders as the between-feet scene. The location-first versions are "Between Maya's feet/shoes is the ball" and "Between the shoes Maya is wearing is the ball." Person-subject versions use "Maya has the ball between their feet/shoes" and "Maya is wearing the shoes with the ball between them."

The owned-shoes condition is tagged `gold_requires_unstated_worn_shoes: true`. Its close label describes the intended worn-shoes reading, not an entailment of ownership alone. Keep it out of the main score and report it as an ambiguity diagnostic. The main seven scenes have `include_in_primary_scene_summary: true`; both shoe controls and shared lexical controls have false.

The footwear comparison holds the far context exactly fixed. Owned versus worn shoes changes explicitness and phrasing/length; feet versus shoes changes the referent. Differences in scores would not by themselves establish that one word caused a failure.

## Size and matching

- **672 main placement pairs / 1,344 context judgments:** 7 scenes × 4 names × 4 eligible objects × 3 context orders × 2 target orders.
- **192 shoe-control pairs / 384 context judgments:** 2 controls × the same entity/order factors.
- **160 shared direct-label/paraphrase pairs / 320 context judgments:** 4 names × 10 objects × 2 evidence types × 2 target orders.
- **Total: 1,024 pairs / 2,048 context judgments.**

For each fixed context-order/target-order combination, the main set has **112 pairs / 224 judgments**, compared with the previous 20 pairs / 40 judgments. It includes seven scene types rather than only changing the entities in the original between-shoes template. However, template, entity, and target variants remain repeated measurements; 224 judgments are not 224 independently sampled scenes.

Direct-label and paraphrase controls are stored once per entity/target-order combination and linked to every eligible scene. The generator creates 3,024 links covering evidence comparisons, context order, target order, feet versus shoes, and explicitly stating wearing. Shared controls and identical far contexts must not be duplicated and treated as independent observations when summarizing.

For person-first targets, alternatives remain "Maya is close to the ball" and "Maya is far from the ball." Object-first targets reverse that order. All targets include final periods. Fronted-location contexts preserve the word multiset and word count of the object-subject contexts, aside from capitalization; person-subject versions also change syntax and pronouns.

## Evaluation plan and limits

This extension has been evaluated on **Qwen3 359M at 19.5k steps**. See the [saved report](../../../runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_scene_expansion_v1/report.md). The generated manifest records generation status; evaluation metadata and input snapshots are stored separately. Retain mean full-target token log likelihood as the primary score, with raw choice, separately labeled PMI choice, fixed-target context sensitivity, and both-side success. Report main placement scenes separately from ambiguity and lexical controls, and break down by scene, context structure, target order, and correct label. Average equally across main scenes, and retain entity-level results to inspect variability. Avoid interpreting pooled percentages as independent-sample significance.

The parent evaluator now accepts `--dataset-root` and reports the main scenes separately from the shoe and lexical controls. No old evaluation files or source datasets were modified by generation.

## Files

- `components.json`: seven main scenes, two shoe controls, object eligibility, context structures.
- `generated/probes.csv` / `.jsonl`: all paired examples and diagnostic metadata.
- `generated/matches.csv`: 3,024 matched comparisons.
- `generated/scene_overview.csv`: one object-subject close/far example per scene/control and eligible objects.
- `generated/review_examples.csv`: 54 Maya examples, one eligible object per scene/control × all context and target orders.
- Component tables and `manifest.json`: counts, hashes, tokenizer metadata, validation status.

Regenerate from the repository root:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/scene_expansion/generate.py
```

Only a local tokenizer is loaded. No GPU evaluation or training is performed.

The [matched wording extension](wording_pairs/README.md) adds close/far, near/distant, short/long distance, and small/large distance families, with close-only substitutions and deduplicated symmetric pairs. It has been evaluated on Qwen3 359M at 19.5k steps; see its README for results.

Evaluate with a free GPU and a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/evaluate.py \
  --dataset-root data/close_far_evidence_probe/scene_expansion \
  --out-dir runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_scene_expansion_v1 \
  --batch-size 16
```

The model defaults to the local Qwen3 19.5k checkpoint. Input snapshots, hashes, per-token likelihoods, target-only priors, pair scores, grouped results, and matched comparison scores are preserved in the output directory. `primary_summary` uses only main-scene items. `grouped_scores.csv` distinguishes `scope=main_scenes` from `scope=all`; use the former for the primary score and the latter for named controls.
