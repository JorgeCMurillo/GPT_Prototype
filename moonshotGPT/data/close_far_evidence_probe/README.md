# Close/far: matched evidence diagnostic

A separate [scene expansion](scene_expansion/README.md) adds seven main scene types (feet, hand, lap, elbow, neighboring seat, shared floor tile, and a worn jacket pocket), plus matched owned/worn-shoes controls. It has been evaluated on Qwen3 19.5k; its README links the results. The original probe and evaluation results described here remain preserved.

The [matched-scene probe](matched_scenes/README.md) crosses direct labels, short/long-distance phrases in both entity orders, and endpoint placements within the same six settings, names, objects, and target orders. It has been evaluated on Qwen3 359M at 19.5k steps; its README links the results.

This diagnostic holds names, objects, and target sentences fixed while comparing three descriptions of proximity. It contains **200 paired items / 400 context judgments**, from **4 names × 5 objects × 5 context conditions × 2 target reference orders**. There are 200 unique context texts, reused across target orders.

Names: Maya, Jesse, Li, Omar. Objects: ball, bag, book, cone, helmet. These portable objects support the between-shoes placement without needing the large stationary objects used in the separate movement probe.

| Evidence | Close context | Far context |
|---|---|---|
| Direct label | The ball is close to Maya. | The ball is far from Maya. |
| Proximity paraphrase | The ball is beside Maya. | The ball is a long way from Maya. |
| Physical placement | The ball is between Maya's shoes. | The ball is across the full length of a long gym from Maya. |
| Placement, location first | Between Maya's shoes is the ball. | Across the full length of a long gym from Maya is the ball. |
| Placement, person subject | Maya has the ball between their shoes. | Maya is across the full length of a long gym from the ball. |

The original three context conditions begin with the object. Version 1.1 adds two placement variants mentioning the person before the object. The location-first version preserves the original word multiset and word count, apart from capitalization; syntax and token order change. The person-subject version also changes grammatical roles and pronouns. Singular they/their avoids assigning gender to the algorithmically substituted names. `context_entity_order` records first mention; `context_structure` distinguishes the constructions. For **person-first targets**, the alternatives are "Maya is close to the ball." and "Maya is far from the ball." For the original object-first contexts, these reverse the reference order as in EWoK's direct symmetry cases. For the new context variants, they preserve mention order. The per-item `reverses_context_entity_order` label is computed from the current context and target orders. The **object-first** control uses "The ball is close to Maya." and "The ball is far from Maya." For direct-label contexts this repeats the correct context verbatim, which is explicitly labeled.

Each target order is crossed with every evidence condition. Targets are byte-identical across evidence variants; contexts are byte-identical across target-order variants. Every cell has 20 pairs, with 20 close and 20 far judgments.

## Interpretation limits

The placement example assumes the shoes are being worn. The far setting supplies a clear endpoint but differs in wording length and specificity from the direct/paraphrase conditions. The matched design isolates the evidence description from entity identities and target wording; it does not independently isolate sentence length, individual lexical cues, or inference difficulty. It uses one close/far template pair per evidence condition, so broader template generalization remains untested.

Numbers, comparative closer/farther wording, and target negation are not used. Score complete targets with mean token log likelihood. Report raw choice, separately labeled mean-token PMI choice, fixed-target context sensitivity, both-side success, and close/far choice frequencies. PMI uses the existing BOS-only baseline convention and cancels from context-sensitivity comparisons. Exact ties count as incorrect.

## Files and reproduction

- `components.json`: entities, evidence templates, target reference orders.
- `generate.py`: deterministic generation and validation; uses only a local tokenizer.
- `generated/probes.csv` / `.jsonl`: 200 paired items, with Context1/Context2 and Target1/Target2.
- `generated/matches.csv`: 80 comparisons of evidence against direct labels, 80 placement-context-order comparisons against the original placement form, and 100 target-order comparisons.
- `generated/review_examples.csv`: ten Maya/ball pairs covering all conditions.
- Component tables and `manifest.json`: counts, source hash, and tokenization metadata.
- `evaluate.py`: scores all 800 conditional sequences and 80 target-only priors; saves input snapshots, per-token scores, all item scores, grouped results, matched comparisons, and a report.

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/generate.py
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/evaluate.py \
  --out-dir runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_v1_1_context_order \
  --batch-size 16
```

Select a free GPU and a new output directory. The default model is Qwen3 359M at 19.5k steps; pass `--model` to change it. The [run report](../../runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_v1_1_context_order/report.md) contains the evaluation results.

The original v1.0 context and target strings and probe IDs are preserved; version 1.1 adds context-order metadata and new placement conditions. Earlier evaluation output remains in its original run directory.
