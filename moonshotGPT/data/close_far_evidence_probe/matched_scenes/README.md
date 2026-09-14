# Matched close/far scenes

This probe expresses each underlying scene with three kinds of evidence while holding the setting, name, object, and close/far targets fixed. The distance evidence has two entity orders.

| Evidence | Close example | Far example |
| --- | --- | --- |
| Direct label | In the gym, the ball is close to Maya. | In the gym, the ball is far from Maya. |
| Distance phrase | In the gym, a short distance separates Maya and the ball. | In the gym, a long distance separates Maya and the ball. |
| Distance phrase, object first | In the gym, a short distance separates the ball and Maya. | In the gym, a long distance separates the ball and Maya. |
| Endpoint placement | In the gym, Maya stands at one end. The ball rests at Maya's feet. | In the gym, Maya stands at one end. The ball rests at the other end. |

The four context conditions are crossed with six settings (gym, hallway, courtyard, field, platform, and auditorium), four names, eight objects, and two target orders. This creates 192 underlying setting/name/object scenes and 1,536 paired probe rows, or 3,072 close/far context judgments. Each context condition has 384 rows; each condition-by-target-order cell has 192.

Targets are identical across the three evidence conditions for a fixed scene and target order:

- Object first: `The ball is close to Maya.` / `The ball is far from Maya.`
- Person first: `Maya is close to the ball.` / `Maya is far from the ball.`

The physical close context requires an inference from an object at the person's feet. The physical far context requires linking the person at one endpoint to the object at the other endpoint. Both contexts use the same setting. Direct and distance conditions state the relationship with progressively less overlap with the target.

The 3,584 match links compare evidence within a scene, the two distance-phrase entity orders, the two target orders under identical contexts, and each setting against the gym while holding entities, evidence, targets, and target order fixed. The two distance forms contain exactly the same words apart from order and capitalization. Setting, template, name, and object variations are repeated measurements rather than independent scenes.

Files in `generated/` include all probes as CSV and JSONL, the comparison links, component tables, a scene overview, review examples for the gym and courtyard, and a manifest containing counts, hashes, tokenizer information, and validation status. Mean target-token log likelihood is the default score.

The probe was evaluated on **Qwen3 359M at 19.5k steps**. Generation status remains recorded in the input manifest; evaluation artifacts and an input snapshot are saved at `runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1/` from the repository root.

| Condition | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct label | 97.66% | 85.68% | 97.14% | 95.31% | 100.00% |
| Distance, person first | 68.88% | 61.46% | 70.83% | 73.96% | 63.80% |
| Distance, object first | 66.28% | 61.72% | 64.84% | 83.33% | 49.22% |
| Endpoint placement | 48.70% | 52.99% | 49.22% | 95.31% | 2.08% |

All 1,536 items were reconstructed from the saved per-token scores and target priors. The grouped counts agree with the input design, all 384 distance-order links are present, and no exact scoring ties occurred.

An additional [applied-score report](../../../runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1/applied_score_report.md) balances the two distance-phrase entity orders within their evidence family and then averages distance phrasing (67.58% raw) with endpoint placement (48.70% raw) equally. The resulting **matched-scene applied mean is 58.14% raw binary accuracy**. Direct labels (97.66%), separate physical arrangements, synonyms, and literal definitions are excluded. This is a score for the matched-scene subset, not a composite of every close/far probe. Recompute it from saved item scores on CPU with:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/matched_scenes/analyze_applied.py --results-dir runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1
```

Regenerate from the repository root:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/matched_scenes/generate.py
```
