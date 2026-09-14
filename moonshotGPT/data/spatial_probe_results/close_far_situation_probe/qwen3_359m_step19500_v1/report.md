# Qwen3 359.4M at step 19,500: close/far situation probe

Completed on GPU 1 (RTX 3090), 2026-09-13. All scores use **mean target-token log likelihood**, the default. The local checkpoint was loaded in float32 with eager attention, using the repository's EWoK conditional likelihood routine and BOS handling. No training occurred. GPU 1 returned to 5 MiB used after the process exited.

## Results

Overall completion-choice accuracy: **503/936 judgments, 53.74%**. Both sides correct: **86/468 probes, 18.38%**. There were no exact or near ties (absolute margin below 1e-6). Equal weighting of the eight scenarios gives 53.47% accuracy, compared with 53.74% when pooling all examples.

| Condition | Correct judgments | Accuracy | Both sides correct |
|---|---:|---:|---:|
| Physical placement baseline | 43/104 | 41.35% | 0/52 (0.00%) |
| Alternate placement phrasing | 46/104 | 44.23% | 3/52 (5.77%) |
| beside / far from | 100/104 | 96.15% | 48/52 (92.31%) |
| alongside / a long way from | 79/104 | 75.96% | 27/52 (51.92%) |
| near / a long distance from | 42/104 | 40.38% | 1/52 (1.92%) |
| Reference reversal, baseline contexts | 55/104 | 52.88% | 6/52 (11.54%) |
| close / not close, baseline contexts | 34/104 | 32.69% | 1/52 (1.92%) |
| not far / far, baseline contexts | 52/104 | 50.00% | 0/52 (0.00%) |
| not far / not close, baseline contexts | 52/104 | 50.00% | 0/52 (0.00%) |

Person-object accuracy was 54.17%; object-object accuracy was 52.31%. No entity assignment passed both sides in all nine conditions. Separate EWoK context-sensitivity accuracy was 50.75% overall; this is not the completion-choice metric used in the table.

## Interpretation

The model performs well on the explicit beside/far-from wording but poorly on these descriptions of physical placement. The beside/far-from condition states "far" directly in its far context, giving it a lexical overlap advantage. The alongside and near conditions also change the far expression, so their differences cannot be assigned solely to understanding beside versus alongside versus near.

Negation is also uneven. In the not-far/far condition, the model always prefers not far, giving 100% accuracy for close contexts and 0% for far contexts. The 50% overall accuracy therefore does not indicate successful handling of both alternatives.

One baseline example illustrates a failure on both sides:

- Close context: Maya stands at one end of a long gym. The ball rests between Maya's shoes.
- Far context: Maya stands at one end of a long gym. The ball rests at the other end of the gym from Maya.
- Targets: The ball is close to Maya. / The ball is far from Maya.

| Context | Mean log likelihood of close target | Mean log likelihood of far target |
|---|---:|---:|
| Close | -2.66735 | -2.58934 |
| Far | -2.20419 | -2.28881 |

Both preferences are wrong in this example. These are diagnostic errors on the authored prompts, not proof that the model has no spatial knowledge. Natural-language category judgments have not received independent human review, and wording, lexical overlap, and target preferences can influence likelihood-based scores.

The earlier original EWoK close/far result was 83.33%, and the original definition probe pooled result was 75.69%. This situation batch's 53.74% is lower, but the datasets differ in their task and condition mixtures; their aggregate gap is not a controlled effect size.

## Files and validation

- `summary.json`: model path, input hashes, settings, versions, and aggregate scores.
- `item_scores.csv` / `.jsonl`: all 468 probes, labels, token counts, four sequence scores, margins, and decisions.
- `grouped_scores.csv`: conditions crossed with scenario, entity type, name, and object, plus evidence types.
- `control_matches_scores.csv`: 416 baseline-versus-variant comparisons.
- `substitution_matches_scores.csv`: 558 linked name/object/reference-object comparisons.
- `consistency.csv`: joint success across the nine conditions for each of 52 entity assignments.
- `token_scores.jsonl`: token likelihoods for 1,456 unique context-target strings, reused for identical strings among the 1,872 required combinations.

Validated dataset count and IDs, stable context tokenization boundaries, finite token scores, mean-only results, and aggregate correctness counts against saved per-item decisions.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
