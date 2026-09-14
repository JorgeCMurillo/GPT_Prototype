# Qwen3 closer/farther evaluation

Dataset v1.2: 7,776 contexts, three targets per context.
Primary scoring: full-target mean token log likelihood, including punctuation. Uniform three-way chance: 33.33%.

| Scenario | Length | N | Raw accuracy | PMI accuracy (secondary) |
|---|---|---:|---:|---:|
| explicit_distance | compact | 1296 | 33.33% | 33.33% |
| explicit_distance | standard | 1296 | 33.33% | 32.87% |
| explicit_distance | expanded | 1296 | 33.33% | 32.48% |
| movement_description | compact | 864 | 0.00% | 49.77% |
| movement_description | standard | 864 | 0.00% | 43.63% |
| movement_description | expanded | 864 | 0.00% | 45.95% |
| orientation_control | compact | 432 | 100.00% | 0.00% |
| orientation_control | standard | 432 | 100.00% | 0.00% |
| orientation_control | expanded | 432 | 100.00% | 0.00% |

Overall raw accuracy: 33.33%.

Lengths also differ in syntax, punctuation, and reference wording; these are matched wording-and-length comparisons.
The three conditions have different outcome coverage. Inspect per-outcome results and confusion matrices before comparing pooled scores.
Rows reuse numeric cases, entities, and templates; they are not independent scenarios.
PMI uses the previous repository convention and remains secondary. Per-token scores, all item scores, matches, and input snapshots are saved alongside this report.

## Interpretation and validation

The primary score is 33.33% (2,592/7,776). Every raw prediction is Target3, the full sentence ending in "at the same distance from the [object] as before." The 100% orientation accuracy therefore does not demonstrate successful orientation reasoning: the same target wins in every closer and farther context too. Compact, standard, and expanded wording each produce 33.33% overall accuracy and identical choices.

| Correct outcome | Raw accuracy | PMI accuracy (secondary) |
|---|---:|---:|
| closer | 0.00% | 14.89% |
| farther | 0.00% | 80.90% |
| unchanged | 100.00% | 0.00% |

PMI accuracy is 31.93% (2,483/7,776). It selects farther in 6,520 contexts, closer in 1,256, and unchanged in zero contexts. This prior adjustment does not resolve the diagnostic issue.

An independent CPU reconstruction of means and choices from all 23,328 saved per-token sequences confirmed every raw prediction. The unchanged target's mean-score advantage over its best alternative ranges from 0.0677 to 0.7225 natural-log units per token; there are no ties. See `validation.json`.

This establishes a strong target preference under this scoring setup. It does not isolate its cause or establish that the model lacks all closer/farther knowledge. The unchanged target has a different structure and length; those are candidate confounds to control next. A matched target-wording comparison and a separately labeled closer-versus-farther analysis using the saved scores would help separate these effects.
