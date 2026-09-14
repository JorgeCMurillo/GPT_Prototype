# Separate binary distance comparisons

Qwen3 359M, 19.5k checkpoint. Primary scoring: mean log likelihood over each full target, including punctuation.
Each comparison includes 5,184 contexts, balanced 2,592 per correct answer. Contexts with the excluded third outcome are omitted.
These scores reuse the saved model likelihoods; no new GPU inference was needed. Binary random-choice accuracy is 50%.

| Comparison | Compact | Standard | Expanded | Overall | Both matched contexts correct |
|---|---:|---:|---:|---:|---:|
| closer_vs_farther | 50.06% | 49.36% | 50.00% | 49.81% | 1.31% |
| closer_vs_unchanged | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |

Matched pairs contain opposite correct outcomes with the same numeric/entity/unit/length assignment and description family. Movement and orientation form the situation pairs for contrasts involving unchanged distance.

| Comparison | Explicit contexts | Situation contexts |
|---|---:|---:|
| closer_vs_farther | 49.88% | 49.73% |
| closer_vs_unchanged | 50.00% | 50.00% |
| farther_vs_unchanged | 50.00% | 50.00% |

The explicit and situation families are each balanced within a contrast. Individual movement or orientation subsets are not balanced for contrasts involving unchanged distance.
Length comparisons also vary syntax and reference wording. The original targets have different structures and lengths. These results alone do not isolate conceptual knowledge from target-form preferences.

## Secondary PMI comparison

| Comparison | Accuracy |
|---|---:|
| closer_vs_farther | 47.90% |
| closer_vs_unchanged | 51.27% |
| farther_vs_unchanged | 50.00% |

PMI reuses the previous mean-token target-only BOS baseline, with no added leading space. Raw mean likelihood remains primary.

## Choice preferences

For closer versus farther, the model selects closer on 4,616/5,184 contexts (89.04%). It answers 88.85% of closer contexts correctly, but only 10.76% of farther contexts correctly. Only 34/2,592 matched pairs (1.31%) have both contexts correct.

For each comparison against unchanged distance, the unchanged target wins every context. Thus the 50% accuracy represents a constant choice, with zero matched pairs having both contexts correct.

Validation checked all binary decisions against the two saved mean scores, balanced outcome counts in each comparison, and inclusion of each original context in exactly two binary comparisons per scoring method.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
