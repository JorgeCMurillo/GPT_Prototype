# Closer/farther event extension: binary evaluation

Qwen3 359M at 19.5k steps. Each decision compares exactly two full target sentences using mean target-token log likelihood. Raw binary choice is primary; target-only PMI is secondary. Uniform-choice chance is 50%.
Every contrast has 2,688 balanced pairs (5,376 judgments). Direct labels are lexical ceiling controls. Numeric cases, wording lengths, and entity orders repeat underlying event patterns.

| Contrast | Evidence | Pairs | Raw choice | PMI choice | First gold | Second gold | Both correct | First chosen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| closer_vs_farther | direct_label | 24 | 91.67% | 97.92% | 100.00% | 83.33% | 83.33% | 58.33% |
| closer_vs_unchanged | direct_label | 24 | 79.17% | 54.17% | 58.33% | 100.00% | 58.33% | 29.17% |
| farther_vs_unchanged | direct_label | 24 | 77.08% | 50.00% | 54.17% | 100.00% | 54.17% | 27.08% |
| closer_vs_farther | absent | 72 | 46.53% | 49.31% | 87.50% | 5.56% | 0.00% | 90.97% |
| closer_vs_unchanged | absent | 72 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| farther_vs_unchanged | absent | 72 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| closer_vs_farther | specific | 2592 | 48.55% | 47.45% | 78.01% | 19.10% | 1.08% | 79.46% |
| closer_vs_unchanged | specific | 2592 | 50.00% | 49.71% | 0.00% | 100.00% | 0.00% | 0.00% |
| farther_vs_unchanged | specific | 2592 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |

'First chosen' is the raw fraction selecting the first named answer in the contrast; a value near 100% can produce 50% accuracy without tracking the context. The two context types in closer/farther are within the same family. Closer/unchanged and farther/unchanged pair an object-movement context with a co-motion context, matched on entities, starting separation, numeric case, unit, wording band, and context entity order; event type and other syntax also change.

## Wording bands

| Contrast | Evidence | Band | Pairs | Raw choice | PMI choice | Both correct |
|---|---|---|---:|---:|---:|---:|
| closer_vs_farther | direct_label | direct | 24 | 91.67% | 97.92% | 83.33% |
| closer_vs_unchanged | direct_label | direct | 24 | 79.17% | 54.17% | 58.33% |
| farther_vs_unchanged | direct_label | direct | 24 | 77.08% | 50.00% | 54.17% |
| closer_vs_farther | absent | compact | 24 | 45.83% | 47.92% | 0.00% |
| closer_vs_unchanged | absent | compact | 24 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | compact | 24 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | absent | standard | 24 | 50.00% | 50.00% | 0.00% |
| closer_vs_unchanged | absent | standard | 24 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | standard | 24 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | absent | expanded | 24 | 43.75% | 50.00% | 0.00% |
| closer_vs_unchanged | absent | expanded | 24 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | expanded | 24 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | compact | 864 | 46.53% | 49.13% | 0.93% |
| closer_vs_unchanged | specific | compact | 864 | 50.00% | 48.50% | 0.00% |
| farther_vs_unchanged | specific | compact | 864 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | standard | 864 | 49.65% | 43.69% | 0.00% |
| closer_vs_unchanged | specific | standard | 864 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | specific | standard | 864 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | expanded | 864 | 49.48% | 49.54% | 2.31% |
| closer_vs_unchanged | specific | expanded | 864 | 50.00% | 50.64% | 0.00% |
| farther_vs_unchanged | specific | expanded | 864 | 50.00% | 50.00% | 0.00% |

## Context entity order

| Contrast | Evidence | Context order | Pairs | Raw choice | PMI choice | Both correct |
|---|---|---|---:|---:|---:|---:|
| closer_vs_farther | direct_label | person_first | 12 | 100.00% | 100.00% | 100.00% |
| closer_vs_unchanged | direct_label | person_first | 12 | 100.00% | 50.00% | 100.00% |
| farther_vs_unchanged | direct_label | person_first | 12 | 95.83% | 50.00% | 91.67% |
| closer_vs_farther | direct_label | object_first | 12 | 83.33% | 95.83% | 66.67% |
| closer_vs_unchanged | direct_label | object_first | 12 | 58.33% | 58.33% | 16.67% |
| farther_vs_unchanged | direct_label | object_first | 12 | 58.33% | 50.00% | 16.67% |
| closer_vs_farther | absent | object_first | 36 | 43.06% | 48.61% | 0.00% |
| closer_vs_unchanged | absent | object_first | 36 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | object_first | 36 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | absent | person_first | 36 | 50.00% | 50.00% | 0.00% |
| closer_vs_unchanged | absent | person_first | 36 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | person_first | 36 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | object_first | 1296 | 47.57% | 47.88% | 0.23% |
| closer_vs_unchanged | specific | object_first | 1296 | 50.00% | 48.30% | 0.00% |
| farther_vs_unchanged | specific | object_first | 1296 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | person_first | 1296 | 49.54% | 47.03% | 1.93% |
| closer_vs_unchanged | specific | person_first | 1296 | 50.00% | 51.12% | 0.00% |
| farther_vs_unchanged | specific | person_first | 1296 | 50.00% | 50.00% | 0.00% |

## Matched entity-order consistency

Each row pairs person-first and object-first contexts with the same event, outcome, entities, distances, targets, and wording band. Order also changes local phrasing or clause position; the flip rate is diagnostic rather than a pure order effect.

| Contrast | Evidence | Order pairs | Person-first raw | Object-first raw | Any raw prediction flip | All four correct |
|---|---|---:|---:|---:|---:|---:|
| closer_vs_farther | direct_label | 12 | 100.00% | 83.33% | 33.33% | 66.67% |
| closer_vs_unchanged | direct_label | 12 | 100.00% | 58.33% | 83.33% | 16.67% |
| farther_vs_unchanged | direct_label | 12 | 95.83% | 58.33% | 75.00% | 16.67% |
| closer_vs_farther | absent | 36 | 50.00% | 43.06% | 13.89% | 0.00% |
| closer_vs_unchanged | absent | 36 | 50.00% | 50.00% | 0.00% | 0.00% |
| farther_vs_unchanged | absent | 36 | 50.00% | 50.00% | 0.00% | 0.00% |
| closer_vs_farther | specific | 1296 | 49.54% | 47.57% | 14.74% | 0.00% |
| closer_vs_unchanged | specific | 1296 | 50.00% | 50.00% | 0.00% | 0.00% |
| farther_vs_unchanged | specific | 1296 | 50.00% | 50.00% | 0.00% | 0.00% |

## Fixed-target context sensitivity

For closer versus farther, the same target is compared across matched toward/away contexts; target-only PMI subtraction cancels. This measure is not applied to the other contrasts because their paired contexts come from different physical event families.

| Evidence | Pairs | Context sensitivity | Both targets correct |
|---|---:|---:|---:|
| direct_label | 24 | 79.17% | 58.33% |
| absent | 72 | 48.61% | 5.56% |
| specific | 2592 | 43.67% | 1.93% |

The archived three-way report offered closer, farther, and same-distance simultaneously and is retained as a secondary response-preference diagnostic. It is not the primary accuracy measure. All binary judgments were calculated from the saved conditional target scores; no model rerun or three-answer ranking was used for the results above.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
