# Closer/farther event extension: binary evaluation

Qwen3 359M at 19.5k steps. Each decision compares exactly two full target sentences using mean target-token log likelihood. Raw binary choice is primary; target-only PMI is secondary. Uniform-choice chance is 50%.
Every contrast has 1,356 balanced pairs (2,712 judgments): 24 direct-label pairs, 36 nonnumeric event pairs, and 1,296 numeric event pairs. Direct labels are lexical ceiling controls. Numeric cases and wording lengths repeat underlying event patterns.

| Contrast | Evidence | Pairs | Raw choice | PMI choice | First gold | Second gold | Both correct | First chosen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| closer_vs_farther | direct_label | 24 | 91.67% | 97.92% | 100.00% | 83.33% | 83.33% | 58.33% |
| closer_vs_unchanged | direct_label | 24 | 79.17% | 54.17% | 58.33% | 100.00% | 58.33% | 29.17% |
| farther_vs_unchanged | direct_label | 24 | 77.08% | 50.00% | 54.17% | 100.00% | 54.17% | 27.08% |
| closer_vs_farther | absent | 36 | 47.22% | 48.61% | 91.67% | 2.78% | 0.00% | 94.44% |
| closer_vs_unchanged | absent | 36 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| farther_vs_unchanged | absent | 36 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| closer_vs_farther | specific | 1296 | 49.07% | 47.45% | 77.47% | 20.68% | 1.31% | 78.40% |
| closer_vs_unchanged | specific | 1296 | 50.00% | 49.88% | 0.00% | 100.00% | 0.00% | 0.00% |
| farther_vs_unchanged | specific | 1296 | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% |

'First chosen' is the raw fraction selecting the first named answer in the contrast; a value near 100% can produce 50% accuracy without tracking the context. The two context types in closer/farther are within the same family. Closer/unchanged and farther/unchanged pair an object-movement context with a co-motion context, matched on entity, starting separation, numeric case, unit, and wording band; event type and syntax also change.

## Wording bands

| Contrast | Evidence | Band | Pairs | Raw choice | PMI choice | Both correct |
|---|---|---|---:|---:|---:|---:|
| closer_vs_farther | direct_label | direct | 24 | 91.67% | 97.92% | 83.33% |
| closer_vs_unchanged | direct_label | direct | 24 | 79.17% | 54.17% | 58.33% |
| farther_vs_unchanged | direct_label | direct | 24 | 77.08% | 50.00% | 54.17% |
| closer_vs_farther | absent | compact | 12 | 41.67% | 45.83% | 0.00% |
| closer_vs_unchanged | absent | compact | 12 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | compact | 12 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | absent | standard | 12 | 50.00% | 50.00% | 0.00% |
| closer_vs_unchanged | absent | standard | 12 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | standard | 12 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | absent | expanded | 12 | 50.00% | 50.00% | 0.00% |
| closer_vs_unchanged | absent | expanded | 12 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | absent | expanded | 12 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | compact | 432 | 45.83% | 49.77% | 0.00% |
| closer_vs_unchanged | specific | compact | 432 | 50.00% | 46.30% | 0.00% |
| farther_vs_unchanged | specific | compact | 432 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | standard | 432 | 49.88% | 42.59% | 0.00% |
| closer_vs_unchanged | specific | standard | 432 | 50.00% | 50.00% | 0.00% |
| farther_vs_unchanged | specific | standard | 432 | 50.00% | 50.00% | 0.00% |
| closer_vs_farther | specific | expanded | 432 | 51.50% | 50.00% | 3.94% |
| closer_vs_unchanged | specific | expanded | 432 | 50.00% | 53.36% | 0.00% |
| farther_vs_unchanged | specific | expanded | 432 | 50.00% | 50.00% | 0.00% |

## Direct-label entity order

| Contrast | Context order | Pairs | Raw choice | PMI choice | Both correct |
|---|---|---:|---:|---:|---:|
| closer_vs_farther | person_first | 12 | 100.00% | 100.00% | 100.00% |
| closer_vs_unchanged | person_first | 12 | 100.00% | 50.00% | 100.00% |
| farther_vs_unchanged | person_first | 12 | 95.83% | 50.00% | 91.67% |
| closer_vs_farther | object_first | 12 | 83.33% | 95.83% | 66.67% |
| closer_vs_unchanged | object_first | 12 | 58.33% | 58.33% | 16.67% |
| farther_vs_unchanged | object_first | 12 | 58.33% | 50.00% | 16.67% |

## Fixed-target context sensitivity

For closer versus farther, the same target is compared across matched toward/away contexts; target-only PMI subtraction cancels. This measure is not applied to the other contrasts because their paired contexts come from different physical event families.

| Evidence | Pairs | Context sensitivity | Both targets correct |
|---|---:|---:|---:|
| direct_label | 24 | 79.17% | 58.33% |
| absent | 36 | 50.00% | 5.56% |
| specific | 1296 | 47.45% | 3.24% |

The archived three-way report offered closer, farther, and same-distance simultaneously and is retained as a secondary response-preference diagnostic. It is not the primary accuracy measure. All binary judgments were calculated from the saved conditional target scores; no model rerun or three-answer ranking was used for the results above.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
