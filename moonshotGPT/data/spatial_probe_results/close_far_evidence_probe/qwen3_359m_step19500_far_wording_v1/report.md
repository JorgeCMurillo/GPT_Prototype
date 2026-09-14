# Close/far: far-description comparison

Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation.
Each primary row uses the same seven main scenes: 672 pairs / 1,344 context judgments per far variant. Footwear controls are excluded from primary rows. The close contexts and targets are identical across far variants.

| Far description | Far-context accuracy | Overall choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context |
|---|---:|---:|---:|---:|---:|---:|
| across_space | 23.51% | 57.07% | 60.94% | 53.20% | 21.43% | 9.82% |
| direct_far | 100.00% | 95.31% | 86.24% | 67.11% | 90.62% | 34.23% |
| synonym_distant | 70.83% | 80.73% | 76.56% | 55.21% | 63.84% | 10.42% |
| large_distance | 83.48% | 87.05% | 80.13% | 59.45% | 74.40% | 19.20% |

## Far accuracy by context and target order

| Far description | Close-context structure | Target order | Far accuracy | PMI far accuracy | Overall choice | Context sensitivity |
|---|---|---|---:|---:|---:|---:|
| across_space | object_subject | person_first | 41.96% | 91.96% | 70.54% | 51.34% |
| direct_far | object_subject | person_first | 100.00% | 100.00% | 99.55% | 87.05% |
| synonym_distant | object_subject | person_first | 78.57% | 97.32% | 88.84% | 57.14% |
| large_distance | object_subject | person_first | 79.46% | 100.00% | 89.29% | 62.50% |
| across_space | object_subject | object_first | 1.79% | 7.14% | 47.77% | 48.66% |
| direct_far | object_subject | object_first | 100.00% | 100.00% | 96.88% | 54.02% |
| synonym_distant | object_subject | object_first | 68.75% | 57.14% | 81.25% | 50.00% |
| large_distance | object_subject | object_first | 60.71% | 57.14% | 77.23% | 60.27% |
| across_space | fronted_location | person_first | 53.57% | 94.64% | 76.79% | 57.14% |
| direct_far | fronted_location | person_first | 100.00% | 100.00% | 100.00% | 87.50% |
| synonym_distant | fronted_location | person_first | 96.43% | 99.11% | 98.21% | 71.43% |
| large_distance | fronted_location | person_first | 100.00% | 100.00% | 100.00% | 61.16% |
| across_space | fronted_location | object_first | 13.39% | 12.50% | 42.41% | 52.23% |
| direct_far | fronted_location | object_first | 100.00% | 100.00% | 85.71% | 60.27% |
| synonym_distant | fronted_location | object_first | 94.64% | 92.86% | 83.04% | 52.68% |
| large_distance | fronted_location | object_first | 100.00% | 92.86% | 85.71% | 55.36% |
| across_space | person_subject | person_first | 25.89% | 83.04% | 61.61% | 65.18% |
| direct_far | person_subject | person_first | 100.00% | 100.00% | 98.66% | 50.00% |
| synonym_distant | person_subject | person_first | 60.71% | 89.29% | 79.02% | 50.00% |
| large_distance | person_subject | person_first | 92.86% | 95.54% | 95.09% | 61.16% |
| across_space | person_subject | object_first | 4.46% | 7.14% | 43.30% | 44.64% |
| direct_far | person_subject | object_first | 100.00% | 100.00% | 91.07% | 63.84% |
| synonym_distant | person_subject | object_first | 25.89% | 48.21% | 54.02% | 50.00% |
| large_distance | person_subject | object_first | 67.86% | 81.25% | 75.00% | 56.25% |

## Repetition control

Some direct-label far contexts repeat the correct target verbatim. These rows are labeled; synonym and magnitude conditions do not repeat the target.

| Far description | Far target repeats context | Far accuracy | Overall choice |
|---|---|---:|---:|
| across_space | False | 23.51% | 57.07% |
| direct_far | False | 100.00% | 94.08% |
| synonym_distant | False | 70.83% | 80.73% |
| large_distance | False | 83.48% | 87.05% |
| direct_far | True | 100.00% | 97.77% |

Far/distant differs by one word within each context order. Large-distance phrasing also changes syntax; new far descriptions do not specify the original setting. This compares kinds of evidence, not identical geometric descriptions.
All baseline/variant and direct/synonym matches were checked for identical close-context scores and targets. Consequently raw/PMI close-context choice accuracy cannot change across variants, while either context-sensitivity judgment can change.
Generic far contexts recur across scenes, and each close context appears in all four versions. Do not interpret repeated judgments as independent scenes or pool all variants into the primary result.
Both metrics use strict positive margins, with ties incorrect. PMI is a separately labeled mean-token target-only adjustment and cancels in fixed-target context sensitivity. Per-token scores, input snapshots, grouped results (including shoe controls), and matched changes are saved.

## Interpretation and validation

Among the 672 main-scene far judgments per variant, raw choice is correct on 158 with across-space descriptions, 672 with direct far labels, 476 with distant, and 561 with large-distance phrasing. Close-context accuracy remains 90.625% for every variant: changing the far description cannot change these fixed-context choice scores.

The large increase for distant and large distance indicates that the earlier low far accuracy is strongly tied to how farness was expressed. Direct far statements are an easier evidence condition, and some repeat the correct target. However, direct far accuracy is also 100% in the subset without verbatim target repetition. Synonym and magnitude cases do not repeat the target.

The narrower supported conclusion is that the checkpoint handles explicit/paraphrased farness evidence more successfully than the repeated across-large-space constructions. It does not establish robust inference of far from arbitrary spatial arrangements. New contexts vary lexical cues, syntax, length, and setting specificity. Context sensitivity remains substantially below choice accuracy in the new conditions.

Independent reconstruction from all 4,512 saved conditional token sequences verified raw, PMI, and context-sensitivity decisions for all 3,456 pairs. All 864 retained baseline pairs reproduced their previous correctness decisions. Close contexts, targets, and close conditional scores were identical across every far variant. No exact ties occurred in the primary results. GPU 1 was released after evaluation.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
