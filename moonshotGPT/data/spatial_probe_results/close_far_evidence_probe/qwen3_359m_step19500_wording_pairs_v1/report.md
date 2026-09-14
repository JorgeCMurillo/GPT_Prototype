# Close/far: matched wording evaluation

Mean full-target token log likelihood, including punctuation. Raw target choice is primary; PMI subtracts the mean target-only BOS likelihood. Context sensitivity compares the same target across the two contexts.

| Mode | Wording family | Pairs | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy | Both correct: choice | Both correct: context |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_baseline | original_placement | 672 | 57.07% | 60.94% | 53.20% | 90.62% | 23.51% | 21.43% | 9.82% |
| close_only | direct_labels | 672 | 59.30% | 61.68% | 50.74% | 95.09% | 23.51% | 19.64% | 1.64% |
| symmetric_pair | direct_labels | 240 | 97.08% | 85.62% | 97.71% | 94.17% | 100.00% | 94.17% | 95.42% |
| close_only | proximity_synonyms | 672 | 33.18% | 35.04% | 50.15% | 42.86% | 23.51% | 6.85% | 1.34% |
| symmetric_pair | proximity_synonyms | 240 | 55.42% | 50.62% | 49.79% | 42.08% | 68.75% | 24.58% | 3.33% |
| close_only | distance_short_long | 672 | 33.63% | 36.01% | 48.59% | 43.75% | 23.51% | 2.53% | 0.15% |
| symmetric_pair | distance_short_long | 240 | 65.00% | 56.46% | 74.17% | 42.92% | 87.08% | 30.00% | 48.33% |
| close_only | distance_small_large | 672 | 31.47% | 36.38% | 48.21% | 39.43% | 23.51% | 2.83% | 0.00% |
| symmetric_pair | distance_small_large | 240 | 60.83% | 54.79% | 78.75% | 38.75% | 82.92% | 21.67% | 57.50% |

## Symmetric pairs by context and target order

| Family | Context structure | Target order | Choice | PMI choice | Context sensitivity |
|---|---|---|---:|---:|---:|
| direct_labels | object_subject | person_first | 95.00% | 80.00% | 98.75% |
| proximity_synonyms | object_subject | person_first | 71.25% | 56.25% | 55.00% |
| distance_short_long | object_subject | person_first | 62.50% | 51.25% | 70.00% |
| distance_small_large | object_subject | person_first | 58.75% | 53.75% | 63.75% |
| direct_labels | object_subject | object_first | 100.00% | 95.00% | 100.00% |
| proximity_synonyms | object_subject | object_first | 40.00% | 38.75% | 42.50% |
| distance_short_long | object_subject | object_first | 73.75% | 63.75% | 70.00% |
| distance_small_large | object_subject | object_first | 66.25% | 53.75% | 85.00% |
| direct_labels | fronted_location | person_first | 92.50% | 72.50% | 97.50% |
| proximity_synonyms | fronted_location | person_first | 65.00% | 55.00% | 50.00% |
| distance_short_long | fronted_location | person_first | 55.00% | 50.00% | 81.25% |
| distance_small_large | fronted_location | person_first | 53.75% | 51.25% | 76.25% |
| direct_labels | fronted_location | object_first | 100.00% | 96.25% | 100.00% |
| proximity_synonyms | fronted_location | object_first | 68.75% | 61.25% | 55.00% |
| distance_short_long | fronted_location | object_first | 73.75% | 58.75% | 83.75% |
| distance_small_large | fronted_location | object_first | 71.25% | 57.50% | 92.50% |
| direct_labels | person_subject | person_first | 98.75% | 85.00% | 97.50% |
| proximity_synonyms | person_subject | person_first | 45.00% | 50.00% | 46.25% |
| distance_short_long | person_subject | person_first | 57.50% | 50.00% | 63.75% |
| distance_small_large | person_subject | person_first | 51.25% | 50.00% | 63.75% |
| direct_labels | person_subject | object_first | 96.25% | 85.00% | 92.50% |
| proximity_synonyms | person_subject | object_first | 42.50% | 42.50% | 50.00% |
| distance_short_long | person_subject | object_first | 67.50% | 65.00% | 76.25% |
| distance_small_large | person_subject | object_first | 63.75% | 62.50% | 91.25% |

Symmetric wording pairs are deduplicated across source scenes: 240 pairs / 480 judgments per family. Close-only substitutions retain the baseline far context and targets: 672 pairs per family. Modes have different weighting and must not be pooled or compared as an isolated intervention without using their match links.
Repeated entities and templates are not independent scenes. Direct labels can repeat targets; repetition flags are included in grouped results. Distance phrasing changes syntax and length. These conditions test explicit or paraphrased distance descriptions, not physical-placement inference.
Matched unchanged-context scores and targets were verified. Exact ties count as incorrect. Target-only PMI cancels in context sensitivity. Input snapshots, token scores, priors, and matched changes are saved.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
