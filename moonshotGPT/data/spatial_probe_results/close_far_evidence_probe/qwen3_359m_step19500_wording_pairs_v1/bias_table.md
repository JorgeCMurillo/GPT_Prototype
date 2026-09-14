# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_baseline / original_placement | close / far | 672 | 57.07% | 90.62% | 23.51% | 83.56% / 16.44% | 21.43% | 69.20% | 2.08% | 7.29% |
| close_only / direct_labels | close / far | 672 | 59.30% | 95.09% | 23.51% | 85.79% / 14.21% | 19.64% | 75.45% | 3.87% | 1.04% |
| symmetric_pair / direct_labels | close / far | 240 | 97.08% | 94.17% | 100.00% | 47.08% / 52.92% | 94.17% | 0.00% | 5.83% | 0.00% |
| close_only / proximity_synonyms | close / far | 672 | 33.18% | 42.86% | 23.51% | 59.67% / 40.33% | 6.85% | 36.01% | 16.67% | 40.48% |
| symmetric_pair / proximity_synonyms | close / far | 240 | 55.42% | 42.08% | 68.75% | 36.67% / 63.33% | 24.58% | 17.50% | 44.17% | 13.75% |
| close_only / distance_short_long | close / far | 672 | 33.63% | 43.75% | 23.51% | 60.12% / 39.88% | 2.53% | 41.22% | 20.98% | 35.27% |
| symmetric_pair / distance_short_long | close / far | 240 | 65.00% | 42.92% | 87.08% | 27.92% / 72.08% | 30.00% | 12.92% | 57.08% | 0.00% |
| close_only / distance_small_large | close / far | 672 | 31.47% | 39.43% | 23.51% | 57.96% / 42.04% | 2.83% | 36.61% | 20.68% | 39.88% |
| symmetric_pair / distance_small_large | close / far | 240 | 60.83% | 38.75% | 82.92% | 27.92% / 72.08% | 21.67% | 17.08% | 61.25% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
