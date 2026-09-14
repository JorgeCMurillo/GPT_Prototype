# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | close / far | 672 | 57.07% | 90.62% | 23.51% | 83.56% / 16.44% | 21.43% | 69.20% | 2.08% | 7.29% |
| ambiguity_control | close / far | 96 | 56.25% | 85.42% | 27.08% | 79.17% / 20.83% | 25.00% | 60.42% | 2.08% | 12.50% |
| explicit_clothing_control | close / far | 96 | 61.98% | 96.88% | 27.08% | 84.90% / 15.10% | 26.04% | 70.83% | 1.04% | 2.08% |
| shared_control | close / far | 160 | 90.62% | 81.25% | 100.00% | 40.62% / 59.38% | 81.25% | 0.00% | 18.75% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
