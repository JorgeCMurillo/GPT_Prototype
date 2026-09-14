# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | close / far | 2688 | 80.04% | 90.62% | 69.46% | 60.58% / 39.42% | 62.57% | 28.05% | 6.88% | 2.49% |
| ambiguity_control | close / far | 384 | 78.39% | 85.42% | 71.35% | 57.03% / 42.97% | 61.20% | 24.22% | 10.16% | 4.43% |
| explicit_clothing_control | close / far | 384 | 84.11% | 96.88% | 71.35% | 62.76% / 37.24% | 68.75% | 28.12% | 2.60% | 0.52% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
