# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| placement__compact | close / far | 52 | 61.54% | 96.15% | 26.92% | 84.62% / 15.38% | 25.00% | 71.15% | 1.92% | 1.92% |
| placement__setting | close / far | 52 | 50.96% | 80.77% | 21.15% | 79.81% / 20.19% | 11.54% | 69.23% | 9.62% | 9.62% |
| placement__endpoint | close / far | 52 | 45.19% | 63.46% | 26.92% | 68.27% / 31.73% | 7.69% | 55.77% | 19.23% | 17.31% |
| person_direct__compact | close / far | 24 | 62.50% | 100.00% | 25.00% | 87.50% / 12.50% | 25.00% | 75.00% | 0.00% | 0.00% |
| person_direct__setting | close / far | 24 | 58.33% | 100.00% | 16.67% | 91.67% / 8.33% | 16.67% | 83.33% | 0.00% | 0.00% |
| person_direct__endpoint | close / far | 24 | 47.92% | 66.67% | 29.17% | 68.75% / 31.25% | 8.33% | 58.33% | 20.83% | 12.50% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
