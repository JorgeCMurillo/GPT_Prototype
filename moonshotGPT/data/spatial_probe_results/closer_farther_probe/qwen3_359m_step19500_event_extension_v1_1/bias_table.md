# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closer_vs_farther / direct_label | closer / farther | 24 | 91.67% | 100.00% | 83.33% | 58.33% / 41.67% | 83.33% | 16.67% | 0.00% | 0.00% |
| closer_vs_unchanged / direct_label | closer / unchanged | 24 | 79.17% | 58.33% | 100.00% | 29.17% / 70.83% | 58.33% | 0.00% | 41.67% | 0.00% |
| farther_vs_unchanged / direct_label | farther / unchanged | 24 | 77.08% | 54.17% | 100.00% | 27.08% / 72.92% | 54.17% | 0.00% | 45.83% | 0.00% |
| closer_vs_farther / absent | closer / farther | 36 | 47.22% | 91.67% | 2.78% | 94.44% / 5.56% | 0.00% | 91.67% | 2.78% | 5.56% |
| closer_vs_unchanged / absent | closer / unchanged | 36 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / absent | farther / unchanged | 36 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| closer_vs_farther / specific | closer / farther | 1296 | 49.07% | 77.47% | 20.68% | 78.40% / 21.60% | 1.31% | 76.16% | 19.37% | 3.16% |
| closer_vs_unchanged / specific | closer / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / specific | farther / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
