# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closer_vs_farther / direct_label | closer / farther | 24 | 91.67% | 100.00% | 83.33% | 58.33% / 41.67% | 83.33% | 16.67% | 0.00% | 0.00% |
| closer_vs_unchanged / direct_label | closer / unchanged | 24 | 79.17% | 58.33% | 100.00% | 29.17% / 70.83% | 58.33% | 0.00% | 41.67% | 0.00% |
| farther_vs_unchanged / direct_label | farther / unchanged | 24 | 77.08% | 54.17% | 100.00% | 27.08% / 72.92% | 54.17% | 0.00% | 45.83% | 0.00% |
| closer_vs_farther / absent | closer / farther | 72 | 46.53% | 87.50% | 5.56% | 90.97% / 9.03% | 0.00% | 87.50% | 5.56% | 6.94% |
| closer_vs_unchanged / absent | closer / unchanged | 72 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / absent | farther / unchanged | 72 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| closer_vs_farther / specific | closer / farther | 2592 | 48.55% | 78.01% | 19.10% | 79.46% / 20.54% | 1.08% | 76.93% | 18.02% | 3.97% |
| closer_vs_unchanged / specific | closer / unchanged | 2592 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / specific | farther / unchanged | 2592 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
