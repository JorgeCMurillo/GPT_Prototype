# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closer_vs_farther / explicit | closer / farther | 1296 | 49.88% | 80.17% | 19.60% | 80.29% / 19.71% | 2.08% | 78.09% | 17.52% | 2.31% |
| closer_vs_farther / situation | closer / farther | 1296 | 49.73% | 97.53% | 1.93% | 97.80% / 2.20% | 0.54% | 96.99% | 1.39% | 1.08% |
| closer_vs_unchanged / explicit | closer / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| closer_vs_unchanged / situation | closer / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / explicit | farther / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| farther_vs_unchanged / situation | farther / unchanged | 1296 | 50.00% | 0.00% | 100.00% | 0.00% / 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
