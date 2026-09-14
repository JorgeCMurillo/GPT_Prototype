# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_label | close / far | 40 | 97.50% | 95.00% | 100.00% | 47.50% / 52.50% | 95.00% | 0.00% | 5.00% | 0.00% |
| proximity_paraphrase | close / far | 40 | 90.00% | 80.00% | 100.00% | 40.00% / 60.00% | 80.00% | 0.00% | 20.00% | 0.00% |
| physical_placement | close / far | 40 | 56.25% | 90.00% | 22.50% | 83.75% / 16.25% | 20.00% | 70.00% | 2.50% | 7.50% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
