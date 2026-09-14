# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_label | close / far | 384 | 97.66% | 95.31% | 100.00% | 47.66% / 52.34% | 95.31% | 0.00% | 4.69% | 0.00% |
| distance_phrase | close / far | 384 | 68.88% | 73.96% | 63.80% | 55.08% / 44.92% | 37.76% | 36.20% | 26.04% | 0.00% |
| distance_phrase_object_first | close / far | 384 | 66.28% | 83.33% | 49.22% | 67.06% / 32.94% | 32.55% | 50.78% | 16.67% | 0.00% |
| endpoint_placement | close / far | 384 | 48.70% | 95.31% | 2.08% | 96.61% / 3.39% | 0.52% | 94.79% | 1.56% | 3.12% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
