# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| word_to_definition | close definition / far definition | 54 | 84.26% | 98.15% | 70.37% | 63.89% / 36.11% | 68.52% | 29.63% | 1.85% | 0.00% |
| definition_to_word | close / far | 18 | 50.00% | 100.00% | 0.00% | 100.00% / 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
