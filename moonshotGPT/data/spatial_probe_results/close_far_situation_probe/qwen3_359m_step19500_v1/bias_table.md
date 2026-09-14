# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | close / far | 52 | 41.35% | 75.00% | 7.69% | 83.65% / 16.35% | 0.00% | 75.00% | 7.69% | 17.31% |
| phrase_alternate | close / far | 52 | 44.23% | 67.31% | 21.15% | 73.08% / 26.92% | 5.77% | 61.54% | 15.38% | 17.31% |
| word_beside | close / far | 52 | 96.15% | 96.15% | 96.15% | 50.00% / 50.00% | 92.31% | 3.85% | 3.85% | 0.00% |
| word_alongside | close / far | 52 | 75.96% | 86.54% | 65.38% | 60.58% / 39.42% | 51.92% | 34.62% | 13.46% | 0.00% |
| word_near | close / far | 52 | 40.38% | 65.38% | 15.38% | 75.00% / 25.00% | 1.92% | 63.46% | 13.46% | 21.15% |
| reference_reversed | close / far | 52 | 52.88% | 84.62% | 21.15% | 81.73% / 18.27% | 11.54% | 73.08% | 9.62% | 5.77% |
| negate_close | close / not close | 52 | 32.69% | 46.15% | 19.23% | 63.46% / 36.54% | 1.92% | 44.23% | 17.31% | 36.54% |
| negate_far | not far / far | 52 | 50.00% | 100.00% | 0.00% | 100.00% / 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| both_negated | not far / not close | 52 | 50.00% | 78.85% | 21.15% | 78.85% / 21.15% | 0.00% | 78.85% | 21.15% | 0.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
