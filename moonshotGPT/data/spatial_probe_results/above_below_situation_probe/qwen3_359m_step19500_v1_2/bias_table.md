# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_placement / target_first | above / below | 144 | 45.14% | 45.83% | 44.44% | 50.69% / 49.31% | 2.78% | 43.06% | 41.67% | 12.50% |
| static_placement / reference_first | below / above | 144 | 51.74% | 61.81% | 41.67% | 60.07% / 39.93% | 7.64% | 54.17% | 34.03% | 4.17% |
| target_crosses / target_first | above / below | 96 | 53.12% | 89.58% | 16.67% | 86.46% / 13.54% | 6.25% | 83.33% | 10.42% | 0.00% |
| target_crosses / reference_first | below / above | 96 | 47.92% | 39.58% | 56.25% | 41.67% / 58.33% | 0.00% | 39.58% | 56.25% | 4.17% |
| reference_crosses / target_first | above / below | 96 | 47.92% | 80.21% | 15.62% | 82.29% / 17.71% | 1.04% | 79.17% | 14.58% | 5.21% |
| reference_crosses / reference_first | below / above | 96 | 53.12% | 31.25% | 75.00% | 28.12% / 71.88% | 6.25% | 25.00% | 68.75% | 0.00% |
| target_moves_without_crossing / target_first | above / below | 96 | 46.35% | 80.21% | 12.50% | 83.85% / 16.15% | 0.00% | 80.21% | 12.50% | 7.29% |
| target_moves_without_crossing / reference_first | below / above | 96 | 52.60% | 44.79% | 60.42% | 42.19% / 57.81% | 7.29% | 37.50% | 53.12% | 2.08% |
| both_move / target_first | above / below | 144 | 50.00% | 97.92% | 2.08% | 97.92% / 2.08% | 0.00% | 97.92% | 2.08% | 0.00% |
| both_move / reference_first | below / above | 144 | 49.65% | 10.42% | 88.89% | 10.76% / 89.24% | 0.69% | 9.72% | 88.19% | 1.39% |
| direct_label_control / target_first | above / below | 48 | 52.08% | 50.00% | 54.17% | 47.92% / 52.08% | 45.83% | 4.17% | 8.33% | 41.67% |
| direct_label_control / reference_first | below / above | 48 | 54.17% | 56.25% | 52.08% | 52.08% / 47.92% | 45.83% | 10.42% | 6.25% | 37.50% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
