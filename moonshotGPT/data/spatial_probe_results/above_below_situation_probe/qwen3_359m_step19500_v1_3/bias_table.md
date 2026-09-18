# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_placement / target_first | above / below | 288 | 48.26% | 70.83% | 25.69% | 72.57% / 27.43% | 4.51% | 66.32% | 21.18% | 7.99% |
| static_placement / reference_first | below / above | 288 | 52.60% | 41.67% | 63.54% | 39.06% / 60.94% | 7.99% | 33.68% | 55.56% | 2.78% |
| target_crosses / target_first | above / below | 192 | 51.56% | 94.79% | 8.33% | 93.23% / 6.77% | 3.12% | 91.67% | 5.21% | 0.00% |
| target_crosses / reference_first | below / above | 192 | 48.96% | 23.96% | 73.96% | 25.00% / 75.00% | 0.00% | 23.96% | 73.96% | 2.08% |
| reference_crosses / target_first | above / below | 192 | 48.96% | 90.10% | 7.81% | 91.15% / 8.85% | 0.52% | 89.58% | 7.29% | 2.60% |
| reference_crosses / reference_first | below / above | 192 | 51.56% | 18.23% | 84.90% | 16.67% / 83.33% | 3.65% | 14.58% | 81.25% | 0.52% |
| target_moves_without_crossing / target_first | above / below | 192 | 48.18% | 90.10% | 6.25% | 91.93% / 8.07% | 0.00% | 90.10% | 6.25% | 3.65% |
| target_moves_without_crossing / reference_first | below / above | 192 | 52.34% | 28.12% | 76.56% | 25.78% / 74.22% | 6.25% | 21.88% | 70.31% | 1.56% |
| both_move / target_first | above / below | 288 | 50.00% | 98.96% | 1.04% | 98.96% / 1.04% | 0.00% | 98.96% | 1.04% | 0.00% |
| both_move / reference_first | below / above | 288 | 49.83% | 6.25% | 93.40% | 6.42% / 93.58% | 0.35% | 5.90% | 93.06% | 0.69% |
| direct_label_control / target_first | above / below | 48 | 52.08% | 50.00% | 54.17% | 47.92% / 52.08% | 45.83% | 4.17% | 8.33% | 41.67% |
| direct_label_control / reference_first | below / above | 48 | 54.17% | 56.25% | 52.08% | 52.08% / 47.92% | 45.83% | 10.42% | 6.25% | 37.50% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
