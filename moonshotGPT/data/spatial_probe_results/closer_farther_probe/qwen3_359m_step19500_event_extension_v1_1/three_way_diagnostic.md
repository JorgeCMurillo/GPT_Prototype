# Qwen3 closer/farther: new event families

Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation. Three-way uniform-choice chance is 33.33%.
Scores for numeric and nonnumeric versions are reported separately. Direct labels are lexical controls and are not pooled with movement inference.

**Reference-object movement is also evaluated as a matched closer/farther binary contrast in [contrast_report.md](contrast_report.md).** The 0% raw figures below are three-way scores that include the same-distance target.

| Condition | Numeric information | Length | N | Raw choice | PMI choice |
|---|---|---|---:|---:|---:|
| direct_label | absent | direct | 72 | 69.44% | 65.28% |
| reference_object_moves | absent | compact | 24 | 0.00% | 45.83% |
| reference_object_moves | absent | standard | 24 | 0.00% | 50.00% |
| reference_object_moves | absent | expanded | 24 | 0.00% | 50.00% |
| both_move_same_separation | absent | compact | 12 | 100.00% | 0.00% |
| both_move_same_separation | absent | standard | 12 | 100.00% | 0.00% |
| both_move_same_separation | absent | expanded | 12 | 100.00% | 0.00% |
| reference_object_moves | specific | compact | 864 | 0.00% | 49.77% |
| reference_object_moves | specific | standard | 864 | 0.00% | 42.59% |
| reference_object_moves | specific | expanded | 864 | 0.00% | 50.00% |
| both_move_same_separation | specific | compact | 432 | 100.00% | 0.00% |
| both_move_same_separation | specific | standard | 432 | 100.00% | 0.00% |
| both_move_same_separation | specific | expanded | 432 | 100.00% | 0.00% |

## Outcome accuracy

| Condition | Numeric information | Length | Gold outcome | N | Raw choice | PMI choice |
|---|---|---|---|---:|---:|---:|
| direct_label | absent | direct | closer | 24 | 58.33% | 95.83% |
| direct_label | absent | direct | farther | 24 | 50.00% | 100.00% |
| direct_label | absent | direct | unchanged | 24 | 100.00% | 0.00% |
| reference_object_moves | absent | compact | closer | 12 | 0.00% | 0.00% |
| reference_object_moves | absent | standard | closer | 12 | 0.00% | 8.33% |
| reference_object_moves | absent | expanded | closer | 12 | 0.00% | 0.00% |
| reference_object_moves | absent | compact | farther | 12 | 0.00% | 91.67% |
| reference_object_moves | absent | standard | farther | 12 | 0.00% | 91.67% |
| reference_object_moves | absent | expanded | farther | 12 | 0.00% | 100.00% |
| both_move_same_separation | absent | compact | unchanged | 12 | 100.00% | 0.00% |
| both_move_same_separation | absent | standard | unchanged | 12 | 100.00% | 0.00% |
| both_move_same_separation | absent | expanded | unchanged | 12 | 100.00% | 0.00% |
| reference_object_moves | specific | compact | closer | 432 | 0.00% | 0.00% |
| reference_object_moves | specific | standard | closer | 432 | 0.00% | 10.19% |
| reference_object_moves | specific | expanded | closer | 432 | 0.00% | 5.32% |
| reference_object_moves | specific | compact | farther | 432 | 0.00% | 99.54% |
| reference_object_moves | specific | standard | farther | 432 | 0.00% | 75.00% |
| reference_object_moves | specific | expanded | farther | 432 | 0.00% | 94.68% |
| both_move_same_separation | specific | compact | unchanged | 432 | 100.00% | 0.00% |
| both_move_same_separation | specific | standard | unchanged | 432 | 100.00% | 0.00% |
| both_move_same_separation | specific | expanded | unchanged | 432 | 100.00% | 0.00% |

## Wording consistency

| Condition | Numeric information | Groups | All three correct (raw) | Same prediction (raw) |
|---|---|---:|---:|---:|
| reference_object_moves | specific | 864 | 0.00% | 100.00% |
| reference_object_moves | absent | 24 | 0.00% | 100.00% |
| both_move_same_separation | specific | 432 | 100.00% | 100.00% |
| both_move_same_separation | absent | 12 | 100.00% | 100.00% |

## Matched comparisons

| Control | N | Reference accuracy (raw) | New accuracy (raw) | Same prediction | Fixed | Broken |
|---|---:|---:|---:|---:|---:|---:|
| direct_label_vs_event | 3996 | 97.22% | 33.33% | 36.11% | 0 | 2553 |
| numeric_vs_non_numeric | 3888 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| same_distances_different_event | 1296 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| same_distances_different_mover_or_action | 1296 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| direct_entity_order | 36 | 97.22% | 41.67% | 44.44% | 0 | 20 |
| length_variant | 2664 | 33.33% | 33.33% | 100.00% | 0 | 0 |

Parent standard comparisons are paired on entity, unit, numeric case, outcome, targets, and initial/final separation; they describe different physical events. Parent item scores were reused only after verifying the checkpoint, mean scoring convention, and source probe hash.
Length groups share event coordinates and target sentences. Their wording also changes syntax, so length effects are not isolated. Numeric cases and entity variants repeat the same templates and are not independent scenarios.
Direct-label rows contain the comparative word and serve as an easy ceiling control. Their score should not be interpreted as movement inference. Exact ties count as incorrect. Saved inputs, token scores, target priors, confusion matrices, match outcomes, and item scores support further diagnosis.
