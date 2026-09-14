# Qwen3 closer/farther: new event families

Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation. Three-way uniform-choice chance is 33.33%.
Scores for numeric and nonnumeric versions are reported separately. Direct labels are lexical controls and are not pooled with movement inference.

| Condition | Numeric information | Length | N | Raw choice | PMI choice |
|---|---|---|---:|---:|---:|
| direct_label | absent | direct | 72 | 69.44% | 65.28% |
| reference_object_moves | absent | compact | 48 | 0.00% | 47.92% |
| reference_object_moves | absent | standard | 48 | 0.00% | 50.00% |
| reference_object_moves | absent | expanded | 48 | 0.00% | 50.00% |
| both_move_same_separation | absent | compact | 24 | 100.00% | 0.00% |
| both_move_same_separation | absent | standard | 24 | 100.00% | 0.00% |
| both_move_same_separation | absent | expanded | 24 | 100.00% | 0.00% |
| reference_object_moves | specific | compact | 1728 | 0.00% | 49.13% |
| reference_object_moves | specific | standard | 1728 | 0.00% | 43.69% |
| reference_object_moves | specific | expanded | 1728 | 0.00% | 49.54% |
| both_move_same_separation | specific | compact | 864 | 100.00% | 0.00% |
| both_move_same_separation | specific | standard | 864 | 100.00% | 0.00% |
| both_move_same_separation | specific | expanded | 864 | 100.00% | 0.00% |

## Outcome accuracy

| Condition | Numeric information | Length | Gold outcome | N | Raw choice | PMI choice |
|---|---|---|---|---:|---:|---:|
| direct_label | absent | direct | closer | 24 | 58.33% | 95.83% |
| direct_label | absent | direct | farther | 24 | 50.00% | 100.00% |
| direct_label | absent | direct | unchanged | 24 | 100.00% | 0.00% |
| reference_object_moves | absent | compact | closer | 24 | 0.00% | 0.00% |
| reference_object_moves | absent | standard | closer | 24 | 0.00% | 8.33% |
| reference_object_moves | absent | expanded | closer | 24 | 0.00% | 0.00% |
| reference_object_moves | absent | compact | farther | 24 | 0.00% | 95.83% |
| reference_object_moves | absent | standard | farther | 24 | 0.00% | 91.67% |
| reference_object_moves | absent | expanded | farther | 24 | 0.00% | 100.00% |
| both_move_same_separation | absent | compact | unchanged | 24 | 100.00% | 0.00% |
| both_move_same_separation | absent | standard | unchanged | 24 | 100.00% | 0.00% |
| both_move_same_separation | absent | expanded | unchanged | 24 | 100.00% | 0.00% |
| reference_object_moves | specific | compact | closer | 864 | 0.00% | 0.00% |
| reference_object_moves | specific | standard | closer | 864 | 0.00% | 5.67% |
| reference_object_moves | specific | expanded | closer | 864 | 0.00% | 3.70% |
| reference_object_moves | specific | compact | farther | 864 | 0.00% | 98.26% |
| reference_object_moves | specific | standard | farther | 864 | 0.00% | 81.71% |
| reference_object_moves | specific | expanded | farther | 864 | 0.00% | 95.37% |
| both_move_same_separation | specific | compact | unchanged | 864 | 100.00% | 0.00% |
| both_move_same_separation | specific | standard | unchanged | 864 | 100.00% | 0.00% |
| both_move_same_separation | specific | expanded | unchanged | 864 | 100.00% | 0.00% |

## Wording consistency

| Condition | Numeric information | Groups | All three correct (raw) | Same prediction (raw) |
|---|---|---:|---:|---:|
| reference_object_moves | specific | 1728 | 0.00% | 100.00% |
| reference_object_moves | absent | 48 | 0.00% | 100.00% |
| both_move_same_separation | specific | 864 | 100.00% | 100.00% |
| both_move_same_separation | absent | 24 | 100.00% | 100.00% |

## Matched comparisons

| Control | N | Reference accuracy (raw) | New accuracy (raw) | Same prediction | Fixed | Broken |
|---|---:|---:|---:|---:|---:|---:|
| direct_label_vs_event | 7992 | 69.44% | 33.33% | 62.50% | 0 | 2886 |
| numeric_vs_non_numeric | 7776 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| same_distances_different_event | 2592 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| same_distances_different_mover_or_action | 2592 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| direct_entity_order | 36 | 97.22% | 41.67% | 44.44% | 0 | 20 |
| length_variant | 5328 | 33.33% | 33.33% | 100.00% | 0 | 0 |
| event_entity_order | 3996 | 33.33% | 33.33% | 100.00% | 0 | 0 |

Parent standard comparisons are paired on entity, unit, numeric case, outcome, targets, and initial/final separation; they describe different physical events. Parent item scores were reused only after verifying the checkpoint, mean scoring convention, and source probe hash.
The reference-object closer/farther minimal-pair analysis is saved separately in contrast_report.md. Its binary choice and fixed-target context-sensitivity scores should be considered alongside the three-way confusion results.
Length groups share event coordinates and target sentences. Their wording also changes syntax, so length effects are not isolated. Numeric cases and entity variants repeat the same templates and are not independent scenarios.
Direct-label rows contain the comparative word and serve as an easy ceiling control. Their score should not be interpreted as movement inference. Exact ties count as incorrect. Saved inputs, token scores, target priors, confusion matrices, match outcomes, and item scores support further diagnosis.
