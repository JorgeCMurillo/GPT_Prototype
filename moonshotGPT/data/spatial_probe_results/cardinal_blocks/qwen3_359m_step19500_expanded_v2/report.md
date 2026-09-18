# Cardinal definitions and observer turns

raw mean full-target conditional log likelihood; no PMI; ties incorrect; explicit per-context gold labels

| Dataset | Axis / block | Pairs | Accuracy | Both contexts correct |
|---|---|---:|---:|---:|
| definitions | north_south / direction_definition | 16 | 71.88% | 56.25% |
| definitions | north_south / opposite_direction | 8 | 87.50% | 87.50% |
| definitions | north_south / relative_map_position | 16 | 50.00% | 6.25% |
| definitions | east_west / direction_definition | 16 | 71.88% | 43.75% |
| definitions | east_west / opposite_direction | 8 | 93.75% | 87.50% |
| definitions | east_west / relative_map_position | 16 | 43.75% | 0.00% |
| observer_turn | east_west / cardinal | 24 | 100.00% | 100.00% |
| observer_turn | east_west / observer_relative | 24 | 50.00% | 0.00% |
| observer_turn | north_south / cardinal | 24 | 62.50% | 58.33% |
| observer_turn | north_south / observer_relative | 24 | 50.00% | 4.17% |

Observer-turn rows compare no turn (C1) with a half-turn (C2). Cardinal gold stays unchanged; observer-relative gold switches. Both switch directions and both cardinal answers are balanced. Prediction stability by itself is not correct invariance.
Relative-position definition rows include both map and globe variants, separately broken down in grouped_scores.csv. These small sets reuse wording and concepts and are not independent scene samples.

All persisted token means and per-context gold choices independently reconstructed.
