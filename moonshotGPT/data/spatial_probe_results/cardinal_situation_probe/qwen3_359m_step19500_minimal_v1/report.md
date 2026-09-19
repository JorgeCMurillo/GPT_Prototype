# minimal_cardinal: spatial pair evaluation

raw mean full-target conditional token likelihood; no PMI; ties incorrect; sum is secondary

equal family, then evidence format, then case; variants averaged within cases

| Block | Pairs | Mean accuracy | Both contexts correct | Sum accuracy |
|---|---:|---:|---:|---:|
| north_south/main_movement | 15 | 37.50% | 25.00% | 37.50% |
| north_south/explicit_movement | 4 | 33.33% | 33.33% | 33.33% |
| north_south/static_controls | 2 | 75.00% | 50.00% | 75.00% |
| north_south/observer_turns | 2 | 50.00% | 0.00% | 50.00% |
| east_west/main_movement | 15 | 34.38% | 12.50% | 34.38% |
| east_west/explicit_movement | 4 | 25.00% | 16.67% | 25.00% |
| east_west/static_controls | 2 | 50.00% | 0.00% | 50.00% |
| east_west/observer_turns | 2 | 50.00% | 0.00% | 50.00% |

## Event families

| Block | Family | Mean accuracy | Both correct |
|---|---|---:|---:|
| north_south/main_movement | target_crosses | 0.00% | 0.00% |
| north_south/main_movement | reference_crosses | 0.00% | 0.00% |
| north_south/main_movement | target_moves_without_crossing | 75.00% | 50.00% |
| north_south/main_movement | both_move | 75.00% | 50.00% |
| north_south/explicit_movement | target_crosses | 0.00% | 0.00% |
| north_south/explicit_movement | reference_crosses | 0.00% | 0.00% |
| north_south/explicit_movement | target_moves_without_crossing | 100.00% | 100.00% |
| north_south/static_controls | static_placement | 75.00% | 50.00% |
| north_south/observer_turns | observer_turn | 50.00% | 0.00% |
| east_west/main_movement | target_crosses | 12.50% | 0.00% |
| east_west/main_movement | reference_crosses | 0.00% | 0.00% |
| east_west/main_movement | target_moves_without_crossing | 75.00% | 50.00% |
| east_west/main_movement | both_move | 50.00% | 0.00% |
| east_west/explicit_movement | target_crosses | 0.00% | 0.00% |
| east_west/explicit_movement | reference_crosses | 0.00% | 0.00% |
| east_west/explicit_movement | target_moves_without_crossing | 75.00% | 50.00% |
| east_west/static_controls | static_placement | 50.00% | 0.00% |
| east_west/observer_turns | observer_turn | 50.00% | 0.00% |

Few hand-built cases; wording variants are not independent samples. No binomial confidence intervals reported.

Explicit persistence uses only a matched baseline subset; use matched_changes.jsonl for controlled wording/cue comparisons, not unmatched group differences.

All saved token means, sums and explicit-gold decisions reconstructed.
