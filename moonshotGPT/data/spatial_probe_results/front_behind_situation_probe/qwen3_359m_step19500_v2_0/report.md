# front_behind: spatial pair evaluation

raw mean full-target conditional token likelihood; no PMI; ties incorrect; sum is secondary

equal family, then evidence format, then case; variants averaged within cases

| Block | Pairs | Mean accuracy | Both contexts correct | Sum accuracy |
|---|---:|---:|---:|---:|
| event | 212 | 50.00% | 0.00% | 44.74% |
| control | 4 | 75.00% | 50.00% | 100.00% |
| observer_turn | 8 | 62.50% | 25.00% | 37.50% |

## Event families

| Block | Family | Mean accuracy | Both correct |
|---|---|---:|---:|
| event | static_placement | 50.00% | 0.00% |
| event | target_crosses | 50.00% | 0.00% |
| event | reference_crosses | 50.00% | 0.00% |
| event | target_moves_without_crossing | 50.00% | 0.00% |
| event | both_move | 50.00% | 0.00% |
| control | direct_relation | 75.00% | 50.00% |
| observer_turn | observer_turn | 62.50% | 25.00% |

Few hand-built cases; wording variants are not independent samples. No binomial confidence intervals reported.

All saved token means, sums and explicit-gold decisions reconstructed.
