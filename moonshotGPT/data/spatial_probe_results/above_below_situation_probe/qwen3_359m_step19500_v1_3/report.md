# Above/below situation probe

Binary choice between two complete target sentences, scored by mean target-token log likelihood. Each pair has one above-compatible and one below-compatible context; exact ties count as wrong.

| Family | Evidence | Pairs | Accuracy | Above gold | Below gold | Both correct | Above word chosen |
|---|---|---:|---:|---:|---:|---:|---:|
| static_placement | measured_height | 144 | 49.31% | 56.94% | 41.67% | 5.56% | 47.22% |
| static_placement | named_shelves | 144 | 47.57% | 50.69% | 44.44% | 4.86% | 43.40% |
| static_placement | numbered_floors | 144 | 54.17% | 61.11% | 47.22% | 9.03% | 88.89% |
| static_placement | numbered_steps | 144 | 50.69% | 56.25% | 45.14% | 5.56% | 87.50% |
| **static_placement** | **all evidence forms** | 576 | **50.43%** | **56.25%** | **44.62%** | **6.25%** | **66.75%** |
| target_crosses | measured_height | 96 | 48.44% | 64.58% | 32.29% | 0.00% | 73.44% |
| target_crosses | named_shelves | 96 | 52.60% | 64.58% | 40.62% | 6.25% | 71.35% |
| target_crosses | numbered_floors | 96 | 50.00% | 51.04% | 48.96% | 0.00% | 98.96% |
| target_crosses | numbered_steps | 96 | 50.00% | 57.29% | 42.71% | 0.00% | 92.71% |
| **target_crosses** | **all evidence forms** | 384 | **50.26%** | **59.38%** | **41.15%** | **1.56%** | **84.11%** |
| reference_crosses | measured_height | 96 | 49.48% | 66.67% | 32.29% | 0.00% | 79.69% |
| reference_crosses | named_shelves | 96 | 51.56% | 44.79% | 58.33% | 7.29% | 74.48% |
| reference_crosses | numbered_floors | 96 | 50.00% | 52.08% | 47.92% | 0.00% | 97.92% |
| reference_crosses | numbered_steps | 96 | 50.00% | 53.12% | 46.88% | 1.04% | 96.88% |
| **reference_crosses** | **all evidence forms** | 384 | **50.26%** | **54.17%** | **46.35%** | **2.08%** | **87.24%** |
| target_moves_without_crossing | measured_height | 96 | 50.00% | 62.50% | 37.50% | 2.08% | 76.04% |
| target_moves_without_crossing | named_shelves | 96 | 48.96% | 62.50% | 35.42% | 5.21% | 65.62% |
| target_moves_without_crossing | numbered_floors | 96 | 51.04% | 54.17% | 47.92% | 2.08% | 96.88% |
| target_moves_without_crossing | numbered_steps | 96 | 51.04% | 57.29% | 44.79% | 3.12% | 93.75% |
| **target_moves_without_crossing** | **all evidence forms** | 384 | **50.26%** | **59.11%** | **41.41%** | **3.12%** | **83.07%** |
| both_move | measured_height | 144 | 50.00% | 60.42% | 39.58% | 0.69% | 89.58% |
| both_move | named_shelves | 144 | 49.65% | 47.92% | 51.39% | 0.00% | 97.57% |
| both_move | numbered_floors | 144 | 50.00% | 50.00% | 50.00% | 0.00% | 100.00% |
| both_move | numbered_steps | 144 | 50.00% | 52.08% | 47.92% | 0.00% | 97.92% |
| **both_move** | **all evidence forms** | 576 | **49.91%** | **52.60%** | **47.22%** | **0.17%** | **96.27%** |
| direct_label_control | direct_label_plain | 48 | 51.04% | 47.92% | 54.17% | 45.83% | 48.96% |
| direct_label_control | direct_label_positioned | 48 | 55.21% | 58.33% | 52.08% | 45.83% | 46.88% |
| **direct_label_control** | **both label forms** | 96 | **53.12%** | **53.12%** | **53.12%** | **45.83%** | **47.92%** |

**Equal-family applied mean:** 50.23% across five physical families. The direct-label control is separate (53.12%).

The family mean first balances the evidence forms and all wording, object, and entity-order variants within each family, then gives the five physical families equal weight. The repeated variants are matched measurements of a small set of physical cases, not independent scene samples.

`target_moves_without_crossing` holds upward or downward motion constant across the above and below contexts. Its score tests whether the model uses the final relative position rather than motion direction alone. `both_move` includes preserved and reversed vertical order.

`variant_consistency_summary.json` records matched prediction flips for context order, target order, evidence type, and wording length. Target order also inverts the relation word in the answer, so its flip rate is not a pure syntax effect. Length-band prefixes vary in wording as well as length. `bias_table.md` separates lexical answer preference by target order.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
