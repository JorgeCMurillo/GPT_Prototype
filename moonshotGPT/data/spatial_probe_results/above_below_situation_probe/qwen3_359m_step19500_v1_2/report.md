# Above/below situation probe

Binary choice between two complete target sentences, scored by mean target-token log likelihood. Each pair has one above-compatible and one below-compatible context; exact ties count as wrong.

| Family | Evidence | Pairs | Accuracy | Above gold | Below gold | Both correct | Above word chosen |
|---|---|---:|---:|---:|---:|---:|---:|
| static_placement | measured_height | 144 | 49.31% | 56.94% | 41.67% | 5.56% | 47.22% |
| static_placement | named_shelves | 144 | 47.57% | 50.69% | 44.44% | 4.86% | 43.40% |
| **static_placement** | **both evidence forms** | 288 | **48.44%** | **53.82%** | **43.06%** | **5.21%** | **45.31%** |
| target_crosses | measured_height | 96 | 48.44% | 64.58% | 32.29% | 0.00% | 73.44% |
| target_crosses | named_shelves | 96 | 52.60% | 64.58% | 40.62% | 6.25% | 71.35% |
| **target_crosses** | **both evidence forms** | 192 | **50.52%** | **64.58%** | **36.46%** | **3.12%** | **72.40%** |
| reference_crosses | measured_height | 96 | 49.48% | 66.67% | 32.29% | 0.00% | 79.69% |
| reference_crosses | named_shelves | 96 | 51.56% | 44.79% | 58.33% | 7.29% | 74.48% |
| **reference_crosses** | **both evidence forms** | 192 | **50.52%** | **55.73%** | **45.31%** | **3.65%** | **77.08%** |
| target_moves_without_crossing | measured_height | 96 | 50.00% | 62.50% | 37.50% | 2.08% | 76.04% |
| target_moves_without_crossing | named_shelves | 96 | 48.96% | 62.50% | 35.42% | 5.21% | 65.62% |
| **target_moves_without_crossing** | **both evidence forms** | 192 | **49.48%** | **62.50%** | **36.46%** | **3.65%** | **70.83%** |
| both_move | measured_height | 144 | 50.00% | 60.42% | 39.58% | 0.69% | 89.58% |
| both_move | named_shelves | 144 | 49.65% | 47.92% | 51.39% | 0.00% | 97.57% |
| **both_move** | **both evidence forms** | 288 | **49.83%** | **54.17%** | **45.49%** | **0.35%** | **93.58%** |
| direct_label_control | direct_label_plain | 48 | 51.04% | 47.92% | 54.17% | 45.83% | 48.96% |
| direct_label_control | direct_label_positioned | 48 | 55.21% | 58.33% | 52.08% | 45.83% | 46.88% |
| **direct_label_control** | **both label forms** | 96 | **53.12%** | **53.12%** | **53.12%** | **45.83%** | **47.92%** |

**Equal-family applied mean:** 49.76% across five physical families. The direct-label control is separate (53.12%).

The family mean first balances the two evidence forms and all wording, object, and entity-order variants within each family, then gives the five physical families equal weight. The repeated variants are matched measurements of a small set of physical cases, not independent scene samples.

`target_moves_without_crossing` holds upward or downward motion constant across the above and below contexts. Its score tests whether the model uses the final relative position rather than motion direction alone. `both_move` includes preserved and reversed vertical order.

`variant_consistency_summary.json` records matched prediction flips for context order, target order, evidence type, and wording length. Target order also inverts the relation word in the answer, so its flip rate is not a pure syntax effect. Length-band prefixes vary in wording as well as length. `bias_table.md` separates lexical answer preference by target order.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
