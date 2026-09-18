# Custom spatial situation benchmark

Raw mean full-target token likelihood; no PMI. Equal weight per included relation category, then event family; evidence formats and declared wording cells are balanced within scene cases. Definitions, direct-label controls and observer-turn diagnostics are excluded.

| Relation category | Paired rows | Scene clusters | Accuracy | Conditional 95% interval | Both contexts correct |
|---|---:|---:|---:|---:|---:|
| above/below | 2,304 | 12 | 50.23% | 49.84%–50.57% | 2.64% |
| close/far | 1,152 | 6 | 58.14% | 54.69%–61.59% | 17.84% |
| closer/farther | 9,324 | 3 | 48.97% | Not estimable (singleton strata) | 0.19% |
| east/west | 648 | 9 | 49.93% | 49.79%–50.00% | 0.00% |
| left/right | 2,304 | 12 | 50.03% | 49.49%–50.45% | 1.74% |
| north/south | 648 | 9 | 50.00% | 50.00%–50.00% | 0.00% |
| **Balanced overall** | — | — | 51.22% | 50.64%–51.80% | 3.73% |

Bootstrap: 20,000 replicates, seed 42; 30 unique cluster units across shared-axis groups. These are not independent template families.

All variants of a selected case travel together. Matched above/below–left/right cases share draws; north/south–east/west cases share draws. Event-family weights remain fixed.

## Interpretation limits

- Intervals are conditional percentile scene-cluster bootstrap intervals, not binomial/Wilson intervals.
- These hand-built cases are not a random sample of spatial reasoning. Intervals do not cover new templates, domains, model seeds, or training uncertainty.
- A zero-width interval means no observed cluster-score variation, not certainty about generalization.
- Binary judgments include reused contexts in distance contrasts; counts are not independent sample sizes.
- closer_farther: singleton strata object_moves_vs_both_move_cross_family, person_moves_reference_stationary, reference_object_moves are fixed in resampling; their scene uncertainty cannot be estimated.
- east_west: singleton strata reference_crosses, target_crosses are fixed in resampling; their scene uncertainty cannot be estimated.
- north_south: singleton strata reference_crosses, target_crosses are fixed in resampling; their scene uncertainty cannot be estimated.
- Close/far resamples six settings with all entity/order variants attached. They reuse a common spatial geometry, so this estimates setting sensitivity, not uncertainty over new spatial structures.
- Closer/farther has one structural cluster per fixed event-family stratum. Its bootstrap contribution is fixed, not an estimated generalization uncertainty; numeric values, names and units are treated as variants, not independent structures.

## Included sources

- `above_below`: `runs/research/bos_aligned_proto/above_below_situation_probe/qwen3_359m_step19500_v1_3/item_scores.jsonl`. Latest applied situation version only; direct-label controls excluded.
- `left_right`: `runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_v1/item_scores.jsonl`. Latest applied situation version only; direct-label controls excluded.
- `north_south`: `runs/research/bos_aligned_proto/cardinal_situation_probe/qwen3_359m_step19500_v1_2/item_scores.jsonl`. Matched axis cases share bootstrap draws. Numeric and named-location evidence get equal weight.
- `east_west`: `runs/research/bos_aligned_proto/cardinal_situation_probe/qwen3_359m_step19500_v1_2/item_scores.jsonl`. Matched axis cases share bootstrap draws. Numeric and named-location evidence get equal weight.
- `close_far`: `runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1/item_scores.jsonl`. Existing matched-scene applied score: distance phrases and endpoint placements; direct labels and other historical close/far versions excluded.
- `closer_farther`: `runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2/overall_applied_pairs.csv`. Existing three-family applied movement macro, with numeric/nonnumeric evidence balanced. Cross-family contrasts share contexts and remain in one structural cluster.

The JSON report records checkpoint, source hashes, category/family intervals, and the exact scope. No model inference was rerun.
