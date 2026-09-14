# Close/far: expanded scene evaluation

Qwen3 359M at 19.5k steps. All metrics use mean full-target token likelihood, including punctuation.
Primary results include only seven main scenes: 672 pairs / 1,344 balanced judgments. Owned/worn-shoes controls and shared direct/paraphrase controls are reported separately. Main scenes contribute equally.

| Context structure | Target order | Judgments | Choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context |
|---|---|---:|---:|---:|---:|---:|---:|
| physical_placement | person_first | 224 | 70.54% | 77.23% | 51.34% | 41.07% | 2.68% |
| physical_placement | object_first | 224 | 47.77% | 45.54% | 48.66% | 0.89% | 1.79% |
| physical_placement_fronted | person_first | 224 | 76.79% | 78.57% | 57.14% | 53.57% | 14.29% |
| physical_placement_fronted | object_first | 224 | 42.41% | 40.18% | 52.23% | 6.25% | 6.25% |
| physical_placement_person_subject | person_first | 224 | 61.61% | 77.23% | 65.18% | 25.89% | 30.36% |
| physical_placement_person_subject | object_first | 224 | 43.30% | 46.88% | 44.64% | 0.89% | 3.57% |

Main-scene overall: choice 57.07%, PMI choice 60.94%, context sensitivity 53.20%.

| Main scene | Judgments | Choice | PMI choice | Context sensitivity |
|---|---:|---:|---:|---:|
| between_feet | 192 | 57.81% | 51.04% | 54.17% |
| in_hand | 192 | 53.65% | 63.02% | 51.04% |
| on_lap | 192 | 53.65% | 65.10% | 51.04% |
| against_elbow | 192 | 54.17% | 53.12% | 50.00% |
| neighboring_seat | 192 | 63.02% | 64.06% | 55.21% |
| shared_floor_tile | 192 | 50.52% | 55.21% | 50.52% |
| worn_jacket_pocket | 192 | 66.67% | 75.00% | 60.42% |

## Feet and footwear comparison

The following averages use all three context structures and both target orders. Each condition has 96 pairs / 192 judgments. Far contexts and targets are identical within each matched comparison; only the close description changes.

| Close description | Choice | Close-context accuracy | Far-context accuracy | PMI choice | Context sensitivity |
|---|---:|---:|---:|---:|---:|
| between_feet | 57.81% | 88.54% | 27.08% | 51.04% | 54.17% |
| between_owned_shoes | 56.25% | 85.42% | 27.08% | 51.56% | 52.08% |
| between_worn_shoes | 61.98% | 96.88% | 27.08% | 55.73% | 53.65% |

## Shared lexical controls

| Condition | Target order | Choice | PMI choice | Context sensitivity |
|---|---|---:|---:|---:|
| direct_label | person_first | 95.00% | 80.00% | 98.75% |
| proximity_paraphrase | person_first | 80.00% | 53.75% | 82.50% |
| direct_label | object_first | 100.00% | 95.00% | 100.00% |
| proximity_paraphrase | object_first | 87.50% | 75.00% | 66.25% |

Choice compares targets within a context; context sensitivity compares the same target across its matched close/far contexts. A fixed target-only PMI baseline cancels from context sensitivity. Exact ties count as incorrect.
Feet directly locates the object relative to the person. Owned shoes only imply proximity under an unstated worn-shoes assumption; that condition is excluded from the primary score. Explicitly worn shoes states the anchor, but also changes sentence structure and length.
Context fronting preserves words and word count, while the person-subject form also changes grammar/pronouns. Near arrangements and far settings vary together across scenes. Object eligibility differs by scene, and variants are repeated measurements rather than independent scenarios.
Do not count shared lexical controls repeatedly through match links. Identical far-context scores in footwear comparisons were checked. Input snapshots, per-token scores, grouped summaries, and matched corrections/regressions are saved with this report.

## Interpretation and validation

Main-scene raw choice accuracy is 767/1,344 (57.07%). The model selects close on 83.56% of judgments: close-context accuracy is 90.63%, but far-context accuracy is 23.51%. PMI yields 819/1,344 (60.94%), with close/far accuracies of 72.47% and 49.40%. Context sensitivity is 715/1,344 (53.20%); both fixed targets prefer their matching contexts in only 9.82% of pairs.

The expanded scenes reproduce the strong interaction with target reference wording. Person-first targets achieve 61.61–76.79% choice accuracy across the three context structures, while object-first targets achieve 42.41–47.77%. The direct-label controls still score 95–100%, indicating that the drop is not universal across these target words and entities.

In matched footwear comparisons, close-context accuracy is 82/96 (85.42%) for owned shoes, 85/96 (88.54%) for feet, and 93/96 (96.88%) for explicitly worn shoes. Thus explicitly stating wearing improves eleven close judgments relative to owned shoes, while replacing shoes with feet improves three. Far-context accuracy remains 26/96 (27.08%) in each case because those contexts and targets are identical. This clarifies a real ambiguity in the initial template, but the result does not resolve the overall far-side failure; explicit wearing also changes syntax and length.

All raw, PMI, and context-sensitivity decisions were independently reconstructed from the saved per-token values. The seven-scene primary subset excludes both footwear controls and the shared lexical controls. No exact ties occurred in the primary results. GPU 1 was released after evaluation. See validation.json.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
