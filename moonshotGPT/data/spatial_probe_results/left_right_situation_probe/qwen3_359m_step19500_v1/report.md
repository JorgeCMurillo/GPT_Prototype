# Fixed-frame left/right evaluation

Raw mean full-target token conditional log likelihood, including punctuation. No PMI. Exact ties count as incorrect.
Left/right gold refer to the designated target object's relation to the reference, regardless of answer entity order.

| Family | Accuracy | Both contexts correct | Left word chosen |
|---|---:|---:|---:|
| static_placement | 48.96% | 2.95% | 7.47% |
| target_crosses | 50.39% | 1.04% | 1.95% |
| reference_crosses | 49.61% | 0.00% | 3.26% |
| target_moves_without_crossing | 50.39% | 2.08% | 2.21% |
| both_move | 50.78% | 2.60% | 3.21% |
| direct_label_control | 52.08% | 50.00% | 47.92% |

Equal-family applied accuracy: 50.03%.

Direct-label controls are separate. Applied results cover 12 physical cases with repeated wording, entity, and labeling variants; 2,304 rows are not independent scenes.

See grouped_scores.csv for evidence form, context/answer order, length, and object breakdowns. item_scores.jsonl preserves all scene annotations and four likelihoods per pair. token_scores.jsonl preserves individual target-token scores.

variant_consistency.csv compares predictions expressed as the designated target object's relation, so equivalent answers remain comparable when answer entity order reverses.
The both-correct pair rate has a 25% baseline for independent uniform binary guesses; a constant answer achieves zero.
Context sensitivity is secondary and compares a fixed target across contexts; it is distinct from choosing the correct answer within each context.
