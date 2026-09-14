# Close/far: matched evidence test

Qwen3 359M at 19.5k steps. Primary scoring is mean full-target token log likelihood, including punctuation.
Each evidence × target-order condition has 20 matched pairs / 40 balanced context judgments. Names and objects are crossed; identical target sentences are used across the three evidence conditions.

| Evidence | Target order | Choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context | Close-choice frequency |
|---|---|---:|---:|---:|---:|---:|---:|
| direct_label | person_first | 95.00% | 77.50% | 100.00% | 90.00% | 100.00% | 45.00% |
| direct_label | object_first | 100.00% | 92.50% | 100.00% | 100.00% | 100.00% | 50.00% |
| proximity_paraphrase | person_first | 87.50% | 55.00% | 82.50% | 75.00% | 65.00% | 37.50% |
| proximity_paraphrase | object_first | 92.50% | 77.50% | 65.00% | 85.00% | 30.00% | 42.50% |
| physical_placement | person_first | 67.50% | 62.50% | 52.50% | 40.00% | 5.00% | 72.50% |
| physical_placement | object_first | 45.00% | 42.50% | 50.00% | 0.00% | 0.00% | 95.00% |

Person-first targets reverse the object-first context relation, as in the EWoK symmetry design. Object-first targets preserve order; for direct labels the correct target repeats the context verbatim, providing an explicit repetition control.
Physical placement uses between the person's shoes versus across the full length of a long gym. It assumes the shoes are worn. Evidence conditions vary in length, wording, and specificity, so this comparison cannot attribute changes solely to inference difficulty.
Both accuracy metrics have a 50% random-choice baseline per judgment. Target-only PMI subtraction cancels in fixed-target context sensitivity.
The same contexts recur across the two target orders; entity and template variants are not independent scenarios. This is a controlled diagnostic, not a replacement for the full EWoK benchmark.

## Findings

With person-first targets fixed across evidence conditions, raw choice accuracy falls from 95% for direct labels to 87.5% for proximity paraphrases and 67.5% for physical placement. Context sensitivity falls from 100% to 82.5% to 52.5%. Thus a substantial evidence-wording effect remains with entities and target sentences held fixed.

Target reference order also matters. The identical physical-placement contexts score 67.5% with person-first targets but 45% with object-first targets. Their close-choice frequency increases from 72.5% to 95%. Direct evidence does not show that preference: close-choice frequencies are 45% and 50% for the two target orders.

In the person-first placement condition, the close target prefers its matching context in all 20 pairs, but the far target does so in only 1/20. Both targets favor their respective correct contexts in only 5% of pairs. The aggregate context-sensitivity score therefore should not be read as symmetric success.

This supports a joint effect of evidence phrasing and target reference wording. It does not isolate whether the placement difficulty comes from body-reference inference, the across-the-gym construction, sentence length, or other wording details. Each evidence level uses a single template pair instantiated with 20 entity combinations.

All raw choice, PMI choice, and context-sensitivity decisions were independently reconstructed from saved per-token values; results agree. See validation.json. GPU 1 was released after evaluation.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
