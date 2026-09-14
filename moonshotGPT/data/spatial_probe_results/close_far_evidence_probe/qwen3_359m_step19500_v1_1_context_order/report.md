# Close/far: matched evidence test

Qwen3 359M at 19.5k steps. Primary scoring is mean full-target token log likelihood, including punctuation.
Each evidence × target-order condition has 20 matched pairs / 40 balanced context judgments. Names and objects are crossed; identical target sentences are used across all context conditions.

| Evidence | Target order | Choice | PMI choice | Context sensitivity | Both correct: choice | Both correct: context | Close-choice frequency |
|---|---|---:|---:|---:|---:|---:|---:|
| direct_label | person_first | 95.00% | 77.50% | 100.00% | 90.00% | 100.00% | 45.00% |
| direct_label | object_first | 100.00% | 92.50% | 100.00% | 100.00% | 100.00% | 50.00% |
| proximity_paraphrase | person_first | 87.50% | 55.00% | 82.50% | 75.00% | 65.00% | 37.50% |
| proximity_paraphrase | object_first | 92.50% | 77.50% | 65.00% | 85.00% | 30.00% | 42.50% |
| physical_placement | person_first | 67.50% | 62.50% | 52.50% | 40.00% | 5.00% | 72.50% |
| physical_placement | object_first | 45.00% | 42.50% | 50.00% | 0.00% | 0.00% | 95.00% |
| physical_placement_fronted | person_first | 72.50% | 55.00% | 50.00% | 50.00% | 0.00% | 67.50% |
| physical_placement_fronted | object_first | 32.50% | 30.00% | 50.00% | 0.00% | 0.00% | 77.50% |
| physical_placement_person_subject | person_first | 70.00% | 70.00% | 62.50% | 40.00% | 25.00% | 80.00% |
| physical_placement_person_subject | object_first | 37.50% | 42.50% | 42.50% | 0.00% | 0.00% | 87.50% |

The original contexts mention the object first. Person-first targets reverse that order; object-first direct-label targets repeat the context verbatim. New fronted-location and person-subject placement contexts mention the person first and are scored under both target orders. Fronted-location contexts preserve the original words and word counts up to capitalization, while changing syntax and token order; person-subject contexts also change grammatical roles and pronouns.
Physical placement uses between the person's shoes versus across the full length of a long gym. It assumes the shoes are worn. Evidence conditions vary in length, wording, and specificity, so this comparison cannot attribute changes solely to inference difficulty.
Both accuracy metrics have a 50% random-choice baseline per judgment. Target-only PMI subtraction cancels in fixed-target context sensitivity.
The same contexts recur across the two target orders; entity and template variants are not independent scenarios. This is a controlled diagnostic, not a replacement for the full EWoK benchmark.

## Context-order comparison

With person-first targets fixed (Maya is close to/far from the ball), changing the original placement context to a fronted location raises raw accuracy from 67.5% to 72.5% (two additional correct judgments out of 40). The person-subject variant scores 70% (one additional correct judgment). Context sensitivity is 52.5%, 50%, and 62.5%, respectively. PMI choice is 62.5%, 55%, and 70%.

With object-first targets fixed (The ball is close to/far from Maya), raw accuracy instead drops from 45% to 32.5% for the fronted location and 37.5% for the person-subject variant. Context sensitivity is 50%, 50%, and 42.5%, respectively.

Thus reordering is not a general fix: the direction of its effect depends on target wording. The fronted condition holds the word multiset and word count fixed, but changes syntax and token positions. The person-subject condition also changes grammatical roles and introduces a pronoun. Each cell contains only 20 entity combinations and one template pair, so the small accuracy differences should be treated as descriptive results for these templates.

All 800 conditional sequences were checked and all three metrics independently reconstructed from saved per-token values. The original 120 paired items retain identical text and correctness decisions. GPU 1 was released after evaluation.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
