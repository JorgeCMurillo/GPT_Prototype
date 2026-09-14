# Qwen3 closer/farther: nonnumeric and matched numeric results

Qwen3 359M, 19.5k checkpoint. All metrics use mean full-target token log likelihood, including punctuation.
Choice holds context fixed and compares targets. Context sensitivity holds target fixed and compares the paired contexts. PMI adjusts choice using target-only means; it leaves context sensitivity unchanged.

| Dataset | Form | Contexts | Choice | PMI choice | Context sensitivity | Both contexts correct (choice) | Both targets correct (context sensitivity) |
|---|---|---:|---:|---:|---:|---:|---:|
| non_numeric | all | 200 | 53.00% | 48.00% | 44.50% | 6.00% | 8.00% |
| numeric_aligned | all | 4320 | 48.54% | 45.35% | 32.50% | 0.05% | 0.05% |
| non_numeric | compact | 40 | 50.00% | 50.00% | 42.50% | 0.00% | 5.00% |
| non_numeric | standard | 40 | 50.00% | 47.50% | 12.50% | 0.00% | 0.00% |
| non_numeric | expanded | 40 | 50.00% | 47.50% | 60.00% | 0.00% | 20.00% |
| non_numeric | minimal | 40 | 50.00% | 45.00% | 50.00% | 0.00% | 0.00% |
| non_numeric | minimal_stationary | 40 | 65.00% | 50.00% | 57.50% | 30.00% | 15.00% |
| numeric_aligned | compact | 1440 | 45.76% | 48.26% | 37.57% | 0.14% | 0.14% |
| numeric_aligned | standard | 1440 | 49.86% | 41.60% | 28.12% | 0.00% | 0.00% |
| numeric_aligned | expanded | 1440 | 50.00% | 46.18% | 31.81% | 0.00% | 0.00% |

Each form is balanced between closer and farther. Both accuracy metrics have a 50% random-choice baseline per individual judgment.
Compare the numeric aligned forms only with their matching compact/standard/expanded nonnumeric forms, not with the full five-form nonnumeric average.
Each of the 120 aligned nonnumeric contexts has 36 numeric counterparts. Repeated contexts and entity/template variants are not independent observations.
Minimal forms use ordinary movement assumptions; the unqualified form does not state that the object is stationary. Minimal forms leave endpoints unstated. Aligned forms make the trajectory constraints explicit.
Input snapshots, individual scores, per-token scores, priors, matches, and diagnostic groupings are saved alongside this report.

## Diagnostic interpretation

Across all five nonnumeric forms, choice accuracy is 53% (106/200), PMI choice is 48% (96/200), and context sensitivity is 44.5% (89/200 target judgments).

The plain minimal sentence and all three longer nonnumeric forms always select the closer target. Their 50% raw choice accuracy therefore reflects a constant choice. For the plain minimal form, both fixed targets always prefer the toward context, explaining its 50% context-sensitivity score and zero both-target successes.

The minimal stationary form scores 65% choice accuracy (26/40): 19/20 closer contexts and 7/20 farther contexts are correct, with both contexts correct in 6/20 pairs. It still favors closer on 32/40 contexts. Context sensitivity is 57.5% (23/40), with both targets correct in 3/20 pairs. The stationary word improves this sample but does not establish robust performance across new templates.

For the comparable aligned forms only (excluding the minimal additions), nonnumeric choice/PMI/context-sensitivity scores are 50.00% / 48.33% / 38.33%. Numeric counterparts score 48.54% / 45.35% / 32.50%. Removing specific distance expressions does not resolve the closer/farther discrimination problem in this set.

Independent CPU reconstruction from all 9,040 saved per-token score sequences confirmed every raw choice and every context-sensitivity decision. Target-only PMI cancellation was checked for all context margins. GPU 1 was released after evaluation.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
