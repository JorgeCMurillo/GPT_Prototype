# Expanded cardinal scenes: Qwen3 359M step 19,500

All 1,344 pairs and 5,376 context/answer sequences completed with raw mean
target-token likelihood, no PMI. The summary bridge is part of the context.
Saved token means and choices were reconstructed successfully. GPU 1 was released.

Balanced event accuracy is 50.00% north/south and 49.93% east/west. Both-context
correctness is zero on both axes. The actual answer word chosen is always
north for north/south events and almost always east for east/west events
(99.65% under the family/format balancing used for the headline).

Numeric event accuracy is 50% on both axes. Named-location pooled accuracy is
50% north/south and 49.77% east/west. The small difference from the balanced
east/west headline reflects different weights assigned to families and formats.

Reversing list presentation changes 5 of 1,296 matched judgments. Reversing
numeric labels changes 0 of 864 matched judgments. These links are repeated,
correlated comparisons. Stable predictions here are consistent with the strong
answer preference and do not establish successful invariance to relabeling.

Direct-relation controls score 54.17% north/south and 50% east/west, with paired
correctness 41.67% and 33.33%. Screen-to-cardinal controls score 50% on each axis
and zero paired correctness. This limits attributing event failures solely to
movement tracking: the prompt/answer format also needs consideration.

The separately generated definition and observer-turn blocks were not evaluated
in this run. See report.md for all five families, grouped_scores.csv for factor
slices, and matched_changes.csv for individual variant comparisons.
