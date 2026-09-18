# Expanded Qwen3 block evaluation

176 matched pairs, 352 binary judgments, 704 scored context/answer sequences.
Same Qwen3 359M step-19,500 checkpoint; raw mean full-target likelihood, no PMI.
Saved token means and gold choices reconstructed successfully. All 176 judgments
on the original 88 pairs reproduced their previous predictions unchanged.
Differences in aggregate scores therefore reflect the added examples, not a
change in model weights or original predictions.

Direction definitions now score 71.875% on each axis. New alternate-edge and
preposition examples score 93.75% north/south and 81.25% east/west. Within this
addition, toward versus to scores are 100% versus 87.5% north/south and 62.5%
versus 100% east/west, each based on only four pairs/eight judgments per word
and axis. The preposition effect does not have the same direction across axes.

Opposite-direction results fall from 100% in the original items to 87.5%
north/south and 93.75% east/west overall; route-reversal additions score 75%
and 87.5%, respectively. Relative-position translations remain weak.

Observer-turn cardinal east/west remains 100%; north/south is 62.5% overall
(75% original structure, 50% fixed-flag structure). Observer-relative choices
remain 50% on each axis. Both-context correctness is zero east/west and 4.17%
north/south, so the aggregate 50% still does not indicate reliable updating.

See confidence_intervals.md and .csv for counts and 95% Wilson reference
intervals. These assume independent judgments, which the paired and repeated
templates do not satisfy. They are not calibrated confidence intervals for
generalization to new scenes. Larger wording coverage is not equivalent to
more independent physical geometries.
