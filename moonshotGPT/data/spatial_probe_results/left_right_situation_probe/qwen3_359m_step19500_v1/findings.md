# Qwen3 359M step 19,500: fixed-frame left/right

Raw mean full-target conditional log likelihood; no PMI. All 2,400 pairs
(9,600 context/target sequences) completed. Independent reconstruction of
token means, choices, lexical preferences, family averages, and matched
prediction flips passed; see validation.json.

The five-family applied mean is **50.03%**. Pooling the 2,304 applied pairs
instead gives **50.00%**, with **1.91%** of pairs correct in both contexts.
The model selects the answer containing **right in 96.09%** of applied judgments.
These results do not show reliable discrimination of opposing situations in
this prompt/answer format.

Direct-label controls are highly dependent on entity order:

| Context and answer entity orders | Pairs | Accuracy | Both contexts correct |
|---|---:|---:|---:|
| Same order | 48 | 4.17% | 0.00% |
| Reversed order | 48 | 100.00% | 100.00% |

For example, a same-order control follows “The ball icon is to the left of
the cone icon” with the same subject/reference order in the candidate answers.
The reversed-order control uses cone-first answers, where “right” is correct.
The aggregate control accuracy of 52.08% hides this large difference.
This is an observed scoring pattern, not evidence identifying its mechanism.
In particular, these controls prevent interpreting the applied result solely
as a failure to track motion or compare final coordinates.

No training or PMI adjustment was performed. GPU 1 was released after scoring.
The scene inventory contains 12 physical cases; thousands of variants are
matched measurements, not independent scenes.
