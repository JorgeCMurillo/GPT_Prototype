# Revised spatial evaluation: Qwen3 359M, step 19,500

Primary scoring: raw mean full-answer conditional token log likelihood, no PMI,
ties incorrect. Fresh inference uses FP32 weights and eager attention, matching
the prior custom spatial evaluators. Candidate answers alone are scored; any
answer bridge stays in the conditioning context.

## General situation score

The comparable six-category score is **51.26%**, with a conditional
scene-cluster bootstrap 95% interval of **50.68–51.85%**. The prior score was
51.22%. This is a descriptive score for the covered scene families, not a
general spatial-reasoning confidence interval. Repeated wording variants are
not independent samples, and singleton structural strata stay fixed in the
bootstrap. Both contexts are correct in only **4.24%** of pairs under the same
hierarchical weighting.

| Relation | Balanced accuracy | Both contexts correct | Source |
|---|---:|---:|---|
| Above/below | 50.23% | 2.64% | Reused unchanged v1.3; current input hash verified |
| Left/right | 50.40% | 3.94% | Fresh shortened v1.2 |
| North/south | 50.00% | 0.00% | Fresh natural-shortened v1.4 |
| East/west | 49.83% | 0.83% | Fresh natural-shortened v1.4 |
| Close/far | 58.14% | 17.84% | Reused matched-scene applied subset |
| Closer/farther | 48.97% | 0.19% | Reused three-family applied movement macro |
| Front/behind (separate) | 50.00% | 0.00% | Fresh compact v2.0, 212 event pairs |

Front/behind is reported separately to preserve the original six-category
definition. Its score balances families, then numeric/nonnumeric evidence,
then cases within each format. Its control score is 75%; observer turns 62.5%.
The front/behind summed-token counterfactual scores 44.74% on events. All scores
in the table use mean likelihood.

General scores and detailed provenance: [report.md](report.md), [summary.json](summary.json).
Versioned input manifest: `data/spatial_benchmark/qwen3_step19500_shortened_v2.json`.

## Minimal relation-first scenes

The new block has 46 paired rows (92 judgments). Main movement results exclude
the explicit stationary-cue subset, static controls, and observer-turn controls.
There are **15 main movement pairs per axis**, covering six cases in four
event families. Average wording variants within a case, cases within a family,
and the four families equally. Consequently these percentages are not pooled
fractions over the 30 judgments per axis.

| Main minimal movement | North/south | East/west |
|---|---:|---:|
| Target crosses reference | 0.00% | 12.50% |
| Reference crosses target | 0.00% | 0.00% |
| Target moves without crossing (toward/away) | 75.00% | 75.00% |
| Both move (equal movement/swap) | 75.00% | 50.00% |
| Balanced main movement accuracy | **37.50%** | **34.38%** |
| Both contexts correct, same weights | 25.00% | 12.50% |

The equal-axis main movement score is **35.94%**. Raw main-movement pooled
accuracy is 36.67% for north/south and 33.33% for east/west. Pooling every row,
including all controls and stationary variants, gives 43.48% and 36.96%; those
are not the primary movement measure.

Across main-movement renderings the model selects the stated initial relation
in 27/30 north/south judgments and 24/30 east/west judgments. This is consistent
with failing to update the starting relation, especially after crossing; it
does not establish an internal mechanism such as copying. Removing the map
setup has not produced successful crossing inference. This is a different
evidence format, not a controlled test of context length alone.

Mean and sum make the same decisions in this minimal block.

### Matched wording and persistence checks

| Change | Matched pairs | Changed judgments | Raw accuracy change over matched judgments |
|---|---:|---:|---:|
| moves → heads (move → head for Both) | 14 | 1/28 | -3.57 pp |
| toward → to (completed crossings only) | 8 | 1/16 | -6.25 pp |
| Add stationary clause | 12 | 3/24 | -4.17 pp |

These tiny, overlapping comparisons include the configured controls for the
stationary-cue check; they do not support broad wording-effect conclusions.
Adding stationary wording did not improve the crossing cases. In the main
short forms, unmentioned locations are assumed to persist. The explicit-only
movement subset covers fewer families, so its group score must not be directly
compared with the full main-movement score as a causal cue effect.

No binomial confidence intervals are assigned to the minimal block: the
paraphrases reuse a few hand-built structures. For example, each crossing
family has one matched structural case per axis.

## Saved new runs

- `runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_v1_2/`
- `runs/research/bos_aligned_proto/cardinal_situation_probe/qwen3_359m_step19500_v1_4/`
- `runs/research/bos_aligned_proto/cardinal_situation_probe/qwen3_359m_step19500_minimal_v1/`
- `runs/research/bos_aligned_proto/front_behind_situation_probe/qwen3_359m_step19500_v2_0/`

Runs retain exact input snapshots, token-level scores, item decisions and
reports. New minimal/front-behind results were validated by reconstructing
both reductions and explicit-gold decisions from saved token arrays; cardinal
v1.4 reconstructs saved means and gold choices. Historical runs were not
overwritten. This directory publishes compact reports; full raw score files
and checkpoint weights remain local.
