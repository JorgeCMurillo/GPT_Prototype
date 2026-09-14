# Qwen3 359.4M, step 19,500: close/far definition probe

Completed on GPU 1 (RTX 3090), 2026-09-13. The process exited successfully and GPU 1 returned to 5 MiB used. The checkpoint was loaded locally with float32 weights and eager attention; scoring reused the repository's EWoK target-only conditional likelihood function, including its BOS handling. No training was performed.

Primary metric: mean-token completion-choice accuracy. Both target alternatives have equal token counts within every context in this dataset. Summed-token scoring produced identical correctness decisions. All 72 probes and 288 context-target combinations were scored; there were no exact or near ties (absolute completion-choice margin below 1e-6).

## Main results

| Direction | Correct judgments | Accuracy | Both sides correct |
|---|---:|---:|---:|
| Word to definition | 91/108 | 84.26% | 37/54 (68.52%) |
| Definition to word | 18/36 | 50.00% | 0/18 (0.00%) |
| All rows pooled | 109/144 | 75.69% | 37/72 (51.39%) |

The equally weighted average of the two direction accuracies is **67.13%**. This prevents the three-times-larger forward set from dominating the score. No definition variant passed all three forward context stems and its reverse counterpart, because every reverse probe failed the far judgment.

## Forward-direction component breakdown

| Factor | Value | Completion-choice accuracy |
|---|---|---:|
| Context | ctx_0: Close/Far means that | 100.00% |
| Context | ctx_1: In physical space ... | 88.89% |
| Context | ctx_2: When two objects are described ... | 63.89% |
| Adjectives | adj_0: small/large | 87.04% |
| Adjectives | adj_1: short/long | 81.48% |
| Definition | def_0: the distance is ... | 75.00% |
| Definition | def_1: the distance between the objects is ... | 83.33% |
| Definition | def_2: the objects are separated by a ... distance. | 94.44% |
| Modifier | mod_0: relatively | 83.33% |
| Modifier | mod_1: comparatively | 86.11% |
| Modifier | mod_2: fairly | 83.33% |

These are controlled, balanced comparisons within the forward direction. The structure and context factors jointly change wording and length, so they do not identify an effect of length alone. The small/large advantage is a descriptive 5.56 percentage points over short/long in these templates, not proof of a general conceptual deficit.

## Reverse-direction preference

For every reverse probe, the model preferred "The objects are close." over "The objects are far." under both contexts. Close judgments were 18/18 correct and far judgments 0/18 correct. However, EWoK-style context sensitivity was 83.33%: holding each target fixed, the matching context was usually preferred. The strict context-sensitivity paired-success rate was 12/18 (66.67%). These are different metrics and should not be conflated.

Example: def_0, relatively, small/large (mean target log likelihood; higher is better):

| Context | The objects are close. | The objects are far. |
|---|---:|---:|
| The distance is relatively small. | -3.45555 | -4.17870 |
| The distance is relatively large. | -3.53255 | -4.04261 |

Both continuations respond in the expected direction to the context change, but "close" remains the more likely continuation in absolute terms. This demonstrates a response preference under these prompts. Unconditional target likelihoods were not measured, so a specific source of that preference has not been established.

Forward context sensitivity was 82.41%, compared with 84.26% forward completion choice. Do not interpret the two as interchangeable evidence.

## Artifacts and interpretation

- `summary.json`: settings, package versions, input hashes, and aggregate results.
- `item_scores.csv` / `item_scores.jsonl`: all component labels, texts, token counts, margins, and decisions for mean and sum scoring; two records per probe, one for each reduction.
- `grouped_scores.csv`: direction, adjective, modifier, structure, context, and adjective-by-modifier breakdowns.
- `consistency.csv`: success across the three forward stems and reverse counterpart for each definition variant.
- `token_scores.jsonl`: target-token log likelihoods for all 288 combinations.

The literal mapping is reliable for the simplest forward prompts, but performance changes with phrasing and scoring direction. These probes alone do not establish robust physical-distance understanding. The existing EWoK close/far score (83.33% completion choice) comes from different examples and is not directly comparable as a difficulty-matched measurement.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
