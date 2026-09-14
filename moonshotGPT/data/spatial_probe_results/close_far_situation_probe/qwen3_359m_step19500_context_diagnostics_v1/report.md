# Qwen3 step 19,500: context expansion and body-reference diagnostics

Evaluated 2026-09-13 on GPU 1 (RTX 3090), using the same local Qwen3 359.4M checkpoint, float32 weights, eager attention, and the existing EWoK target-only likelihood routine. All reported completion-choice scores use **mean target-token log likelihood**. The process completed and GPU 1 returned to 5 MiB used.

## Context expansion: same 52 placement assignments

| Context form | Accuracy | Correct judgments | Close accuracy | Far accuracy | Both sides correct |
|---|---:|---:|---:|---:|---:|
| Compact | 61.54% | 64/104 | 96.15% | 26.92% | 13/52 (25.00%) |
| Setting | 50.96% | 53/104 | 80.77% | 21.15% | 6/52 (11.54%) |
| Endpoint detail | 45.19% | 47/104 | 63.46% | 26.92% | 4/52 (7.69%) |

The setting prefix lowers accuracy by 10.58 percentage points relative to compact prompts, and the endpoint prefix lowers it by 16.35 points. Location sentences and targets are identical within these comparisons. The major decline is in close judgments; compact prompts still perform poorly on far judgments.

Person scenes have eight entity assignments and object-object scenes four. Giving all eight scenarios equal weight yields 65.63%, 54.69%, and 48.44% for compact, setting, and endpoint forms, respectively. The qualitative trend remains.

## Body/worn-item versus direct-person reference

This comparison is restricted to the same 24 assignments in the gym, playing field, and platform scenes. It compares "immediately beside Maya's shoes/feet/ankle" with "immediately beside Maya." Far contexts and both targets are identical between reference forms.

| Context form | Body/worn-item pair accuracy | Direct-person pair accuracy | Body/worn-item close accuracy | Direct-person close accuracy |
|---|---:|---:|---:|---:|
| Compact | 58.33% | 62.50% | 91.67% | 100.00% |
| Setting | 41.67% | 58.33% | 66.67% | 100.00% |
| Endpoint detail | 35.42% | 47.92% | 41.67% | 66.67% |

Each pair-accuracy entry averages 48 context-level judgments. Each close-accuracy entry uses 24 judgments. Far accuracy is identical across the two reference forms by construction: 25.00%, 16.67%, and 29.17% for compact, setting, and endpoint prompts. Saved score equality for these identical far combinations was checked explicitly.

Removing the body/worn-item reference improves performance, especially with setting prefixes. Endpoint detail also hurts direct-person prompts, so the decline cannot be attributed solely to body-reference inference.

## Overall and interpretation

Across all 228 probes: **245/456 judgments correct (53.73%)**, with **35/228 pairs fully correct (15.35%)**. Separate context-sensitivity accuracy is 49.78%. There are no exact or near completion-choice ties below an absolute margin of 1e-6. These pooled figures mix conditions; the matched tables above are the main diagnostic results.

The findings support sensitivity to added context and to body/worn-item reference constructions. They do not isolate sentence length alone: the prefixes add different spatial information, and the body phrase itself adds words. A matched-length nonspatial prefix control has not been tested. The weak far results persist even without a prefix, so reducing context length does not fully resolve the failure.

The original placement wording was revised for self-contained compact descriptions and controlled immediately-beside comparisons. Therefore, compare context forms within this batch rather than interpreting the difference from the original batch's 41.35% baseline as a pure effect of shortening.

## Artifacts and validation

- `summary.json`: input hashes, model path, settings, aggregate and matched-subset results.
- `item_scores.csv` / `.jsonl`: 228 probes with component labels, mean scores, token counts, margins, and correctness.
- `grouped_scores.csv`: condition, entity type, scenario, name, object, and body-comparison eligibility breakdowns.
- `diagnostic_matches_scores.csv`: 224 controlled comparisons with accuracy/margin changes.
- `consistency.csv`: success across all applicable conditions for each entity assignment.
- `token_scores.jsonl`: 760 unique context-target sequences, reused across the 912 required combinations.

Validated row/ID count, mean-only results, stable context tokenization boundaries, finite token scores, aggregate counts from saved decisions, and equality of far scores in all body-reference matches. No training was performed.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
