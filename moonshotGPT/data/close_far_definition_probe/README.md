# Close/far literal-definition probe, version 1.0

An additive [version 1.1](v1_1/README.md) provides 18 matched explicit naming probes with 9-, 11-, and 12-word contexts. Its combined dataset has 90 probes and preserves these original 72 records. The added probes have not yet been evaluated.

This probe measures whether a language model associates close/far with descriptions of relatively small/large physical distance. It contains 72 paired probes: 54 word-to-definition and 18 definition-to-word. It does not test symmetry, movement, orientation independence, comparative versus categorical reasoning, or robust use in situations. Dataset generation does not evaluate a model. A separate Qwen3 step-19,500 evaluation is saved in `runs/research/bos_aligned_proto/close_far_definition_probe/qwen3_359m_step19500_v1/` relative to the repository root.

## Files and regeneration

- `components.json`: reviewed wording options and stable component IDs; the source of truth.
- `generate.py`: deterministic standard-library generator and design validation.
- `evaluate.py`: local-checkpoint GPU evaluation using the existing EWoK likelihood routine; retains component labels and exports per-item, grouped, and consistency scores. Mean-token log likelihood is the default and primary scoring method. Summed-token scores are included only with `--include-sum`, using the same model predictions. Requires the repository's torch/transformers/pandas environment.
- `generated/probes.csv`: all 72 probes and diagnostic labels in a spreadsheet-friendly table.
- `generated/probes.jsonl`: the same records with EWoK-style text and category fields.
- `generated/context_stems.csv`, `definition_structures.csv`, `modifiers.csv`, `adjective_pairs.csv`, `directions.csv`, `label_targets.csv`: component lookup tables.
- `generated/direction_matches.csv`: 18 groups joining each reverse-direction probe to its three forward-direction counterparts.
- `generated/manifest.json`: counts and generation validation status.

Regenerate from any working directory with:

```sh
python /home/jorge/tokenPred/moonshotGPT/data/close_far_definition_probe/generate.py
```

The generator rewrites only its generated outputs. To expand the design, update the component tables and the explicit expected-count checks in the generator together.

## Controlled components

| Factor | Values |
|---|---|
| Direction | `word_to_definition`, `definition_to_word` |
| Context stem | `ctx_0`: Close/Far means that; `ctx_1`: In physical space ...; `ctx_2`: When two objects are described ... |
| Definition structure | `def_0`: the distance is ...; `def_1`: the distance between the objects is ...; `def_2`: the objects are separated by a ... distance. |
| Modifier | `mod_0`: relatively; `mod_1`: comparatively; `mod_2`: fairly |
| Adjective pair | `adj_0`: small/large; `adj_1`: short/long |
| Reverse-direction targets | `label_0`: The objects are close. / The objects are far. |

All three structures cross with all three modifiers and both adjective pairs, producing 18 definition variants. Each variant occurs after three concept stems (54 forward probes) and as a standalone context before fixed label targets (18 reverse probes). Reverse contexts capitalize the first letter of the definition. The `context_stem_id` is null in reverse probes, where the definition itself is the context.

Within every pair, C1 describes close and C2 describes far; T1 is the close-compatible target and T2 the far-compatible target. Word-to-definition contexts have no final punctuation because their targets complete the sentence. Reverse contexts and targets are full sentences. Join a context and target with one space, without adding an answer-choice instruction.

Example forward probe:

- C1: Close means that
- C2: Far means that
- T1: the distance is relatively small.
- T2: the distance is relatively large.

Matched reverse probe:

- C1: The distance is relatively small.
- C2: The distance is relatively large.
- T1: The objects are close.
- T2: The objects are far.

## Scoring and diagnosis

Compute target-only conditional log likelihoods for all four combinations. Let Sij be the mean log likelihood across target tokens for Tj after Ci. This produces 288 sequence scores and 144 individual completion-choice judgments. Mean-token scoring is the default; optional summed-token results must be labeled separately. Whitespace word counts are not tokenizer counts.

- Completion-choice close correctness: S11 > S12.
- Completion-choice far correctness: S22 > S21.
- Individual accuracy: average of those correctness indicators.
- Paired success: both inequalities hold, a stricter criterion than individual accuracy.
- Optional EWoK context sensitivity: S11 > S21 for the close definition/label and S22 > S12 for the far definition/label. Report this separately from completion choice.
- Report ties separately; do not silently count them as successful pairs. Record score margins as well as correctness.

Save results by `probe_id` and join to component labels. Report direction-specific accuracy, close/far accuracy, paired success, and matched component contrasts. Avoid letting the 54:18 direction count imbalance determine the headline result; if one combined accuracy is needed, average the two direction accuracies equally.

For the adjective comparison, match direction, context stem (when applicable), definition structure, and modifier, changing only `adj_0` versus `adj_1`. For a modifier comparison, hold the other factors fixed. Inspect interactions, such as whether short/long errors occur only with fairly. The same approach applies to structures and stems.

Use `definition_variant_id` and `direction_matches.csv` to compare both mapping directions. Each reverse probe matches three forward stems; do not treat those repeated uses of the same reverse result as independent observations. Report how many variants pass in all three forward stems, in the reverse direction, and across all four probes. Agreement alone is not success: consistently wrong predictions must remain distinguishable from consistently correct ones.

## Interpretation and limits

- The dataset contains two mapping directions and repeated wording variations, not 72 independent conceptual skills. Do not interpret row-level counts as independent statistical replications.
- The lexical alternatives are controlled but not perfectly synonymous: fairly can express a different degree from relatively/comparatively. A performance difference diagnoses wording sensitivity and needs follow-up before concluding the model lacks a concept.
- Structure changes also change length, syntax, and wording. `context_structure_label` and per-text character/word counts describe them; they do not isolate a causal length effect. Forward context `ctx_2` uses the natural close-to/far-from distinction. Definitions do not supply a numeric threshold or comparison class.
- All rows place close in slot 1. This is harmless for separate conditional likelihood scoring, which presents no slot labels to the model. If converting to displayed multiple-choice questions, counterbalance option order and track it; do not count order reversals as new semantic probes.
- The little/a-lot-of wording family is excluded from version 1.0 to keep the modifier/adjective cross product grammatical and balanced.
- Definition recognition alone does not establish physical reasoning or robust situational understanding. The short reverse descriptions also offer limited information about the distance scale.

## Existing evaluator integration

The JSONL preserves the standard `Domain`, `ConceptA`, `ConceptB`, `ContextType`, `ContextDiff`, `TargetDiff`, `Context1`, `Context2`, `Target1`, and `Target2` fields. The additional IDs support diagnosis. The existing `evaluation/ewok.py` record builder does not automatically copy these custom IDs, so an evaluation wrapper must preserve `probe_id` or join records to the exact input table. `probe_row_index` is the zero-based index within this file, not a guaranteed row index after combining it with other datasets. Keep the generated file as a separate probe dataset rather than mixing its results with existing EWoK benchmark scores.
