# Close/far situation probe: first diagnostic batch

An additive [context/body-reference diagnostic batch](context_diagnostics/README.md) contains 228 new probes with compact, setting-only, and endpoint-detail contexts, plus matched body/worn-item versus direct-person references. The original batch and its results remain unchanged.

This dataset tests categorical close/far judgments in described physical situations. It contains **468 paired probes**, each with C1/C2 and T1/T2, across **52 entity assignments** and **eight scenario templates**. It does not test closer/farther, arithmetic, movement, changing standards, or negated premises. Mean target-token log likelihood is the default scoring convention. Qwen3 step-19,500 results are saved under `runs/research/bos_aligned_proto/close_far_situation_probe/qwen3_359m_step19500_v1/` relative to the repository root.

## Scenarios and substitutions

| Scenario | Reference entity | Target objects | Close placement | Far placement |
|---|---|---|---|---|
| Gym | person at one end | ball, bag | between shoes | other end / across full gym length |
| Dining room | person seated at one end | book, bag | beneath or immediately beside chair | across full room length |
| Playing field | person at one end | ball, cone | between feet | other end / entire field between entities |
| Train platform | person at one end | suitcase, backpack | against or immediately next to ankle | other end / full platform length between entities |
| Auditorium | person in front row | bag, coat | adjacent seat | last row / across auditorium depth |
| Gym, objects | bench or mat | ball, cone | adjacent edge / narrow gap | across full gym length |
| Kitchen, objects | plate or bowl on table | cup, mug | immediately adjacent on table | counter across full kitchen length |
| Workshop, objects | toolbox or crate | helmet, bucket | immediately adjacent / narrow gap | full workshop length between objects |

The four names are Maya, Jesse, Li, and Omar. Each person scenario crosses all four names with its two eligible target objects (8 assignments per scenario; 40 total). Each object-object scenario crosses two reference objects with two target objects (4 assignments per scenario; 12 total). Names do not imply gender, and templates avoid gendered pronouns.

Objects are balanced within each scenario, not across the entire dataset. Some objects are appropriate in several settings and others in only one. Compare substitutions within the same setting. The eligible-object lists prevent arbitrary swaps such as a bench between someone's shoes.

## Nine conditions per assignment

| Condition ID | Change from baseline | Contexts or targets held fixed |
|---|---|---|
| baseline | physical arrangement, target object first, positive targets | reference condition |
| phrase_alternate | alternate location phrasing | same targets and close/far categories |
| word_beside | beside / far from | same targets and intended categories |
| word_alongside | alongside / a long way from | same targets and intended categories |
| word_near | near / a long distance from | same targets and intended categories |
| reference_reversed | ask about the reference entity relative to the target object | identical contexts |
| negate_close | close / not close | identical contexts |
| negate_far | not far / far | identical contexts |
| both_negated | not far / not close | identical contexts |

There are **52 probes per condition**, 360 person-object probes and 108 object-object probes. The design deliberately changes one configured control at a time. It does not fully cross phrasing, reversal, and negation. Sentence order stays fixed: reference setup, then target location. Interactions and sentence-order variation can be added later.

Every paired probe uses a clearly close-supporting C1 and a far-supporting C2 under an ordinary physical-distance reading. T1 matches C1 and T2 matches C2. Some negated targets express non-farness or non-closeness rather than explicitly asserting close or far; they are supported because these contexts describe clear endpoints. **Not close is not defined as far, and not far is not defined as close.** Intermediate cases and insufficient-information responses are outside this first batch.

## Example: object-object scene

Baseline:

- C1: The plate rests on a table at one end of a large kitchen. The cup rests on the table immediately beside the plate.
- C2: The plate rests on a table at one end of a large kitchen. The cup rests on a counter at the other end of the kitchen from the plate.
- T1: The cup is close to the plate.
- T2: The cup is far from the plate.

Reference reversal keeps those contexts and uses "The plate is close to the cup." / "The plate is far from the cup." The negate-close condition keeps those contexts and uses "The cup is close to the plate." / "The cup is not close to the plate."

## Diagnostic labels and matching

`components.json` is the source of truth for names, objects, scenarios, phrase pairs, reference orders, target patterns, and conditions. Every record retains these IDs, plus the following:

- `matched_group_id`: the same scenario and entity assignment across nine conditions.
- `evidence_type`: physical_arrangement or proximity_word. Physical-arrangement descriptions may still contain words such as next to or beside, but locate the object relative to body parts, furniture, or a spatial extent; proximity-word conditions directly state the relation between the evaluated entities. This is a coarse source-of-evidence label, not a claim that vocabularies never overlap.
- `location_phrase_pair_id`: scenario plus phrase-pair ID, to identify exact template wording.
- `semantic_match`: category_matched for alternate descriptions. For example, between shoes and next to shoes both support close but do not specify identical positions.
- `reference_order_id`, `target_negation_pattern_id`, and `sentence_structure_id`.
- `name_id`, `object_id`, `anchor_object_id`, `scenario_id`, and `entity_type`.
- Per-text word and character counts. These are not model-token counts.

`control_matches.csv` links every variant to its own baseline (416 comparisons). `substitution_matches.csv` links name, target-object, and reference-object substitutions while holding the other configuration components fixed. The reference substitution is selected by sorted component ID; these links support comparisons, not additional independent examples.

## Scoring and reporting

Compute Sij as the mean log likelihood of target Tj conditioned on Ci, scoring only target tokens. Close-context completion-choice success is S11 > S12, and far-context success is S22 > S21. Report individual accuracy and the fraction of probes where both inequalities hold. Keep ties separate.

Report results separately by condition and by entity type. Use the match tables for paired accuracy differences, changes in margins, and correctness stability. Agreement alone is insufficient: consistently wrong variants are not successful. Evidence-wording results should stay separate from physical-arrangement results because explicit synonyms give a more direct cue.

Keep mean-token scoring as the primary result, particularly because adding "not" changes target length. Log actual target-token counts and margins. Context-sensitive likelihood comparisons can supplement completion choice but must be separately named. Do not count order reversals, matched wording variants, or repeated uses of a baseline as independent conceptual tests.

Because each person scene has eight assignments and each object-object scene four, pooling all rows weights person scenes twice as much. For a scenario-balanced overview, average the eight scenario accuracies within each condition; also retain the raw counts and separate person-object/object-object results. Do not replace the condition breakdown with a single pooled score.

## Generation and review

Run from the repository root:

```sh
python data/close_far_situation_probe/generate.py
```

Outputs in `generated/`:

- `probes.csv` and `probes.jsonl`: the complete batch with labels and text.
- Component CSVs: `names`, `objects`, `scenarios`, `phrase_pairs`, `reference_orders`, `target_patterns`, `conditions`.
- `control_matches.csv` and `substitution_matches.csv`: diagnostic joins.
- `review_examples.csv`: one entity assignment for every scenario and condition (72 representative probes).
- `manifest.json`: counts, source hash, and structural validation status.

Generation checks counts, unique IDs/text combinations, balanced substitutions, exact preservation of contexts for reversal/negation, positive/negated target mappings, absence of unfilled placeholders, and preservation of targets for phrase changes. Structural checks do not certify human agreement on natural-language proximity. The examples are authored to make the categories clear, but human semantic review is still pending. Wording and scene size affect how people interpret close/far; these are diagnostic stimuli, not formally entailed numeric labels.

The JSONL uses EWoK-style text/category fields and additional diagnostic metadata. `evaluate.py` scores a local checkpoint on CUDA, retains all metadata, and exports per-item results, condition breakdowns, matched differences, and consistency. Its arguments are `--model`, `--out-dir` (a new output directory), and optional `--batch-size` (default 8). It uses mean target-token log likelihood only. The generator does not launch evaluation; its manifest records generation-time status.
