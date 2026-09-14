# Context expansion and body-reference diagnostics

This additive batch contains **228 paired probes** using the same eight scenarios and 52 entity assignments as the original situation dataset. The original 468 probes and their results are preserved. Qwen3 step-19,500 results are saved under `runs/research/bos_aligned_proto/close_far_situation_probe/qwen3_359m_step19500_context_diagnostics_v1/` relative to the repository root. Mean target-token log likelihood remains the default.

## Three context forms

The location sentence and both target alternatives stay exactly the same across the three forms. Only a shared prefix is added to C1 and C2.

| Form | Example close context |
|---|---|
| compact | The ball rests immediately beside Maya's shoes. |
| setting | Maya stands in a gym. The ball rests immediately beside Maya's shoes. |
| endpoint | Maya stands at one end of a long gym. The ball rests immediately beside Maya's shoes. |

The corresponding far location sentence is always:

> The ball rests across the full length of a long gym from Maya.

The same setting or endpoint prefix is added to this sentence in its respective condition. Targets remain "The ball is close to Maya." and "The ball is far from Maya."

Far descriptions are self-contained, so the compact version does not depend on a deleted sentence to define "the other end" or locate the person. Compact dining contexts specify the chair where the person sits; object-object contexts likewise avoid relying on a removed setup for the relation itself.

## Body/worn-item comparison

Three scenarios have an additional matched close description:

| Scenario | Placement reference | Direct person reference |
|---|---|---|
| Gym | immediately beside Maya's shoes | immediately beside Maya |
| Field | immediately beside Maya's feet | immediately beside Maya |
| Platform | immediately beside Maya's ankle | immediately beside Maya |

Each comparison uses the same predicate, far context, targets, name, object, and prefix form. Only the possessive body/worn-item phrase is removed. Shoes are explicitly labeled `worn_item`; feet and ankle are labeled `body_part`. This does not claim shoes are anatomically part of a person.

For this contrast, the original between-shoes/between-feet/against-ankle descriptions were rewritten with a shared "immediately beside" predicate. Those original examples remain in the original dataset. Comparing the new batch directly to an old baseline can therefore change wording as well as context length; use the new within-batch match table for controlled comparisons.

There is no body-reference condition for chairs, seats, benches, or other objects. Those five other scenarios still receive all three context forms.

## Counts and variable entities

- All 52 original entity assignments x 3 context forms = **156 placement probes**.
- 3 eligible body/worn-item scenarios x 4 names x 2 objects x 3 context forms = **72 direct-person probes**.
- Total **228 paired probes**, or **456 context-level judgments**.
- Each context form has 76 probes (52 placement + 24 direct-person).
- Names: Maya, Jesse, Li, Omar. Every name is crossed with both eligible objects in every person scenario and condition. Names are consistently substituted in contexts and targets.
- Object-object scenarios retain all four eligible target/reference-object combinations per scenario. No name is inserted into object-only scenes.

## Diagnostic metadata and comparison

Use `context_form_id`, `reference_form_id`, `body_reference_type`, `body_reference`, `scenario_id`, and name/object IDs for grouping. The location sentences and `prefix_text` are also saved separately. Per-text word and character counts are descriptive; actual model-token counts must be saved during evaluation.

`diagnostic_matches.csv` contains **224 comparisons**:

- 152 prefix comparisons: compact versus setting and compact versus endpoint, with identical location sentences and targets.
- 72 body-reference comparisons: placement versus direct-person, with identical far context and targets.

The separate `original_matches.csv` links new probes to their original baseline entity assignment, but marks those comparisons as potentially changing wording. `close_evidence_type` and `far_evidence_type` distinguish the mixed direct-person-close/physical-placement-far conditions. A whole-item evidence label is coarser than these per-context labels.

For the primary context comparison, use the 156 placement probes so that all eight scenarios are represented. For the body-reference comparison, restrict both sides to the three eligible scenarios. Within each condition, also report scenario-balanced results because person scenarios contain eight assignments and object-object scenarios four.

Mean-token completion-choice accuracy, both-sides success, changes in close/far margins, and correctness stability should be reported separately. Shared far context/target strings in the body comparisons should yield identical far judgments under deterministic scoring; they are a useful scoring consistency check and not independent evidence of improvement.

## Limits of the experiment

The prefix conditions test sensitivity to added context, not a pure causal effect of sentence length. Setting and endpoint prefixes add different spatial information. The matched body phrase is also longer than the direct-person phrase; an improvement supports sensitivity to this reference construction but does not isolate body inference from length and lexical familiarity. A length-matched nonspatial prefix control has not been added.

These remain natural-language clear-close/clear-far judgments under an ordinary physical-distance reading. Independent human semantic review has not been conducted.

## Files and generation

Run `python data/close_far_situation_probe/context_diagnostics/generate.py` from the repository root.

- `components.json`: prefix forms, location sentences, and body-reference eligibility.
- `generated/probes.csv` / `.jsonl`: all 228 labeled probes.
- `generated/review_examples.csv`: one entity assignment for each scenario and diagnostic condition.
- `generated/diagnostic_matches.csv`: controlled comparison links.
- `generated/original_matches.csv`: links to original baseline IDs.
- `generated/context_forms.csv`, `reference_forms.csv`, `scenarios.csv`, `names.csv`, `objects.csv`: component tables.
- `generated/manifest.json`: counts, source hashes, and structural validation.

Generation validates unique text/IDs, variable-name/object balance, prefix-only expansion, identical targets, body-phrase removal, identical far contexts in body comparisons, and preservation of the original dataset hash. Evaluate with the parent `evaluate.py --dataset context_diagnostics --model CHECKPOINT --out-dir NEW_OUTPUT_DIRECTORY`. This scores the 228 probes and reports body-reference comparisons on the same eligible subset. Generation manifests retain generation-time evaluation status.
