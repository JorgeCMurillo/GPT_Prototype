# Version 1.1: matched explicit naming probes

This extension adds 18 definition-to-word probes, one matched to each of the 18 original reverse probes. The original 72 records and version 1 results are preserved. The combined dataset contains 90 paired probes. This extension has not yet been evaluated.

## Context structures

All contexts end in "described by the word" and all targets are `close.` / `far.`. Contexts differ only in the distance adjective within each pair. Modifiers and adjective pairs retain their original IDs and values.

| ID | Matched original structure | Example C1 / C2 | Words |
|---|---|---|---:|
| name_0 | def_0 | A relatively small/large distance is described by the word | 9 |
| name_1 | def_1 | A relatively small/large distance between objects is described by the word | 11 |
| name_2 | def_2 | Objects separated by a relatively small/large distance are described by the word | 12 |

The lengths stay between 9 and 12 words for all reviewed modifiers and adjective pairs. These are word counts, not tokenizer counts. The contexts are incomplete sentences on purpose: appending the target completes them. No names, movement, numbers, distractors, or additional spatial facts are introduced. The longer structures add explicit objects or separation wording rather than an unrelated story.

The three structures cross with relatively/comparatively/fairly and small/large versus short/long: 3 x 3 x 2 = 18 new probes. They are matched in concept, modifier, adjective pair, and underlying definition structure to the original reverse probes. They are not a pure experiment on length: syntax and wording also differ. The naming structure and original definition structure are deliberately linked, not independently crossed factors.

## What the comparison can establish

The original form asks the model to continue a distance statement with "The objects are close/far." The added form explicitly asks for the spatial word and uses a short label continuation. If scores improve, that supports sensitivity to the task/response format. It does not by itself distinguish explicit task wording from the changed target sentence length, because both change together.

Mean target-token log likelihood remains the default. Report original reverse, explicit naming, and forward results separately before aggregating. Matched differences can be calculated with `naming_matches.csv`. Group the new condition by `naming_context_id`, modifier, and adjective pair. Do not treat matched variants as independent conceptual skills.

## Files

- `naming_contexts.json`: controlled templates and IDs.
- `generate.py`: deterministic extension generator, run with `python data/close_far_definition_probe/v1_1/generate.py` from the repository root.
- `generated/probes.csv` / `.jsonl`: 90 records; the first 72 match version 1 exactly.
- `generated/naming_probes.csv` / `.jsonl`: the 18 added probes only.
- `generated/naming_contexts.csv`: context lookup table.
- `generated/naming_matches.csv`: one-to-one links to original reverse probes.
- `generated/manifest.json`: counts, original dataset hash, and validation status.

The current parent `evaluate.py` targets the original 72-item dataset. It must be given support for the new input and condition grouping before evaluating this extension; do not silently mix the naming probes into the original reverse-condition aggregate.
