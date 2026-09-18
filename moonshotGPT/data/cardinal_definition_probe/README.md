# Cardinal direction definitions and map-position controls

Expanded version 1.2 evaluation: `runs/research/bos_aligned_proto/cardinal_blocks/qwen3_359m_step19500_expanded_v2/`.
See its report, findings, and confidence-interval files; the earlier run below
is retained for comparison.

A small inventory of 80 matched pairs (160 binary judgments), balanced between
north/south and east/west. Version 1.2 preserves all 40 prior IDs and texts,
adding alternate edge expressions with matched prepositions, route-reversal
wording, and reciprocal cabin/lake relative-position examples.

| Block | Pairs | Scope |
|---|---:|---|
| Direction definitions | 32 | Original mappings plus upper/lower or right-hand/left-hand edges, both mapping directions, traveling/going, toward/to |
| Opposite directions | 16 | Original four phrasings plus four straight-route reversal forms per axis |
| Relative position | 32 | Original map/globe mappings plus reciprocal cabin/lake answers |

`design_extension`, `preposition`, and `preposition_match_id` identify additions.
Eight preposition links hold answers and all context wording fixed except
“toward” versus “to”. These words differ in directional versus destination
meaning even though gold directions agree here. `reciprocal_named_entities`
changes entity names as well as answer reference order; it is not an isolated
entity-order intervention. Existing `gold*_cardinal_direction` fields describe
the answer's subject relative to its reference, while `context*_direction`
describes the original subject. Route-reversal items specify a straight route.

All map items state: “On this map, north is at the top and east is on the
right.” The edge expressions are top/bottom and right/left edge; movement
expressions are upward/downward and rightward/leftward. Relative positions use
above/below and to the right/left of marker B. These refer to positions on a
map, not real-world altitude. Opposite-direction items do not require a map.
The half-turn wording requires applying a directional change, so keep its
template ID visible in analysis; it is not merely a synonym substitution.

The original eight opposite pairs are **two axes × four templates**: “The direction
opposite…”, “Going the opposite way…”, “reversing that direction…”, and “turn
halfway around…”. Each north/south pair contains both a north context with a
south answer and a south context with a north answer; east/west works likewise.
They are eight paired examples, not eight different opposing direction concepts.

Globe versions explicitly specify an upright North Pole and a central visible
region where north appears upward and east rightward, with both markers nearby.
An arbitrary view of a globe does not guarantee that correspondence. The globe
prefix is longer and supplies local orientation, so map/globe links are not
pure single-word substitutions and cannot isolate the effect of the noun alone.
`reference_surface` labels the medium; the legacy `relative_map_position` block
name is retained for both media. Eight `reference_surface` links hold candidate
answers, axis, mapping, and wording template fixed.

Within each pair, the wording structure is held fixed while direction changes.
Both contexts share both candidate answers: C1 supports T1, C2 supports T2.
Across wording variants, candidate answers stay identical. Word-to-meaning and
meaning-to-word variants are linked, as are edge and movement expressions;
these change what is being asked and are not pure wording controls.

Metadata records axis, mapping direction, wording family and template, reference
frame, intended cardinal direction, and map-axis sign (+up or +right). Marker
A remains the subject in relative-position items. Reciprocal entity reversal,
rotated maps, and the five physical event families are outside this inventory.
No heuristic explanations of errors are assigned.

Run from the repository root:

```bash
python data/cardinal_definition_probe/generate.py
```

Edit `components.json` to change wording. Generated files include `probes.csv`,
`probes.jsonl`, `variant_matches.csv`, `manifest.json`, and a readable
`review_examples.md` contains all 80 pairs. Qwen3 359M step-19,500 results for
the earlier 40-pair version are
saved in `runs/research/bos_aligned_proto/cardinal_blocks/qwen3_359m_step19500_v1/`.
The shared runner is `data/cardinal_situation_probe/evaluate_blocks.py` with
`--model CHECKPOINT --out-dir RESULTS`; it evaluates this block and observer turns.
It scores all four context/answer combinations
using raw mean target-token conditional log likelihood, counting ties as wrong,
and report binary accuracy, paired correctness, and answer preference by block,
axis, wording, and mapping direction. Do not pool these controls into physical
situation scores.

This checks recognition and application of an explicitly supplied map
convention. Because the prompt supplies the convention, success does not show
that a model independently knows geographic orientation. Wording variants are
repeated measurements of a few conceptual distinctions, not independent scenes.
