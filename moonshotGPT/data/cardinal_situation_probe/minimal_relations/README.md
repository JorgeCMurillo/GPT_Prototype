# Minimal relation-first cardinal scenes

An additive **direct-initial-relation evidence format** for the cardinal
families: 46 matched pairs / 92 context judgments, 23 pairs per axis. The
numeric/named-location inventory, definitions, observer-turn extension and
all previous evaluation results are unchanged. This block has now been evaluated
on Qwen3 359M step 19,500: [report](../../spatial_probe_results/cardinal_situation_probe/qwen3_359m_step19500_minimal_v1/report.md).
The generator manifest records generation-time status only. The primary
minimal-movement scores are 37.50% north/south and 34.38% east/west, with
static controls, observer turns, and explicit-persistence variants separate.

Eight cases per axis cover static placement, target crossing, reference
crossing, approaching without crossing, moving away, equal-distance co-motion,
swapping positions, and turning without translation. Each row pairs opposing
initial directions under the same wording; north/south and east/west share
the same structures. These are repeated renderings, not 46 independent
spatial structures.

Examples (each ends with the conditioning bridge “Final positions:”):

- “A is south of B. A moves straight toward B and continues past B.” → north.
- “A is south of B. A heads straight to B and continues past B.” → north.
- “A is south of B. B moves straight toward A and continues past A.” → north.
- “A is south of B. A heads toward B but stops before reaching B.” → south.
- “A is south of B. A moves farther south.” → south.
- “A is south of B. Both head north for the same distance.” → south.
- “A is south of B. A and B swap positions.” → north.
- “A is south of B. B turns around without moving.” → south.

## Controlled wording and assumptions

The main forms omit redundant stationary clauses. Their labels assume that
unmentioned positions persist; `requires_implicit_persistence` makes this
assumption explicit in metadata. Twelve paired rows form a small explicit
subset, adding “B stays still,” “A stays still,” or “Neither moves” to the
baseline wording. This is not a full crossing of all cue and wording factors.
Equal co-motion and swapping already describe both entities and receive no
stationary variants. Static restatements are controls, not movement inference.

`movement_verb` matches change only moves/heads (move/head after “Both”).
`preposition` matches change only toward/to **in completed crossing scenes**.
“To” implies reaching the reference whereas “toward” alone does not; the
following “continues past” clause makes both crossing outcomes agree. No
“to ... stops before reaching” variant is generated. “Heads” is interpreted
as motion in the described event, not merely an intention or facing change.
These lexical variants share intended outcomes but are not claims of full
semantic synonymy. Orientation-only events use “turns,” never “heads.”

No map, numbers, location list, or up/right definition is needed: starting
relations are explicitly cardinal. The stored coordinates are representative
ordinal geometries used to validate outcomes, not distances supplied by the
text. A swap explicitly states a reversal, unlike inferring it from endpoints.

## Answers, matches, and evaluation

Targets are “A is north/south of B” or “A is east/west of B.” Follow
`correct_target_for_context1/2`: **C1 does not always support Target1**.
Both directions are balanced within each row. Preserve the scored answer
separately from the context and its “Final positions:” bridge.

`variant_matches.csv` links verbs, prepositions, persistence cues, axes, and
static/observer-turn contexts. The latter link tests cardinal invariance under
turning; in explicit variants it also replaces “Neither moves” with a turn
statement and “A stays still,” so it is not a pure single-clause addition.
Report static controls, observer turns, and movement families separately;
keep implicit/explicit persistence and wording factors visible. Do not simply
pool this block into the existing numeric/named score. There are no exact
cross-format matches to those scenes: this block supplies less geometric detail.

Generate and validate:

```bash
python data/cardinal_situation_probe/minimal_relations/generate.py
python -m pytest tests/test_cardinal_minimal_relations.py -q
```

Evaluate with `scripts/evaluate_spatial_pairs.py --dataset minimal_cardinal
--model CHECKPOINT --out-dir NEW_RESULTS_DIRECTORY` (CUDA required). It follows
explicit per-context answer keys, saves mean and summed scores from the same
token arrays, and reports matched wording/cue changes.

Generated outputs: `probes.jsonl`, `probes.csv`, `variant_matches.csv`,
`review_examples.md`, and `manifest.json`. Word counts are whitespace-based.
