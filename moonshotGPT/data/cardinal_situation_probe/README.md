# Cardinal situations: fixed map

The additive [minimal relation-first block](minimal_relations/README.md)
provides short scenes with explicit starting relations, moves/heads wording,
toward/to crossing variants, and a small matched stationary-cue subset.
It covers both axes without changing the numeric/named inventory below and
has its own evaluation report linked there.

Version 1.4 uses short, natural event wording in **both numeric and named-location
scenes**, without removing cases, pairs, or ordering variants. The 1,344 IDs,
answers, endpoints, and 4,424 match links are unchanged. Event contexts change;
the 48 simple-control pairs retain their previous text. Saved v1.2 evaluation
results remain historical and do not evaluate this revised wording.
The [v1.4 evaluation](../spatial_probe_results/cardinal_situation_probe/qwen3_359m_step19500_v1_4/report.md)
on Qwen3 359M step 19,500 scores 50.00% north/south and 49.83% east/west on
the balanced event families, using raw mean-token likelihood without PMI.
Version 1.3 shortened the scenes; v1.4 smooths only the setup sentences,
keeping the movements and “Final positions:” bridge unchanged. The map
orientation and fixed frame remain explicit.

| Event wording tier | Original v1.2 mean words | Current v1.4 mean words |
|---|---:|---:|
| Compact | 51.3 | 40.3 |
| Standard | 60.4 | 42.3 |
| Expanded | 87.4 | 47.2 |
| All tiers | 66.4 | 43.3 |

These are whitespace-delimited context-word averages, excluding answers and
controls; north/south and east/west have identical averages. Current event
contexts range from 35 to 52 words. Existing per-text `*_word_count` metadata
retains its regex-based counting convention, which splits hyphenated words.

Compact examples (the same formats exist on both axes):

- Numeric: “On this fixed map, north is up and east is right.
  Rows run south to north: 1, 2, 3. Both markers share a column.
  A moves from row 1 to row 3. B stays at row 2.
  Final positions:”
- Nonnumeric: “On this fixed map, north is up and east is right.
  Along one line, the locations run west to east: the school, the station, the garden.
  A moves from the school to the garden. B stays at the station. Final positions:”

The separate [observer-turn extension](observer_turn/README.md) adds matched
no-turn/half-turn contexts with cardinal and observer-relative answers, testing
invariance versus left/right reversal. It preserves this original dataset and
uses explicit per-context gold labels rather than fixed diagonal matches.

1,344 matched pairs (2,688 context judgments): 1,296 event pairs and 48 simple
controls. Every scene explicitly uses a north-up, east-right map. Two markers,
A (target) and B (reference), occupy positions on one axis; the other axis is
held fixed. North/south and east/west share the same underlying event states.

## Cases and counting

| Event family | Underlying matched cases | Expanded pairs |
|---|---:|---:|
| Static placement | 2: adjacent, one intervening position | 288 |
| Target crosses reference | 1 | 144 |
| Reference crosses target | 1 | 144 |
| Target moves without crossing | 2: positive/negative movement | 288 |
| Both move | 3: preserve order in either direction, swap positions | 432 |

Each underlying case already includes opposing outcomes: C1 makes A north/east
of B; C2 makes A south/west of B. Counting a crossing again with its two
contexts exchanged would duplicate the same pair. This audit reduces the
earlier proposed 12 cases to nine. Expansion is nine cases × two axes × two
evidence formats × three wording lengths × two context entity orders × two
answer entity orders = 432 base event pairs. Version 1.2 adds two list
presentation orders to both formats and two numbering assignments to numeric
scenes: 864 numeric plus 432 named-location event pairs = 1,296.

Direct cardinal statements and screen-to-cardinal mappings contribute two
control types × two axes × three lengths × two context orders × two answer
orders = 48 more pairs. They remain separate
from the event score. The 36 family/format/length template combinations share
reusable clause renderers, rather than being 36 independent semantic tests.
Version 1.1 added both entity-order dimensions while preserving the original
120 pairs' IDs and then-current texts; added order combinations receive ID suffixes. Entity
names remain A and B. `template_id` retains the 36 family/format/length
combinations; context and answer order are separate factors.

## Formats, length, and names

Numeric versions explicitly state the directional order of rows or columns.
Named versions explicitly list locations along that same axis. A location label
maps to the exact ordinal position used in its numeric counterpart. Neither
format requires interpreting an unstated geographic arrangement.

### Separately tagged ordering factors

- `location_list_order`: `negative_to_positive` (south→north / west→east) or
  `positive_to_negative`. Reverses the list and its stated traversal direction
  together while leaving each location's physical position and label fixed.
- `numeric_label_order`: `increasing` or `decreasing` along the internal
  negative→positive axis. Reverses numeric labels and updates every start/end
  label in the scene while keeping physical coordinates fixed. Named locations
  and simple controls use `not_applicable` for this field.

For example, “rows 1, 2, 3 from south to north” and “rows 3, 2, 1 from north
to south” have identical numbering but different presentation. Assigning rows
3, 2, 1 from south to north instead changes the numbering itself. These factors
are crossed independently. Named locations only receive list reversals, not
numeric relabeling. Simple controls receive neither factor. Separate matched
links change one factor at a time and preserve correct answers and endpoints.
Version 1.2 preserved all previous 480 IDs and texts as the original-factor
subset. Version 1.3 retains those IDs but shortens event texts across all factors.

The pool is lake, forest, village, bridge, tower, garden, station, school, mill,
field, cabin, and fountain. Each case uses only two to four named anchors.
Static adjacent and swap cases use two; crossing cases use three. No-crossing
and order-preserving motion use four anchors across their two contexts.
Only the separated-static case intentionally includes an unused intermediate
anchor, to distinguish adjacent from separated positions. Location spacing
and physical distances are not specified.

Names stay fixed across length variants and opposing contexts. A cyclic
assignment distributes all 12 names across cases, with reversed assignments
between the two axes, so every name has balanced ordinal rank when pooling
axes. This is not full name counterbalancing within each axis; axis comparisons
also change names in the named format. Numeric axis comparisons do not.

Compact, standard, and expanded variants preserve endpoints and events.
Standard adds the explicit “Marker” noun; expanded states initial and final
conditions more explicitly without adding another event or marker. All retain
the fixed north-up/east-right map convention and shared row, column, or axis.
Length and syntax vary together. Version 1.3 uses the short event answer bridge
“Final positions:” in all length/format variants. Controls retain the original
“After the scene ends, to summarize the final positions:” bridge and frame.
The event shortening changes both setup and bridge, so a before/after evaluation
would not isolate the bridge's effect. Observer-turn and definition blocks are
unchanged.

In events, context order reverses the complete A/B description blocks without
changing positions. In simple controls, it rewrites the stated relation with B
as subject and the inverse direction, preserving the same fact. Answer order
independently switches between “Marker A is north/east of marker B.” and
“Marker B is south/west of marker A.” The opposite answer is also inverted.
C1 still supports T1 and C2 T2. Gold-relation metadata describes A relative to B;
`target1_relation_word` and `target2_relation_word` track actual answer words.
Matched links hold answers fixed when context order changes and contexts fixed
when answer order changes. Score only the candidate continuation after the context, including
the bridge in the context rather than in the scored answer.

## Generate and review

```bash
python data/cardinal_situation_probe/generate.py
python -m pytest -q tests/test_cardinal_situation_probe.py
```

`components.json` contains the nine endpoint pairs, location pool, conventions,
and entity-clause templates. `generated/` contains:

- `probes.csv` and `probes.jsonl`: all texts and observable metadata.
- `case_catalog.json`: underlying initial/final states.
- `review_examples.md`: all 1,344 readable pairs.
- `variant_matches.csv`: length, evidence-format, axis, both entity orders,
  list presentation, and numeric-label matches.
- `manifest.json`: counts and source hash.

Metadata includes initial/final ordinal positions and their rendered labels,
movement directions, initial/final relations, order reversal, location names,
entity order, format, wording length, and reference frame. No speculative
heuristic explanations are attached. Direct controls leave unspecified
coordinates blank.

Generation checks distinct paired states, non-coincident initial and final
positions, correct final relations, family constraints, and matched endpoints.
Tests independently decode rendered location labels, check lexical balance and
opposing-motion invariants, and confirm invalid states are rejected.

`evaluate.py --model CHECKPOINT --out-dir RESULTS` scores the dataset on CUDA,
using the repository's EWoK conditional-likelihood routine and local checkpoint
files. It saves input snapshots, per-token and per-item scores, grouped scores,
matched changes, and a report. The output directory must not already exist.
It reconstructs saved token means and decisions before reporting completion.
The Qwen3 359M step-19,500 evaluation of version 1.2 is saved in
`runs/research/bos_aligned_proto/cardinal_situation_probe/qwen3_359m_step19500_v1_2/`;
see `report.md`, `findings.md`, and `summary.json` for results and validation.
The evaluator reports raw mean-token answer accuracy, both-context correctness, and directional answer
preference by family and format; exact ties count as incorrect. Average the
five event-family means equally if producing an overall event score, because
families have different numbers of cases. Also balance numeric and named
format means within each family: numeric has twice as many variants due to
the extra numbering factor. Keep controls separate. The 1,344 pairs
are repeated measurements of nine event structures and two control types,
not 1,344 independent scenes. Globe events, reference motion
without crossing, and arbitrary trajectories are outside this initial batch.
