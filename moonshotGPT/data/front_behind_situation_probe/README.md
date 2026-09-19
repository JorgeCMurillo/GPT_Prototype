# Compact front/behind probe (v2.0)

This **replaces** the detailed v1.0 set. Only compact contexts remain: no signs,
location lists, long tiers, definition prefixes, or answer bridges.
[Browse examples](generated/examples.md).

The compact v2.0 set has now been evaluated on Qwen3 359M step 19,500:
[report](../spatial_probe_results/front_behind_situation_probe/qwen3_359m_step19500_v2_0/report.md).
Balanced event accuracy is 50.00% using mean-token likelihood, with 0.00%
both-context correctness. Direct controls and observer turns are separate.
Use `scripts/evaluate_spatial_pairs.py --dataset front_behind --model CHECKPOINT
--out-dir NEW_RESULTS_DIRECTORY` to evaluate these explicit-gold pairs.

Examples:

- “B faces the door. A stands between B and the door.” → Front.
- “B faces A, then walks straight past A without turning. A stays still.” → Behind.
- “B faces increasing position numbers. A moves from 1 to 5. B stays at 3.” → Front.
- “B faces increasing position numbers. A moves from 3 to 5. B moves from 1 to 3 without turning.” → Front.

Front/behind is relative to **B's facing**, not A's facing, the reader's viewpoint,
or the front of an object. Contexts preserve the orientation information needed
for the answer but do not explain the definition. Current contexts contain
**4–19 whitespace-delimited words** (about 15 on average); the generator enforces
a 24-word maximum.

## Coverage

| Block | Paired rows |
|---|---:|
| Numeric situations | 160 |
| Nonnumeric situations | 52 |
| Direct-label controls | 4 |
| Observer-turn diagnostics | 8 |
| **Total** | **224** |

Main situations cover static placement, target crossing, reference crossing,
movement without crossing, and both moving. All 12 proposed seed geometries
remain in the numeric set, grouped into 10 opposing pairs: forward/backward full
crossings were already each other's matched counterpart.

Numeric scenes retain two entity pairs, two placement-clause orders, two answer
surface orders, and increasing/decreasing facing variants. No location list or
extra numbering scheme is needed.

Nonnumeric scenes cover eight qualitative cases: one static anchor case and the
seven movement cases. The static anchor has door/window/gate and facing-toward/
away variants. Numeric static distance/offset cases are **not** duplicated as
identical nonnumeric sentences. Natural-language motion clauses have their own
short order; no artificial placement-order variants are added.

The simple anchor scenes are now part of the nonnumeric static situation family,
not a separate long “minimal” block. Controls and observer turns remain separate.
No existing evaluated benchmark scores have been changed.

## Evidence and answer keys

- `initial_relation_explicit` flags literal starting front/behind statements in
  target-crossing and equal-motion scenes. Those are not the same evidence as
  inferring the starting relation from a person's facing.
- `final_relation_explicit` flags direct-label controls. The main situations
  require interpreting the scene, even when the starting relation is stated.
- `geometry_precision` distinguishes exact numeric coordinates from qualitative
  language. Coordinates in nonnumeric rows are representative satisfying
  geometries, not distances explicitly supplied to the model.
- Follow `correct_target_for_context1/2`; gold is **not always diagonal**.
- The answer forms “A is in front of B” and “In front of B is A” preserve B as the
  reference. Do not substitute “B is behind A” unless A's orientation is known.

`variant_matches.csv` links names, answer order, numeric placement-clause order,
fixed facing, and anchor nouns. Facing reversal flips answers while holding
physical positions fixed. Other links preserve answers. Numeric and nonnumeric
sentences are not presented as exact wording-only minimal pairs.

When evaluated, use raw binary mean-token likelihood and retain sums/counts:
“in front of” and “behind” can have unequal token lengths. Report initial-explicit
versus implicit cases separately. For an overall score, balance event families
and numeric/nonnumeric evidence within each family, then available cases and
surface variants. Do not pool the 160 numeric and 52 nonnumeric rows directly.
Cluster on `scene_cluster_id`, keeping renderings and facing variants together.
The crossing families still have only one matched geometry each; more renderings
do not make their scene uncertainty estimable.

## Regenerate

From the repository root:

```bash
python data/front_behind_situation_probe/generate.py
python -m pytest tests/test_front_behind_situation_probe.py -q
```

The generated JSONL, CSV, links, manifest, and example sheet replace the old
detailed artifacts in place. No detailed archive is retained in this probe.
Tests parse numeric positions and qualitative movement descriptions independently
to check the gold labels and reject long contexts or noncompact tiers.

**Not evaluated yet**, and not included in the existing scored six-category
composite.
