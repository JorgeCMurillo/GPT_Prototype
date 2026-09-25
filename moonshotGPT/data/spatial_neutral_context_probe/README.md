# Matched neutral spatial contexts

This additive probe gives each selected matched spatial pair `(C1, C2)` one
uninformative context `C0`. It is designed for the contextual answer baseline

`b0 = log P(T1 | C0) - log P(T2 | C0)`.

Neutral means that neither target is entailed or ruled out. It does not mean
that the model must assign equal scores, and some contexts also permit a third
outcome such as equality or unchanged distance.

## Scope

The dataset covers the six-category custom composite plus the compact
front/behind and minimal-cardinal event supplements. Definitions, direct-label
controls, observer turns, and minimal-cardinal static controls are excluded.
The six-category composite has 16,380 rows; the supplements add 212 and 38,
for 16,630 rows total.

Original contexts, targets, answer keys, and source files are not modified.
Every output row copies the original text and IDs, adds `Context0`, records a
versioned neutralization rule, and links Target1 and Target2 to their respective
source-scene witnesses. The witness records include available coordinates,
motions, facing, or distance-setting metadata. `correct_target_for_context0`
is always null.

The closer/farther cross-family contrasts require a broader before/after
distance baseline because their original contexts describe incompatible event
types. Those rows are tagged `contrast_level` and receive high-priority manual
review. Other tradeoffs—including endpoint-position neutralization and removal
of an explicit front/behind facing value—are also tagged.

## Generation and review

Run these commands from the `moonshotGPT` directory. The published JSONL and CSV
are losslessly compressed; the generator and manifest are preserved unchanged.
The manifest's output hash refers to the **decompressed JSONL** bytes.

```bash
gzip -dk data/spatial_neutral_context_probe/generated/neutral_probes.jsonl.gz
gzip -dk data/spatial_neutral_context_probe/generated/neutral_probes.csv.gz
python -m pytest tests/test_spatial_neutral_context_probe.py -q
```

Tests read the compressed snapshot directly; extraction is only needed by tools
that expect plain JSONL or CSV. The full dataset includes C0, both original
contexts and targets, source hashes, and witness metadata.

To regenerate from the original inputs:

```bash
python data/spatial_neutral_context_probe/generate.py
```

Regeneration requires every source file listed in `generated/manifest.json` at
its recorded path under `moonshotGPT`, with the recorded SHA-256. In particular,
the closer/farther inputs under `runs/research/bos_aligned_proto/` are local run
artifacts and are not included in this checkout. The published snapshot and its
validation tests do not require those run files. Regeneration produces plain
JSONL and CSV; it does not replace the published compressed snapshot.

Generated artifacts:

- `neutral_probes.jsonl.gz` and `.csv.gz`: full matched triplets and metadata.
- `review_examples.md`: one example per source/family/evidence/rule cell.
- `manifest.json`: counts, source hashes, output hash, and validation status.

All rows remain `manual_review_status: pending` until a human review pass is
completed. Generation fixes the rules before model scores are inspected.
`Context0_whitespace_token_count` is a model-independent whitespace count;
the later evaluator must save the actual tokenizer-specific target counts.

## Scoring convention

No model scores are generated here. A later evaluator should score the exact
targets after C0 using the same separator/BOS convention as the spatial
benchmark and report both summed and mean-token differences. The literal `b0`
is the summed-token log-likelihood difference. Preserve original T1/T2 for
audit, while also reporting the stored alphabetical canonical order so that
aggregate signed biases do not flip when candidate slots are reversed.
