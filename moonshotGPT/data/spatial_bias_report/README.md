# Spatial answer-preference tables

`generate.py` reads the saved, matched two-answer scores for a spatial probe. It
does not score the model again. Supported close/far, closer/farther, and
above/below evaluators
generate a table after writing their scores. To backfill or regenerate a run:

```bash
python data/spatial_bias_report/generate.py --results-dir runs/research/bos_aligned_proto/close_far_situation_probe/qwen3_359m_step19500_v1
```

`--all-spatial` regenerates every supported report under
`runs/research/bos_aligned_proto`. Unsupported reports are listed as skipped.
Each supported result directory gets `bias_table.md`, `bias_table.csv`, and
`bias_table_summary.json`. The CSV includes all available finer groupings and
PMI results. The JSON records the source file hashes. An existing `report.md`
gets a link to its table; repeated generation does not duplicate the link.

Each matched pair contains two judgments: one context supports the first
answer, and the other supports the second. The table reports accuracy on each
gold answer, how often each answer was chosen, and these mutually exclusive
pair outcomes:

- **Both correct:** the model switches answers with the context.
- **Always first / always second:** it chooses the same answer in both contexts.
- **Both wrong:** it switches in the wrong direction.
- **Tie in pair:** at least one comparison tied exactly; ties count as incorrect.

Thus 50% accuracy with 100% `always second` diagnoses a constant answer choice,
while 50% accuracy with 50% `both correct` and 50% `both wrong` is a different
pattern. For negated targets, the answer labels explicitly say `not close` or
`not far`. The table only describes preference within the given binary target
pair; it does not infer that `not close` entails `far` in an intermediate case.

The primary table groups by probe condition or binary contrast. The CSV splits
further by event family, scenario, wording, sentence length, or entity order
when those fields are available. Names and objects are repeated stimuli, so a
large pair count is not the number of independent semantic situations. PMI and
fixed-target context sensitivity are retained as secondary diagnostics; raw
mean target-token log likelihood is the primary choice score.
