# Balanced custom spatial benchmark

This report combines **six selected situation categories**: above/below,
fixed-frame left/right, north/south, east/west, close/far, and closer/farther.
It uses saved Qwen3 359M step-19,500 **raw binary mean-token likelihood**
decisions, without PMI. It does not rerun inference or modify source results.

The versioned [manifest](qwen3_step19500.json) defines the exact input runs,
filters, balancing fields, clustering and exclusions. This is a selected
situation composite, not a pool of every historical spatial test. Definitions,
direct-label controls, observer-turn diagnostics, older probe versions and
unevaluated extensions are excluded. Close/far uses the existing matched-scene
applied subset; closer/farther uses the existing three-family movement macro.

## Weighting

1. Average paired-context accuracy within each declared wording cell of a
   scene/evidence combination. Each paired row has two binary judgments.
2. Average wording cells equally, then evidence formats equally within a scene.
3. Average scenes within each event family, then event families within a category.
4. Average the six relation categories equally, irrespective of row counts.

Both-contexts-correct follows the same hierarchy. This helps distinguish
consistent matched-pair success from a fixed answer preference that scores 50%.
Do not interpret 50% paired accuracy as successful spatial reasoning.

The manifest's wording cells balance length and entity orders in the directional
probes, list/number order in the cardinal probes, and contrast/length/entity order
in the movement probe. Numeric and nonnumeric evidence have equal weight even
where numeric renderings greatly outnumber nonnumeric ones. Regression tests
verify all six category estimates reproduce their existing balanced reports.

## Conditional cluster bootstrap

The default uses 20,000 replicates (seed 42) and percentile 95% intervals.
Within each fixed event-family stratum, sample the original number of scene
clusters with replacement. Keep both paired contexts, every evidence format,
entity substitution, numeric variant and wording/order variant attached.
Recompute the same balanced score in every replicate.

Structurally matched above/below and left/right cases share draws, as do the
matched north/south and east/west cases. Thus the axes are not treated as
independent case samples. Different source groups are sampled independently.
The source schemas must contain matching case sets and complete case/evidence
crossings. Duplicate selected pair identifiers, wrong row counts, incompatible
scoring reductions and mixed checkpoints are rejected. Source and metadata
hashes are included in the JSON output.

Important limitations:

- These hand-built cases are not random samples from all spatial reasoning.
  The interval describes **sensitivity to resampling the covered scenes**,
  conditional on these families, templates, model and scoring method.
- Close/far has six setting clusters, not 192 independent name/object scenes.
  Those settings reuse the same spatial geometry; its interval measures setting
  variation, not new-structure generalization.
- Closer/farther has one structural cluster per fixed family. Numbers, names,
  units and repeated cross-family comparisons are variants, not new structures.
  All its strata are singletons: scene uncertainty is **not estimable**. It
  contributes its fixed point score to the composite bootstrap. The JSON retains
  the resulting degenerate resampling range and an `all_strata_singleton` flag;
  the Markdown table says “Not estimable” rather than presenting that as precision.
- Some cardinal families also have singleton strata. These stay fixed, so the
  overall interval does not estimate uncertainty from novel cases in them.
- Constant cluster scores can give zero-width intervals, e.g. always choosing
  north gives exactly 50% on every matched pair. This is not certainty about
  competence. Adding copies or paraphrases does not create independent evidence.

Use the composite as a descriptive summary alongside category and paired-success
scores. More genuinely distinct scene structures are needed for broader CIs.

## Run

From the repository root (Python with NumPy):

```bash
python data/spatial_benchmark/report.py \
  --output-dir runs/research/bos_aligned_proto/spatial_benchmark/qwen3_359m_step19500_v1
python -m pytest tests/test_spatial_benchmark_report.py -q
```

Outputs: `summary.json` (provenance, counts, family/category/composite intervals,
warnings) and `report.md`. Optional `--categories above_below left_right
north_south east_west` produces a directional-only score. To use a different
checkpoint, copy the manifest and replace its source and metadata paths; do not
mix checkpoints. `--bootstrap`, `--seed`, `--manifest`, and `--root` are configurable.
