# EWoK Linear Probe

This package tests whether EWoK context-sensitivity information is linearly
accessible in frozen causal-LM hidden states.

V1 uses one representation: `post_target_last`, the final non-pad hidden state
after the model reads `Context + " " + Target`. It trains one logistic-regression
probe per candidate layer and regularization value, selects `(layer, C)` on the
validation split, then reports test metrics and probe-vs-LM case buckets.

```bash
python -m research.bos_aligned_proto.analysis.linear_probe.run_ewok_linear_probe \
  --model /path/to/ckpt_final_step0020000 \
  --output-dir runs/research/bos_aligned_proto/linear_probe/example \
  --ewok-variant fast \
  --score-view ewok_context_sensitivity \
  --batch-size 8 \
  --device auto \
  --dtype auto \
  --seed 42
```

By default, the runner also writes the standard figure set to
`<output-dir>/plots`. Use `--no-plots` for a table-only run, or
`--skip-layer-domain-plots` when you want the lightweight figures without
refitting the per-layer domain curves from the saved activation cache.

## Plotting

You can regenerate the standard figure set from an existing run at any time:

```bash
python -m research.bos_aligned_proto.analysis.linear_probe.plot_ewok_linear_probe \
  --run-dir runs/research/bos_aligned_proto/linear_probe/example
```

This writes:

- `layer_validation_curve.png/.svg`
- `probe_lm_test_summary.png/.svg`
- `probe_vs_lm_bucket_counts.png/.svg`
- `domain_bucket_heatmap.png/.svg`
- `probe_correct_lm_wrong_by_domain.png/.svg`
- `layer_domain_directional_avg_4x3.png/.svg`
- `layer_domain_scores.csv`
- `plot_manifest.json`

The layer-domain plot retrains small probes from the saved activation cache only;
it does not rerun the language model.
