# Spatial probe result snapshots

These are compact snapshots of the Qwen3 approximately 359M-parameter
step-19,500 evaluations. Each run directory preserves its report, aggregate
JSON, grouped scores, and available bias tables. The full item-level and
token-level outputs remain in the local `runs/` directory; the checkpoint is
also local. Generated input examples and their component tables are in the
neighboring probe directories.

| Probe | Primary result | Diagnostic control |
|---|---|---|
| Close/far literal definition | [Word to definition: 84.26%; definition to word: 50.00%](close_far_definition_probe/qwen3_359m_step19500_v1/report.md) | [Response preference](close_far_definition_probe/qwen3_359m_step19500_v1/bias_table.md) |
| Close/far matched situations | [Applied mean: 58.14%](close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1/applied_score_report.md) | [Direct labels: 97.66%](close_far_evidence_probe/qwen3_359m_step19500_matched_scenes_v1_1/applied_score_report.md) |
| Closer/farther movement | [Applied mean: 48.97%](closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2/overall_report.md) | [Direct labels: 82.64%](closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2/overall_report.md) |
| Above/below literal definition | [Word to definition: 56.25%; definition to word: 61.25%](above_below_definition_probe/qwen3_359m_step19500_v1_2/report.md) | [Answer preference and entity order](above_below_definition_probe/qwen3_359m_step19500_v1_2/above_below_bias.md) |
| Above/below situations v1.3 | [Equal-family applied mean: 50.23%](above_below_situation_probe/qwen3_359m_step19500_v1_3/report.md) | [Target-order bias](above_below_situation_probe/qwen3_359m_step19500_v1_3/bias_table.md) |
| Fixed-frame left/right, evaluated v1.0 | [Applied mean: 50.03%](left_right_situation_probe/qwen3_359m_step19500_v1/report.md) | [Explicit summary bridge](left_right_situation_probe/qwen3_359m_step19500_summary_bridge_v1/report.md) |
| Cardinal situations v1.2 | [North/south 50.00%; east/west 49.93%](cardinal_situation_probe/qwen3_359m_step19500_v1_2/report.md) | [Failure analysis](cardinal_situation_probe/qwen3_359m_step19500_v1_2/findings.md) |
| Cardinal definitions and observer turns | [Expanded-block results](cardinal_blocks/qwen3_359m_step19500_expanded_v2/report.md) | [Intervals](cardinal_blocks/qwen3_359m_step19500_expanded_v2/confidence_intervals.md) |
| Selected six-category situation composite | [Balanced accuracy: 51.22%](spatial_benchmark/qwen3_359m_step19500_v1/report.md) | Conditional scene-cluster bootstrap; limitations in report |

These scores use different evidence and different averaging rules. They are
diagnostic summaries; only the explicitly defined six-category composite
combines selected situation runs with equal category/family/evidence weighting. Repeated
names, objects, wordings, and numbers are related examples rather than
independent situations. The matched-situation and movement means exclude
their direct-label controls.

## Current generated probes versus evaluated versions

The current [left/right v1.2](../left_right_situation_probe/README.md) shortens
compact/standard/expanded contexts to averages of 25/29/36 words while preserving
all 2,400 pairs. The results above still describe the longer **v1.0** wording.
The [front/behind v2.0](../front_behind_situation_probe/README.md) set has 224
compact pairs; the discarded detailed front/behind versions are not published.
Neither these front/behind examples nor the revised left/right wording has been
evaluated or substituted into the saved composite score.

The [composite method and input manifest](../spatial_benchmark/README.md) document
the scene clusters, controls excluded, and uncertainty limitations. Recomputing
it requires the original local per-item runs; compact snapshots alone are not
enough. Absolute checkpoint paths in provenance are identifiers, not bundled
weights or portable download locations.

Separate [30-seed initialization estimates](initialization_distributions/llama_qwen3_30seeds_v1/report.md)
and the [20-versus-30-seed comparison](initialization_distributions/llama_qwen3_30seeds_v1/sample_comparison.md)
describe untrained-model output distributions, not spatial benchmark scores.
