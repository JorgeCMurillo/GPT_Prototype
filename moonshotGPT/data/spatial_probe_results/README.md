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

These scores use different evidence and different averaging rules. They are
diagnostic summaries, not a single cross-concept benchmark ranking. Repeated
names, objects, wordings, and numbers are related examples rather than
independent situations. The matched-situation and movement means exclude
their direct-label controls.
