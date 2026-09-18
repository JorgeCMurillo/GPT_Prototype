# Explicit summary bridge: direct-label controls

Appended **To summarize the positions:** to each context, keeping both candidate answers unchanged.

Same Qwen3 359M step-19,500 checkpoint and raw mean full-target likelihood scoring as the baseline; no PMI. The bridge is conditioning text and is not included in the scored target.

| Entity order | Pairs | Baseline accuracy | Bridge accuracy | Baseline both correct | Bridge both correct |
|---|---:|---:|---:|---:|---:|
| same_entity_order | 48 | 4.17% | 97.92% | 0.00% | 95.83% |
| reversed_entity_order | 48 | 100.00% | 44.79% | 100.00% | 0.00% |
| all | 96 | 52.08% | 71.35% | 50.00% | 47.92% |

This is a matched prompt-format diagnostic on 96 direct-label pairs, not a rerun of the physical-situation families. A bridge effect supports sensitivity to how restatement is prompted; it does not uniquely establish repetition as the mechanism. The bridge also changes length and discourse cues.

saved token means and choices reconstructed successfully
