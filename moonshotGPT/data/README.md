# Spatial-relation probes

This directory contains the controlled spatial probes developed for the Qwen3
step-19,500 analysis. Each probe has a generator, component tables, generated
examples, an evaluator, and a README describing its design. The primary model
metric is binary choice accuracy between two complete target sentences, scored
by mean target-token log likelihood. PMI and context sensitivity are reported
separately where available.

| Concept | Probe directories | What they vary |
|---|---|---|
| Close / far | [definition](close_far_definition_probe/README.md), [situations](close_far_situation_probe/README.md), [matched evidence](close_far_evidence_probe/README.md) | Literal meaning, physical placement, distance wording, entity order, and direct labels |
| Closer / farther | [movement events](closer_farther_probe/README.md) | Numeric and nonnumeric evidence, moving entity, unchanged distance, wording length, and entity order |
| Above / below | [definition and synonyms](above_below_definition_probe/README.md), [situations](above_below_situation_probe/README.md) | Height paraphrases, static placement, target/reference movement, movement without crossing, and entity order |

The [bias reporter](spatial_bias_report/README.md) creates a common table of
answer preferences and matched-context behavior. [Compact result snapshots](spatial_probe_results/README.md)
preserve the reports and diagnostic tables from the local Qwen3 evaluation.
The model checkpoint and full token-level run outputs are not part of this
repository.

Run generators from the `moonshotGPT` directory, for example:

```bash
python data/above_below_definition_probe/generate.py
```
