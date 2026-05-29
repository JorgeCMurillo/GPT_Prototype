# EWoK Query Specs

This directory contains JSON filter specs for slicing the EWoK query set used
by attribution runs.

Pass one of these files to:

```bash
--ewok_filter_spec /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/<spec>.json
```

The filter language is conjunctive across fields: if a spec lists both
`domains` and `context_types`, a row must satisfy both constraints to be kept.

## Supported Keys

```json
{
  "name": "optional-short-name",
  "description": "optional human-readable description",
  "variant": "fast",
  "domains": ["social-relations"],
  "context_types": ["indirect"],
  "context_diffs": ["negation"],
  "target_diffs": ["concept swap"],
  "row_indices": [17, 42]
}
```

Singular aliases such as `domain`, `context_type`, `context_diff`, and
`target_diff` are also accepted.

## Normalized Values

The current fast EWoK split has 1,100 rows and the following normalized filter
values:

- `domains`
  `agent-properties`, `material-dynamics`, `material-properties`,
  `physical-dynamics`, `physical-interactions`, `physical-relations`,
  `quantitative-properties`, `social-interactions`, `social-properties`,
  `social-relations`, `spatial-relations`
- `context_types`
  `direct`, `indirect`
- `context_diffs`
  `active-passive`, `antonym`, `material`, `negation`, `number`, `other`,
  `variable swap`
- `target_diffs`
  `concept swap`, `variable swap`

One normalization detail matters: raw `ContextDiff="variable_swap"` is treated
as `variable swap` by the filter layer so specs can use one consistent label.

## Catalog

See [fast_catalog.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/fast_catalog.json) for counts by category on the current fast split.

## Example Specs

- [all_fast.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/all_fast.json)
- [domain_social_relations.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/domain_social_relations.json)
- [domain_material_dynamics.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/domain_material_dynamics.json)
- [context_type_indirect.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/context_type_indirect.json)
- [context_diff_negation.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/context_diff_negation.json)
- [target_diff_variable_swap.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/target_diff_variable_swap.json)
- [social_relations_indirect_variable_swap.json](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/social_relations_indirect_variable_swap.json)
