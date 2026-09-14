# Closer/farther applied movement score

Each accuracy is a binary choice between two full targets, scored by mean target-token log likelihood. The raw score is primary; PMI is secondary. Uniform-choice chance is 50%.
The overall mean gives equal weight to three comparison families. Within a family, it averages matched case judgments, then wording lengths and available entity orders, numeric and nonnumeric evidence, and finally the family's contrasts. Direct labels and literal definitions are separate controls.

| Comparison family | Contrasts | Pairs | Raw accuracy | PMI accuracy | Both contexts correct |
|---|---|---:|---:|---:|---:|
| Person moves; reference stationary | closer_vs_farther | 1,332 | 49.38% | 46.18% | 0.04% |
| Reference object moves | closer_vs_farther | 2,664 | 47.54% | 48.38% | 0.54% |
| Reference object moves vs both move (cross-family) | closer_vs_unchanged, farther_vs_unchanged | 5,328 | 50.00% | 49.93% | 0.00% |
| **Overall applied mean** | Equal family weight | 9,324 | **48.97%** | 48.16% | 0.19% |

The cross-family row averages closer-versus-unchanged and farther-versus-unchanged equally. Its two contexts differ in physical event and syntax; it does not isolate a within-family unchanged judgment. The person-moving row uses the previously evaluated matched numeric/nonnumeric probe and has no matched entity-order variants. Numeric and nonnumeric evidence receive equal weight despite very different row counts.

## Contrast details

| Family | Contrast | Evidence | Pairs | Raw accuracy | Closer/farther-side accuracy | Other-side accuracy | First answer chosen |
|---|---|---|---:|---:|---:|---:|---:|
| Person moves; reference stationary | closer_vs_farther | absent | 36 | 50.00% | 100.00% | 0.00% | 100.00% |
| Person moves; reference stationary | closer_vs_farther | specific | 1,296 | 48.77% | 95.99% | 1.54% | 97.22% |
| Reference object moves | closer_vs_farther | absent | 72 | 46.53% | 87.50% | 5.56% | 90.97% |
| Reference object moves | closer_vs_farther | specific | 2,592 | 48.55% | 78.01% | 19.10% | 79.46% |
| Reference object moves vs both move (cross-family) | closer_vs_unchanged | absent | 72 | 50.00% | 0.00% | 100.00% | 0.00% |
| Reference object moves vs both move (cross-family) | closer_vs_unchanged | specific | 2,592 | 50.00% | 0.00% | 100.00% | 0.00% |
| Reference object moves vs both move (cross-family) | farther_vs_unchanged | absent | 72 | 50.00% | 0.00% | 100.00% | 0.00% |
| Reference object moves vs both move (cross-family) | farther_vs_unchanged | specific | 2,592 | 50.00% | 0.00% | 100.00% | 0.00% |

## Controls and scope

Direct comparative labels: 82.64% raw, 67.36% PMI across the three binary contrasts. They are excluded from the applied mean.
Literal close/far definitions (separate categorical probe): 84.26% word-to-definition and 50.00% definition-to-word; equal direction mean 67.13%. This is excluded from the closer/farther movement mean.
The three family rows are diagnostic categories, not independent random samples: names, objects, numbers, units, lengths, and entity orders reuse underlying cases. Inspect `overall_applied_cells.csv`, `overall_applied_pairs.csv`, and `entity_order_summary.csv` for failures hidden by the mean.
