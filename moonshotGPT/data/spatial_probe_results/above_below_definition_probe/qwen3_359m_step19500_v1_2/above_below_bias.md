# Above/below answer preference

Each probe has one upper-compatible and one lower-compatible context, and both targets are scored after each context. Gold answers are balanced 50/50. ‘Upper chosen’ measures the model's response preference; accuracy measures whether the chosen answer matches the context. ‘Same upper answer twice’ counts pairs in which the same answer wins in both contexts. In word-to-definition rows the targets are height descriptions, not the literal words *above* and *below*.

## Literal height definitions

| Set | Pairs | Accuracy | Upper chosen | Lower chosen | Ties | Upper gold correct | Lower gold correct | Both contexts correct | Same upper answer twice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All literal pairs (row-weighted) | 64 | 59.38% | 114/128 (89.06%) | 14/128 (10.94%) | 0 | 98.44% | 20.31% | 20.31% | 78.12% |
| Above/below → height definition | 24 | 56.25% | 45/48 (93.75%) | 3/48 (6.25%) | 0 | 100.00% | 12.50% | 12.50% | 87.50% |
| Height definition → above/below | 40 | 61.25% | 69/80 (86.25%) | 11/80 (13.75%) | 0 | 97.50% | 25.00% | 25.00% | 72.50% |

Direction-balanced accuracy: 58.75%. The row-weighted accuracy above differs because there are 24 forward and 40 reverse pairs. Both measures exclude over/under.

## Over/under synonyms

The upper target is *over* in word-to-synonym rows and *above* in synonym-to-word rows. These are lexical mappings and remain separate from height definitions.

| Set | Pairs | Accuracy | Upper chosen | Lower chosen | Ties | Upper gold correct | Lower gold correct | Both contexts correct | Same upper answer twice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All synonym pairs (row-weighted) | 16 | 62.50% | 22/32 (68.75%) | 10/32 (31.25%) | 0 | 81.25% | 43.75% | 43.75% | 37.50% |
| Above/below → over/under | 6 | 100.00% | 6/12 (50.00%) | 6/12 (50.00%) | 0 | 100.00% | 100.00% | 100.00% | 0.00% |
| Over/under → above/below | 10 | 40.00% | 16/20 (80.00%) | 4/20 (20.00%) | 0 | 70.00% | 10.00% | 10.00% | 60.00% |

Direction-balanced synonym accuracy: 70.00%.

## Height-phrase sensitivity

| Set | Pairs | Accuracy | Upper chosen | Lower chosen | Ties | Upper gold correct | Lower gold correct | Both contexts correct | Same upper answer twice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vertically higher / lower | 16 | 56.25% | 30/32 (93.75%) | 2/32 (6.25%) | 0 | 100.00% | 12.50% | 12.50% | 87.50% |
| greater / lesser height | 16 | 62.50% | 28/32 (87.50%) | 4/32 (12.50%) | 0 | 100.00% | 25.00% | 25.00% | 75.00% |
| higher up / lower down | 16 | 56.25% | 30/32 (93.75%) | 2/32 (6.25%) | 0 | 100.00% | 12.50% | 12.50% | 87.50% |
| higher / lower | 16 | 62.50% | 26/32 (81.25%) | 6/32 (18.75%) | 0 | 93.75% | 31.25% | 31.25% | 62.50% |

## Reverse-context entity order

These rows compare first-subject and second-subject contexts within the added inverse-relation block. The noun, phrase pair, clause structure, and target sentences are matched; the relational wording also inverts.

| Set | Pairs | Accuracy | Upper chosen | Lower chosen | Ties | Upper gold correct | Lower gold correct | Both contexts correct | Same upper answer twice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first subject | 16 | 53.12% | 31/32 (96.88%) | 1/32 (3.12%) | 0 | 100.00% | 6.25% | 6.25% | 93.75% |
| second subject | 16 | 78.12% | 23/32 (71.88%) | 9/32 (28.12%) | 0 | 100.00% | 56.25% | 56.25% | 43.75% |

The phrase and order cells reuse names, templates, and targets, so their counts are repeated diagnostic variants rather than independent semantic cases. See `bias_table.csv` for every split and `entity_order_report.md` for direct matched-order flips.
