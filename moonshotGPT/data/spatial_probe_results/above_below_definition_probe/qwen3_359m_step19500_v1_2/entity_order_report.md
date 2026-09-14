# Matched entity-order diagnostics

Each match holds the phrase pair, noun, target sentences, and clause structure fixed. The second-subject context inverts the relational wording—for example, ‘second item is lower than first item’ corresponds to ‘first item is above second item.’ This is an inverse-relation test, not an order-only syntax manipulation.

| Probe family | Matched pairs | First-subject accuracy | Second-subject accuracy | Both contexts correct in both orders | Any prediction flip |
|---|---:|---:|---:|---:|---:|
| literal_definition | 16 | 53.12% | 78.12% | 6.25% | 50.00% |
| lexical_synonym | 4 | 62.50% | 12.50% | 0.00% | 75.00% |

`entity_order_pair_scores.csv` contains each matched contrast and separate above- and below-context prediction flips. The matched variants reuse phrase pairs and nouns; percentages describe this probe set rather than independent semantic cases.
