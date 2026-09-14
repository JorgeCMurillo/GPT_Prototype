# Above/below definition and synonym probe

Qwen3 step-19,500 checkpoint; binary choice between two full target sentences, scored by mean target-token log likelihood. Each pair includes an above-compatible and a below-compatible context. Exact ties count as incorrect.

| Probe family | Direction | Pairs | Choice accuracy | Above gold | Below gold | Both correct | Above choice share | Context sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| literal_definition | word_to_definition | 24 | 56.25% | 100.00% | 12.50% | 12.50% | 93.75% | 83.33% |
| literal_definition | definition_to_word | 40 | 61.25% | 97.50% | 25.00% | 25.00% | 86.25% | 63.75% |
| lexical_synonym | word_to_synonym | 6 | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 91.67% |
| lexical_synonym | synonym_to_word | 10 | 40.00% | 70.00% | 10.00% | 10.00% | 80.00% | 55.00% |

Direction-balanced choice accuracy averages the two mapping directions within each family. The over/under lexical-synonym rows are separate because those words can suggest vertical alignment. The entity-order extension adds cases where the second entity is the subject; these invert the relational phrase while keeping the first entity as the target subject. Repeated stems, nouns, and structures are related stimuli, not independent semantic cases.

See `bias_table.md` for constant-answer and paired-context behavior; `grouped_scores.csv` and `bias_table.csv` include phrase-pair, noun, structure, and entity-order splits.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.

See [matched entity-order diagnostics](entity_order_report.md) for the first- versus second-subject comparison.

See [above/below preference table](above_below_bias.md) for explicit upper- and lower-answer counts.

See [above/below word priors](word_priors.md) for BOS-only and prefix-conditioned likelihoods.
