# Above/below definition and synonym probe

Qwen3 step-19,500 checkpoint; binary choice between two full target sentences, scored by mean target-token log likelihood. Each pair includes an above-compatible and a below-compatible context. Exact ties count as incorrect.

| Probe family | Direction | Pairs | Choice accuracy | Above gold | Below gold | Both correct | Above choice share | Context sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| literal_definition | word_to_definition | 24 | 56.25% | 100.00% | 12.50% | 12.50% | 93.75% | 83.33% |
| literal_definition | definition_to_word | 8 | 43.75% | 87.50% | 0.00% | 0.00% | 93.75% | 50.00% |
| lexical_synonym | word_to_synonym | 6 | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 91.67% |
| lexical_synonym | synonym_to_word | 2 | 50.00% | 100.00% | 0.00% | 0.00% | 100.00% | 75.00% |

Direction-balanced choice accuracy averages the two mapping directions within each family. The over/under lexical-synonym rows are separate because those words can suggest vertical alignment. Repeated stems and structures are related stimuli, not independent semantic cases.

See `bias_table.md` for constant-answer and paired-context behavior; `grouped_scores.csv` and `bias_table.csv` include phrase-pair and structure splits.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
