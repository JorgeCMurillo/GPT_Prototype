# Answer-preference diagnostics

Each pair has one context supporting the first answer and one supporting the second. Scores compare exactly two full target sentences using the run's mean target-token likelihoods. The first-choice fraction is a response preference across balanced gold contexts, not an accuracy measure.
`Always first` and `always second` mean the model selects the same answer in both matched contexts. `Both wrong` means it reverses both answers. Exact ties are shown separately. Repeated names, objects, and wording variants are not independent semantic cases.

| Group | Answers | Pairs | Accuracy | First gold | Second gold | Chosen first / second | Both correct | Always first | Always second | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| literal_definition / word_to_definition | above definition / below definition | 24 | 56.25% | 100.00% | 12.50% | 93.75% / 6.25% | 12.50% | 87.50% | 0.00% | 0.00% |
| literal_definition / definition_to_word | above / below | 40 | 61.25% | 97.50% | 25.00% | 86.25% / 13.75% | 25.00% | 72.50% | 0.00% | 2.50% |
| lexical_synonym / word_to_synonym | over / under | 6 | 100.00% | 100.00% | 100.00% | 50.00% / 50.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| lexical_synonym / synonym_to_word | above / below | 10 | 40.00% | 70.00% | 10.00% | 80.00% / 20.00% | 10.00% | 60.00% | 0.00% | 30.00% |

`bias_table.csv` contains the same metrics for finer groupings and, when available, PMI choice and fixed-target context sensitivity. PMI is a separate target-prior adjustment; it does not turn a response preference into evidence of correct situational reasoning. These tables describe this report's stimulus set and are not pooled into the benchmark headline score.
