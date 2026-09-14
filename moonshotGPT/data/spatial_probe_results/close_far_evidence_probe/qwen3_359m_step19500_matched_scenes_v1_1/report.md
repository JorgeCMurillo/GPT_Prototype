# Close/far: matched-scene evaluation

Qwen3 359M at 19.5k steps. Mean full-target token log likelihood, including punctuation. Each condition has 384 pairs / 768 balanced context judgments.

| Evidence condition | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy | Both correct: choice | Both correct: context |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_label | 97.66% | 85.68% | 97.14% | 95.31% | 100.00% | 95.31% | 94.27% |
| distance_phrase | 68.88% | 61.46% | 70.83% | 73.96% | 63.80% | 37.76% | 41.67% |
| distance_phrase_object_first | 66.28% | 61.72% | 64.84% | 83.33% | 49.22% | 32.55% | 29.69% |
| endpoint_placement | 48.70% | 52.99% | 49.22% | 95.31% | 2.08% | 0.52% | 5.73% |

## Distance phrase order by target order

| Context order | Target order | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |
|---|---|---:|---:|---:|---:|---:|
| person_first | object_first | 70.05% | 65.62% | 73.70% | 83.85% | 56.25% |
| person_first | person_first | 67.71% | 57.29% | 67.97% | 64.06% | 71.35% |
| object_first | object_first | 69.27% | 67.45% | 67.19% | 88.54% | 50.00% |
| object_first | person_first | 63.28% | 55.99% | 62.50% | 78.12% | 48.44% |

The two distance contexts have the same word multiset apart from order and capitalization. Both target orders are evaluated under each context order. Direct-label and endpoint-placement contexts differ in syntax and length from the distance phrases.
Settings, names, objects, target orders, and templates are crossed repeated measurements rather than independent scenes. Choice compares the two targets within a context; context sensitivity compares one target across close and far contexts. Target-only PMI cancels from context sensitivity. Exact ties count as incorrect.

See [answer-preference diagnostics](bias_table.md) for per-answer choice and paired-context behavior.
