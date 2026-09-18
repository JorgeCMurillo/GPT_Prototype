# Initialization sample extension

Extended 20 to 30 seeds, preserving all original samples.

| Model (BOS only) | 20-seed central 90% | 30-seed central 90% | Spread ratio (30 / 20) |
|---|---:|---:|---:|
| qwen3 | 0.763–1.283 × uniform | 0.803–1.228 × uniform | 0.817 |
| llama | 0.765–1.282 × uniform | 0.805–1.227 × uniform | 0.819 |

For independent seed noise around equal expectations, standard deviation scales approximately as 1/sqrt(n), predicting a spread ratio of 0.816. This comparison is descriptive; the samples overlap and are not independent experiments. It does not prove equal expectations for every token.

Original seed records unchanged; all average distributions finite and normalized.
