# Above versus below: target-only likelihoods

Same Qwen3 checkpoint and float32/eager scoring as the definition probe. Higher log likelihood means more preferred. BOS-only targets are scored exactly as written, without an added leading space or EOS. Full-sentence means match the probe's primary per-target-token reduction; full-sentence sums are also saved in `word_priors.json`.

| Scoring condition | Above log likelihood | Below log likelihood | Above minus below | Target tokens (above/below) |
|---|---:|---:|---:|---:|
| Bare word from BOS | -13.4271 | -12.7529 | -0.6742 | 1/1 |
| After ‘One object is’ | -7.0143 | -7.6206 | +0.6062 | 1/1 |
| After ‘The first object is’ | -8.2569 | -8.6063 | +0.3493 | 1/1 |
| After ‘The second object is’ | -8.0077 | -7.7509 | -0.2568 | 1/1 |
| After ‘The first item is’ | -9.4815 | -9.5452 | +0.0637 | 1/1 |
| After ‘The second item is’ | -9.0207 | -8.1020 | -0.9187 | 1/1 |

The bare-word row is a no-context prior. Prefix rows score the word after a neutral sentence fragment and are not BOS-only priors. A preference in either setting can help explain response bias, but neither by itself measures whether the model uses the above/below context correctly.

## Full sentence targets from the probe

| Above target | Below target | Pairs using form | Above mean | Below mean | Above minus below |
|---|---|---:|---:|---:|---:|
| One object is above another. | One object is below another. | 5 | -4.6458 | -5.2038 | +0.5581 |
| The first object is above the second. | The first object is below the second. | 5 | -3.9158 | -4.1180 | +0.2022 |
| The first object is above the second object. | The first object is below the second object. | 20 | -3.8057 | -3.9901 | +0.1845 |
| The first item is above the second item. | The first item is below the second item. | 20 | -4.0128 | -3.9910 | -0.0218 |

Sentence-level prior gaps include all words and punctuation. They should be compared to the conditional full-target scores in `item_scores.csv`, rather than treated as the probability of the isolated relation word.
