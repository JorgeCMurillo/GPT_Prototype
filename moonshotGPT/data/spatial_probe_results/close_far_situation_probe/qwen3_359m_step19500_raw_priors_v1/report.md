# Qwen3 step 19,500: close/far priors without a descriptive spatial context

Scored on GPU 1 using the same local checkpoint, float32/eager attention, and the repository EWoK likelihood routines. Logarithms are natural logs (nats), and higher values are more likely. Mean target-token log likelihood remains the primary convention. BOS token 50256 is prepended; no EOS is appended.

| Scoring condition | Close | Far | Mean difference, close minus far |
|---|---:|---:|---:|
| Bare word after BOS | -11.972879 | -12.298231 | +0.325352 |
| Leading-space word after BOS | -14.762218 | -14.867821 | +0.105602 |
| Bare word plus period after BOS | -8.110636 | -8.594584 | +0.483948 |
| Full sentence: The objects are close/far. | -5.780874 | -6.520782 | +0.739909 |
| Word only after The objects are | -7.429005 | -7.427178 | -0.001826 |

Bare close/far and their leading-space variants are single tokens. Thus their mean log likelihood equals their raw token log probability. Bare "close" is about 1.38 times as likely as bare "far" after BOS. Leading-space close is about 1.11 times as likely as leading-space far after BOS. Those document-start priors do not restrict "close" to its spatial meaning.

## Full-sentence preference is dominated by the period

Both full target sentences contain five tokens. The initial three tokens (The, objects, are) have identical likelihoods. The remaining token log probabilities from the full-sentence evaluation are:

| Token position | Close sentence | Far sentence |
|---|---:|---:|
| close / far | -7.429004 | -7.427175 |
| final period | -5.282255 | -8.983629 |

The adjective probabilities are nearly equal, with a tiny preference for far. The period is much more likely after close. This explains essentially the entire no-description sentence-level preference. The sequence-likelihood ratio of complete close/far sentences is about 40.43, calculated from the summed log probabilities; exponentiating the mean difference alone would not give that sequence ratio.

This result distinguishes a small document-start lexical preference from a substantial preference for these complete target sentences. It does not by itself establish how much punctuation contributes under each spatial context or explain all placement failures. A contextual token-level analysis can check that separately without changing the primary mean-token metric.

`scores.json` contains all texts, token IDs as token strings, per-token log probabilities, means, sums, ratios, BOS ID, and checkpoint path. `data/close_far_situation_probe/score_priors.py` reproduces the measurement. No training was performed.
