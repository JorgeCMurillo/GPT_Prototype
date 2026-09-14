# Fixed-target context sensitivity

Qwen3 359M at 19.5k steps; mean full-target log likelihood, reusing the saved scores.
For each target, compare its likelihood in the matching context with its likelihood in the opposite-outcome context. Each contrast has 2,592 matched pairs and 5,184 target judgments. Accuracy averages the two target judgments; both-correct requires success on both.

| Contrast | Compact | Standard | Expanded | Overall | Both targets correct |
|---|---:|---:|---:|---:|---:|
| closer_vs_farther | 50.52% | 33.22% | 42.65% | 42.13% | 6.44% |
| closer_vs_unchanged | 69.56% | 53.94% | 58.51% | 60.67% | 21.99% |
| farther_vs_unchanged | 68.17% | 49.83% | 51.74% | 56.58% | 15.24% |

| Contrast | Description family | Accuracy | Target A accuracy | Target B accuracy | Both correct |
|---|---|---:|---:|---:|---:|
| closer_vs_farther | explicit | 49.07% | 67.98% | 30.17% | 10.11% |
| closer_vs_farther | situation | 35.19% | 21.91% | 48.46% | 2.78% |
| closer_vs_unchanged | explicit | 59.34% | 30.17% | 88.50% | 19.75% |
| closer_vs_unchanged | situation | 62.00% | 54.78% | 69.21% | 24.23% |
| farther_vs_unchanged | explicit | 59.68% | 31.48% | 87.89% | 19.75% |
| farther_vs_unchanged | situation | 53.47% | 47.22% | 59.72% | 10.73% |

Target A/B follow the contrast name: closer/farther, closer/unchanged, farther/unchanged.
Subtracting a fixed target-only PMI baseline cancels when comparing that same target across contexts. This invariance was verified for every margin.
A correct result here can coexist with wrong target choice within a context. Requiring both target judgments to be correct helps reveal whether both targets merely prefer the same context.
Situation comparisons involving unchanged distance match movement descriptions against orientation descriptions, so they also change evidence type and sentence structure. Their scores do not isolate distance reasoning alone. Explicit comparisons hold the template fixed while varying the endpoint.
The same scenes, targets, and contexts recur across contrasts and wording variants; these judgments are not independent samples. Exact ties count as incorrect.
