# Close/far matched-scene applied score

Qwen3 359M at step 19,500. Each choice compares exactly two full close/far target sentences using mean target-token log likelihood. Raw choice is primary; PMI and fixed-target context sensitivity are secondary.
The distance family averages its two context entity orders and two target orders within each scene. Endpoint placement averages its two target orders. The 192 scenes are averaged within each family, then the two applied families receive equal weight. Direct labels are a separate control.

| Evidence family | Probe rows | Raw choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |
|---|---:|---:|---:|---:|---:|---:|
| distance_phrase | 768 | 67.58% | 61.59% | 67.84% | 78.65% | 56.51% |
| endpoint_placement | 384 | 48.70% | 52.99% | 49.22% | 95.31% | 2.08% |
| direct_label_control | 384 | 97.66% | 85.68% | 97.14% | 95.31% | 100.00% |
| **Matched-scene applied mean** | 1,152 | **58.14%** | 57.29% | 58.53% | 86.98% | 29.30% |

The direct-label row is excluded from the applied mean. The separate physical-arrangement, synonym, and literal-definition probes are also excluded, so this is not yet a full close/far benchmark composite. The distance and endpoint families differ in evidence and difficulty, and the repeated settings, entities, and target orders are not independent scene samples.
