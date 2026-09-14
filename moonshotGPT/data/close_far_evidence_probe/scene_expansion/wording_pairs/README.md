# Matched close/far wording controls

| Family | Close context | Far context |
| --- | --- | --- |
| Direct labels | The ball is close to Maya. | The ball is far from Maya. |
| Proximity synonyms | The ball is near Maya. | The ball is distant from Maya. |
| Short/long distance | The ball and Maya are separated by a short distance. | The ball and Maya are separated by a long distance. |
| Small/large distance | The ball and Maya are separated by a small distance. | The ball and Maya are separated by a large distance. |

Names and objects are substituted algorithmically. Each family includes three context structures and both person-first and object-first targets. Targets always express close versus far, with periods. Distance constructions use coordinated entities or a distance subject; metadata records this syntax change instead of treating it as a pure word-order manipulation.

## Controlled comparisons

- **Original baseline:** 672 pairs from the seven main placement scenes, excluding footwear controls and shared lexical controls.
- **Close-only substitutions:** 672 pairs per family, 2,688 total. Replace only C1; retain the original across-space C2 and both targets. Scene eligibility and the far setting remain fixed.
- **Symmetric wording pairs:** 240 pairs per family, 960 total. Replace both contexts with corresponding close/far expressions. Generic contexts are stored once per name, object, context structure, target order, and family, rather than repeated for each source scene.

Total: **4,320 pairs / 8,640 context judgments**, including the baseline. These are controlled repeated measurements, not independent scenes. Report modes and families separately rather than pooling them into one diagnostic score.

The 6,288 comparison links hold targets fixed and cover baseline versus close-only, close-only versus symmetric, short versus small with C2 fixed, and short/long versus small/large. The generator checks these invariants, IDs, counts, and deduplication. It records target/context repetition and tokenizer lengths. Length is measured, not matched across all wording families.

## Files and status

- [components.json](components.json): editable families and structure templates.
- [generated/paired_overview.csv](generated/paired_overview.csv): four concise Maya/ball examples.
- [generated/review_examples.csv](generated/review_examples.csv): examples across structures, target orders, and modes.
- `generated/probes.csv` / `.jsonl`: all examples and diagnostic labels.
- `generated/matches.csv`: links for controlled comparisons.
- `generated/manifest.json`: counts, validation, input hashes, and tokenizer information.

Evaluated on **Qwen3 359M at 19.5k steps**, using mean target-token log likelihood. The parent evaluator now groups by `mode` and `wording_family_id`, separately reporting scene-based close-only rows and deduplicated symmetric wording controls. Generation manifests retain generation-time status; evaluation snapshots and results are stored separately.

Results: `runs/research/bos_aligned_proto/close_far_evidence_probe/qwen3_359m_step19500_wording_pairs_v1/` from the repository root.

| Symmetric family | Choice | PMI choice | Context sensitivity | Close accuracy | Far accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| close/far | 97.08% | 85.62% | 97.71% | 94.17% | 100.00% |
| near/distant | 55.42% | 50.62% | 49.79% | 42.08% | 68.75% |
| short/long distance | 65.00% | 56.46% | 74.17% | 42.92% | 87.08% |
| small/large distance | 60.83% | 54.79% | 78.75% | 38.75% | 82.92% |

Each row contains 240 pairs / 480 judgments. All 4,320 item scores were checked against saved token means and target priors; grouped accuracies were independently reconstructed. All 672 baseline rows reproduce the prior scene evaluation's raw, PMI, and context-sensitivity decisions. No exact ties occurred in the symmetric families.

Regenerate from the repository root:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/close_far_evidence_probe/scene_expansion/wording_pairs/generate.py
```
