# Closer versus farther: matched numeric and nonnumeric wording, v1.3

This extension contains **280 nonnumeric contexts / 140 binary pairs**, crossing **4 names × 5 objects × 7 wording forms × 2 directions**. Names are Maya, Jesse, Li, and Omar. Objects are cone, flag, bicycle, **bench**, and **mailbox**. The two additional objects are local to this extension; the original evaluated numeric dataset is preserved.

Three aligned wording lengths contribute 120 nonnumeric contexts; four short forms contribute 160 more. The extension also contains **4,320 aligned numeric contexts / 2,160 binary pairs**, crossing the three aligned lengths with the original 12 numeric cases and 3 units. The previous v1.2 set of 200 nonnumeric contexts and its numeric counterparts were evaluated on Qwen3 19.5k; see the [saved report](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_non_numeric_v1_2/report.md). The generated manifest describes generation only; evaluation metadata and snapshots are in the run directory. Version 1.3 adds 80 unevaluated contexts using a matched stationary/still relative clause. All 200 previously evaluated nonnumeric context texts, targets, and IDs are preserved. Existing scores describe v1.2, not the expanded v1.3 set. Row indices may change when forms are added; join preserved examples by probe ID.

## Minimal forms

| Form | Closer context | Farther context |
|---|---|---|
| Minimal | Maya walked toward the cone. | Maya walked away from the cone. |
| Stationary reference | Maya walked toward the stationary cone. | Maya walked away from the stationary cone. |
| Stationary clause | Maya walked toward the cone, which remained stationary. | Maya walked away from the cone, which remained stationary. |
| Still clause | Maya walked toward the cone, which remained still. | Maya walked away from the cone, which remained still. |

Each form contributes 40 contexts / 20 pairs, spanning all names and objects. Targets are identical to those in the aligned versions below. The nearer/farther labels use the ordinary interpretation of these movements. The unqualified form assumes a stationary reference; adding `stationary` states that assumption explicitly. All four short forms leave the endpoint and no-overshoot condition unstated. The clause versions make stationarity explicit; the word still denotes lack of movement in this construction. They therefore test a less explicit description rather than just sentence length.

`reference_stationarity`, `no_overshoot_status`, and `interpretation_basis` distinguish stated constraints from assumptions. `object_stationary` is null in the unqualified form; `passes_object` is null in all four short forms, because the text does not explicitly establish these facts. `template_family` separates minimal from aligned forms. `length_band` uses `minimal`, `minimal_stationary`, `clause_stationary`, and `clause_still` for the short forms; they do not enter the original compact/standard/expanded length matches.

`minimal_control_matches.csv` contains 280 comparisons: 160 short-to-aligned-compact comparisons, 40 adding the stationary adjective, 40 moving stationary into a relative clause, and 40 swapping stationary for still in an otherwise identical clause. The stationary/still comparison changes exactly one word and preserves word count. Short-to-aligned comparisons change multiple wording features and the explicitness of movement constraints. Minimal forms are generated once per entity/direction; they are not duplicated across unused numeric cases or units.

## What changes between numeric and nonnumeric versions

Both versions use the same template. Only two slots change: the initial separation and movement distance. Each numeric value plus unit becomes `some distance`.

| Version | Standard closer example |
|---|---|
| Numeric | Maya was **32 ft** from a stationary cone. Maya walked **23 ft** directly toward the cone without reaching or passing it. |
| Nonnumeric | Maya was **some distance** from a stationary cone. Maya walked **some distance** directly toward the cone without reaching or passing it. |

The farther context replaces only `toward` with `away from` in both aligned versions. These aligned conditions explicitly keep the object stationary and rule out reaching or passing it. The shared trajectory clause includes negation. It is identical across the numeric/nonnumeric and closer/farther contrasts.

These are the three nonnumeric wording forms:

| Length | Closer context |
|---|---|
| Compact | Starting some distance from a stationary cone, Maya walked some distance directly toward it without reaching or passing it. |
| Standard | Maya was some distance from a stationary cone. Maya walked some distance directly toward the cone without reaching or passing it. |
| Expanded | Initially, Maya was some distance from a cone that remained stationary. From that starting position, Maya walked some distance directly toward the cone without reaching or passing it. |

Targets remain identical across all versions for each name/object combination:

- Target1: Maya is now closer to the cone than before.
- Target2: Maya is now farther from the cone than before.

The aligned numeric contexts preserve the original numeric movement wording and append the same no-reaching/no-passing clause. For the original three objects, the generator validates this relationship exactly. **Previously saved numeric scores cannot be reused for these amended contexts.** New objects also require new inference.

## Interpretation

This design controls the surrounding wording much more closely than v1.0. It tests sensitivity to specific distance expressions versus an unspecified distance phrase. Tokenization, physical specificity, and the meaning of those distance expressions still differ; it is not a test of digits alone. Word/token counts are stored for measurement rather than assumed equal.

Both numeric and nonnumeric movement examples can be solved from the toward/away relation without computing the final distance. An improvement without specific numbers would not by itself establish an arithmetic deficit. The task is comparative change in distance, not a categorical close/far threshold. Local EWoK close/far cases also use nonnumeric evidence, but test a different relation.

## Matches and scoring

Each of the 120 aligned nonnumeric contexts has 36 aligned numeric variants (12 numeric cases × 3 units), linked in `numeric_matches.csv`. These links do not create 4,320 independent nonnumeric observations. The 160 short contexts have no numeric counterparts. The numeric case seed, unit conventions, names, and original objects are inherited from parent components/generated cases; the additional objects and shared templates are in this extension's components.

`template_id`, `length_band`, `evidence_type`, `numeric_information`, `trajectory_constraint_id`, name/object IDs, and actual word/character/token counts support diagnosis. Each numeric context normalizes exactly to its nonnumeric counterpart when both numeric distance expressions are replaced with `some distance`. Compact/standard/expanded also vary syntax, so differences among lengths do not isolate length alone.

Use binary full-target **mean token log likelihood** as the primary score. For each paired row, evaluate all four Context1/Context2 × Target1/Target2 combinations. Report target-choice accuracy, per-outcome accuracy, both-context correctness, fixed-target context sensitivity, both-target correctness, and results by wording length and entity. Label mean-token PMI choice scores separately. Fixed-target PMI subtraction cancels for context sensitivity.

`probes` files are context rows with two targets and a correct target; `pairs` files are EWoK-style paired rows with closer mapped to Context1/Target1 and farther to Context2/Target2. They are different views of the same examples. The parent three-way evaluator needs adaptation for these binary datasets.

## Generated files

- `probes.csv` / `.jsonl`: 280 nonnumeric contexts.
- `pairs.csv` / `.jsonl`: 140 nonnumeric pairs.
- `numeric_aligned_probes.csv` / `.jsonl`: 4,320 matched numeric contexts.
- `numeric_aligned_pairs.csv` / `.jsonl`: 2,160 matched numeric pairs.
- `numeric_matches.csv`: 4,320 exact distance-slot substitution links. `original_numeric_probe_id` refers to a previous example before adding the shared clause, or is blank for a new object.
- `length_matches.csv`: 80 nonnumeric wording comparisons against the standard reference.
- `minimal_control_matches.csv`: 280 comparisons involving short forms.
- `review_examples.csv`: fourteen nonnumeric Maya/cone examples, covering all seven forms.
- `numeric_comparison_review.csv`: six side-by-side numeric/nonnumeric Maya/cone examples using 32 ft and 23 ft.
- `templates.csv`, `names.csv`, `objects.csv`: component tables.
- `manifest.json`: counts, tokenizer provenance, source hashes, and validation status.

Regenerate from the repository root:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/non_numeric/generate.py
```

Generation uses the local Qwen3 checkpoint tokenizer on CPU, without loading model weights. Pass `--tokenizer` for another local tokenizer.

## Evaluation

`evaluate.py` scores the two binary datasets together, loading the model once. It reports raw mean target-choice accuracy, separately labeled mean-token PMI choice accuracy, fixed-target context sensitivity, and both-judgment correctness. Results are grouped by dataset, wording form, name, and object. Saved matching tables link numeric/nonnumeric and minimal/longer target-choice decisions. Use an available GPU and a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/non_numeric/evaluate.py \
  --out-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_non_numeric_v1_3 \
  --batch-size 16
```

The default model is the local Qwen3 19.5k checkpoint; override it with `--model`. The evaluator preserves input snapshots, hashes, per-token scores, target-only priors, all item/pair scores, grouped summaries, and a report. Full targets include punctuation, and exact ties count as incorrect. Target-only prior subtraction is checked to cancel from context-sensitivity margins.
