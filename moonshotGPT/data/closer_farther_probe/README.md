# Closer/farther: matched conditions, version 1.2

This batch tests distance decrease, distance increase, and unchanged distance with three answer alternatives. It contains **7,776 contexts**, generated from **12 numeric cases × 4 names × 3 objects × 3 units × 6 condition/outcome combinations × 3 wording lengths**. These are controlled variations of **nine context templates**, not independent scenarios. The Qwen3 19.5k checkpoint has been evaluated; see the [saved report](../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2/report.md). The generated manifest describes generation only; evaluation metadata and exact input snapshots are stored in the separate run directory.

Version 1.2 adds compact and expanded versions to the existing standard sentences. All 2,592 version 1.1 contexts retain their text, targets, and probe IDs as standard versions. Row indices change; join by `probe_id`.

## Conditions and wording lengths

A separate [non-numeric closer/farther extension](non_numeric/README.md) adds 280 movement contexts / 140 binary pairs across the same four names, with bench and mailbox added to the object inventory. This includes 120 contexts using three aligned wording lengths and 160 using four short forms (unqualified, stationary adjective, and matched stationary/still relative clauses). It also generates 4,320 aligned numeric contexts: each specific distance is replaced by `some distance` in the aligned nonnumeric version, keeping the surrounding wording and no-reaching/no-passing clause fixed. The minimal forms label their unstated movement assumptions separately. The v1.2 set of 200 nonnumeric contexts and its numeric counterparts were evaluated on Qwen3 19.5k; the 80 stationary/still clause contexts added in v1.3 are unevaluated. The extension README links the prior results. They preserve the closer/farther target texts. The original evaluated numeric dataset here is unchanged.

An additive [event extension](event_extension/README.md) generates direct comparative-label controls, reference-object movement, and unchanged-separation co-motion with numeric and nonnumeric versions in compact, standard, and expanded wording. It has been evaluated on Qwen3 359M at step 19,500; the original set remains unchanged.

For current and future diagnostic summaries, **binary target comparisons are primary**: closer versus farther, closer versus unchanged, and farther versus unchanged, each with 50% uniform-choice chance. The original version 1.2 three-way run remains an archived historical result. The event extension's binary report and pair tables demonstrate the current scoring convention.

| Condition | Outcomes | Per length | All lengths | Evidence type |
|---|---|---:|---:|---|
| Explicit initial/final distance | closer, farther, unchanged | 1,296 | 3,888 | explicit_distance_comparison |
| Movement description | toward/closer, away/farther | 864 | 2,592 | directed_movement_inference |
| Orientation control | turn without moving/unchanged | 432 | 1,296 | rotation_without_translation_inference |

Each length band has 2,592 contexts: 864 per outcome. Across the batch, each outcome has 2,592 contexts. Report each condition and outcome separately. Compare movement with the explicit closer/farther subset, and orientation with the explicit unchanged subset, within each length band.

| Condition | Length | Example |
|---|---|---|
| Explicit distance | Compact | Maya's distance from the stationary cone: initially 32 ft, now 9 ft. |
| Explicit distance | Standard | Maya was 32 ft from a stationary cone. Maya is now 9 ft from the cone. |
| Explicit distance | Expanded | Initially, Maya was 32 ft from a cone that remained stationary. The distance between Maya and that cone is now 9 ft. |
| Movement | Compact | Starting 32 ft from a stationary cone, Maya walked 23 ft directly toward it. |
| Movement | Standard | Maya was 32 ft from a stationary cone. Maya walked 23 ft directly toward the cone. |
| Movement | Expanded | Initially, Maya was 32 ft from a cone that remained stationary. From that starting position, Maya walked 23 ft directly toward the cone. |
| Orientation | Compact | Maya, 32 ft from a stationary cone, turned around in place. |
| Orientation | Standard | Maya was 32 ft from a stationary cone. Maya turned around without changing position. |
| Orientation | Expanded | Initially, Maya was 32 ft from a cone that remained stationary. Maya then turned to face the opposite direction, keeping both feet in the same place. |

For explicit distance, substitute the farther or unchanged endpoint. For movement, substitute `away from` for `toward`. The compact explicit wording avoids saying the distance "changed" when the endpoint is unchanged. All three orientation versions require inferring unchanged distance from rotation without translation; none explicitly states the final distance. Names are repeated where needed so gendered pronouns do not require assumptions about algorithmically substituted names.

All versions use identical targets for a given name/object pair:

- Target1: Maya is now closer to the cone than before.
- Target2: Maya is now farther from the cone than before.
- Target3: Maya is now at the same distance from the cone as before.

`template_id`, `length_band`, and `evidence_type` identify the controlled components. `length_match_id` links each set of three versions; `length_matches.csv` contains 5,184 comparisons against the standard reference. Endpoints, movement magnitude, entities, units, targets, and evidence type are identical within each set. These are **wording-and-length comparisons**: syntax, punctuation, and reference wording also vary, so a score difference cannot be attributed to length alone.

Each text has actual word, character, and tokenizer-specific token counts. Word counts use regex word boundaries, so possessives count as two words. Standalone token counts exclude special tokens and leading spaces. Additional `TargetN_conditional_token_count` fields use the evaluator's context + space + target boundary. The manifest identifies the tokenizer; generation loads it locally without loading model weights or using a GPU. `length_summary.csv` reports word/token ranges by condition and band. Compact < standard < expanded is validated for both word and token counts within every matched set; bands are relative within a scenario, not universal length thresholds.

## Number and unit controls

Seed **20260913** produces 12 distinct starting distances sampled from 6 through 90. For each starting distance, a positive movement is sampled subject to:

- Final distances remain positive and at most 99 in the selected unit.
- Toward movement stops before reaching the object, excluding overshoot.
- Toward and away conditions use the same movement magnitude.
- The unchanged endpoint equals the starting distance.

Cases include 18→13/23, 85→73/97, 53→21/85, 12→4/20, and 32→9/55. `numeric_cases.csv` contains all 12. Movement magnitudes and endpoints may repeat; starting distances are unique. Numbers use digits. A seed change should be versioned if the dataset has been evaluated.

Units are metre/metres, ft, and inch/inches. Each contributes 2,592 contexts. Every case is crossed with all units, with consistent units throughout each example and singular wording supported for a value of one. The same numbers are reused, not converted: 18→13 metres and 18→13 inches share a comparative answer but have different physical distances. Abbreviation and physical scale vary together for ft; this is not an isolated abbreviation manipulation or a unit-conversion test.

`unit_match_id` and the 5,184 rows in `unit_matches.csv` link unit variants to the metre reference within each wording length. Only unit expressions change in those matches.

## Entity and reference controls

Names are Maya, Jesse, Li, and Omar. Objects are cone, flag, and bicycle. Every name/object combination appears with every numeric case, unit, wording length, condition, and eligible outcome. Objects are explicitly stationary; targets describe the person's distance relative to the object.

Reference reversal remains a later control. Physical separation is symmetric, while which entity moves is a different question. Further/nearer wording, passing an object, both objects moving, insufficient information, and closer-but-still-far conditions also remain separate future additions.

## Scoring and diagnosis

This is a **three-way target-choice dataset**, not the EWoK two-context/two-target schema. Each row has `Context`, `Target1`, `Target2`, `Target3`, and `correct_target`. Default scoring is mean target-token log likelihood over complete target continuations. Report PMI-style mean scores separately if evaluating target-only baselines. Keep per-token scores because target lengths, prepositions, and punctuation may affect preferences.

Report accuracy and confusion matrices by condition, outcome, length, and evidence type; matched wording agreement and joint correctness; explicit/situation agreement; and joint correctness across the three outcomes for a numeric/entity/unit/length group. The 3,888 rows in `matched_examples.csv` link explicit and situation descriptions with identical endpoints and targets within a length band. Random uniform choice has expected accuracy 1/3. Fixed target-slot identities are for likelihood scoring; counterbalance option order if converting to displayed multiple choice.

Use `numeric_case_id`, initial/final distance, signed change, movement magnitude, name/object IDs, unit, and template IDs to locate errors. Treat variants as repeated measurements of shared scenarios rather than independent evidence when estimating uncertainty.

Directly toward/away wording can solve the initial movement set without calculating the final distance because overshoot is excluded. It tests the movement-to-distance-change mapping, not multi-step arithmetic or arbitrary trajectories.

## Regeneration and files

From the repository root, using the installed environment:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/generate.py
```

The default tokenizer is the local Qwen3 19.5k checkpoint. Pass `--tokenizer /path/to/local/checkpoint` to use another tokenizer; token counts and their ordering are then revalidated.

- `components.json`: seed, limits, units, entities, nine context templates, evidence labels, targets.
- `generated/probes.csv` / `.jsonl`: 7,776 labeled contexts.
- `generated/numeric_cases.csv`: 12 reproducible distance/movement cases.
- `generated/matched_examples.csv`: 3,888 explicit-to-situation matches.
- `generated/unit_matches.csv`: 5,184 unit comparisons.
- `generated/length_matches.csv`: 5,184 comparisons against standard wording.
- `generated/templates.csv`: nine labeled context templates.
- `generated/length_summary.csv`: actual word/token ranges by condition and length band.
- `generated/review_examples.csv`: all 648 Maya/cone contexts, covering every numeric case, unit, and length.
- `generated/names.csv`, `objects.csv`, `outcomes.csv`, `conditions.csv`, `units.csv`: component tables.
- `generated/manifest.json`: counts, tokenizer source, component hash, and validation status.

Generation validates arithmetic, positive distances, no overshoot, outcome/unit/length/entity balance, unique contexts and IDs, identical facts and targets within length matches, increasing word/token lengths, identical endpoints/targets within explicit/situation matches, and unit-only substitutions in unit matches.

## Evaluation command

The requested separate binary comparisons are saved in the [binary report](../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2_binary/report.md). Each compares only two targets and includes only contexts where one of those targets is correct: closer/farther, closer/unchanged, and farther/unchanged. Each has 5,184 balanced context judgments. `analyze_binary_choices.py` reuses saved likelihoods; it does not require a GPU:

```bash
python data/closer_farther_probe/analyze_binary_choices.py \
  --source-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2 \
  --out-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2_binary
```

Binary results include accuracy by length and description family, outcome-specific accuracy, choice preferences, and both-context correctness for matched pairs. PMI remains secondary. Each original context participates in two binary comparisons, so results across contrasts are not independent.

The [context-sensitivity report](../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2_context_sensitivity/report.md) holds each target fixed and compares its mean likelihood across the two matched contexts. `analyze_context_sensitivity.py` reuses the same source scores and binary pair table:

```bash
python data/closer_farther_probe/analyze_context_sensitivity.py \
  --source-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2 \
  --binary-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2_binary \
  --out-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2_context_sensitivity
```

Each contrast has 2,592 matched pairs and 5,184 target judgments. The report separates average target-judgment accuracy from the fraction of pairs where both targets favor their matching contexts. Fixed target-only PMI baselines cancel in this comparison, which is checked for every margin. Situation comparisons involving unchanged distance pair movement and orientation descriptions; their evidence type and wording differ as well as their distance outcomes.

`evaluate.py` implements three-way choice using the existing EWoK conditional likelihood functions. It defaults to the local Qwen3 19.5k model; pass `--model` to change it. Select a free GPU and a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/evaluate.py \
  --out-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_v1_2 \
  --batch-size 16 --include-pmi
```

Mean full-target log likelihood is always the primary score. `--include-pmi` adds separately labeled mean-token prior adjustment using the previous repository convention: target text as written after BOS, with no leading space or appended EOS. Exact maximum ties count as incorrect. The run saves input snapshots, per-token conditional scores, target-only priors when requested, item scores, grouped summaries, confusion matrices, matched comparisons, consistency tables, and a report. Output directories must be new to protect existing runs.
