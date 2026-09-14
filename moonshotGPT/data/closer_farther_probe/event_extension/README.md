# Closer/farther: direct labels and matched movement events

This additive probe extends the evaluated closer/farther set without changing its generated examples or scores. It uses the same names, objects, twelve numeric cases, three units, and three complete targets as the parent probe.

| Condition | Example | Gold outcome |
| --- | --- | --- |
| Direct label, person first | Compared with before, Maya is now closer to the cone. | Closer |
| Direct label, object first | Compared with before, the cone is now closer to Maya. | Closer |
| Reference object moves | Maya remained in place. The cone was moved toward Maya without crossing Maya's position. | Closer |
| Both move, separation fixed | Maya and the cone each moved 5 ft to the right. | Unchanged |

Direct labels are lexical ceiling controls, generated once per name/object/outcome and both entity orders. They deliberately use the answer word in the context and are flagged `target_relation_repeated`; they do not demonstrate movement inference. There are 72 direct-label rows: 12 name/object combinations × 3 outcomes × 2 orders.

The two new event families each have numeric and nonnumeric renderers in **compact, standard, and expanded** wording. The object-movement family includes closer and farther outcomes with the person stationary. The co-motion family includes unchanged separation while both entities translate equally to the right. The numeric generator stores before/after coordinates for both entities and computes the gold outcome from the absolute distance change. It validates final distances against the parent numeric cases. Object movement toward the person stops before crossing their position. Nonnumeric examples avoid assuming specific numerical endpoints but retain the same event logic.

Version 1.2 renders **both person-first and object-first contexts at every wording length** for both movement families, with identical entities, event outcome, distances, and answer targets. Examples at standard length:

| Event | Person first | Object first |
| --- | --- | --- |
| Reference object moves | Initially, Maya and the cone were 18 ft apart. Maya remained in place. The cone was moved 5 ft directly toward Maya without crossing Maya's position. | Initially, the cone and Maya were 18 ft apart. The cone was moved 5 ft directly toward Maya without crossing Maya's position. Maya remained in place. |
| Both move equally | Initially, Maya and the cone were 18 ft apart. Maya and the cone each moved 5 ft to the right. | Initially, the cone and Maya were 18 ft apart. The cone and Maya each moved 5 ft to the right. |

The original v1.1 context texts and probe IDs remain as one member of each pair. A new ID appends `__person_first` or `__object_first` to identify the alternate. The old **compact reference-object** context already began with the object, although it was mislabeled `person_first`; its label is now corrected and a person-first compact context is added. Order pairs share `entity_order_match_id` and appear in `matches.csv` as `event_entity_order`. The generator verifies the first named entity in every context. Clause position or voice can change along with order, so an observed difference is an order-and-phrasing diagnostic, not a pure causal effect of order alone.

| Length | Reference object moves: numeric closer | Both move: numeric unchanged |
| --- | --- | --- |
| Compact | The cone, 18 ft from Maya, moved 5 ft directly toward Maya without crossing Maya's position; Maya stayed still. | Starting 18 ft apart, Maya and the cone each moved 5 ft right. |
| Standard | Initially, Maya and the cone were 18 ft apart. Maya remained in place. The cone was moved 5 ft directly toward Maya without crossing Maya's position. | Initially, Maya and the cone were 18 ft apart. Maya and the cone each moved 5 ft to the right. |
| Expanded | At the start, Maya and the cone were 18 ft apart. Maya remained in exactly the same position throughout. From its starting position, the cone was moved 5 ft in a straight line directly toward Maya, without crossing Maya's position. | At the beginning, Maya and the cone were 18 ft apart. From that starting arrangement, Maya moved 5 ft to the right, and the cone also moved 5 ft to the right. |

The original version 1.0 event contexts and probe IDs are preserved as `standard`. Original short and long IDs append `__compact` or `__expanded`. Each three-length set now shares an **order-specific** `length_match_id`, so length consistency is measured while holding entity order fixed. All forms share the event coordinates, outcome, and target sentences. Generation validates strict compact < standard < expanded word and tokenizer-token counts in every matched group. The bands are relative **within an event family**; a compact object-movement context need not have the same length as a compact co-motion or parent person-movement context. Wording also changes syntax, so score differences do not isolate length alone. The direct-label controls remain single short forms.

There are 7,776 numeric event rows (12 numeric cases × 4 names × 3 objects × 3 units × 3 outcome/event combinations × 3 lengths × 2 entity orders) and 216 nonnumeric event rows (4 names × 3 objects × 3 combinations × 3 lengths × 2 orders). Alongside the 72 direct-label rows, the extension has **8,064 contexts**, balanced at 2,688 each for closer, farther, and unchanged. Numeric examples should not dominate a pooled score simply because they have more parameter combinations.

The 30,312 comparison links cover all 2,664 order-specific compact/standard/expanded groups, all 3,996 person-first/object-first event pairs, numeric versus nonnumeric versions within each order and length, standard new events versus the parent explicit-distance and movement/orientation rows at the same initial/final separation, direct labels versus each event, and the two direct-label entity orders. Parent comparisons preserve targets and distances, but generally describe **different physical events**. Co-motion moves both entities, while the parent's unchanged orientation case does not translate either entity. A direct-label row is deliberately reused as a control for many situations; these links are not independent direct-label examples.

The new families test who moves and whether two movements preserve distance. They do not test passing or overshooting, coordinate/landmark reasoning, or comparative-versus-categorical distinctions. The original person-movement examples still cover toward/away motion with a stationary reference. The direct-label controls are a separate evidence category, not another physical event family.

Generated files include `probes.csv` / `.jsonl`, `matches.csv`, direct, event, and length review tables, template tables, and a manifest with source hashes and validation. Score complete targets using mean target-token log likelihood, reporting direct labels, numeric object movement, nonnumeric object movement, numeric co-motion, and nonnumeric co-motion separately. Keep closer/farther/unchanged outcomes and compact/standard/expanded bands separate and inspect confusion matrices and matched consistency.

For the evaluated v1.2 probe, `report.md` gives all three **binary** contrasts by evidence mode, wording length, and entity order; `entity_order_pair_scores.csv` and `entity_order_summary.csv` quantify paired prediction flips. `overall_report.md` and `overall_summary.json` add an equal-weight **applied movement mean**. Its three comparison families are (1) person moves while the reference stays still, reusing the previously scored matched numeric/nonnumeric probe restricted to the original three objects; (2) reference object moves, scored closer versus farther within the family; and (3) reference object moves versus both entities moving together, scored as the mean of closer versus unchanged and farther versus unchanged. Within each family, the calculation averages cases, wording bands and available entity orders, numeric and nonnumeric evidence equally, and then contrasts. It then averages the three family scores equally. Direct labels and the categorical close/far literal-definition probe are separate controls, excluded from this applied movement mean. The older person-moving contexts have no matched entity-order variant yet, and cross-family comparisons change event type and syntax.

## Qwen3 19.5k evaluation of v1.2

The [overall applied report](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2/overall_report.md) gives an **equal-family raw mean of 48.97%**: person moves/reference stationary 49.38%, reference object moves 47.54%, and reference object moves versus both move (cross-family) 50.00%. The [binary report](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2/report.md) gives each two-answer contrast by evidence mode, wording band, and entity order, with matched prediction-flip statistics. In the cross-family contrasts, the model always selects unchanged, so 50% is a response preference rather than successful discrimination. Direct comparative labels score 82.64% raw across contrasts and are excluded from the applied mean. The older categorical close/far literal-definition probe is also shown separately in the overall report.

The v1.2 run scored all 8,064 contexts and saved 24,192 context-target token-score sequences. All 4,068 v1.1 context texts, targets, and gold outcomes were preserved; their regenerated mean log likelihoods differ from the prior run by at most 9.8e-6, with no three-way prediction changes. GPU 1 was released. Source hashes, model configuration, and all per-item results are in the result directory.

## Qwen3 19.5k evaluation of the earlier v1.1 rows

The **v1.1 rows** were evaluated on the local **Qwen3 359M, step-19,500** checkpoint. The [binary report](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_1/report.md) is the prior result: closer versus farther, closer versus unchanged, and farther versus unchanged are each scored using only two targets at a time. Saved scores remain available alongside it. The v1.2 run above supersedes this version for entity-order analysis.

The following archived three-way results diagnose the answer preference that initially obscured the matched-pair design. They are retained in [three_way_diagnostic.md](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_1/three_way_diagnostic.md) and are **not** the headline accuracy measure.

| Condition | Contexts | Archived three-way raw | Archived three-way PMI |
| --- | ---: | ---: | ---: |
| Direct label | 72 | 69.44% | 65.28% |
| Reference object moves, nonnumeric | 72 | 0.00% | 48.61% |
| Reference object moves, numeric | 2,592 | 0.00% | 47.45% |
| Both move, nonnumeric | 36 | 100.00% | 0.00% |
| Both move, numeric | 1,296 | 100.00% | 0.00% |

Under raw mean likelihood, the model selected the **same-distance** target in all 3,996 movement contexts. The apparent perfect co-motion score is therefore a constant-response effect. After target-only PMI adjustment, it selected farther in 3,753 of those contexts and closer in 243, with no unchanged predictions. The direct-label raw score is 97.22% when the person is the context subject and 41.67% when the object is. In the reference-object movement pairs, raw closer-versus-farther binary choice is 49.07% for numeric and 47.22% for nonnumeric contexts; fixed-target context sensitivity is 47.45% and 50.00%, respectively. These results show strong answer-form sensitivity and little evidence of robust use of the new movement descriptions. Compact/standard/expanded predictions are identical within all 1,332 event groups under raw choice, including consistently incorrect groups.

The reference-object movement examples are matched closer/farther context pairs: **toward Maya** versus **away from Maya**, with targets and other event constraints fixed. Their 0% figures above are **three-way** scores, because the same-distance target was offered. For the intended binary contrast, use the separate [paired report](../../../runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_1/contrast_report.md), which also reports context sensitivity and label-choice rates. Binary choice near 50% means the model still does not reliably distinguish the two contexts after removing the same-distance option. The direction phrases differ in word count, so these are matched semantic contrasts rather than exact one-token substitutions.

The binary report also balances the other two contrasts by pairing object-movement **closer** or **farther** contexts with co-motion **unchanged** contexts at the same starting separation, entity assignment, unit, numeric case, and wording band. Those pairs change physical event family and syntax, so they diagnose categorical discrimination without isolating a single lexical cue. Direct-label pairs use only direct-label contexts. Each contrast has 1,356 pairs / 2,712 binary judgments, including 24 direct-label, 36 nonnumeric event, and 1,296 numeric event pairs. Numeric and nonnumeric categories should remain separate when forming an overall score.

All 4,068 item decisions were independently reconstructed from the 12,204 saved conditional token means and 36 target-only priors. Parent comparisons reuse prior scores only after the checkpoint, mean-scoring convention, and exact parent probe hash were verified. GPU memory was released after evaluation.

Regenerate from the repository root:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/event_extension/generate.py
```

For an already scored output, regenerate only the paired contrast analysis on CPU:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/event_extension/analyze_contrasts.py --results-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_1
```

Regenerate the **primary binary** report and pair scores on CPU from the saved token scores:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/event_extension/analyze_binary.py --results-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_1
```

For an evaluated v1.2 directory, recompute the applied macro after binary postprocessing:

```bash
/home/jorge/miniconda3/envs/babylm/bin/python data/closer_farther_probe/event_extension/analyze_overall.py --results-dir runs/research/bos_aligned_proto/closer_farther_probe/qwen3_359m_step19500_event_extension_v1_2
```
