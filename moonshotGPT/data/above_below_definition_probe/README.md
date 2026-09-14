# Above/below definition and synonym probe

The reviewed phrase inventory in `components.json` includes both
`higher up than` / `lower down than` and the simpler everyday
`higher than` / `lower than`. It also retains `vertically higher/lower than`
and `at a greater/lesser height than` as separate wording families. These are
phrasing diagnostics, not four independent spatial concepts. The separate
`lexical_synonym_pairs` table adds `over` / `under`. This pair can suggest
vertical alignment, so its scores should not be pooled with literal height
definitions.

Run `python data/above_below_definition_probe/generate.py` from the repository
root to rebuild `generated/`. Version 1.2 creates 80 matched two-answer
probes: 24 word-to-definition, 40 definition-to-word, 6 word-to-synonym,
and 10 synonym-to-word. The original 40 texts and IDs remain in the dataset.
Every phrase pair occurs in both definition structures and in both mapping
directions. Three context stems provide short, medium, and long word-to-definition
forms. `direction_matches.csv` and `synonym_direction_matches.csv` join the
original reverse probes to their three forward counterparts. The new
`entity_order_matches.csv` joins 20 first-subject and second-subject pairs.
`probes.csv` and `probes.jsonl` retain all component IDs,
the four texts, and the correct target for each context.

Example using the two requested phrase pairs:

| Phrase pair | C1 | T1 | C2 | T2 |
|---|---|---|---|---|
| Colloquial | Above means that | one object is higher up than another. | Below means that | one object is lower down than another. |
| Everyday | One object is higher than another. | One object is above another. | One object is lower than another. | One object is below another. |
| Synonym | One object is over another. | One object is above another. | One object is under another. | One object is below another. |

The reverse-direction extension crosses two entity nouns (`object`, `item`),
two clause structures (plain, `is positioned`), and two context orders. For
example, “The second item is lower than the first item” supports “The first
item is above the second item.” Reversing the subject also reverses the gloss
polarity, so `context_entity_order` is a separate semantic diagnostic, not just
a wording variant. The 40 added rows consist of 32 height-definition and 8
over/under synonym probes. The noun and structure factors remain labeled for
matched comparisons; repeated combinations are not independent concepts.

The C1/T1 and C2/T2 assignments mark the intended gold matches. Run
`evaluate.py --model CHECKPOINT --out-dir RESULTS` to compare both full target
sentences after each context using mean target-token log likelihood, as in the
close/far definition probe. Generation alone does not evaluate a model. The
Qwen3 step-19,500 results for the earlier 40-row version are saved under
`runs/research/bos_aligned_proto/above_below_definition_probe/qwen3_359m_step19500_v1_1/`.
The current 80-row version is saved under
`runs/research/bos_aligned_proto/above_below_definition_probe/qwen3_359m_step19500_v1_2/`.
The evaluator saves per-item scores, direction and phrase-pair summaries, and
the shared answer-preference table. `render_bias.py` formats an explicit
upper/lower choice table by direction, phrase, and entity order.
`analyze_entity_order.py` also produces a
matched first- versus second-subject report from the saved scores, without
another model run. Score the two directions separately, and
use the phrase IDs to diagnose wording effects. `probe_family` keeps the lexical synonym rows
separate from the literal definition rows. The literal definitions express
relative vertical position; they do not assert direct alignment or test
interpretation of scenes.

`score_priors.py` scores the isolated words `above` and `below` from BOS, their
continuations after five neutral prefixes, and the four full-sentence target
pairs used by the reverse-direction probes. The saved Qwen3 results are in
`word_priors.md` and `word_priors.json` alongside the version 1.2 evaluation.
The bare-word score is not interchangeable with a word score inside a sentence.
