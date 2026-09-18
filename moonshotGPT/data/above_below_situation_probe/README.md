# Above/below situation probe

This probe asks whether a model can choose between two full target sentences
from a controlled vertical scene. Every row contains an above-compatible
context (`Context1`), a below-compatible context (`Context2`), and two target
sentences. `Target1` is correct after `Context1`; `Target2` is correct after
`Context2`. The primary metric is binary accuracy using mean target-token
conditional log likelihood. Exact ties are wrong. Direct labels are a separate
control and are excluded from the applied mean.

| Physical family | Case IDs | Diagnostic question |
|---|---|---|
| Static placement | `static_*` | Can the model compare two final positions? |
| Target crosses a fixed reference | `target_cross_*` | Can it track the moving target across the reference? |
| Reference crosses a fixed target | `reference_cross_*` | Can it track motion when the answer's subject stays still? |
| Target moves without crossing | `no_cross_*` | Does it confuse movement direction with final relative position? |
| Both move | `both_*` | Can it track preserved versus reversed vertical order? |

The physical cases and their initial/final levels are specified in
`components.json`; the generator validates the correct final relation for both
contexts. The five shelf levels are named bottom, lower, middle, upper, and top
in that order. Measured-height variants use the same levels as 1–5 feet from
the floor. Numbered-step variants put objects on steps 1–5, which go up in
order. Numbered-floor variants put them on building floors 1–5, numbered
upward. Across all four forms, the correct answer follows the objects' final
heights. The `no_cross_up` and `no_cross_down` pairs keep the target's movement direction
the same across both gold answers.

Generation crosses 12 physical cases with four object pairs, four evidence
forms (`named_shelves`, `measured_height`, `numbered_steps`, `numbered_floors`),
three wording lengths, two context entity orders, and two target entity orders.
This yields 2,304 applied
rows. A further 96 direct-label rows use plain and `is positioned` forms
with the same object, length, and order controls. The `target_entity_order=reference_first` variants express the
inverse relation (for example, “The cone is below the ball” when the ball is
above the cone), so a preference for the *word* `above` can be separated from
a preference for `Target1`.

Example from the target-crossing family:

| Field | Text |
|---|---|
| `Context1` | The rack's shelves run from bottom through lower, middle, and upper to top. The ball moves from the bottom shelf to the top shelf. The cone remains on the middle shelf. |
| `Target1` | The ball is above the cone. |
| `Context2` | The rack's shelves run from bottom through lower, middle, and upper to top. The ball moves from the top shelf to the bottom shelf. The cone remains on the middle shelf. |
| `Target2` | The ball is below the cone. |

The same case in the compact steps scene uses: “Steps 1 to 5 go up. The ball
moves from step 1 to step 5. The cone stays on step 3.” Its
matched below context changes the ball's path to step 5 → step 1. Each step
scene shares the original case ID, object pair, and target choices with its
shelf, floor, and measured-height versions. This holds ordinal level assignments
and event structure fixed, but changes wording, number cues, and physical scale.

Run from the `moonshotGPT` directory:

```bash
python data/above_below_situation_probe/generate.py
CUDA_VISIBLE_DEVICES=1 conda run -n babylm python \
  data/above_below_situation_probe/evaluate.py \
  --model CHECKPOINT --out-dir RESULTS
```

The generator writes `probes.csv`, `probes.jsonl`, a case catalog, review
examples, and `variant_matches.csv`. The evaluator writes item-level and
family-level scores, a family-balanced applied mean, the shared answer-bias
table, and matched-variant consistency. The overall applied score averages
the five physical family means equally; all four evidence forms and the
available wording and order variants are balanced within each family. Related
variants reuse the same cases and should not be treated as independent scenes.
The two direct-label forms are reported separately so exact sentence
repetition can be diagnosed rather than hidden in the control mean.

All scenes use an upright vertical frame. Step and building-floor scenes test
height order without requiring the objects to overlap on a vertical line.
This batch does not test rotated viewpoints, screen coordinates, tilted
supports, or ambiguous object extent.
