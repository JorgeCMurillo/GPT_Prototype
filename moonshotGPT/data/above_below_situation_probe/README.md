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
the floor. The
`no_cross_up` and `no_cross_down` pairs keep the target's movement direction
the same across both gold answers.

Generation crosses 12 physical cases with four object pairs, two evidence
forms (`named_shelves`, `measured_height`), three wording lengths, two
context entity orders, and two target entity orders. This yields 1,152 applied
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
the five physical family means equally; numeric and shelf evidence and the
available wording and order variants are balanced within each family. Related
variants reuse the same cases and should not be treated as independent scenes.
The two direct-label forms are reported separately so exact sentence
repetition can be diagnosed rather than hidden in the control mean.

The vertical frame is fixed to a room and its floor. This first batch does not
test rotated viewpoints, screen coordinates, tilted supports, or ambiguous
object extent.
