# Minimal both-moving closer/farther probe

This additive probe gives the `both_entities_move` family all three distance
outcomes without changing the evaluated event-extension v1.2 data. Every
example starts with the person to the left of the object, and both entities
move to the right.

| Outcome | Person movement | Object movement | Result |
|---|---:|---:|---|
| Closer | fast | slow | Separation decreases |
| Farther | slow | fast | Separation increases |
| Unchanged | fast | fast | Separation is preserved |

For a parent case with initial separation `D` and distance change `m`, slow is
1 unit and fast is `m + 1` units. This makes the final separations exactly
`D - m`, `D + m`, and `D`, matching the parent numeric cases. Both movements
are nonzero, and the person does not reach or pass the object in closer cases.

The batch deliberately has one wording form rather than compact, standard, and
expanded variants. It crosses the existing 12 numeric cases, four names, three
objects, three units, and two entity orders. A small nonnumeric form uses
`covered more ground` or `equal distances` with the same event structure.

Every underlying outcome group produces three two-answer pairs:

- closer versus farther
- closer versus unchanged
- farther versus unchanged

Thus closer, farther, and unchanged are each correct in their own contexts
inside the same physical event family. `pairs.jsonl` uses the EWoK-style
`Context1`, `Context2`, `Target1`, and `Target2` layout. Primary evaluation
should use binary mean target-token log likelihood, with fixed-target context
sensitivity and both-context correctness reported separately. Numeric and
nonnumeric evidence should receive equal weight because their generated row
counts differ substantially.

Regenerate from the repository root:

```bash
python data/closer_farther_probe/both_move_minimal/generate.py
```

The generated manifest marks this additive batch as unevaluated.
