# Observer turns: matched reference frames

Expanded version 1.1 evaluation: `runs/research/bos_aligned_proto/cardinal_blocks/qwen3_359m_step19500_expanded_v2/`.
The report includes both description structures; confidence intervals are
reported with the paired/template dependence caveat.

An additive extension, separate from the cardinal event/control inventory
(originally 120 pairs, expanded to 480 in version 1.1 and 1,344 in version 1.2).
Eight fixed geometries cover all four initial facing directions, with an object
on either side of the observer. Each has compact, standard, and expanded wording
and two answer frames. Version 1.1 adds a second description structure using a
fixed flag and “continues looking” versus “faces the opposite direction”:
96 paired rows, or 192 judgments. All original 48 IDs and texts are preserved.
`description_structure` and `object_description` identify the two structures.
The structure links also change Object A to Flag A in the answers, so their
effects do not isolate context wording alone. The eight physical geometries
are unchanged, not doubled.

Within each row, Context1 describes no turn and Context2 a half-turn in place.
The observer and object never change location. Cardinal answers must remain
unchanged; observer-relative left/right answers must reverse. Both contexts are
identical between matched cardinal and observer-relative rows; only candidate
answers change. Absolute north/south and east/west each have balanced answers.

Example: B faces north and A is east of B. Before and after B's half-turn, A is
east of B. A starts on B's right and ends on B's left. The answers explicitly
refer to observer B's side, not the reader's screen coordinates.

**This extension does not use the usual C1→T1, C2→T2 assumption.** Follow
`correct_target_for_context1` and `correct_target_for_context2`. Cardinal rows
have the same correct target in both contexts, counterbalanced across object
directions. Relative rows require opposite targets, with both switch directions
represented. Do not pass these rows to an evaluator that hardcodes diagonal
gold matches. Report accuracy and both-context correctness plus correct
invariance/switching. Unchanged predictions alone do not prove understanding:
an always-first answer passes invariance while failing half the cardinal rows.

Initial/final facing, positions, action, answer frame, and expected relation
change are recorded explicitly. Half-turn endpoints are computed from vectors;
left/right is computed from the object's projection onto the observer's right
axis. Quarter-turns, rotating maps, and moving objects are not included.

Run from the repo root:

```bash
python data/cardinal_situation_probe/observer_turn/generate.py
python -m pytest -q tests/test_cardinal_observer_turn.py
```

Generated CSV/JSONL, readable examples, and frame/length match links are under
`generated/`. This extension uses direct cardinal placement statements rather
than numeric or named-location inference. Wording length and syntax vary
together. Qwen3 359M step-19,500 results for the earlier 48-pair version are saved in
`runs/research/bos_aligned_proto/cardinal_blocks/qwen3_359m_step19500_v1/`.
The runner `data/cardinal_situation_probe/evaluate_blocks.py --model CHECKPOINT
--out-dir RESULTS` evaluates this block and the cardinal definitions using
explicit per-context gold labels. Saved token means and choices are validated.
