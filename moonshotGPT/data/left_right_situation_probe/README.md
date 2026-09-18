# Fixed-frame left/right situation probe

This dataset tests relative horizontal position on a stationary screen viewed
straight on by the reader. Left and right always refer to that viewpoint.
Objects are icons; no character-facing direction, rotation, or mirror is involved.
Targets describe the final state after all described movements finish.

Version **1.2** uses shorter situation wording in all three tiers. Average
whitespace-delimited words per context: **compact 25**, **standard 29**,
**expanded 36** (previously 43, 51, and 61 before the wording revisions).
The compact wording from v1.1 is unchanged. For example:

> Fixed screen slots, left to right: 1, 2, 3, 4, 5. The ball icon moves from 5 to 1. The cone icon stays at 3.

All 2,400 paired rows, geometry, answers, slot labels and matched variants are
preserved. Standard adds one brief screen reminder; expanded describes starting
and ending positions explicitly. Direct controls are unchanged. Saved
v1.0 evaluation results remain untouched; the shortened wording has **not**
been evaluated. Stored word-count columns retain their existing regex-based
convention, which counts hyphen-separated parts separately.

Each row has two opposing situations and two complete answer sentences.
`Context1` places the designated target object left of the reference;
`Context2` places it right of the reference. `Target1` is correct for
`Context1`, and `Target2` for `Context2`. Reversing the answer's entity order
also reverses its relation word: “The ball icon is to the left of the cone
icon” becomes “The cone icon is to the right of the ball icon.”

| Family | Cases | Question |
|---|---:|---|
| Static placement | 3 | Can it compare two horizontal positions? |
| Target crosses reference | 2 | Can it update the moving target? |
| Reference crosses target | 2 | Can it update the relation when the target stays still? |
| Target moves without crossing | 2 | Can it use final position when movement direction is insufficient? |
| Both move | 3 | Can it track preserved and reversed order? |

The 12 cases cross four object pairs, four evidence forms, three wording
lengths, two context mention orders, and two answer entity orders: 2,304
applied pairs. Another 96 pairs provide plain and `is positioned` direct-label
controls. The evidence forms are named slots, numeric slots increasing left
to right, numeric slots increasing right to left, and lettered slots. Every
context states the complete left-to-right slot order. Internal coordinates
always increase left to right; displayed labels are stored separately.

For example, with slots 1–5 ordered left to right:

- C1: The ball moves from slot 1 to slot 2; the cone stays in slot 5.
- C2: The ball moves from slot 4 to slot 5; the cone stays in slot 1.
- T1: The ball icon is to the left of the cone icon.
- T2: The ball icon is to the right of the cone icon.

Both contexts move the ball rightward, but require opposite answers. Actual
generated contexts include the explicit fixed-screen frame and slot order.

## Observable annotations and matched comparisons

Rows retain initial/final coordinates and displayed labels for both entities
in both contexts, their movement directions, initial/final relations, whether
order reverses, context mention order, last-mentioned entity, answer entity
order, relation words, evidence form, reference frame, and text word counts.
Direct controls leave unspecified positions and movements blank. There are
no heuristic predictions or inferred explanations of model errors.

`variant_matches.csv` links changes in mention order, answer order, wording
length, evidence form, numbering direction, and direct-label style. Links
preserve the underlying coordinates. The numbering-direction comparison
changes labels and the stated order while preserving physical states and
answers. These are matched descriptions, not independent physical scenes.

## Generate and inspect

From the repository root:

```bash
python data/left_right_situation_probe/generate.py
```

`components.json` is the editable case and factor inventory. Generation
validates final relations, event-family constraints, unchanged motion direction
across no-crossing pairs, unique IDs, expected family counts, and matched-link
invariants. Output under `generated/` includes CSV/JSONL probes, a case catalog,
matched links, a manifest, and readable `review_examples.md` plus CSV examples.

Run the evaluator with a local Hugging Face causal-LM checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n babylm python data/left_right_situation_probe/evaluate.py \
  --model CHECKPOINT --out-dir RESULTS
```

The evaluator scores all four context/target combinations using raw mean
full-target token conditional log likelihood, including punctuation, with no
PMI. Exact choice ties count as incorrect. It reports binary accuracy,
both-contexts-correct rates, lexical answer preference, and matched consistency.
Direct-label controls remain separate; the five applied family means receive
equal weight. Left/right gold and matched predictions refer to the designated
target object's relation, even when the answer sentence reverses entity order.
Lexical choice frequency separately tracks the actual word in the answer.
Outputs include input snapshots, per-token and per-item scores, grouped scores,
matched consistency, model/runtime provenance, a summary, and a readable report.
The output directory must not already exist.

Qwen3 359M step-19,500 results are saved under
`runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_v1/`.
See `report.md` for family scores, `findings.md` for the direct-control
entity-order split, and `validation.json` for reconstruction checks. Recheck
saved results with `python data/left_right_situation_probe/validate_scores.py RESULTS`.

### Explicit summary bridge diagnostic

`evaluate_bridge.py` appends `To summarize the positions:` to each of the
96 direct-label control contexts while keeping the answers unchanged. It
loads the same checkpoint recorded in a baseline run and compares raw accuracy
and paired correctness by same versus reversed entity order. The bridge is
context, not scored answer text. Original datasets and results are preserved.

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n babylm python data/left_right_situation_probe/evaluate_bridge.py \
  --baseline-dir runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_v1 \
  --out-dir runs/research/bos_aligned_proto/left_right_situation_probe/qwen3_359m_step19500_summary_bridge_v1
```

This tests sensitivity to an explicit restatement cue. Changes cannot uniquely
identify a repetition mechanism, since the bridge also changes wording and
length. It is a separate control diagnostic, not a change to the main probe.

## Scope

This is a controlled slot-based horizontal-order probe, with 12 physical cases.
The four evidence forms share one scene structure. Named slots contain explicit
left/right position cues; lettered and numbered slots require using the stated
ordering. Successful ordinal comparison is a valid solution here and does not
alone establish broad spatial understanding. Wording lengths vary wording as
well as word count. Both-move cases specify endpoints, not timing, speed, or
collision behavior; order reversal refers to initial versus final order.
Person-relative viewpoints, turning, equal positions, and ambiguous overlap
are outside this batch.
