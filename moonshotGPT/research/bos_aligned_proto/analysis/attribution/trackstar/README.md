# TrackStar Backend

This directory contains the Bergson-backed attribution method used by the
`moonshotGPT` BOS attribution pipeline.

The name `TrackStar` is the local backend label in this repo. Under the hood,
it uses EleutherAI Bergson programmatically for gradient collection, gradient
loading, and query-time scoring.

This implementation is inspired by the TrackStar paper, but it is adapted to
the repo’s own BOS-packed, EWoK-focused attribution workflow. It should be read
as a practical research implementation, not as a claim of exact paper
reproduction.

## Current Files

```text
trackstar/
  README.md
  METHOD_NOTES.md
  __init__.py
  config.py
  backend.py
  bergson_datasets.py
  bergson_queries.py
```

- `config.py`
  TrackStar-specific configuration and CLI surface.
- `backend.py`
  Main Bergson-backed scoring implementation.
- `bergson_datasets.py`
  Candidate-row dataset adapter used when building or reading Bergson indices.
- `bergson_queries.py`
  EWoK query construction and loss logic for the custom paired objective.
- `METHOD_NOTES.md`
  Deeper method notes on the scoring geometry, Bergson integration contracts,
  projection defaults, and current differences from the paper setup.

If you want the technical version of this document, read
[`METHOD_NOTES.md`](METHOD_NOTES.md).

## What This Backend Ranks

This backend is not mainly ranking text similarity. It is ranking BOS-packed
training rows by how much their training gradients align with reducing an EWoK
query loss at a checkpoint.

Interpretation:

- large positive scores suggest rows whose gradient direction points toward
  better EWoK behavior;
- large negative scores suggest rows whose gradient direction pushes against the
  desired EWoK behavior;
- scores near zero suggest rows that are mostly irrelevant to that particular
  checkpoint-local EWoK signal.

Because the exported score is cosine-normalized, the main interpretation is
directional alignment, not raw gradient magnitude.

## EWoK Query Definition

Each EWoK item contributes four conditional scores:

- `s11 = log P(T1 | C1)`
- `s12 = log P(T2 | C1)`
- `s22 = log P(T2 | C2)`
- `s21 = log P(T1 | C2)`

Two scoring views are supported:

- `babylm_completion_choice`
  Uses `m1 = s11 - s12` and `m2 = s22 - s21`.
- `ewok_paper_context_sensitivity`
  Uses `m1 = s11 - s21` and `m2 = s22 - s12`.

The paired query loss is:

$$
L(t) = \frac{1}{2}\left[softplus\left(-\frac{m_1}{\tau}\right) + softplus\left(-\frac{m_2}{\tau}\right)\right]
$$

where `tau` is the configured temperature.

That loss makes the attribution question concrete:

Which training rows appear most aligned with reducing the EWoK mistake signal
for this checkpoint?

## Current Implemented Score

For each target item `t` and candidate row `x`, the backend builds corrected,
optionally projected candidate and query gradients, applies the mixed
preconditioner, concatenates per-module features, and exports cosine similarity:

$$
score(t, x) = \frac{\langle q(t), g(x) \rangle}{\|q(t)\|_2 \, \|g(x)\|_2}
$$

Important implementation notes:

- when `optimizer.pt` exists next to the checkpoint, candidate and query
  gradients can both use Adam second-moment correction;
- if `optimizer.pt` is missing, the backend falls back to raw gradients;
- current defaults use projected gradients with `use_fast_jl=True` and
  `proj_dim=16`;
- the exported outputs still follow the shared repo format from
  `analysis/attribution/common/export.py`.

## Example Command

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir runs/research/bos_aligned_proto/<run_name> \
  --data_dir data/processed/bos_aligned_proto/<data_view> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

## When To Prefer TrackStar

Use this backend when:

- you want the Bergson-backed path that is most aligned with the repo’s current
  EWoK attribution direction;
- you want cosine-style gradient alignment rather than the original TRAK
  baseline;
- you want the backend that the output-inspection notebook is most often used
  with.

If you want the simpler baseline path for comparison, read `../trak/README.md`.
