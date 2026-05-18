# TrackStar Method Notes

This document keeps the more technical notes for the Bergson-backed TrackStar
backend. The main [README.md](README.md) is the entrypoint. This file is the
deeper reference for how the current implementation should be interpreted,
where it intentionally differs from the papers, and which practical constraints
shape the code.

## Scope

This repo's TrackStar path should be read as:

- a TrackStar-inspired, checkpoint-local attribution backend;
- adapted to EWoK rather than generic factual prompt-completion tracing;
- adapted to BOS-packed rows and stream windows rather than open-set C4
  retrieval;
- implemented on top of Bergson's indexing and scoring machinery where useful,
  while keeping the repo's own run structure, candidate selection, and exports.

So this file is not claiming exact paper reproduction. It is documenting the
current research implementation used in this repo.

## High-Level View

The outer attribution pipeline is:

`config -> checkpoints -> row manifest -> exposures -> candidates -> EWoK targets -> backend -> export -> compare`

TrackStar only changes the `backend` stage. The shared pipeline still decides:

- which checkpoints are scored;
- which exposed rows are eligible as candidates;
- which EWoK targets are used as queries;
- which exports get written after scoring.

The backend's job is to score candidate rows against EWoK query targets in a
geometry that is closer to local training influence than raw text similarity.

## What The Backend Is Trying To Rank

For a checkpoint with parameters `theta`, a query target `t`, and a candidate
training example `x`, the practical question is:

Which exposed training rows point most in the direction that would reduce the
chosen EWoK loss at this checkpoint?

This is not the same as:

- nearest-neighbor retrieval in embedding space;
- lexical overlap with the EWoK prompt;
- raw training loss on the candidate row alone.

The intended interpretation is directional helpfulness:

- large positive scores suggest the row points toward lower EWoK loss;
- large negative scores suggest the row points against that objective;
- scores near zero suggest little checkpoint-local alignment.

## Paper-Style Feature View

A useful paper-style summary is:

$$
\phi_\theta(z) \approx normalize\left(H^{-1/2} P M^{-1/2} \nabla_\theta \ell(z; \theta)\right)
$$

and then

$$
score(t, x) \approx \langle \phi_\theta(t), \phi_\theta(x) \rangle
$$

where:

- `M` is an optimizer second-moment correction;
- `P` is random projection into a smaller feature space;
- `H` is a projected gradient autocorrelation / Hessian-style correction.

That compact view is helpful because it explains the backend as a sequence of
corrections rather than as "just cosine similarity."

## Why EWoK Is The Query

The repo does not use a generic LM prompt-completion loss as the main query.
The point of the attribution pipeline is to ask which data supports controlled
world-knowledge behavior on EWoK.

Each EWoK item supplies two contexts and two targets:

- `C1`, `T1`
- `C2`, `T2`

The intended structure is:

- `T1` should fit `C1`;
- `T2` should fit `C2`;
- `T2` should not fit `C1`;
- `T1` should not fit `C2`.

The backend therefore evaluates four conditional scores:

- `s11 = log P(T1 | C1)`
- `s12 = log P(T2 | C1)`
- `s22 = log P(T2 | C2)`
- `s21 = log P(T1 | C2)`

Two views are supported:

- `babylm_completion_choice`
  Uses `m1 = s11 - s12` and `m2 = s22 - s21`.
- `ewok_paper_context_sensitivity`
  Uses `m1 = s11 - s21` and `m2 = s22 - s12`.

The paired query loss is:

$$
L_{EWoK}(t) = \frac{1}{2}\left[softplus\left(-\frac{m_1}{\tau}\right) + softplus\left(-\frac{m_2}{\tau}\right)\right]
$$

where `tau` is the configured temperature.

This gives the backend a smooth, differentiable objective that still matches
the benchmark question: which data seems most aligned with making the model
prefer the plausible concept-context pairing over the implausible one?

## Candidate-Side CE Convention

For a raw candidate token chunk

$$
r = (r_0, r_1, \ldots, r_{L-1}),
$$

the TrackStar adapter now hands Bergson the full unshifted chunk as both
`input_ids` and `labels`.

That is deliberate. Bergson's causal-LM CE collector applies the autoregressive
shift internally, so giving it already-shifted labels would double-shift the
targets and score the wrong prediction problem.

One practical wrinkle is context length. The repo's GPT-2 checkpoints were
trained with externally shifted examples of length `seq_len`, which correspond
to raw chunks of length `seq_len + 1`. A plain Hugging Face GPT-2 forward pass
would fail on that longer chunk when `n_positions == seq_len`. The current
backend therefore keeps the full unshifted candidate chunk at the dataset
layer, but patches the model forward only during Bergson candidate indexing: it
runs the real model on the first `seq_len` tokens, appends one dummy logits
row, and lets Bergson's own `logits[:, :-1]` shift recover the intended
`seq_len` next-token targets without ever indexing `wpe` out of range.

With the current fixed adapter, the default candidate-side scalar is standard
mean next-token cross-entropy:

$$
CE(r) = \frac{1}{L-1}\sum_{i=0}^{L-2} -\log p_\theta(r_{i+1} \mid r_0, \ldots, r_i)
$$

when Bergson is using its default `loss_reduction="mean"` setting.

Earlier local TrackStar versions accidentally passed already-shifted labels into
Bergson. Because Bergson then shifted again internally, that older path was
effectively misaligned by one token. The adapter now avoids that double-shift.

## Current Implemented Score

Two projection layouts now exist:

- `module`
  The legacy Bergson-backed path that projects each 2D weight module
  independently before concatenation.
- `paper_blocks`
  The paper-faithful GPT-2 path that first pools corrected gradients into
  eight contiguous layer blocks, keeps attention and MLP separate, and then
  applies two-sided random projection per pooled block.

For the current 24-layer GPT-2 checkpoints, `paper_blocks` uses:

- 8 layer blocks of 3 layers each
- 2 families per block: `attn` and `mlp`
- 16 pooled blocks total

Within one pooled block, the code treats the constituent module gradients as an
implicit block-diagonal matrix and applies the TrackStar two-sided projection
as:

$$
L \; diag(G_1, \ldots, G_k) \; R^T
=
\sum_i L_i G_i R_i^T
$$

where `L_i` and `R_i` are the deterministic row/column slices of the pooled
block projection matrices that correspond to module `i`. This keeps the
projection mathematically faithful to the paper's "pool first, then project"
design without materializing an enormous sparse matrix.

For a checkpoint `theta`, target item `t`, candidate row `x`, and module `m`,
the current backend can be summarized as:

$$
g_m(x) = H_{mix,m}^{-1/2} \; proj(corr_m(x))
$$

$$
q_m(t) = H_{mix,m}^{-1/2} \; proj(corr^q_m(t))
$$

Candidate-side and query-side features are then concatenated across modules:

$$
g(x) = [g_1(x); \ldots; g_M(x)]
$$

$$
q(t) = [q_1(t); \ldots; q_M(t)]
$$

and the exported score is:

$$
score(t, x) = \frac{\langle q(t), g(x) \rangle}{\|q(t)\|_2 \, \|g(x)\|_2}
$$

So the current backend is best read as:

- query = gradient of the custom EWoK paired loss;
- candidate = gradient of training cross-entropy on the candidate row;
- optional optimizer correction = Adam second-moment correction from
  `optimizer.pt` when available;
- curvature correction = mixed Hessian-style whitening in projected feature
  space;
- final score = cosine-style alignment in that corrected feature space.

## Optimizer-State Correction

When `optimizer.pt` is available next to the checkpoint, the backend can use
checkpoint-local Adam second moments on both sides.

Candidate-side correction:

$$
corr_m(x) = D_m^{-1/2} \nabla_{\theta_m} CE(x)
$$

Query-side correction:

$$
corr^q_m(t) = D_m^{-1/2} \nabla_{\theta_m} L_{EWoK}(t)
$$

If optimizer state is missing, the backend falls back to raw gradients:

$$
corr_m(x) = \nabla_{\theta_m} CE(x)
$$

$$
corr^q_m(t) = \nabla_{\theta_m} L_{EWoK}(t)
$$

This correction matters because raw gradients can be dominated by consistently
large or noisy coordinates. Using checkpoint-local second moments moves the
comparison geometry closer to the update geometry actually used during AdamW
training.

## Mixed Hessian-Style Correction

The backend does not form the exact full Hessian. Instead it builds a
projected, mixed curvature approximation. For each module:

$$
H_{mix,m} = (1 - \lambda) H_{ce,m} + \lambda H_{ewok,m}
$$

where:

- `H_ce,m` is the projected gradient autocorrelation from candidate-side CE
  gradients;
- `H_ewok,m` is the projected gradient autocorrelation from EWoK query
  gradients;
- `lambda` mixes train-side and query-side curvature information.

By default, the backend follows Bergson's `compute_lambda` rule rather than a
fixed hard-coded mixture. A fixed override is still possible with
`--hessian_lambda`.

The purpose of this step is to downweight high-variance or overly common
directions before scoring. Intuitively, it tries to move the ranking closer to:

$$
influence(q, x) \approx - \nabla L(q)^T H^{-1} \nabla L(x)
$$

instead of a plain raw dot product.

## First-Order Intuition

Under a small SGD-style update on row `x`,

$$
\theta' = \theta - \eta \nabla CE(x)
$$

the query loss changes approximately like:

$$
\Delta L_{EWoK}(t) \approx -\eta \langle \nabla L_{EWoK}(t), \nabla CE(x) \rangle
$$

The implemented TrackStar score is a more corrected version of that same
intuition:

- replace raw gradients with optimizer-corrected gradients when possible;
- score in a projected feature space;
- whiten common directions with a mixed Hessian-style correction;
- normalize so the score behaves like directional alignment rather than raw
  norm comparison.

That is why a positive score is best read as "this row points toward reducing
the EWoK mistake signal" rather than "this row is globally important in all
senses."

## How This Differs From The Paper Setup

The current backend captures several important TrackStar-style ingredients:

- custom query gradients rather than generic lexical similarity;
- optimizer-state correction on both candidate and query sides when available;
- mixed Hessian-style preconditioning;
- cosine-normalized scoring;
- reusable checkpoint-local indexing.

But it still differs from the paper's Section 6 setup in a few important ways:

- it uses checkpoint-local candidate retrieval rather than open-set retrieval
  over C4;
- it uses the repo's EWoK paired loss as the query objective;
- its projection details and layer/module handling follow the current Bergson
  integration rather than the exact experimental setup in the paper;
- the runs here are based on AdamW-trained checkpoints, not Adafactor-trained
  checkpoints.

So the backend is closer to "TrackStar-style influence adapted to EWoK" than to
"paper-faithful reproduction."

## Bergson Integration Boundaries

The wrapper intentionally uses Bergson for only part of the pipeline.

This repo still owns:

- checkpoint discovery;
- candidate-row selection from exposures;
- EWoK target construction;
- final export format and checkpoint comparisons.

Bergson mainly owns:

- candidate gradient collection;
- candidate index persistence and loading;
- lower-level scoring machinery in the corrected feature space.

The intended mental model is:

- Bergson owns candidate-side indexing mechanics;
- this repo owns the meaning of the query.

## Important Bergson Contracts

Several implementation details are easy to miss unless you have already read
the code carefully.

### Dataset Access Contract

Bergson does not only call `dataset[i]`. It may also call:

- `dataset[[i, j, ...]]`
- `dataset[slice(...)]`

and it expects batched `input_ids` and `labels` payloads as Python
`list[list[int]]`, not pre-padded tensors. That is why
`bergson_datasets.py` implements custom batched indexing behavior.

### Dataset Teardown Contract

After collection, Bergson expects a Dataset-like object and may call methods
such as:

- `remove_columns(...)`
- `add_column(...)`
- `save_to_disk(...)`

That is why the candidate adapter exposes a narrow teardown bridge rather than
behaving like a plain Python list.

### Partial Run Layout

Bergson writes build artifacts to `run_path.part/` before promoting them to the
stable cache path. The wrapper treats `.part` as a real intermediate artifact
location and only promotes it after a successful build.

### Gradient Loading Contract

The installed Bergson path here returns structured `numpy.memmap` gradient
artifacts rather than a friendly `dict`. The wrapper normalizes those memmaps
into a module-name keyed mapping so query-side and candidate-side features can
be aligned.

### Module Naming Contract

Bergson's stored GPT-2 module names are base-model relative, for example:

- `h.0.attn.c_attn`

while Hugging Face module names look like:

- `transformer.h.0.attn.c_attn`

So the wrapper has to normalize the two namespaces onto one shared module set.

## Why Projection Is Mandatory In Practice

For GPT-2 Medium scale models, raw per-example module gradients are enormous.
For a weight matrix shaped `1024 x 4096`, one example already yields more than
four million coordinates.

A full covariance-style preconditioner over that space is therefore far too
large to build directly. This is why naive raw-preconditioner attempts can show
impossible memory requests.

The default TrackStar path therefore works in a projected feature space rather
than forming full raw-gradient preconditioners.

## Why `proj_dim=16`

The default TrackStar settings are:

- `use_fast_jl = True`
- `proj_dim = 16`

This number is deliberately conservative.

In Bergson, projection is applied on both the left and right sides of a module
gradient, so the effective feature count scales roughly like:

- `proj_dim = 16` -> about `256` features per module;
- `proj_dim = 32` -> about `1024` features per module;
- `proj_dim = 64` -> about `4096` features per module.

That scaling is why a TRAK-style value such as `2048` would be completely
inappropriate here. In Bergson semantics, that would imply a per-module feature
space on the order of millions of coordinates.

`16` is therefore the practical default because it:

- keeps indexing feasible on modest GPU budgets;
- keeps candidate and query features in a shared tractable space;
- still leaves room to scale to `32` for higher-fidelity experiments.

If you need a smaller memory footprint, try `--proj_dim 8`. If you want a
larger approximation budget and have the hardware for it, try `--proj_dim 32`.

## Projection Basis Contract

Projected scores are only meaningful when candidate-side and query-side
gradients use exactly the same deterministic projection matrices. A useful
debugging pattern is:

- raw first-order signal `-<grad L_Q, grad loss_x>` correlates with negative
  one-step target-loss deltas;
- projected TrackStar scores stop correlating.

That pattern is coherent, but it should raise suspicion that the two projected
feature spaces are not actually the same. For Bergson's default Rademacher
projection, the matrix entries come from a NumPy `PCG64` byte stream seeded by
the module/side identifier. Replacing that with `torch.randint` under the same
integer seed produces a different matrix, so raw-gradient agreement can vanish
only after projection.

## Cache Behavior

TrackStar caches checkpoint-local backend artifacts under:

`<output_dir>/cache/stepXXXXXXXX/trackstar/`

That cache is designed for reuse when:

- the checkpoint is unchanged;
- the ordered candidate row set is unchanged;
- index-affecting settings such as projection remain unchanged.

This is what makes it practical to rerun query scoring and export logic without
rebuilding the candidate side every time.

## Practical Reading Guide

If you want to understand the code path behind these notes:

1. read `backend.py` for the end-to-end scoring flow;
2. read `bergson_queries.py` for the EWoK paired loss and query-gradient side;
3. read `bergson_datasets.py` for the candidate adapter and Bergson-facing
   dataset contracts;
4. read `../common/export.py` to see how checkpoint-local scores become the
   exported CSV and JSONL artifacts.
