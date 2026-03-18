# TrackStar Backend

This directory contains the Bergson-backed attribution method.

The name `TrackStar` is the local backend label used in this repo. Under the
hood, it uses EleutherAI Bergson programmatically for gradient collection,
gradient loading, and query-time scoring.

If you have not seen TrackStar before, the main reference point is the paper
"Scalable Influence and Fact Tracing for Large Language Model Pretraining":

- arXiv abstract:
  https://arxiv.org/abs/2410.17413
- arXiv HTML:
  https://arxiv.org/html/2410.17413

In the paper's framing, TrackStar is a scalable gradient-based attribution
method for asking which pretraining examples most influence a model prediction
or evaluation target. The core idea is not simple text similarity. Instead, it
represents examples using corrected gradient features and ranks training
examples by how much their update directions align with improving the query of
interest.

Its practical goal is:

Which training rows look most likely to improve EWoK performance if the model
were pushed a little further in the direction of those rows?

That is the attribution question this backend is built to answer.

So this README is doing two things at once:

- explaining the general TrackStar idea for readers who do not know the paper
- documenting the specific TrackStar-inspired implementation used in this repo
  for EWoK-focused BOS-row attribution

This distinction matters because the implementation here is adapted to the
repo's setting. It borrows important ideas from the paper, but it is not meant
to be a claim of exact paper reproduction unless stated explicitly.

## What We Are Actually Ranking

This backend is not primarily trying to find rows that are merely "similar" to
an eval prompt.

It is trying to rank BOS-packed training rows by how much their training
gradients align with reducing the chosen EWoK query loss at a checkpoint.

That means:

- large positive scores suggest rows whose gradient direction should help the
  model do better on the EWoK target bundle
- large negative scores suggest rows whose gradient direction pushes against
  good EWoK behavior
- scores near zero suggest rows that are mostly irrelevant to this particular
  EWoK signal

Because the scorer now uses cosine-normalized similarity between candidate and
query gradients, the main interpretation is directional alignment: which rows
point most toward improving the EWoK objective, not merely which rows have the
largest raw gradient norm.

## Why EWoK Matters

EWoK is important here because it is not just another generic next-token
benchmark. The broader EWoK project frames the task as testing whether a model
can use concept-level world knowledge to connect the right targets with the
right contexts in controlled minimal-pair settings.

In the EWoK setup, concepts are the first-class object. Each item is designed
so that one target fits one context and the other target fits the other
context. That makes EWoK useful for attribution because it asks a sharper
question than "what data makes this string more likely?":

which training examples appear to teach the model the right world knowledge so
that it prefers plausible concept-context pairings over implausible ones?

This repo currently uses the fast EWoK bundle, but the underlying motivation is
the same: use a controlled world-knowledge test to study what the model seems
to have learned and which training rows appear to support that knowledge.

## What Makes This Backend Different

The outer pipeline is still the repo’s own:

`config -> checkpoints -> row manifest -> exposures -> candidates -> EWoK targets -> backend -> export -> compare`

TrackStar changes only the backend stage.

Instead of TRAK featurization, it:

1. builds a checkpoint-local candidate dataset from selected BOS row ids
2. builds or reuses a Bergson gradient index under the cache directory
3. computes query gradients from the custom EWoK paired softplus loss
4. scores candidate rows against those query gradients with cosine-normalized
   similarity, so high positive rows are rows whose training gradients point
   most toward improving the EWoK objective
5. returns the same `CheckpointScores` shape expected by the shared export code

## Why The Query Loss Matters

This backend is not using a generic language-model loss as its main query,
because the thing we care about is not raw likelihood by itself. The thing we
care about is whether the model uses world knowledge to separate the right
pairing from the wrong pairing.

Each EWoK item is a minimal pair-of-pairs. There are two contexts and two
targets:

- `C1`, `T1`
- `C2`, `T2`

The intended structure is:

- `T1` should fit `C1`
- `T2` should fit `C2`
- `T2` should *not* fit `C1`
- `T1` should *not* fit `C2`

So the backend evaluates four conditional target scores:

- `s11 = log P(T1 | C1)`  correct pairing
- `s12 = log P(T2 | C1)`  wrong target in context `C1`
- `s22 = log P(T2 | C2)`  correct pairing
- `s21 = log P(T1 | C2)`  wrong target in context `C2`

From those scores, it builds margins that measure whether the model prefers the
plausible pairing over the implausible one.

There are two supported views:

- `babylm_completion_choice`
  Measure completion selection under a fixed prompt. In plain language:
  given a context, does the model prefer the world-knowledge-consistent target
  completion over the distractor completion?
  It holds context fixed and asks whether the correct target beats the
  distractor. This is the most direct "does the model complete this prompt with
  the right continuation?" view.
  This uses:
  `m1 = s11 - s12`
  `m2 = s22 - s21`
- `ewok_paper_context_sensitivity`
  Hold target fixed and ask whether the correct context beats the distractor.
  This uses:
  `m1 = s11 - s21`
  `m2 = s22 - s12`

If both margins are positive, the model is behaving the way we want on both
halves of the item. If either margin is negative, the model is favoring an
implausible pairing somewhere.

To turn that into a differentiable attribution query, the backend uses a paired
softplus loss:

`L = 0.5 * [softplus(-m1 / tau) + softplus(-m2 / tau)]`

This objective is trying to do three things at once:

- reward the model for making both intended pairings score higher than their
  distractors
- penalize confident mistakes more than near-ties
- provide a smooth gradient signal so we can ask which training rows would most
  reduce this loss

So the custom softplus objective is not arbitrary. It is the differentiable
version of the benchmark question:

which data points seem most responsible for making the model choose the
plausible world-knowledge pairing over the implausible one?

That is the main reason this backend exists: it lets the repo keep its
task-specific query definition while swapping in a different gradient/indexing
engine.

## Current Implemented Score

For a checkpoint `theta`, an EWoK target item `t`, and a candidate training row
`x`, the backend currently computes:

- a candidate-side gradient for each module `m`:
  `g_m(x) = H_mix,m^(-1/2) proj(corr_m(x))`
- a query-side gradient for each module `m`:
  `q_m(t) = H_mix,m^(-1/2) proj(grad_{theta_m} L_EWoK(t))`

Here:

- `CE(x)` is the causal-LM cross-entropy loss on the BOS training row `x`
- `L_EWoK(t)` is the paired softplus query loss defined above for target item
  `t`
- `corr_m(x)` is the candidate training gradient after optional Adam
  second-moment correction
- `proj(...)` means:
  flatten the raw module gradient if projection is disabled, or
  apply Bergson's left/right random projection and then flatten if projection
  is enabled

When `optimizer.pt` is available next to the checkpoint, the candidate-side
correction is:

`corr_m(x) = D_m^{-1/2} grad_{theta_m} CE(x)`

where `D_m` is the saved Adam second-moment estimate for module `m`, scaled so
the corrected candidate direction is closer to the parameter-update geometry of
AdamW. For Hugging Face `Conv1D` modules, this same correction is applied after
transposing the saved second-moment matrix into Bergson's `[out, in]` layout.

If `optimizer.pt` is missing, the backend falls back to:

`corr_m(x) = grad_{theta_m} CE(x)`

After projection, the backend computes a mixed Hessian-style preconditioner for
each module:

`H_mix,m = (1 - lambda) H_ce,m + lambda H_ewok,m`

where:

- `H_ce,m` is the autocorrelation of the projected candidate CE gradients for
  module `m`
- `H_ewok,m` is the autocorrelation of the projected EWoK query gradients for
  module `m`
- `lambda` is the paper-style query-side mixing coefficient
- the default is:
  `lambda = 0.9`
  so the effective mix is:
  `0.1 * H_ce,m + 0.9 * H_ewok,m`

The wrapper then applies the split TrackStar-style preconditioner:

`H_mix,m^(-1/2)`

to both sides before scoring.

With projection enabled, the per-module feature is:

`proj(G_m) = vec(A_m,left  G_m  A_m,right^T)`

where `A_m,left` and `A_m,right` are the deterministic Bergson random
projection matrices for module `m`.

After that, the backend concatenates all module features:

- `g(x) = [g_1(x); ...; g_M(x)]`
- `q(t) = [q_1(t); ...; q_M(t)]`

and the implemented score is cosine similarity:

`score(t, x) = <q(t), g(x)> / (||q(t)||_2 ||g(x)||_2)`

This is the score that is currently exported in the dense score matrix and in
the top/bottom row summaries.

So the present backend is best understood as:

- query = gradient of the custom EWoK softplus loss
- candidate = Adam-corrected gradient of training CE on a BOS row when
  optimizer state is available, otherwise the raw CE gradient
- score = cosine similarity between the mixed-Hessian-corrected projected
  gradient features

Under a plain first-order SGD-style view of one small training step on row `x`:

`theta' = theta - eta * grad CE(x)`

the EWoK query loss changes approximately like:

`Delta L_EWoK(t) ~= -eta * <grad L_EWoK(t), grad CE(x)>`

The Adam-corrected candidate variant keeps the same intuition, but replaces the
raw candidate gradient with the optimizer-scaled direction `corr(x)` so the
ranking is closer to AdamW update geometry. The mixed Hessian correction then
downweights high-variance shared directions in the projected feature space
before cosine scoring. After cosine normalization, a positive score means the
row points in a direction that should tend to reduce EWoK loss, while a
negative score means the row points against that direction. The normalization
means the backend is mostly ranking directional helpfulness, not raw gradient
magnitude.

## What Paper TrackStar Still Adds

The current backend now has:

- query gradients from the custom EWoK softplus loss
- candidate gradients from BOS-row CE
- optional Adam second-moment correction on the candidate side
- mixed Hessian-style split preconditioning on both sides
- cosine normalization

That already makes it much closer to a TrackStar-style influence score than a
plain raw gradient dot product.

However, the Section 6 paper setup still differs in a few important ways:

- optimizer-state correction on both sides
- automatic lambda selection rather than a fixed lambda
- different projection details and layer blocking
- open-set retrieval over C4 rather than checkpoint-local BOS candidates

### What Adafactor Is

The paper's models were trained with `Adafactor`, not AdamW.

Adafactor is an optimizer in the same family as Adam: it keeps a running
estimate of gradient second moments so that large, consistently high-magnitude
coordinates do not dominate the update. The important practical difference is
that for a large weight matrix it does not store a full second-moment matrix.
Instead it stores a row statistic and a column statistic and factorizes the
estimate.

So for a weight matrix `W in R^{O x I}`:

- Adam stores something like a full `O x I` second-moment estimate
- Adafactor stores a cheaper factorized approximation using:
  `row in R^O` and `col in R^I`

This is why Adafactor is attractive at LLM scale: it captures most of the
"which coordinates are usually huge?" information without the full memory cost
of Adam.

In TrackStar, this optimizer-state correction matters because it rescales the
gradient before comparison. Intuitively:

- raw gradients can be dominated by rogue high-magnitude coordinates
- optimizer correction downweights those coordinates
- the corrected gradient is closer to the direction the optimizer would really
  use during training

This repo now does that with Adam second moments on the candidate side when
`optimizer.pt` is available, because AdamW is what these runs were actually
trained with. The query side is still not optimizer-corrected.

### What The Hessian Correction Is

Even after second-moment correction, a plain gradient similarity still ignores
curvature.

The Hessian is the matrix of second derivatives of the loss with respect to the
parameters:

`H = nabla_theta^2 L`

and it describes which parameter-space directions are:

- steep or flat
- coupled to one another
- common nuisance directions rather than task-specific directions

Influence-style methods therefore try to use a corrected similarity like:

`influence(q, x) ~= - grad L(q)^T H^{-1} grad L(x)`

instead of just:

`<grad L(q), grad L(x)>`

The reason this matters is that raw dot products can be misleading:

- some directions have large gradients simply because they are common
- some directions correspond to shared templates or formatting
- some useful directions only make sense jointly, not coordinate by coordinate

So the Hessian correction is trying to answer a better question:

if we move the model a little in the direction of training row `x`, after
accounting for the local geometry of the loss surface, how much should the
query loss change?

### What TrackStar Actually Uses For The Hessian

The paper does not form the exact full Hessian. That would be intractable.

Instead, it uses a projected Gauss-Newton / gradient-autocorrelation
approximation in a blockwise feature space, then applies the inverse square
root of that approximation to both the train and query vectors.

In practice, this does two useful things:

- it whitens high-variance common directions
- it makes similarity closer to actual influence than plain cosine similarity

For the open-set setup, the paper goes one step further and mixes:

- a train-side curvature estimate
- a query-task curvature estimate

This mixture is meant to suppress directions that are common for the task
itself, such as template-like query components.

This wrapper now does the same style of split preconditioning, but with a
fixed paper-style default:

- `lambda = 0.9`

which means:

- `10%` candidate CE Hessian
- `90%` EWoK query Hessian

That is much closer to the paper's intended semantics than the earlier local
two-weight interface. It is still not identical to the paper's Section 6 setup,
because the paper chooses lambda automatically rather than fixing it by hand,
and for C4 the effective lambda is often even more query-heavy.

### Why We Still Care About It

With the Hessian correction, the current backend is no longer just
optimizer-scaled projected gradient retrieval. It is much closer to a true
TrackStar-style score.

That is already useful for our EWoK question, because it asks which training
rows point most toward improving the EWoK objective while downweighting common
high-variance directions. But the remaining gaps mean the backend can still
depart from paper TrackStar in how it values:

- common high-variance directions
- task-template directions
- coordinates that are large but not especially influential after curvature is
  accounted for

So if we want to move this backend closer to paper TrackStar, the biggest
remaining method gap is:

- add query-side optimizer correction, then replace the fixed
  `lambda = 0.9` with either the paper's automatic rule or a better-justified
  EWoK-specific choice

## Bergson Audit Summary

This section reflects a direct code review of the local Bergson checkout under
`/home/jorge/tokenPred/bergson/bergson/`.

The key Bergson files for understanding this backend are:

- `trackstar.py`
- `build.py`
- `collection.py`
- `collector/collector.py`
- `collector/gradient_collectors.py`
- `builder.py`
- `data.py`
- `score/score.py`
- `score/scorer.py`

### How Native Bergson Works

Native Bergson TrackStar is a five-step pipeline:

1. compute value-side normalizers and preconditioners
2. compute query-side normalizers and preconditioners
3. mix those preconditioners
4. build a query index
5. score the value dataset against that query

That is a good fit for Bergson's own CLI workflow, but it is not the workflow
this repo wants. This repo already owns:

- checkpoint selection
- BOS row manifesting
- candidate selection
- EWoK query construction
- export shape
- checkpoint comparison

So this integration uses Bergson more narrowly as:

1. candidate gradient collector / index writer
2. candidate gradient loader
3. scorer backend

### What This Repo Uses From Bergson

The TrackStar wrapper intentionally keeps the repo's outer pipeline and only
borrows the backend mechanics.

Concretely:

- candidate rows are still BOS rows selected by this repo
- candidate gradients are still collected with Bergson
- the candidate-side index uses Bergson's causal-LM CE collection path
- EWoK query gradients are *not* built with Bergson's generic query dataset
  path
- EWoK query gradients are computed manually from the custom paired softplus
  loss in `bergson_queries.py`
- candidate/query scoring is then done in the Bergson-compatible feature space
  with cosine normalization enabled by default

So the mental model should be:

- Bergson owns candidate-side indexing machinery
- this repo owns the meaning of the query

### Important Bergson Contracts

The Bergson code review surfaced several contracts that are easy to miss if you
only look at the top-level API.

#### Dataset Access Contract

Bergson's collector does not only call `dataset[i]`.

It also calls:

- `dataset[[i, j, ...]]`
- `dataset[slice(...)]`

and expects batched `input_ids` and `labels` as Python `list[list[int]]`
payloads, not pre-padded tensors. That is why `bergson_datasets.py` has a
custom batched indexing path.

#### Dataset Teardown Contract

After collection, Bergson mutates the dataset into a Hugging Face
Dataset-like object by calling methods such as:

- `remove_columns(...)`
- `add_column(...)`
- `save_to_disk(...)`

That is why the candidate adapter also exposes a narrow Dataset-like teardown
bridge.

#### Partial Run Layout

Bergson writes build artifacts to `run_path.part/` first and only later
promotes them to `run_path/`.

When we call lower-level collection functions directly, we have to account for
that layout ourselves. The wrapper now treats `.part` as a real intermediate
artifact location and promotes it into the stable cache path after a successful
build.

#### Gradient Loading Contract

`load_gradients(...)` returns a structured `numpy.memmap` in the installed
Bergson version here, not a friendly dict. That memmap uses `dtype.names` as
the module list, so the wrapper normalizes it into
`module_name -> [num_examples, feature_dim]`.

#### Module Naming Contract

Bergson's indexed module names are base-model relative for GPT-2 style models,
for example:

- `h.0.attn.c_attn`

whereas Hugging Face `GPT2LMHeadModel.named_modules()` yields:

- `transformer.h.0.attn.c_attn`

So the query-side module lookup has to normalize those two name spaces onto the
same module set.

### Why This Backend Skips Full TrackStar Preconditioners

For GPT-2-medium-scale modules, raw per-example weight gradients are enormous.

For example, a matrix like `1024 x 4096` has:

- `4,194,304` gradient coordinates per example

A full covariance-style preconditioner over that space is on the order of:

- `4,194,304 x 4,194,304`

which is tens of terabytes in `float32`. That is why an OOM can appear to ask
for absurd amounts of memory such as `65536 GiB`: Bergson is trying to form
`P^T P` over raw gradients.

The wrapper therefore does **not** use Bergson's full raw preconditioner path
in the default TrackStar run. Instead it uses projected candidate gradients and
projected query gradients in a shared feature space.

## Why `proj_dim=16`

The TrackStar runner now defaults to:

- `use_fast_jl = True`
- `proj_dim = 16`

This choice is deliberate.

First, `16` matches Bergson's own `IndexConfig` default.

Second, Bergson's `projection_dim` is not analogous to the larger default
dimension used by the TRAK backend in this repo. In Bergson, projection is
applied on the left and right sides of each module gradient. So a module's
stored feature size is roughly:

- `proj_dim x proj_dim`

which means the effective feature count scales quadratically:

- `proj_dim = 16` gives about `256` features per module
- `proj_dim = 32` gives about `1024` features per module
- `proj_dim = 64` gives about `4096` features per module

That is why a TRAK-style value such as `2048` is wildly inappropriate here.
For Bergson semantics, `2048` would imply a per-module feature space on the
order of millions of coordinates.

`16` is therefore the conservative default because it:

- keeps candidate indexing practical on a single GPU
- keeps scoring dimensions aligned across candidate and query paths
- preserves the ability to increase `--proj_dim` later if you want a larger
  approximation budget

If you need a more aggressive memory reduction, try:

- `--proj_dim 8`

If you want a larger feature space and your GPU budget allows it, try:

- `--proj_dim 32`

## Files

- `config.py`
  CLI parsing and the `TrackstarConfig` dataclass.
- `backend.py`
  Main Bergson integration, checkpoint scoring, shard assembly, and cache reuse.
- `bergson_datasets.py`
  Candidate dataset adapter plus candidate-index metadata and fingerprints.
- `bergson_queries.py`
  Query gradient collection for the custom EWoK loss.

## CUDA and Distributed Execution

This is the only backend that supports distributed execution today.

Supported patterns:

- single-process Python
- `torchrun`
- `accelerate launch` compatibility through environment-variable detection

The preferred multi-GPU launcher is `torchrun`.

Examples:

Note on naming:

- any `conda run -n <your_env_name> ...` command below is just an example of
  how this was run locally
- earlier local notes used `babylm` as the conda environment name because that
  is Jorge's personal environment on this machine
- you should replace it with whatever environment name you actually use
- this is unrelated to `babylm_completion_choice`, which is the name of the
  EWoK score view, not the name of a required environment

Single GPU:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 30000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --topk 20 \
  --bottomk 20 \
  --device cuda \
  --distributed none
```

DDP with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_ddp \
  --checkpoint_steps 30000 \
  --max_candidate_rows 50000 \
  --device cuda \
  --distributed ddp
```

Accelerate-compatible launch:

```bash
accelerate launch --num_processes 4 -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trackstar_accelerate \
  --checkpoint_steps 30000 \
  --device cuda \
  --distributed ddp
```

## TrackStar CLI Reference

`run_trackstar.py` does not expose a `--backend` flag because the runner is
already fixed to the TrackStar backend. The CLI surface is:

### Required

- `--run_dir`
  Completed BOS run directory.
- `--data_dir`
  BOS row-packed data directory.

### Checkpoints And Candidates

- `--exp_name`
  Output subdirectory name under `analysis/attribution/`.
- `--output_dir`
  Override the output directory.
- `--cache_dir`
  Override the cache directory.
- `--checkpoint_steps`
  Optional explicit checkpoint steps. If omitted, TrackStar uses all discovered
  checkpoints.
- `--candidate_strategy`
  Candidate selection mode.
- `--max_candidate_rows`
  Cap on checkpoint-local candidate rows after deterministic subsampling.
- `--candidate_seed`
  Seed used when candidate subsampling is needed.
- `--recent_window_steps`
  Step window used by `recent_window`.

### EWoK Query Definition

- `--ewok_score_view`
  One of `babylm_completion_choice` or
  `ewok_paper_context_sensitivity`.
- `--ewok_target_scope`
  One of `overall`, `per_domain`, or `both`.
- `--score_reduction`
  One of `mean` or `sum`.
- `--temperature`
  Temperature used inside the paired softplus loss.
- `--max_targets`
  Optional limit on the number of EWoK items. Useful for smoke tests.

### Export Controls

- `--topk`
  Number of top positively scored rows exported per target.
- `--bottomk`
  Number of bottom negatively scored rows exported per target.
  This is the lightweight way to preserve exact per-target negative examples.
- `--write_dense_scores`
  Write the full dense target-by-row score matrix as `.npy`.

### Execution

- `--device`
  `cuda`, `auto`, or `cpu`.
- `--distributed`
  `none`, `ddp`, or `fsdp`.
- `--batch_size`
  Batch size used for target batching and candidate collection.

### Bergson / TrackStar-Specific

- `--proj_dim`
  Per-side Bergson projection dimension.
- `--use_fast_jl`
  Shared inherited flag that enables projected gradients.
  For TrackStar this is already the default, so passing it is usually
  redundant.
- `--no_fast_jl`
  TrackStar-specific convenience flag that disables projected gradients.
  This is mainly for debugging or tiny models and is not recommended for
  GPT-2-medium-scale runs.

In practice, the most important TrackStar-specific knobs are:

- `--distributed`
- `--proj_dim`
- `--topk`
- `--bottomk`
- `--write_dense_scores`

Important constraints:

- `--distributed ddp|fsdp` requires CUDA.
- `--distributed ddp|fsdp` requires a multi-process launcher.
- `--device cuda` fails loudly if CUDA is unavailable.
- `--device auto` permits CPU fallback for single-process work.
- TrackStar defaults to projected candidate/query gradients with `proj_dim=16`.

## What You Should See In The Terminal

The runner now prints explicit progress lines while TrackStar is running.

On a healthy single-process run, you should see messages along the lines of:

- starting run, with launcher, device, and distributed mode
- resolved checkpoint list
- loaded row manifest
- built exposure index
- built target bundle
- loaded tokenizer and model
- constructed backend
- candidate selection for the current checkpoint
- loading checkpoint weights
- reusing or building the candidate index
- loading index gradients
- collecting query gradients
- scoring targets against candidate rows
- writing artifacts
- finished run

Example status lines:

```text
18:42:11 [attribution][trackstar] starting run launcher=python device=cuda:0 distributed=none exp_name=trackstar_step16000
18:42:14 [attribution][trackstar] checkpoint step=16000: selected 512/18437 candidate row(s) with strategy=between_checkpoints
18:42:15 [trackstar][rank 0/1] loading checkpoint step=16000 from /.../ckpt_periodic_step0016000
18:42:19 [trackstar][rank 0/1] building candidate index for step=16000 with 512 candidate row(s)
18:43:02 [trackstar][rank 0/1] collecting query gradients for 32 local target(s) across 24 module(s)
18:43:11 [trackstar][rank 0/1] finished checkpoint step=16000; score matrix shape=(32, 512)
```

If the terminal stays completely blank, that suggests you are still on an older
checkout or an earlier invocation path that does not include the new status
printing.

## Cache Behavior

TrackStar caches backend artifacts under:

`<output_dir>/cache/stepXXXXXXXX/trackstar/`

The goal is to make candidate-index reuse easy when:

- the checkpoint is unchanged
- the ordered candidate row ids are unchanged
- index-affecting settings such as projection settings are unchanged

This is especially useful when rerunning query scoring or iterating on analysis
without wanting to rebuild the candidate side every time.

One subtle detail: the cache validator now rejects structurally broken Bergson
indexes, such as earlier trial runs that accidentally wrote zero-width module
gradients. So if a previous TrackStar run used an incompatible cache layout or
bad projection setting, the backend should rebuild rather than silently reuse
bad artifacts.

## Reduction Notes

The query layer can internally reduce gradients in three ways:

- item-level
- per-domain
- overall

The main run path currently stays item-level so that exports remain aligned with
the shared pipeline contract. Domain and overall summaries are still produced by
the shared export code after scoring.

## When To Prefer TrackStar

Use TrackStar when:

- you want the Bergson-backed backend
- you want checkpoint-local candidate index reuse
- you want multi-GPU execution
- you want the custom EWoK query loss to remain the query definition

## Dependency Reminder

The environment must be able to import `bergson`.

The backend does not shell out to a Bergson CLI. It imports Bergson directly,
so the Python package itself must be installed in the environment you use to
run the command.

A straightforward setup is:

```bash
cd /home/jorge/tokenPred
git clone https://github.com/EleutherAI/bergson.git

conda run -n <your_env_name> pip install -e /home/jorge/tokenPred/bergson
conda run -n <your_env_name> python -c "import bergson; print(bergson.__file__)"
```

The clone directory is not special. It can live anywhere you want. The key
requirement is that your chosen Python environment can import `bergson` after
installation.

If you are reading old local notes or terminal logs from this repo, you may see
`babylm` used as the environment name. That is just one developer's local conda
environment, not a requirement of the backend.
