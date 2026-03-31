# TrackStar Backend

This directory contains the Bergson-backed attribution method used by the
`moonshotGPT` BOS attribution pipeline.

The name `TrackStar` is the local backend label in this repo. Under the hood,
it uses EleutherAI Bergson programmatically for gradient collection, gradient
loading, and query-time scoring.

This implementation is inspired by the TrackStar method introduced in
[`Scalable Influence and Fact Tracing for Large Language Model Pretraining`](https://arxiv.org/html/2410.17413v1),
and it should be understood relative to the original
[`TRAK: Attributing Model Behavior at Scale`](https://proceedings.mlr.press/v202/park23c.html)
baseline.

This repo adapts that method to its own EWoK-focused attribution workflow. It
supports both BOS-packed runs and stream-trained runs, and it should be read as
a practical research implementation, not as a claim of exact paper
reproduction.

## Current Files

```text
trackstar/
  README.md
  METHOD_NOTES.md
  __init__.py
  config.py
  backend.py
  cpt_ablation.py
  bergson_datasets.py
  bergson_queries.py
  run_cpt_ablation.py
  plot_cpt_ablation.py
```

- `config.py`
  TrackStar-specific configuration and CLI surface.
- `backend.py`
  Main Bergson-backed scoring implementation.
- `cpt_ablation.py`
  Shared helper layer for TrackStar continued-pretraining ablations.
- `bergson_datasets.py`
  Candidate-example dataset adapter used when building or reading Bergson
  indices.
- `bergson_queries.py`
  EWoK query construction and loss logic for the custom paired objective.
- `run_cpt_ablation.py`
  Paired treated/control continued-pretraining runner.
- `plot_cpt_ablation.py`
  Plotter for treated/control EWoK margin curves and treated-minus-control
  effects.
- `METHOD_NOTES.md`
  Deeper method notes on the scoring geometry, Bergson integration contracts,
  projection defaults, and current differences from the paper setup.

If you want the technical version of this document, read
[`METHOD_NOTES.md`](METHOD_NOTES.md).

## What This Backend Ranks

TrackStar is a gradient-based training-data influence method. In the paper, the
core object is not lexical similarity and not nearest-neighbor retrieval in raw
embedding space. Instead, for a query example `t` and a training example `x`
under checkpoint parameters `theta`, the method constructs corrected versions of
their gradients and scores them by directional alignment. At a high level, the
paper’s pipeline is:

1. compute per-example gradients with respect to model parameters;
2. correct those gradients using optimizer second-moment statistics so that a
   few high-variance dimensions do not dominate the score;
3. randomly project the corrected gradients into a smaller feature space so
   that large-scale indexing and retrieval are tractable;
4. apply a Hessian-style / curvature-style correction based on projected
   gradient autocorrelation, with an additional query-task-aware mixture in
   TrackStar;
5. unit-normalize the resulting vectors and score them with an inner product,
   which becomes a cosine-style alignment score.

One useful way to summarize the paper is:

$$
\phi_\theta(z) \approx normalize\left(H^{-1/2} P M^{-1/2} \nabla_\theta \ell(z; \theta)\right)
$$

and then

$$
score(t, x) \approx \langle \phi_\theta(t), \phi_\theta(x) \rangle
$$

where:

- `M` stands for an optimizer second-moment correction;
- `P` stands for random projection;
- `H` stands for a projected gradient autocorrelation / Hessian approximation.

The important intuition is that TrackStar tries to rank training examples by
approximate local influence on the query loss, not by surface-form overlap. In
the paper, the query is typically a factual prompt plus a desired completion (or
model completion) used for pretraining-data fact tracing.

This repo keeps that influence-style interpretation, but changes what the query
is. Here, the candidate examples are whichever examples the checkpoint was
actually trained on:

- BOS-packed runs use BOS-packed training rows;
- stream-trained runs use exact contiguous stream windows of length
  `seq_len + 1`.

The query is an EWoK-derived paired objective. So the backend ranks faithful
training examples by how much their training gradients align with reducing an
EWoK query loss at a particular checkpoint.

That distinction matters operationally: a top-ranked example here is not
necessarily the example that looks most semantically similar to the EWoK item.
It is the example whose local training gradient is most aligned with pushing
the model toward a better EWoK decision under this repo’s query loss. This
follows the paper’s broader lesson that textual attribution and causal
influence do not have to be the same thing.

### How This Differs From TRAK

TrackStar is best understood as building on TRAK rather than replacing its core
intuition wholesale. The main differences emphasized by the TrackStar paper, and
relevant to this repo, are:

- output function:
  TRAK is framed around a margin-style output function with an output-to-loss
  conversion factor, whereas the TrackStar paper reports that token-level loss
  gradients work slightly better in its setting. This repo goes one step
  further and uses the gradient of the repo’s custom EWoK paired softplus loss
  for the query side.
- optimizer correction:
  TRAK’s main curvature correction is based on projected gradient
  autocorrelation. TrackStar adds a per-parameter optimizer second-moment
  correction before projection, which the paper argues is both more faithful to
  training dynamics and better at suppressing outlier dimensions. In this repo,
  that correction is taken from `optimizer.pt` when available; otherwise the
  backend falls back to raw gradients.
- task-specific Hessian mixing:
  TRAK uses a projected-gradient autocorrelation approximation. TrackStar mixes
  information from pretraining-example gradients with information from the query
  task gradients, so directions that are common to the task template can be
  downweighted. In this repo, that same idea is instantiated as a mixed
  Hessian-style preconditioner built from candidate-side and EWoK query-side
  projected gradients.
- score interpretation:
  the TrackStar paper explicitly unit-normalizes corrected vectors, making the
  final score a cosine-style alignment quantity. That is also the interpretation
  exported by this backend: direction matters more than raw gradient magnitude.
- repo role:
  the `trak/` backend is the simpler baseline path and preserves parity with
  earlier `traker`-based experiments; the `trackstar/` backend is the more
  query-adaptive, Bergson-backed path used when the EWoK-specific attribution
  question is the main object of interest.

Interpretation:

- large positive scores suggest candidate examples whose gradient direction
  points toward better EWoK behavior;
- large negative scores suggest candidate examples whose gradient direction
  pushes against the desired EWoK behavior;
- scores near zero suggest candidate examples that are mostly irrelevant to
  that particular checkpoint-local EWoK signal.

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

In this backend, the query-side gradient `q(t)` is the gradient of that paired
loss with respect to model parameters.

That loss makes the attribution question concrete:

Which training examples appear most aligned with reducing the EWoK mistake signal
for this checkpoint?

When you want to restrict that question to a specific EWoK slice, the shared
runner now supports file-based target filtering via `--ewok_filter_spec`. The
example JSON specs under
[ewok_query_specs/](/home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs)
let you isolate queries by `domain`, `context_type`, `context_diff`,
`target_diff`, or explicit row index without editing code.

## Current Implemented Score

For each target item `t` and candidate example `x`, the backend builds candidate
and query gradients, optionally applies optimizer-state correction, projects the
per-module gradients, applies the mixed Hessian-style preconditioner, and then
exports cosine similarity between the resulting feature vectors:

$$
score(t, x) = \frac{\langle q(t), g(x) \rangle}{\|q(t)\|_2 \, \|g(x)\|_2}
$$

This mirrors the paper’s corrected-and-normalized influence geometry, but the
two sides are instantiated differently here:

- `g(x)` comes from candidate-side training gradients for BOS-packed rows,
  or exact stream windows, using the example’s training cross-entropy
  objective and stored/loaded through Bergson;
- `q(t)` is computed at query time from the repo’s EWoK paired loss, not from a
  factual prompt-completion objective from the paper.

So the score should be read as:

Which exposed training example is most directionally aligned with lowering this
EWoK loss at this checkpoint?

Important implementation notes:

- when `optimizer.pt` exists next to the checkpoint, candidate and query
  gradients can both use Adam second-moment correction;
- candidate-side sequences are handed to Bergson as full unshifted token
  chunks, because Bergson's causal-LM CE path applies the next-token shift
  internally;
- for GPT-2 checkpoints trained at `seq_len = 1024`, candidate indexing now
  uses a small forward shim so Bergson can score the full `1025`-token raw
  chunk without asking the model for position embeddings past `1024`; this
  avoids the `wpe(position_ids)` CUDA assert while preserving the intended
  external-shift training convention;
- if `optimizer.pt` is missing, the backend falls back to raw gradients;
- current defaults use projected gradients with `use_fast_jl=True` and
  `proj_dim=16`;
- the exported outputs still follow the shared repo format from
  `analysis/attribution/common/export.py`.

For downstream selection, the exported artifacts now support two especially
useful views:

- pooled candidate ranking from `row_summary_stepXXXXXXXX.csv`, including
  `positive_score_sum`:

$$ 
S_+(x_i) = \sum_j max(s(x_i, q_j), 0)
$$

- signed net pooling for treated/control selection:

$$
S_{net}(x_i) = \sum_j s(x_i, q_j)
$$

- per-query candidate ranking from `dense_scores_stepXXXXXXXX.npy`, where one
  target row gives the full set of `s(x_i, q_j)` values for a chosen EWoK item.

If you want to turn those scores into fair continued-pretraining datasets,
`../build_matched_cpt_pools.py` builds treated/control fixed-row pools while
matching on observable candidate metadata such as token count and shard
membership. This now works for both BOS-packed checkpoints and stream-trained
checkpoints: stream-window candidates are materialized as exact fixed windows so
the continued-pretraining pool still contains the exact selected examples.

## Continued-Pretraining Ablation Workflow

The TrackStar-specific downstream workflow is now:

1. run TrackStar and export `row_summary_stepXXXXXXXX.csv`, with
   `dense_scores_stepXXXXXXXX.npy` if you want per-query selection;
2. build a matched pool with `../build_matched_cpt_pools.py`;
3. run `run_cpt_ablation.py` to continue pretraining from one checkpoint on
   both the treated and control pools;
4. inspect `ablation_summary.json` and the generated plots.

This is meant to answer a more causal question than ranking alone:

Do the high-TrackStar rows improve EWoK more than a matched low-score control
pool, when both pools get the same continued-pretraining budget?

### Runner Defaults

V1 uses weights-only continuation from the checkpoint. It does not yet resume
the original optimizer state for the ablation arm itself. The default continued
pretraining setup is:

- `micro_batch_size = 4`
- `total_batch_tokens = 32768`
- effective global batch target = `32` sequences at `seq_len = 1024`
- `num_epochs = 3`
- `ewok_every = steps_per_epoch`
- `hellaswag_every = 0`
- `core_every = 0`

The runner prepares fixed-row training views under its own output directory and
creates a synthetic `val_000000.bin` split by concatenating the matched-pool
train shards, so the existing matched-pool directories can stay immutable.

Recommended first LR sweep:

- `1e-5`
- `2e-5`
- `4e-5`
- `8e-5`

### What Gets Written

`run_cpt_ablation.py` writes:

- `ablation_manifest.json`
- `baseline/baseline_ewok_items.jsonl`
- `baseline/baseline_summary.json`
- `ablation_runs.json`
- `ablation_curves.jsonl`
- `ablation_summary.json`
- `plots/`

The baseline files capture the checkpoint before any ablation updates. The
curves file then stores per-eval-point margins with:

$$
\Delta_{\text{arm}}(t) = M_{\text{arm}}(t) - M_{\text{base}}
$$

and the paired ablation effect is:

$$
\Delta\Delta(t) = \Delta_{\text{treated}}(t) - \Delta_{\text{control}}(t)
$$

which is the same as:

$$
M_{\text{treated}}(t) - M_{\text{control}}(t)
$$

In practical terms:

- positive `delta_from_baseline` means that arm improved over the starting
  checkpoint;
- positive `treated_minus_control` means the high-TrackStar pool beat the
  matched control pool.

### Example Commands

Build the matched pool first:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.build_matched_cpt_pools \
  /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/outputs \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --step 16000 \
  --output_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/matched_positive_pooled \
  --score_mode positive_pooled \
  --num_treated 2048 \
  --max_control_score 0.0
```

Then launch the ablation:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_cpt_ablation \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name>/ckpt_final_step0016000 \
  --matched_pool_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/matched_positive_pooled \
  --output_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/cpt_ablation_step16000 \
  --learning_rates 1e-5,2e-5,4e-5,8e-5 \
  --seeds 42,43,44
```

If you want to inspect the ablation plan without launching child training runs,
add `--dry_run`.

`run_cpt_ablation.py` now also shows an outer tqdm bar for:

- baseline evaluation
- each treated/control child run
- aggregation
- plotting

The child trainer still prints its own logs inside each run. If you want the
older quieter behavior, add `--no_progress`.

By default, the ablation runner now auto-generates:

- the average treated/control and treated-minus-control plots
- a per-domain plot set
- grouped plot sets for `ContextDiff`, `TargetDiff`, and `ContextType`

The per-domain effect plot is laid out as a 4x3 grid when all EWoK domains are
present. The grouped category plots are written under subdirectories such as
`plots/ContextDiff_mean/`, `plots/TargetDiff_mean/`, and
`plots/ContextType_mean/`. The plot titles now also include the matched-pool
selection mode when `summary.json` is available under the matched-pool root,
for example `selection=positive_pooled` or `selection=net_pooled`. The PNG
filenames now include the same information as a tag such as
`selection_positive_pooled` or `selection_net_pooled`.

To regenerate plots from an existing ablation directory:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.plot_cpt_ablation \
  --ablation_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<attrib_run>/cpt_ablation_step16000 \
  --group_by average \
  --reduction mean
```

`plot_cpt_ablation.py` can also render per-domain or per-category views with:

- `--group_by domain`
- `--group_by ContextDiff`
- `--group_by TargetDiff`
- `--group_by ContextType`

## Example Command

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --exp_name trackstar_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --write_dense_scores \
  --device cuda
```

If you want to decouple the scored checkpoint from the candidate-exposure
window, use `--candidate_from_step` and `--candidate_to_step`. For example, to
score the model at checkpoint `16000` against examples exposed between steps
`16000 -> 20000`:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --exp_name trackstar_varswap_ckpt16000_window16000_20000 \
  --checkpoint_steps 16000 \
  --candidate_from_step 16000 \
  --candidate_to_step 20000 \
  --ewok_filter_spec /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/target_diff_variable_swap.json \
  --ewok_score_view babylm_completion_choice \
  --score_reduction mean \
  --write_dense_scores \
  --device cuda
```

Without these explicit overrides, `between_checkpoints` now uses the real
previous discovered checkpoint for a requested step. So `--checkpoint_steps
20000` will default to the `16000 -> 20000` window when `16000` is the previous
saved checkpoint in the run.

By default, TrackStar now shows progress feedback for the main long-running
phases:

- a checkpoint-level tqdm bar in the shared runner;
- a target-level tqdm bar while EWoK diagnostics are being scored;
- a target-level tqdm bar while query gradients are being collected;
- a periodic elapsed-time heartbeat during Bergson's candidate-index build,
  which is the longest opaque step.

If you want the old quieter behavior, add `--no_progress`.

## When To Prefer TrackStar

Use this backend when:

- you want the attribution question to be stated rigorously as:
  which exposed training examples are most aligned with reducing the EWoK query
  loss at this checkpoint?
- you want the repo’s most query-adaptive backend, where the query is not a
  generic prompt-completion score but the paired EWoK objective defined above;
- you want the extra conditioning used by TrackStar-style scoring, especially
  optimizer second-moment correction plus mixed Hessian-style downweighting of
  directions that are globally strong but weakly diagnostic for the current
  EWoK query set;
- you want cosine-style gradient alignment scores whose signs and magnitudes are
  easiest to interpret as helpful vs harmful local training directions for EWoK;
- you want the Bergson-backed backend that the output-inspection notebook is
  most often used with.

In practical terms, prefer TrackStar when the main research question is not
"which training examples look most similar to this target?" but rather "which
training examples appear most responsible, in a checkpoint-local gradient sense, for
the model’s current EWoK behavior?"

Prefer the `trak/` backend instead when you mainly want:

- the simpler baseline path;
- closer parity with earlier `traker` experiments in this repo;
- a less specialized comparison point before adding TrackStar’s optimizer
  correction and mixed-Hessian machinery.

If you want the simpler baseline path for comparison, read `../trak/README.md`.
