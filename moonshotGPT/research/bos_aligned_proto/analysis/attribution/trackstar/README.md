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
  one_step_sanity.py
  bergson_datasets.py
  bergson_queries.py
  run_cpt_ablation.py
  run_one_step_sanity.py
  plot_cpt_ablation.py
```

- `config.py`
  TrackStar-specific configuration and CLI surface.
- `backend.py`
  Main Bergson-backed scoring implementation.
- `cpt_ablation.py`
  Shared helper layer for TrackStar continued-pretraining ablations.
- `one_step_sanity.py`
  Shared helper layer for tiny-update target-loss sanity checks.
- `bergson_datasets.py`
  Candidate-example dataset adapter used when building or reading Bergson
  indices.
- `bergson_queries.py`
  EWoK query construction and loss logic for the custom paired objective.
- `run_cpt_ablation.py`
  Paired treated/control continued-pretraining runner.
- `run_one_step_sanity.py`
  Tiny-update evaluator for top/matched-random/bottom candidate groups.
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

By default, the query-side gradient `q(t)` is the gradient of that paired loss
with respect to model parameters.

The TrackStar runner also supports `--query_objective completion_ce`. That mode
keeps one exported row per EWoK item, but uses a TrackStar-style completion
loss on the two correct prompt-completion pairs:

$$
L_{completion}(t) =
-\frac{1}{2}\left[\log P(T_1 \mid C_1) + \log P(T_2 \mid C_2)\right]
$$

where each log-probability uses the configured `--score_reduction` (`mean` or
`sum`) over target tokens. This is meant as an interpretable semantic-support
probe; it does not encode the two counterfactual completions directly.

For the more paper-faithful completion-query shape, use
`--query_objective completion_side_ce`. That expands each EWoK item into two
query rows:

$$
L_{C_1 \to T_1}(t) = -\log P(T_1 \mid C_1),
\qquad
L_{C_2 \to T_2}(t) = -\log P(T_2 \mid C_2)
$$

The exported target IDs get `:completion_side:c1_t1` or
`:completion_side:c2_t2` suffixes, so downstream rankings can inspect which
training examples support each correct side separately.

### Interpreting Paired Completion Queries

Be careful when interpreting one-row paired EWoK queries, especially with
`--query_objective completion_ce`. For the BabyLM completion-choice view, the
paired margin is:

$$
M = \frac{1}{2}\left[(s_{11} - s_{12}) + (s_{22} - s_{21})\right]
$$

so `M > 0` does not imply that both facts are learned. It only means the
average contrast is positive. A checkpoint can satisfy the paired item because
one side is strong enough to outweigh a failed side:

```text
side 1 margin < 0
side 2 margin > 0
paired margin > 0
```

The `completion_ce` query gradient has the same averaging issue:

$$
\nabla_\theta L_{completion}
= \frac{1}{2}\nabla_\theta CE(T_1 \mid C_1)
 + \frac{1}{2}\nabla_\theta CE(T_2 \mid C_2)
$$

so a high-scoring candidate can align with either side, with a failure mode, or
with generic completion features. It is not automatically evidence for both
intended concept relations.

This also changes which windows are retrieved. The candidate pool may be fixed
at the same 250k raw stream windows, but the top-ranked windows are determined
by the query gradient. If

$$
g_{pair} = \frac{1}{2}g_{C_1 \to T_1} + \frac{1}{2}g_{C_2 \to T_2}
$$

then directions that are strong for one side can be cancelled by the other side,
and the resulting nearest windows can be nearest to the mixed vector rather
than to either fact by itself. Running `completion_side_ce` over the same 250k
candidate windows is therefore not just a re-labeling of the paired run: it can
retrieve a different top set because it scores windows against
`g_{C_1 \to T_1}` and `g_{C_2 \to T_2}` separately.

In a 20k-step material-dynamics audit over 250k raw stream windows, the direct
lexical sanity check was weak:

```text
all material top100 query-stem overlap:      7.53%
all material random100 query-stem overlap:   6.37%

strong-item top100 query-stem overlap:      10.8%
strong-item random100 query-stem overlap:    7.5%
strong-item top5 query-stem overlap:        12.0%
```

Here "strong" meant both EWoK sides were correct with
`min(margin_1, margin_2) > 0.2`. Even in that subset, only `6 / 50` top-5 rows
had direct query-term overlap, and `19 / 50` had neither query-term nor broad
material-keyword support.

Two illustrative failure modes from that audit:

- `fabric -> wrinkles` vs. `liquid -> splashes` had a positive paired margin
  because the liquid/splash side was strong, even though the fabric/wrinkle
  side failed. Its top-5 retrieved windows had `0 / 5` query-term hits.
- `twill -> drapes` vs. `sand -> stirs` barely passed as a pair, but the
  twill/drape side failed and the top-5 retrieved windows again had `0 / 5`
  query-term hits.

The practical takeaway is that paired `completion_ce` rankings can contain weak
signal but low precision. For cleaner interpretation, prefer
`completion_side_ce` and restrict downstream analyses to the individual sides
the checkpoint actually scores correctly.

These query losses make the attribution question concrete:

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
- `q(t)` is computed at query time from either the repo's EWoK paired loss, an
  averaged correct-completion CE loss via `--query_objective completion_ce`, or
  separate correct prompt-completion losses via
  `--query_objective completion_side_ce`.

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

For raw token-stream runs, the exact training surface is still
`stream_window`: fixed contiguous windows of `seq_len + 1` tokens. Those
windows can cross document boundaries because that is what the stream trainer
saw. If you want cleaner human interpretation, pass
`--candidate_kind document_aligned_row`. That alternate candidate view scans
the raw shards for the BOS/EOS document marker and scores the first full
model-context row inside each document. Exposure selection then picks documents
whose full document span overlaps the requested training window. This is better
for asking "which documents look relevant?", but it is no longer an exact
reconstruction of the original SGD examples.

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
- `ewok_frac_per_epoch = 1/2`
- `ewok_every = ceil(0.5 * steps_per_epoch)`
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

If you want a different EWoK cadence, `run_cpt_ablation.py` accepts:

- `--ewok_frac_per_epoch 1/2` for half-epoch evals
- `--ewok_frac_per_epoch 1.0` for once-per-epoch evals

### What Gets Written

`run_cpt_ablation.py` writes:

- `ablation_manifest.json`
- `baseline/baseline_ewok_items.jsonl`
- `baseline/baseline_summary.json`
- `ablation_runs.json`
- `ablation_curves.jsonl`
- `ablation_summary.json`
- `plots/`

To reduce disk usage, ablation child runs do not save final model checkpoints
by default. The runner still preserves the metric files, `ewok_items.jsonl`,
aggregated summaries, and plots. If you want child runs to emit final model
artifacts, add `--save_final_checkpoint`.

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

If `--matched_pool_dir` points to a batch root with child condition directories
such as `cluster_2/`, `cluster_4/`, `cluster_5/`, and `cluster_mix/`, the
runner now auto-discovers each child directory containing
`treated_dataset/` and `control_dataset/`, and launches one ablation per
condition under `--output_dir/<condition_name>/`. It also writes a root
`ablation_batch_manifest.json` summarizing those child runs.

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

## One-Step Sanity Check

When you want to verify that the attribution direction itself is sensible
before committing to a full continued-pretraining ablation, the repo now
includes a one-step target-loss check.

The idea is:

1. fix one checkpoint `theta`;
2. build three candidate pools from the saved attribution scores:
   - top-ranked
   - matched-random
   - bottom-ranked
3. repeatedly sample tiny candidate minibatches `B` from each pool;
4. reset to the same checkpoint each trial;
5. compute the targeted EWoK softplus loss `L_Q(theta)`;
6. take one tiny SGD step on the candidate CE loss;
7. measure:

$$
\Delta_Q(B) = L_Q(\theta') - L_Q(\theta)
$$

If the attribution direction is behaving sensibly, top-ranked batches should
tend to make `Delta_Q(B)` more negative than matched-random batches, while
bottom-ranked batches should be less helpful or harmful.

By default this sanity check uses:

- `score_mode = net_pooled`
- the same EWoK filter/view/reduction/temperature stored in the attribution
  run config when `config.json` is available
- a tiny plain SGD step, not a full optimizer-state resume

Example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_one_step_sanity \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/ckpt_periodic_step0016000 \
  --attribution_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/trackstar_varswap_ckpt16000_window16000_20000 \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --step 16000 \
  --group_size 1000 \
  --candidate_batch_size 4 \
  --num_trials_per_group 64 \
  --update_lr 1e-5
```

This writes:

- `config.json`
- `group_top_candidates.csv`
- `group_matched_random_candidates.csv`
- `group_bottom_candidates.csv`
- `trial_results.jsonl`
- `summary.json`

The `summary.json` file reports the baseline query loss and per-group
statistics for:

- `delta_q`
- candidate batch CE loss
- candidate gradient norm
- mean batch selection score

The one-step sanity runner shows tqdm progress for:

- setup / pool construction
- the baseline target-loss pass
- the sampled one-step trial loop

## Raw Gradient Audit

When you want an even stricter sign/orientation check, the repo also includes a
raw-gradient audit runner:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_raw_dot_audit \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/ckpt_periodic_step0016000 \
  --attribution_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/trackstar_varswap_ckpt16000_window16000_20000 \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --step 16000 \
  --num_examples_per_group 64 \
  --update_lr 2e-5
```

This runner samples single examples from the `top`, `matched_random`, and
`bottom` pools and records three quantities for each sampled example:

- exported attribution `selection_score`
- raw first-order signal `-<grad L_Q, grad loss_x>`
- actual tiny-step `Delta_Q(x)` on that single example

It writes:

- `sampled_top_candidates.csv`
- `sampled_matched_random_candidates.csv`
- `sampled_bottom_candidates.csv`
- `audit_results.jsonl`
- `summary.json`

The summary includes global Pearson/Spearman correlations between exported
scores, raw dots, and negative actual deltas so you can see whether the sign
and ranking are coherent before trusting the longer ablation workflow.

## Projection Geometry Audit

When the raw dot path works but projected scores look suspicious, use the
projection-geometry audit before rebuilding a full TrackStar index at a larger
dimension. It recomputes raw and Adam-corrected query/candidate gradients for
sampled single examples, sweeps projection side ranks, and compares:

- `raw_dot` vs `projected_raw_dot`
- `raw_dot` vs `projected_raw_dot_rescaled`
- `adam_dot` vs `projected_adam_dot`
- `adam_dot` vs `projected_adam_dot_rescaled`

The `*_rescaled` metrics compensate for the expected scale introduced by
Bergson's row-normalized two-sided projection, so they are useful for
distinguishing random-basis bugs from ordinary projection/module reweighting.

Example:

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.trackstar.run_projection_geometry_audit \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/ckpt_periodic_step0016000 \
  --attribution_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/trackstar_varswap_ckpt16000_window16000_20000 \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --step 16000 \
  --score_mode net_pooled \
  --group_size 100 \
  --num_examples_per_group 16 \
  --projection_ranks 16 32 64 128 \
  --device cuda
```

If correlations improve smoothly with rank, the projected path is likely
dimension-limited rather than broken. If raw projection improves but Adam
projection stays near zero, inspect the Adam normalizer shapes/scales and the
Adam-corrected feature distribution. If both projected paths stay bad, suspect
projection layout or basis mismatch.

## Adam Projection Diagnostic

When the projection-geometry audit shows healthy raw projection but weak
Adam-corrected projection, run the per-block diagnostic. It samples the same
top/matched-random/bottom pools, then writes per-example and per-paper-block
comparisons between:

- pre-projection raw and Adam dots
- rescaled projected raw and Adam dots
- raw-to-Adam norm amplification
- Adam normalizer scale summaries

Example:

```bash
conda run --no-capture-output -n <your_env_name> python -u -m research.bos_aligned_proto.analysis.attribution.trackstar.run_adam_projection_diagnostic \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/ckpt_periodic_step0016000 \
  --attribution_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/trackstar_varswap_ckpt16000_window16000_20000_paper_blocks_fresh \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --step 16000 \
  --score_mode net_pooled \
  --group_size 100 \
  --num_examples_per_group 12 \
  --projection_ranks 64 \
  --device cuda
```

It writes:

- `example_results.jsonl`
- `block_results.jsonl`
- `normalizer_module_summary.csv`
- `normalizer_block_summary.csv`
- `summary.json`

In `summary.json`, start with
`block_correlations_by_rank.<rank>.top_problem_blocks_by_adam_projection_error`
and `top_blocks_by_mean_abs_adam_dot`. If the same blocks dominate both lists,
the issue is probably a block-specific Adam scale/projection interaction. If
many blocks have poor Adam projection despite normal raw projection, suspect a
global Adam-normalization convention mismatch.

### Current Variable-Swap Findings

For the checkpoint-16000 FineWeb variable-swap run, the score-path audit and
projection diagnostics currently support the following interpretation:

- the raw first-order path is the trusted sanity check for tiny SGD updates
- `paper_blocks` raw projection is behaving like a normal dimension/fidelity
  tradeoff, not like a projection-basis bug
- Adam-corrected gradients are substantially spikier and need more projection
  dimension than raw gradients
- `2^16` total paper-block features is useful for cheap exploratory retrieval,
  but it under-resolves Adam-corrected geometry for this task
- `2^18` total paper-block features is the practical compromise for a fresh
  Adam/TrackStar-style index
- `2^20` is a useful diagnostic upper bound, but likely too expensive for
  ordinary iteration

In the rank sweep, `projection_rank` is the side length per paper block:

```text
rank 64:   16 blocks * 64^2   = 65,536 dims   = 2^16
rank 128:  16 blocks * 128^2  = 262,144 dims  = 2^18
rank 256:  16 blocks * 256^2  = 1,048,576 dims = 2^20
```

The 36-example Adam projection diagnostic showed:

```text
Example-level raw projection
rank 64:   Pearson 0.753, Spearman 0.699, RMSE/RMS 0.950
rank 128:  Pearson 0.808, Spearman 0.755, RMSE/RMS 0.666
rank 256:  Pearson 0.905, Spearman 0.892, RMSE/RMS 0.451

Example-level Adam projection
rank 64:   Pearson 0.318, Spearman 0.286, RMSE/RMS 3.318
rank 128:  Pearson 0.571, Spearman 0.529, RMSE/RMS 1.362
rank 256:  Pearson 0.785, Spearman 0.729, RMSE/RMS 0.769
```

This does not contradict the TrackStar paper's `2^16` setting. The paper treats
`2^16` as a memory/fidelity compromise, notes that higher projection dimension
improves fidelity, and reports that their 8B model had not clearly plateaued at
`2^16`. In this project, the variable-swap Adam geometry appears to need more
dimension than raw geometry; a `2^18` index should be treated as the next
production-ish test.

The added audit code lives in:

- `adam_projection_diagnostic.py`
- `run_adam_projection_diagnostic.py`

The relevant test coverage is in `tests/test_paper_blocks.py`, especially the
paper-block projection and collector parity checks.

That projection-basis fix is query-independent: candidate gradients and query
gradients now share the same deterministic Bergson/TrackStar projection basis.
Projection quality can still vary by query family, especially after
Adam-style scaling, because the gradient distribution can be spikier or less
spiky. So material-dynamics should get its own score-path/projection audit if
the final interpretation depends on Adam-corrected scores.

For a fresh full `2^18` paper-block index, use `--paper_block_features 16384`
because features are specified per block:

```bash
conda run --no-capture-output -n <your_env_name> python -u -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --exp_name trackstar_varswap_ckpt16000_window16000_20000_paper_blocks_2p18 \
  --checkpoint_steps 16000 \
  --candidate_from_step 16000 \
  --candidate_to_step 20000 \
  --max_candidate_rows 50000 \
  --ewok_filter_spec /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/target_diff_variable_swap.json \
  --ewok_score_view babylm_completion_choice \
  --score_reduction mean \
  --projection_layout paper_blocks \
  --paper_block_features 16384 \
  --write_dense_scores \
  --device cuda
```

For a document-aligned material-dynamics run at the cheaper `2^16` setting,
use:

```bash
conda run --no-capture-output -n <your_env_name> python -u -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --exp_name trackstar_material_dynamics_ckpt16000_window16000_20000_doc_rows_paper_blocks \
  --checkpoint_steps 16000 \
  --candidate_from_step 16000 \
  --candidate_to_step 20000 \
  --candidate_kind document_aligned_row \
  --max_candidate_rows 50000 \
  --ewok_filter_spec /home/jorge/tokenPred/moonshotGPT/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/domain_material_dynamics.json \
  --ewok_score_view babylm_completion_choice \
  --score_reduction mean \
  --projection_layout paper_blocks \
  --paper_block_features 4096 \
  --write_dense_scores \
  --device cuda
```

For the current 19,196-candidate window, the main `gradients.bin` storage would
be roughly 18.7 GiB at `2^18`, before smaller metadata and normalizer files.

If you want to reuse one expensive candidate-gradient pass across multiple
query sets and ranking metrics, build a projected feature bank instead of
immediately scoring one query bundle:

```bash
conda run --no-capture-output -n <your_env_name> python -u -m research.bos_aligned_proto.analysis.attribution.trackstar.run_projected_feature_bank build \
  --base_ckpt /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/ckpt_periodic_step0016000 \
  --attribution_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/trackstar_material_dynamics_context_sensitivity_ckpt16000_window16000_20000_doc_rows_paper_blocks \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --step 16000 \
  --score_mode net_pooled \
  --candidate_subset first \
  --max_candidates 0 \
  --banks raw adam \
  --paper_block_features 16384 \
  --storage_dtype float16 \
  --device cuda \
  --output_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/projected_feature_bank_material_dynamics_context_sensitivity_step00016000_2p18
```

Then score any query slice from the saved bank:

```bash
conda run --no-capture-output -n <your_env_name> python -u -m research.bos_aligned_proto.analysis.attribution.trackstar.run_projected_feature_bank score \
  --feature_bank_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/projected_feature_bank_material_dynamics_context_sensitivity_step00016000_2p18 \
  --query_selection highest_margin \
  --num_queries 10 \
  --metrics projected_raw_dot projected_raw_cosine projected_adam_dot projected_adam_cosine trackstar_no_hessian \
  --topk 10 \
  --bottomk 0 \
  --chunk_size 128 \
  --device cuda \
  --output_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name>/analysis/attribution/projected_feature_bank_scores_material_dynamics_context_sensitivity_step00016000_2p18_highest_margin
```

`trackstar_no_hessian` is the Adam-corrected projected cosine score without the
mixed Hessian correction. The full Hessian-corrected TrackStar score is not a
good first target for a `2^18` feature bank, because it requires large
per-block covariance eigendecompositions; use the normal `run_trackstar` path
at a smaller paper-block dimension when you specifically need that comparison.

### Projection Sanity Failure Mode

If the raw first-order signal correlates with tiny one-step loss improvement
but the projected TrackStar score does not, treat that as a projection-space
bug until proven otherwise. Candidate and query gradients must be projected
with the same deterministic Bergson matrices. In particular, Bergson's
Rademacher projection uses a NumPy `PCG64` byte stream seeded from the module
identifier, not `torch.randint` from the same seed. Using different random
generators leaves the two sides in incompatible bases, so a good raw-gradient
correlation can disappear after projection.

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

For direct corpus sweeps, `raw_window_range` bypasses exposure logs and selects
raw stream-window manifest ids directly. This is useful when you want a fixed
candidate bank such as the first 250k training windows rather than the subset
recorded in exposure JSONL:

```bash
torchrun --nproc_per_node=2 -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir /home/jorge/tokenPred/moonshotGPT/experiments/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/fineweb_edu_10B \
  --output_dir /SSD-2/trackstar_raw_windows_2p16/results \
  --cache_dir /SSD-2/trackstar_raw_windows_2p16/cache \
  --checkpoint_steps 20000 \
  --candidate_kind stream_window \
  --candidate_strategy raw_window_range \
  --raw_window_start_id 0 \
  --max_candidate_rows 250000 \
  --projection_layout paper_blocks \
  --paper_block_features 4096 \
  --score_candidate_chunk_size 4096 \
  --distributed ddp \
  --device cuda
```

Use `raw_window_random` with the same range flags for a deterministic random
sample instead of a contiguous prefix. Both raw-window modes require
`candidate_kind=stream_window`.

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
