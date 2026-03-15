# Word2Vec Lexical Probe

This research package trains a gensim-based Word2Vec baseline from GPT-2 token shards and evaluates it on EWoK with cosine similarity.

## Layout

```text
research/w2v_lexical_probe/
  __init__.py
  corpus.py
  text.py
  model.py
  train_word2vec.py
  eval_ewok_word2vec.py
  README.md
```

## What It Does

- `train_word2vec.py`
  - Reads BOS-delimited `train_*.bin` shards such as `fineweb_edu_10B`
  - Detokenizes them with the tokenizer recorded in `meta.json`
  - Applies moderate text cleanup for word-level Word2Vec
  - Trains a gensim skip-gram negative-sampling model with CPU worker threads
  - Saves `vectors.kv`, `vocab.json`, and `train_summary.json`
  - Can also emit intermediate checkpoints under `checkpoints/`
- `eval_ewok_word2vec.py`
  - Loads a saved Word2Vec run
  - Tokenizes EWoK contexts and targets with the same normalization rules
  - Mean-pools vectors and compares the two targets within each context
  - Supports new gensim runs and legacy PyTorch runs through the same loader

## Purpose

This probe is a lexical baseline for Moonshot, not a replacement for GPT.

Its job is to estimate how much EWoK-relevant lexical semantics are recoverable from the dataset alone using a simple co-occurrence model. In practice, that means:

- train Word2Vec on a chosen corpus view such as `fineweb_edu_10B` or `fineweb_edu_10B_bosrow`
- evaluate those vectors on EWoK
- treat that score as a corpus-level lexical baseline

That gives you a reference point for later GPT-2 comparisons:

- if a GPT-2 model performs around this level, it may mostly be recovering lexical signal already available from local word co-occurrence
- if a GPT-2 model performs clearly above this level, that suggests it is using stronger compositional or contextual representations than the lexical baseline alone

So the practical research question is not "is Word2Vec good enough?" but "how much of EWoK can be explained by lexical semantics in the training corpus, and how far above that baseline do the GPT-2 models get?"

## Default Text Cleanup

The default normalization is intentionally moderate:

- Unicode normalize and lowercase text
- Collapse whitespace and line breaks
- Replace URLs with `<url>`
- Replace emails with `<email>`
- Strip HTML-like tags
- Keep ordinary words, digits, contractions, and hyphenated words
- Drop punctuation-only tokens

This is enough to make detokenized GPT-2 shard text cleaner for Word2Vec without heavily rewriting the corpus.

## Setup

Run from `tokenPred/moonshotGPT`.

Install `gensim` into the active Python environment if needed:

```bash
pip install gensim
```

EWoK sources expected by this package now live in the Moonshot repo root:

- `ewok_fast_jsonl.zip`
  Fast subset used by default.
- `ewok_full_jsonl.zip`
  Full filtered EWoK set. Use this when you want the complete benchmark rather than the fast subset.

## Training

Basic run:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --data_dir fineweb_edu_10B
```

More explicit small-run example:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --data_dir fineweb_edu_10B \
  --max_train_shards 1 \
  --max_docs 2000 \
  --embedding_dim 128 \
  --epochs 1 \
  --workers 8
```

Exposure-aligned example using all BOS-row shards:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --data_dir fineweb_edu_10B_bosrow \
  --max_train_shards 0 \
  --max_docs 0 \
  --workers 44 \
  --vocab_report_every_secs 120
```

Full-EWoK periodic evaluation example:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --data_dir fineweb_edu_10B \
  --workers 44 \
  --eval_every_words 1000000000 \
  --ewok_variant full
```

### Choosing A Dataset

There are two different research questions you may want this probe to answer:

- `fineweb_edu_10B`
  Use this when you want a clean lexical baseline for the underlying source corpus.
- `fineweb_edu_10B_bosrow`
  Use this when you want an exposure baseline for what GPT training actually saw after BOS-row packing.

The BOS-row dataset is intentionally not the same thing as natural document text. Its metadata marks it as `bos_row_packed_bestfit`, and the packing process can crop documents to finish fixed-width rows. That makes it worse for training a "nice" Word2Vec model, but often better for analyzing lexical signal in the actual LM training exposure.

The trainer records this distinction in the run summary and prints it at startup:

- `baseline_role=corpus_baseline`
  Clean source-corpus lexical semantics baseline.
- `baseline_role=exposure_baseline`
  BOS-row lexical exposure baseline, including packing and crop artifacts.

### CPU Worker Guidance

Training is CPU-threaded. On the current host, the default `--workers` resolves to `32`, which is the balanced default for shared use.

If this machine is dedicated to the run, try `--workers 44` first. That matches the physical core count more closely than the full 88 logical CPUs and is usually the best first throughput setting to test.

### Important Flags

- `--workers`
  Number of gensim CPU worker threads.
- `--batch_words`
  Gensim batching knob. Default is `10000`.
- `--sample`
  Downsampling threshold for frequent tokens. Default is `1e-3`.
- `--checkpoint_every_epochs`
  Intermediate checkpoints are disabled by default for speed. Set a positive value to save `checkpoints/epoch_0001/`, `checkpoints/epoch_0002/`, and so on.
- `--eval_every_words`
  Run in-memory EWoK evaluation every N post-normalization Word2Vec training word tokens. This writes interval metrics without saving an intermediate `vectors.kv` checkpoint.
- `--eval_margin_eps`
  Near-tie threshold used when logging periodic EWoK interval metrics.
- `--ewok_variant`
  Choose which EWoK set to use during evaluation. `fast` is the default subset; `full` uses the full filtered benchmark.
- `--ewok_text_preprocessing`
  Choose how EWoK text is tokenized at evaluation time. `probe` uses the lexical-probe tokenizer; `paper` uses the official notebook's simpler lowercase, punctuation-strip, whitespace-split preprocessing.
- `--vocab_report_every_secs`
  Emit an explicit heartbeat during vocabulary building. Set to `0` to disable it.

### Compatibility Notes

- `--batch_size` is still accepted as a deprecated alias for `--batch_words`.
- `--device` is still accepted as a deprecated no-op so older commands do not break.

Runs are written to `runs/research/w2v_lexical_probe/` by default.

## Run Artifacts

New gensim runs contain:

- `vectors.kv`
- `vocab.json`
- `train_summary.json`
- `ewok_interval_metrics.jsonl` when `--eval_every_words` is enabled

Legacy PyTorch runs may still contain `model.pt`. The evaluator can still read those.

## Relation to World Models Literature

As a framing reference, the survey [*Understanding World or Predicting Future? A Comprehensive Survey of World Models*](https://github.com/tsinghua-fib-lab/World-Model) describes two broad views of world models: one centered on internal representations for understanding the present world, and one centered on predicting future states for simulation and decision-making. EWoK is most naturally related to the first view: it probes whether a model can use conceptual world knowledge to distinguish more plausible from less plausible context-target pairings across physical and social domains, rather than serving as a direct benchmark of full future simulation.

- In that sense, EWoK is best read as a benchmark of world-knowledge-sensitive representation and discrimination. Some subsets, especially dynamics-like items, are more prediction-like than others, but the benchmark as a whole should not be described as a direct future-prediction task.
- Different scoring conventions emphasize slightly different capabilities. The original EWoK paper's context-swap scoring is closer to context or world-state sensitivity ("understanding the world"), while a BabyLM-style target-choice scoring is closer to conditional continuation choice or predictive fit. The latter can still be informative, but it should not be overinterpreted as full future simulation across the entire benchmark.

## EWoK Evaluation

The evaluator turns each EWoK text field into a single Word2Vec vector by mean-pooling its in-vocabulary token embeddings.

For a text span `x`, let `T(x)` be the normalized word tokens that survive lookup in the trained Word2Vec vocabulary, and let `e(w)` be the embedding for token `w`. The pooled representation is:

`v(x) = (1 / |T(x)|) * sum_{w in T(x)} e(w)`

If `T(x)` is empty, the code treats that pooled vector as missing. Any cosine that depends on a missing side is set to `0.0`.

### Evaluation Preprocessing Modes

The evaluator supports two tokenization modes for EWoK text:

- `probe`
  The default. Uses the lexical-probe normalization pipeline described above, which is shared with the training/evaluation code in this package.
- `paper`
  A paper-alignment mode added for direct comparison with the official EWoK Word2Vec notebook. This mode lowercases, strips punctuation, splits on whitespace, and applies the same filler-agent filtering.

Why this exists:

- The official paper notebook uses a much simpler text pipeline than the default lexical-probe tokenizer.
- When comparing local Google News Word2Vec numbers to the paper baseline, it is useful to separate "metric differences" from "preprocessing differences."

Example:

```bash
python -m research.w2v_lexical_probe.eval_google_word2vec \
  --ewok_variant full \
  --ewok_text_preprocessing paper
```

For each EWoK item, the code builds four pooled vectors:

- `C1 = v(Context1)`
- `C2 = v(Context2)`
- `T1 = v(Target1)`
- `T2 = v(Target2)`

Then it computes the four cosine similarities:

- `S11 = cos(C1, T1)`
- `S12 = cos(C1, T2)`
- `S22 = cos(C2, T2)`
- `S21 = cos(C2, T1)`

From those four scores, the code reports two evaluation conventions:

### BabyLM Completion Choice

This is the local Moonshot / BabyLM-style convention. It asks whether the correct target beats the distractor within each fixed context.

- `S(C1, T1) > S(C1, T2)`
- `S(C2, T2) > S(C2, T1)`

Equivalently, the code forms two margins:

- `m1 = S11 - S12`
- `m2 = S22 - S21`

Here `m1` is the "official" direction and `m2` is the symmetric reverse check. Positive margins mean the correct target is preferred in that context.

The per-item outputs store three correctness views:

- `correct_official = 1 / 0.5 / 0` for `m1 > / == / < 0`
- `correct_symmetric = 1 / 0.5 / 0` for `m2 > / == / < 0`
- `correct_combined = 1 / 0.5 / 0` for `m > / == / < 0`

The combined margin used by `acc_combined` is:

- `m = 0.5 * (m1 + m2)`

So `acc_combined` is the mean of the per-item combined tie-aware scores.

At aggregation time, the code reports:

- `domain_scores_official[domain]`
  Mean of the tie-aware `correct_official` values within that domain.
- `domain_scores_full[domain] = (acc1, acc2)`
  Where `acc1 = mean(correct_official)` and `acc2 = mean(correct_symmetric)`.
- `domain_margin_stats[domain]`
  Includes `acc_combined`, the mean signed combined margin, the mean absolute combined margin, and the tie rate under `|m| < margin_eps`.

The final `"average"` entries are macro-averages across domains, not a single pooled micro-average over all items.

### EWoK Context Sensitivity

This is the paper-style convention. It holds the target fixed and asks whether the correct context scores higher for that same target:

- `S(C1, T1) > S(C2, T1)`
- `S(C2, T2) > S(C1, T2)`

For this view, the Word2Vec evaluator now follows the paper's released analysis tie handling:

- `Accuracy_T1 = 1` if `S11 > S21`, `0` if `S11 < S21`, and `0.5` if they are exactly equal.
- `Accuracy_T2 = 1` if `S22 > S12`, `0` if `S22 < S12`, and `0.5` if they are exactly equal.
- Combined context sensitivity is `1` iff `Accuracy_T1 == Accuracy_T2`, otherwise `0`.

The evaluator reports this method as:

- `ewok_context_sensitivity_domain_scores_official`
  Mean of `Accuracy_T1` within each domain.
- `ewok_context_sensitivity_domain_scores_full`
  Per-domain pair `(mean(Accuracy_T1), mean(Accuracy_T2))` plus an `"average"` entry.
- `metrics_by_method["ewok_context_sensitivity"]["domain_context_sensitivity_stats"]`
  Includes the paper-style combined context-sensitivity rate for each domain.

Run:

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name> \
  --write_per_item
```

To evaluate on the full filtered EWoK set instead of the fast subset:

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name> \
  --ewok_variant full
```

To evaluate with the paper notebook's simpler text preprocessing:

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name> \
  --ewok_variant full \
  --ewok_text_preprocessing paper
```

By default, EWoK evaluation excludes the recurring filler agent names used in the benchmark, matching the paper repo's `config/fillers/filler-agent.csv` and the notebook's simple `name+s` handling after punctuation removal. To keep those names in the tokenized contexts and targets, pass `--no-filter_ewok_agent_names`.

To materialize and evaluate the pretrained Google News Word2Vec baseline, run:

```bash
python -m research.w2v_lexical_probe.eval_google_word2vec \
  --write_per_item
```

Google News on the full filtered EWoK set:

```bash
python -m research.w2v_lexical_probe.eval_google_word2vec \
  --ewok_variant full
```

This will reuse an existing lexical-probe run directory if the pretrained vectors have already been imported there; otherwise it downloads the Google model via `gensim.downloader`, saves it as a normal run under `runs/research/w2v_lexical_probe/`, and writes the usual EWoK outputs.

This writes:

- `ewok_metrics.json`
- `ewok_items.jsonl` when `--write_per_item` is enabled
- `ewok_word2vec_predictions.csv`, a merged EWoK table with per-item Word2Vec scores, BabyLM-style margins, and EWoK context-sensitivity correctness flags

When `--ewok_variant full` is used, the evaluator writes variant-specific artifact names so the fast results remain untouched in the same run directory:

- `ewok_full_metrics.json`
- `ewok_full_items.jsonl`
- `ewok_full_word2vec_predictions.csv`

When `--ewok_text_preprocessing paper` is used, the evaluator adds a preprocessing suffix so the probe-style outputs are preserved:

- `ewok_paperprep_metrics.json` for fast EWoK
- `ewok_full_paperprep_metrics.json` for full EWoK
- matching `*_paperprep_items.jsonl` and `*_paperprep_word2vec_predictions.csv`

### Paper Alignment Notes

The `paper` preprocessing mode is intended to match the official Word2Vec notebook's text cleanup more closely, but it is not the only difference between the released paper analysis and this package.

- The official analysis pipeline also applies item-level inclusion/reversal logic from paper-side CSV files after scoring.
- The `ewok_filtered` / `ewok_full_jsonl` data used here already appears to incorporate at least some of that cleanup: items from `remove_from_results.csv` are absent, and several officially reversed cases already appear in swapped orientation.
- Because of that, this package does not automatically re-apply the paper's exclusion/reversal tables on top of `ewok_filtered`; doing so risks double-applying corrections.

### Periodic EWoK Evaluation During Training

Training can also run EWoK periodically without saving an intermediate Word2Vec model:

```bash
python -m research.w2v_lexical_probe.train_word2vec \
  --data_dir fineweb_edu_10B \
  --max_train_shards 1 \
  --max_docs 2000 \
  --epochs 1 \
  --workers 8 \
  --eval_every_words 100000
```

This path evaluates directly from the in-memory gensim vectors and writes only `ewok_interval_metrics.jsonl` under the run directory. It does not save a temporary or permanent `vectors.kv` checkpoint at each evaluation boundary.

For `--ewok_variant full`, the periodic log is written as `ewok_full_interval_metrics.jsonl` so it does not overwrite the default fast-subset interval log.

For `--ewok_text_preprocessing paper`, the periodic log also gets a preprocessing suffix, e.g. `ewok_paperprep_interval_metrics.jsonl` or `ewok_full_paperprep_interval_metrics.jsonl`.

The interval is counted in post-normalization Word2Vec word tokens, not raw GPT-2 BPE shard tokens.

For the "every 1B words" case, set `--eval_every_words 1000000000`.
