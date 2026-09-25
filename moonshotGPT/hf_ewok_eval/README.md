# Hugging Face EWoK evaluation

Minimal import of the local HF evaluator into the Moonshot Git checkout.
Import it as `moonshotGPT.hf_ewok_eval`; a separate sibling `hf_ewok_eval`
checkout is not required.

## Included

- `run_queue.py`: HF model loading, raw EWoK evaluation and optional download queue.
- `prompted_ewok_eval.py`: target/context choice and statement True/False scoring.
- `prompts/`: four statement True/False formulations, plain target choice and
  the answer-only context-choice prompt used in the experiments.
- `tests/`: loader, prompt-boundary, metric and scoring tests.
- `../scripts/compare_ewok_true_false.py`: compare raw completion scores with
  `log P(True) - log P(False)` on a local checkpoint.
- `../scripts/eval_ewok_context_choice.py`: generate a context choice (1/2).
- `../scripts/score_ewok_context_choice_labels.py`: score context-choice labels.

The shared implementation remains `moonshotGPT/evaluation/ewok.py`; its imported
version includes the PMI metric required by the queue. The existing portable
`evaluation/ewok_data.py` loader is retained.

Combined accuracy averages the two decision accuracies for each scoring method.
For example, one correct decision and one incorrect decision score 50%, even
when their average margin is positive.

This copy contains code, prompts, a small example configuration and tests. It
does not contain credentials, model weights, local run outputs or additional
EWoK source/derived datasets. The fine-tuning experiments are separate from this
evaluator import.

## Run from the Git repository root

The root `requirements.txt` supplies runtime dependencies; `requirements-dev.txt`
adds pytest. The imported code was checked with the existing environment:
PyTorch 2.7.0 and Transformers 4.57.6.

For the local checkpoint comparison:

```bash
CUDA_VISIBLE_DEVICES=1 EWOK_FULL_SRC=/path/to/ewok_full_jsonl.zip \
  python -m moonshotGPT.scripts.compare_ewok_true_false \
  --model /path/to/local_checkpoint --variant full --dtype float32 \
  --output-dir moonshotGPT/runs/hf_ewok_eval/local_true_false
```

`--model` is required in these local experiment scripts; model/tokenizer files
are loaded locally. For the fast split, use `--variant fast` and optionally
`EWOK_FAST_SRC`. The shared loader accepts either a JSONL directory or ZIP archive.
Use `EWOK_ZIP_PASSWORD` when a local encrypted archive needs a password override.

Direct context-choice generation:

```bash
CUDA_VISIBLE_DEVICES=1 python -m moonshotGPT.scripts.eval_ewok_context_choice \
  --model /path/to/local_checkpoint --variant fast --dtype float32 \
  --output-dir moonshotGPT/runs/hf_ewok_eval/local_context_choice
```

The package also retains the original Hub-oriented CLI:

```bash
python -m moonshotGPT.hf_ewok_eval.prompted_ewok_eval --help
python -m moonshotGPT.hf_ewok_eval.run_queue --help
```

The Hub queue can download models and cleans up its staged model snapshots when
finished. Its default staging directory is `~/.cache/moonshotGPT/hf_models`,
overridable through `HF_EWOK_DOWNLOADS_ROOT` or `--downloads-root`. HF cache
discovery respects `HF_HUB_CACHE` / `HF_HOME`. Output defaults are under the
ignored `moonshotGPT/runs/hf_ewok_eval/` directory. No host-specific SSD paths
are required.

## Scoring conventions

- **Completion choice:** hold C fixed and compare T1/T2 scores.
- **Context sensitivity:** hold T fixed and compare C1/C2 scores.
- **Statement margin:** score each statement with log P(` True`) minus
  log P(` False`), then use those margins for either comparison above.
- **Choice-label scoring:** compare the answer strings `1` and `2`.

Preserve prompt whitespace, BOS handling, answer separators, mean/sum reduction
and tie conventions when comparing runs. The local True/False comparison uses
mean target-token scores. For our Qwen3 runs with the GPT-2 tokenizer, each
True/False label is one token, so mean and sum are identical for those labels.
Context-choice generation uses the existing permissive response parser;
consult response validity/raw outputs rather than equating its score with strict
answer-only compliance.

Run the focused tests without downloading models:

```bash
python -m pytest moonshotGPT/hf_ewok_eval/tests moonshotGPT/tests/test_ewok_eval.py -q
```

## Attribution

EWoK: Ivanova et al. (2025), *Elements of World Knowledge*,
https://ewok-core.github.io/. Keep local EWoK plaintext and derived training data
out of public commits; publish selection rules and result summaries instead.
