# TRAK Backend

This directory contains the original TRAK backend used by the BOS attribution
pipeline.

## What It Does

The TRAK path keeps the repo’s outer BOS attribution flow intact and swaps in
TRAK only for the backend stage.

For each checkpoint it:

1. loads the checkpoint into the model
2. builds a checkpoint-local candidate row dataset
3. featurizes candidate rows with TRAK
4. scores EWoK targets against those features
5. returns a dense score matrix shaped
   `[num_targets, num_candidates]`

The exported artifacts are produced by the shared `common/` modules, not by the
backend itself.

## Files

- `config.py`
  CLI parsing and the `TRAKConfig` dataclass.
- `backend.py`
  Direct integration with the `traker` library plus the train-side BOS scalar.

## Runtime Expectations

- The `traker` package must be installed.
- This backend is single-process only.
- `--distributed` must remain `none`.
- `--device cuda` is the default and fails loudly if CUDA is unavailable.

## Main Command

From the repo root:

```bash
conda run -n babylm python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/<bos_data_dir> \
  --exp_name trak_smoke \
  --checkpoint_steps 30000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

## When To Prefer TRAK

Use TRAK when:

- you want the original backend path
- you want the simplest run surface
- you are not trying to do multi-GPU attribution
- you want parity with earlier TRAK-style experiments

## Caveat

TRAK is the simpler baseline path, but it does not currently provide the
distributed checkpoint-local indexing behavior that the TrackStar/Bergson path
does.
