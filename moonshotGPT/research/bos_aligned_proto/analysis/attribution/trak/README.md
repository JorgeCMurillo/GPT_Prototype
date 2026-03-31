# TRAK Backend

This directory contains the original TRAK backend used by the BOS attribution
pipeline.

It keeps the repo’s shared attribution flow intact and swaps in TRAK only for
the backend stage.

## Current Files

```text
trak/
  README.md
  __init__.py
  config.py
  backend.py
```

- `config.py`
  CLI parsing and the `TRAKConfig` dataclass.
- `backend.py`
  Integration with the `traker` library plus the BOS candidate/query scoring
  wrapper expected by the shared export code.

## What It Does

For each checkpoint, the backend:

1. loads the checkpoint into the model;
2. builds a checkpoint-local candidate row dataset;
3. featurizes candidate rows with TRAK;
4. scores EWoK query targets against those features;
5. returns a dense score matrix shaped `[num_targets, num_candidate_rows]`.

Export files such as `top_rows_step*.csv` and `row_summary_step*.csv` are still
written by the shared `common/` layer, not by this backend directly.

## Runtime Expectations

- `traker` must be installed.
- This backend is single-process only.
- `--distributed` should remain `none`.
- `--device cuda` is the normal default and should fail loudly if CUDA is not
  available.

## Example Command

```bash
conda run -n <your_env_name> python -m research.bos_aligned_proto.analysis.attribution.run_trak \
  --run_dir /home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/<run_name> \
  --data_dir /home/jorge/tokenPred/moonshotGPT/data/processed/bos_aligned_proto/<data_view> \
  --exp_name trak_smoke \
  --checkpoint_steps 16000 \
  --max_candidate_rows 512 \
  --max_targets 32 \
  --device cuda
```

## When To Prefer TRAK

Use TRAK when:

- you want the simplest backend surface;
- you want parity with earlier attribution experiments in this repo;
- you are happy with a strong single-process baseline.

If your main interest is the newer Bergson-backed retained-data analysis path,
move next to `../trackstar/README.md`.
