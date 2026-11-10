# Packed document attention

The Qwen3 trainer has an optional native PyTorch variable-length FlashAttention
adapter. The default remains the existing `--attention_mode sdpa` path.

| CLI mode | Training attention segments |
| --- | --- |
| `sdpa` | Existing Transformers SDPA, causal within each packed row |
| `flash_varlen` | Native varlen FlashAttention, one causal segment per row |
| `flash_varlen_document` | Same native kernel, one causal segment per document fragment |

Use the two varlen modes for the controlled masking comparison. Both require
CUDA BF16/FP16, Qwen3 full-attention layers, and the native
`torch.nn.attention.varlen.varlen_attn` API. There is no separate `flash-attn`
package dependency. Attention dropout must be zero.

## Semantics

The adapter uses the same tokenized stream, packing, block shuffle, next-token
labels, and position IDs as the original trainer. It does not repack or pad
documents. The tokenizer BOS marker starts a new document; ClimbMix metadata
records this as GPT-2 token 50256. BOS belongs to the following document.
Each row boundary also starts a segment, including a row beginning in the
middle of a document. Adjacent BOS markers form valid one-token segments.

Only attention changes. The loss still includes every original target,
including prediction of a BOS token from the last token of the previous
document. The BOS query predicts the first text token of its own document
without attending to the previous document. Position IDs are not reset.

Offsets are constructed once per microbatch from the CPU loader tensor,
copied to the GPU, and shared across the layers. The varlen kernel applies
causality within these segments without constructing a dense attention mask.
Liger RoPE/RMSNorm/SwiGLU and Muon remain on their existing paths.

## Validation, benchmarks, and checkpoints

Both varlen modes score the same validation prefix twice: once with ordinary
row-level causal attention and once with document isolation. In `scalars.jsonl`,
`val_loss` always means ordinary stream loss; `document_isolated_val_loss` is
the second view. These fields have identical definitions for both training arms.

Calls without explicit segment metadata, including existing EWoK/CORE evaluators
and generation, use the normal SDPA path with causal/padding masks and KV-cache
support. A BOS inside a benchmark prompt does not automatically split that
prompt. Saved model weights load through standard Hugging Face APIs; custom
attention registration is only needed for the varlen training calls.

`run_config.json`, optimizer-step scalar records, and `trainer_state.json`
record `attention_mode`. Resuming with a different mode fails before modifying
the run. Older checkpoints without the field are treated as `sdpa`.
`--init_from_ckpt` remains available for deliberate weight-only interventions;
those are different experiments from a matched pretraining comparison.

## Integration check

Run from `moonshotGPT`, in the environment used for training, on idle GPUs:

```bash
python -u \
  scripts/check_document_attention_integration.py \
  --gpus 6 7 \
  --output-dir runs/evaluation/document_attention_integration_NEW
```

This uses the full 778,114,048-parameter Qwen3 architecture and the existing
ClimbMix 10B artifact. It tests all three modes sequentially, five optimizer
steps per mode plus replay of steps 4 and 5 from the step-3 checkpoint.
Microbatch size is 2, sequence length 1024, and accumulation is 30. With two
GPUs this is 122,880 tokens per step; it preserves the original per-GPU batch
geometry, not the eight-GPU global batch size.

The check verifies finite training and validation, identical token traces
across arms and resume, matching resumed weights and losses, and usable
optimizer checkpoints. The uninterrupted reference checkpoint is preserved.
The JSON report records median optimizer-step time, peak training memory on
rank 0, validation views, and resume error. Timing includes the full training
step and excludes validation/checkpoint I/O. Five-step results do not establish
an effect on language-model quality or benchmarks.

For a later quality pilot, use identical initialization, corpus/token order,
global batch, token budget, and learning-rate schedule for `flash_varlen` and
`flash_varlen_document`. Do not compare a short new run with an old final model
as evidence for the masking effect.
