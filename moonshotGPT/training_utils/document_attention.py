"""Native packed FlashAttention for controlled document-isolation experiments.

Only calls supplying explicit segment metadata use varlen attention. Ordinary
benchmark/generation calls retain Transformers' SDPA masking and cache behavior.
No token, label, position ID, or model parameter is changed here.
"""

from __future__ import annotations

import torch

ATTENTION_MODES = ("sdpa", "flash_varlen", "flash_varlen_document")
HF_BACKEND = "native_document_varlen"


def validate_attention_resume(saved_state: dict, attention_mode: str) -> None:
    saved_mode = saved_state.get("attention_mode", "sdpa")
    if saved_mode != attention_mode:
        raise ValueError(
            f"Resume attention_mode mismatch: checkpoint={saved_mode!r}, requested={attention_mode!r}. "
            "Use the original mode to resume; use init_from_ckpt for a deliberate intervention."
        )


def attention_forward_kwargs(
    input_ids: torch.Tensor,
    *,
    attention_mode: str,
    bos_token_id: int,
    device: torch.device | str,
) -> dict:
    """Build cumulative offsets once per microbatch, shared by every layer.

    Call with CPU loader tensors to avoid GPU synchronization for boundary
    discovery. BOS starts a new segment; a row starting mid-document is a
    separate fragment. Adjacent BOS markers are valid one-token segments.
    Both modes always split at row boundaries, preserving the original context
    window. Position IDs and next-token labels remain untouched.
    """
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention_mode={attention_mode!r}")
    if attention_mode == "sdpa":
        return {}
    if input_ids.ndim != 2 or not input_ids.numel():
        raise ValueError("Expected a nonempty [batch, sequence] input_ids tensor")
    batch_size, sequence_length = input_ids.shape
    if attention_mode == "flash_varlen":
        cu = torch.arange(batch_size + 1, dtype=torch.int32) * sequence_length
        maximum = sequence_length
    else:
        # Work on CPU even for a caller passing GPU tensors. The training loop
        # passes the original CPU tensor, avoiding this device-to-host copy.
        starts = input_ids.detach().to("cpu").eq(int(bos_token_id))
        starts[:, 0] = True
        indices = starts.reshape(-1).nonzero(as_tuple=True)[0]
        cu = torch.cat((indices, indices.new_tensor([input_ids.numel()]))).to(torch.int32)
        maximum = int((cu[1:] - cu[:-1]).max())
    return {
        # Bypass generic mask construction: segment metadata supplies both the
        # document boundaries and causal constraint without a dense T*T mask.
        "attention_mask": {"full_attention": None},
        "document_cu_seqlens": cu.to(device=device, non_blocking=True),
        "document_max_seqlen": int(maximum),
    }


def _run_varlen(query, key, value, cu, maximum, scale):
    if not query.is_cuda or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Native varlen FlashAttention requires CUDA BF16/FP16 queries; enable mixed precision")
    from torch.nn.attention.varlen import varlen_attn

    return varlen_attn(
        query, key, value, cu, cu, maximum, maximum,
        window_size=(-1, 0), enable_gqa=True, scale=scale,
    )


def native_varlen_attention_forward(
    module, query, key, value, attention_mask,
    dropout=0.0, scaling=None,
    document_cu_seqlens=None, document_max_seqlen=None, **kwargs,
):
    if document_cu_seqlens is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    if attention_mask is not None:
        raise ValueError("Explicit varlen segments cannot be combined with a padding/attention mask")
    if dropout != 0.0 or kwargs.get("sliding_window") is not None:
        raise ValueError("The document attention experiment supports full attention with zero attention dropout")
    if query.shape[2] != key.shape[2] or query.shape[:1] != key.shape[:1]:
        raise ValueError("Explicit document segments require equal query/key lengths without a KV cache")
    if document_max_seqlen is None:
        raise ValueError("document_max_seqlen is required with document_cu_seqlens")
    batch, _, sequence_length, _ = query.shape
    packed = [x.transpose(1, 2).reshape(-1, x.shape[1], x.shape[-1]) for x in (query, key, value)]
    output = _run_varlen(
        *packed, document_cu_seqlens, document_max_seqlen, scaling,
    )
    return output.reshape(batch, sequence_length, query.shape[1], query.shape[-1]), None


def configure_attention(model, attention_mode: str) -> None:
    """Install a registered attention adapter; serialized weights stay standard HF."""
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention_mode={attention_mode!r}")
    if attention_mode == "sdpa":
        return
    if model.config.model_type != "qwen3":
        raise ValueError("The native document attention experiment currently supports Qwen3 only")
    if any(kind != "full_attention" for kind in model.config.layer_types):
        raise ValueError("The native document attention experiment requires full-attention layers")
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS

    AttentionInterface.register(HF_BACKEND, native_varlen_attention_forward)
    # Calls without explicit segments (benchmarks, generation) must still get
    # the normal causal/padding mask, including when using a KV cache.
    AttentionMaskInterface.register(HF_BACKEND, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    model.set_attn_implementation(HF_BACKEND)
    model.config.pretraining_attention_mode = attention_mode
