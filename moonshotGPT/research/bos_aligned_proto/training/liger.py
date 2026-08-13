"""Optional Liger Kernel integration for Llama and Qwen3 training."""

from __future__ import annotations


def apply_liger_kernel_if_requested(
    *,
    use_liger_kernel: bool,
    model_arch: str,
) -> bool:
    """Patch supported Hugging Face decoder modules with Liger kernels when requested."""
    if not use_liger_kernel:
        return False
    if model_arch not in {"llama", "qwen3"}:
        raise ValueError("--use_liger_kernel is only supported for --model_arch llama or qwen3.")

    try:
        if model_arch == "llama":
            from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel
        else:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    except Exception as exc:
        raise RuntimeError(
            "--use_liger_kernel requires a working Liger Kernel install and a CUDA-visible "
            "Triton runtime. Try running inside the babylm environment on a GPU node. "
            f"Original error: {exc}"
        ) from exc

    apply_liger_kernel(
        rope=True,
        swiglu=True,
        rms_norm=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    return True


__all__ = ["apply_liger_kernel_if_requested"]
