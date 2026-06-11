"""Optional Liger Kernel integration for Llama training."""

from __future__ import annotations


def apply_liger_kernel_if_requested(
    *,
    use_liger_kernel: bool,
    model_arch: str,
) -> bool:
    """Patch Hugging Face Llama modules with Liger kernels when requested."""
    if not use_liger_kernel:
        return False
    if model_arch != "llama":
        raise ValueError("--use_liger_kernel is only supported for --model_arch llama.")

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
    except Exception as exc:
        raise RuntimeError(
            "--use_liger_kernel requires a working Liger Kernel install and a CUDA-visible "
            "Triton runtime. Try running inside the babylm environment on a GPU node. "
            f"Original error: {exc}"
        ) from exc

    apply_liger_kernel_to_llama(
        rope=True,
        swiglu=True,
        rms_norm=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    return True


__all__ = ["apply_liger_kernel_if_requested"]
