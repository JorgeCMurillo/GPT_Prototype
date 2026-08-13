import sys
import types

import pytest

from research.bos_aligned_proto.training.liger import apply_liger_kernel_if_requested


def test_liger_kernel_noop_when_disabled() -> None:
    assert apply_liger_kernel_if_requested(use_liger_kernel=False, model_arch="gpt2") is False


def test_liger_kernel_requires_llama_arch() -> None:
    with pytest.raises(ValueError, match="model_arch llama"):
        apply_liger_kernel_if_requested(use_liger_kernel=True, model_arch="gpt2")


def test_liger_kernel_applies_safe_llama_patches(monkeypatch) -> None:
    calls = []

    def fake_apply_liger_kernel_to_llama(**kwargs):
        calls.append(kwargs)

    liger_pkg = types.ModuleType("liger_kernel")
    transformers_mod = types.ModuleType("liger_kernel.transformers")
    transformers_mod.apply_liger_kernel_to_llama = fake_apply_liger_kernel_to_llama
    liger_pkg.transformers = transformers_mod

    monkeypatch.setitem(sys.modules, "liger_kernel", liger_pkg)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_mod)

    assert apply_liger_kernel_if_requested(use_liger_kernel=True, model_arch="llama") is True
    assert calls == [
        {
            "rope": True,
            "swiglu": True,
            "rms_norm": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,
        }
    ]


def test_liger_kernel_applies_safe_qwen3_patches(monkeypatch) -> None:
    calls = []

    def fake_apply_liger_kernel_to_qwen3(**kwargs):
        calls.append(kwargs)

    liger_pkg = types.ModuleType("liger_kernel")
    transformers_mod = types.ModuleType("liger_kernel.transformers")
    transformers_mod.apply_liger_kernel_to_qwen3 = fake_apply_liger_kernel_to_qwen3
    liger_pkg.transformers = transformers_mod

    monkeypatch.setitem(sys.modules, "liger_kernel", liger_pkg)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_mod)

    assert apply_liger_kernel_if_requested(use_liger_kernel=True, model_arch="qwen3") is True
    assert calls == [
        {
            "rope": True,
            "swiglu": True,
            "rms_norm": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,
        }
    ]
