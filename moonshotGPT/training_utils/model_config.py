"""Model config helpers for architecture ablations on the same token streams."""

from __future__ import annotations

from typing import Any

from transformers import GPT2Config, LlamaConfig


SUPPORTED_MODEL_ARCHES = ("gpt2", "llama")


def normalize_model_arch(model_arch: str) -> str:
    arch = str(model_arch or "gpt2").strip().lower().replace("_", "-")
    aliases = {
        "gpt2": "gpt2",
        "gpt-2": "gpt2",
        "llama": "llama",
        "llama2": "llama",
        "llama-2": "llama",
        "llama3": "llama",
        "llama-3": "llama",
    }
    if arch not in aliases:
        raise ValueError(f"Unsupported model_arch={model_arch!r}; expected one of {SUPPORTED_MODEL_ARCHES}.")
    return aliases[arch]


def default_llama_intermediate_size(n_embd: int) -> int:
    return int(4 * int(n_embd))


def build_causal_lm_config(
    *,
    model_arch: str,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int | None,
    seq_len: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    llama_intermediate_size: int = 0,
    llama_num_key_value_heads: int = 0,
    llama_tie_word_embeddings: bool = True,
    rope_theta: float = 10000.0,
):
    """Build a from-scratch causal-LM config while keeping trainer knobs stable."""

    arch = normalize_model_arch(model_arch)
    if arch == "gpt2":
        config = GPT2Config(
            vocab_size=int(vocab_size),
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            n_ctx=int(seq_len),
            n_positions=int(seq_len),
            n_embd=int(n_embd),
            n_head=int(n_head),
            n_layer=int(n_layer),
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            summary_first_dropout=0.0,
        )
    elif arch == "llama":
        intermediate_size = (
            int(llama_intermediate_size)
            if int(llama_intermediate_size) > 0
            else default_llama_intermediate_size(int(n_embd))
        )
        num_key_value_heads = int(llama_num_key_value_heads) if int(llama_num_key_value_heads) > 0 else int(n_head)
        if int(n_head) % int(num_key_value_heads) != 0:
            raise ValueError(
                "llama_num_key_value_heads must divide n_head, "
                f"got n_head={n_head}, llama_num_key_value_heads={num_key_value_heads}."
            )
        config = LlamaConfig(
            vocab_size=int(vocab_size),
            hidden_size=int(n_embd),
            intermediate_size=int(intermediate_size),
            num_hidden_layers=int(n_layer),
            num_attention_heads=int(n_head),
            num_key_value_heads=int(num_key_value_heads),
            max_position_embeddings=int(seq_len),
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            pad_token_id=None if pad_token_id is None else int(pad_token_id),
            attention_dropout=0.0,
            tie_word_embeddings=bool(llama_tie_word_embeddings),
            rope_theta=float(rope_theta),
        )
    else:  # pragma: no cover - normalize_model_arch guards this.
        raise AssertionError(f"Unhandled model_arch={arch!r}")

    config.use_cache = False
    return config


def expected_config_values(
    *,
    model_arch: str,
    seq_len: int,
    vocab_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    llama_intermediate_size: int = 0,
    llama_num_key_value_heads: int = 0,
) -> dict[str, Any]:
    arch = normalize_model_arch(model_arch)
    if arch == "gpt2":
        return {
            "model_type": "gpt2",
            "vocab_size": int(vocab_size),
            "n_embd": int(n_embd),
            "n_head": int(n_head),
            "n_layer": int(n_layer),
            "n_positions": int(seq_len),
        }
    intermediate_size = (
        int(llama_intermediate_size)
        if int(llama_intermediate_size) > 0
        else default_llama_intermediate_size(int(n_embd))
    )
    num_key_value_heads = int(llama_num_key_value_heads) if int(llama_num_key_value_heads) > 0 else int(n_head)
    return {
        "model_type": "llama",
        "vocab_size": int(vocab_size),
        "hidden_size": int(n_embd),
        "num_attention_heads": int(n_head),
        "num_hidden_layers": int(n_layer),
        "max_position_embeddings": int(seq_len),
        "intermediate_size": int(intermediate_size),
        "num_key_value_heads": int(num_key_value_heads),
    }


def validate_checkpoint_config_alignment(
    cfg: dict[str, Any],
    *,
    model_arch: str,
    seq_len: int,
    vocab_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    llama_intermediate_size: int = 0,
    llama_num_key_value_heads: int = 0,
) -> list[str]:
    """Return human-readable architecture mismatches for a saved HF config."""

    expected = expected_config_values(
        model_arch=model_arch,
        seq_len=seq_len,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        llama_intermediate_size=llama_intermediate_size,
        llama_num_key_value_heads=llama_num_key_value_heads,
    )
    arch = normalize_model_arch(model_arch)
    mismatches: list[str] = []

    got_model_type = cfg.get("model_type")
    if got_model_type is not None and str(got_model_type) != expected["model_type"]:
        mismatches.append(f"model_arch: ckpt={got_model_type}, requested={expected['model_type']}")

    def check(keys: tuple[str, ...], expected_value: int, label: str) -> None:
        got = None
        for key in keys:
            if key in cfg:
                got = int(cfg[key])
                break
        if got is None:
            return
        if int(got) != int(expected_value):
            mismatches.append(f"{label}: ckpt={got}, requested={int(expected_value)}")

    if arch == "gpt2":
        check(("vocab_size",), expected["vocab_size"], "vocab_size")
        check(("n_embd",), expected["n_embd"], "n_embd")
        check(("n_head",), expected["n_head"], "n_head")
        check(("n_layer",), expected["n_layer"], "n_layer")
        check(("n_positions", "n_ctx"), expected["n_positions"], "seq_len")
    else:
        check(("vocab_size",), expected["vocab_size"], "vocab_size")
        check(("hidden_size",), expected["hidden_size"], "n_embd/hidden_size")
        check(("num_attention_heads",), expected["num_attention_heads"], "n_head/num_attention_heads")
        check(("num_hidden_layers",), expected["num_hidden_layers"], "n_layer/num_hidden_layers")
        check(("max_position_embeddings",), expected["max_position_embeddings"], "seq_len/max_position_embeddings")
        check(("intermediate_size",), expected["intermediate_size"], "llama_intermediate_size")
        check(("num_key_value_heads",), expected["num_key_value_heads"], "llama_num_key_value_heads")

    return mismatches


__all__ = [
    "SUPPORTED_MODEL_ARCHES",
    "build_causal_lm_config",
    "default_llama_intermediate_size",
    "expected_config_values",
    "normalize_model_arch",
    "validate_checkpoint_config_alignment",
]
