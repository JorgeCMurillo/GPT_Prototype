import torch
from transformers import AutoModelForCausalLM

from training_utils.model_config import (
    build_causal_lm_config,
    default_llama_intermediate_size,
    validate_checkpoint_config_alignment,
)


def test_build_llama_config_keeps_gpt2_tokenizer_surface() -> None:
    config = build_causal_lm_config(
        model_arch="llama",
        vocab_size=50257,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
        seq_len=1024,
        n_embd=128,
        n_head=4,
        n_layer=2,
        llama_intermediate_size=0,
        llama_num_key_value_heads=0,
        llama_tie_word_embeddings=True,
    )

    assert config.model_type == "llama"
    assert config.vocab_size == 50257
    assert config.bos_token_id == 50256
    assert config.eos_token_id == 50256
    assert config.pad_token_id == 50256
    assert config.hidden_size == 128
    assert config.intermediate_size == default_llama_intermediate_size(128)
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 4
    assert config.tie_word_embeddings is True
    assert config.use_cache is False


def test_tiny_llama_model_instantiates_from_config() -> None:
    config = build_causal_lm_config(
        model_arch="llama",
        vocab_size=128,
        bos_token_id=127,
        eos_token_id=127,
        pad_token_id=127,
        seq_len=32,
        n_embd=32,
        n_head=4,
        n_layer=1,
        llama_intermediate_size=64,
        llama_num_key_value_heads=2,
    )

    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    assert model.config.model_type == "llama"


def test_tiny_qwen3_model_instantiates_and_runs_from_config() -> None:
    config = build_causal_lm_config(
        model_arch="qwen3",
        vocab_size=128,
        bos_token_id=127,
        eos_token_id=127,
        pad_token_id=None,
        seq_len=32,
        n_embd=32,
        n_head=4,
        n_layer=1,
        qwen_intermediate_size=96,
        qwen_num_key_value_heads=2,
        qwen_head_dim=8,
        qwen_tie_word_embeddings=True,
        rope_theta=1_000_000.0,
    )

    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    output = model(input_ids=torch.randint(0, 128, (2, 16)))
    assert model.config.model_type == "qwen3"
    assert model.config.pad_token_id is None
    assert model.config.num_key_value_heads == 2
    assert model.config.head_dim == 8
    assert output.logits.shape == (2, 16, 128)


def test_parameter_matched_qwen3_shape_is_within_point_two_percent_of_llama() -> None:
    config = build_causal_lm_config(
        model_arch="qwen3",
        vocab_size=50257,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        seq_len=1024,
        n_embd=1024,
        n_head=16,
        n_layer=24,
        qwen_intermediate_size=3152,
        qwen_num_key_value_heads=8,
        qwen_head_dim=64,
        qwen_tie_word_embeddings=True,
        rope_theta=1_000_000.0,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")

    qwen_params = sum(param.numel() for param in model.parameters())
    llama_params = 359_794_688
    assert qwen_params == 359_404_544
    assert abs(qwen_params - llama_params) / llama_params < 0.002


def test_checkpoint_alignment_understands_llama_keys() -> None:
    cfg = {
        "model_type": "llama",
        "vocab_size": 50257,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 1024,
        "intermediate_size": 512,
        "num_key_value_heads": 4,
    }

    assert not validate_checkpoint_config_alignment(
        cfg,
        model_arch="llama",
        seq_len=1024,
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=2,
        llama_intermediate_size=512,
        llama_num_key_value_heads=4,
    )
    mismatches = validate_checkpoint_config_alignment(
        cfg,
        model_arch="gpt2",
        seq_len=1024,
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=2,
    )
    assert any("model_arch" in mismatch for mismatch in mismatches)


def test_checkpoint_alignment_understands_qwen3_keys() -> None:
    cfg = {
        "model_type": "qwen3",
        "vocab_size": 50257,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "max_position_embeddings": 1024,
        "intermediate_size": 3152,
        "num_key_value_heads": 8,
        "head_dim": 64,
    }

    assert not validate_checkpoint_config_alignment(
        cfg,
        model_arch="qwen3",
        seq_len=1024,
        vocab_size=50257,
        n_embd=1024,
        n_head=16,
        n_layer=24,
        qwen_intermediate_size=3152,
        qwen_num_key_value_heads=8,
        qwen_head_dim=64,
    )
