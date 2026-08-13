from research.bos_aligned_proto.training.config import parse_args


def test_bos_core_config_defaults_and_overrides() -> None:
    defaults = parse_args(["--data_dir", "/tmp/bos_rows"])
    assert defaults.vocab_size == 0
    assert defaults.model_arch == "gpt2"
    assert defaults.llama_intermediate_size == 0
    assert defaults.llama_num_key_value_heads == 0
    assert defaults.llama_tie_word_embeddings is True
    assert defaults.qwen_intermediate_size == 0
    assert defaults.qwen_num_key_value_heads == 0
    assert defaults.qwen_head_dim == 0
    assert defaults.qwen_tie_word_embeddings is True
    assert defaults.rope_theta == 10000.0
    assert defaults.use_liger_kernel is False
    assert defaults.tokenizer_name_or_path == ""
    assert defaults.rho_granularity == "token"
    assert defaults.core_every == 2000
    assert defaults.core_max_per_task == 500
    assert defaults.core_bundle_dir == ""
    assert defaults.core_local_files_only is False
    assert defaults.ewok_reductions == "mean"
    assert defaults.save_final_checkpoint is True
    assert defaults.muon_batch_updates is True
    assert defaults.profile_optimizer_steps is False

    overridden = parse_args(
        [
            "--data_dir",
            "/tmp/bos_rows",
            "--vocab_size",
            "60000",
            "--tokenizer_name_or_path",
            "/tmp/tokenizer",
            "--model_arch",
            "llama",
            "--llama_intermediate_size",
            "2048",
            "--llama_num_key_value_heads",
            "4",
            "--no-llama_tie_word_embeddings",
            "--rope_theta",
            "500000",
            "--use_liger_kernel",
            "--rho_granularity",
            "sequence",
            "--core_every",
            "100",
            "--core_max_per_task",
            "25",
            "--core_bundle_dir",
            "/tmp/eval_bundle",
            "--core_local_files_only",
            "--ewok_reductions",
            "both",
            "--no-muon_batch_updates",
            "--profile_optimizer_steps",
            "--no-save_final_checkpoint",
        ]
    )
    assert overridden.vocab_size == 60000
    assert overridden.model_arch == "llama"
    assert overridden.llama_intermediate_size == 2048
    assert overridden.llama_num_key_value_heads == 4
    assert overridden.llama_tie_word_embeddings is False
    assert overridden.rope_theta == 500000
    assert overridden.use_liger_kernel is True
    assert overridden.tokenizer_name_or_path == "/tmp/tokenizer"
    assert overridden.rho_granularity == "sequence"
    assert overridden.core_every == 100
    assert overridden.core_max_per_task == 25
    assert overridden.core_bundle_dir == "/tmp/eval_bundle"
    assert overridden.core_local_files_only is True
    assert overridden.ewok_reductions == "both"
    assert overridden.save_final_checkpoint is False
    assert overridden.muon_batch_updates is False
    assert overridden.profile_optimizer_steps is True

    qwen = parse_args(
        [
            "--data_dir",
            "/tmp/bos_rows",
            "--model_arch",
            "qwen3",
            "--qwen_intermediate_size",
            "3152",
            "--qwen_num_key_value_heads",
            "8",
            "--qwen_head_dim",
            "64",
            "--qwen_tie_word_embeddings",
            "--rope_theta",
            "1000000",
            "--use_liger_kernel",
        ]
    )
    assert qwen.model_arch == "qwen3"
    assert qwen.qwen_intermediate_size == 3152
    assert qwen.qwen_num_key_value_heads == 8
    assert qwen.qwen_head_dim == 64
    assert qwen.qwen_tie_word_embeddings is True
    assert qwen.rope_theta == 1_000_000
    assert qwen.use_liger_kernel is True
