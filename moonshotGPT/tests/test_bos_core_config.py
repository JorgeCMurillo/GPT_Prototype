from research.bos_aligned_proto.training.config import parse_args


def test_bos_core_config_defaults_and_overrides() -> None:
    defaults = parse_args(["--data_dir", "/tmp/bos_rows"])
    assert defaults.core_every == 2000
    assert defaults.core_max_per_task == 500
    assert defaults.core_bundle_dir == ""
    assert defaults.core_local_files_only is False
    assert defaults.ewok_reductions == "mean"

    overridden = parse_args(
        [
            "--data_dir",
            "/tmp/bos_rows",
            "--core_every",
            "100",
            "--core_max_per_task",
            "25",
            "--core_bundle_dir",
            "/tmp/eval_bundle",
            "--core_local_files_only",
            "--ewok_reductions",
            "both",
        ]
    )
    assert overridden.core_every == 100
    assert overridden.core_max_per_task == 25
    assert overridden.core_bundle_dir == "/tmp/eval_bundle"
    assert overridden.core_local_files_only is True
    assert overridden.ewok_reductions == "both"
