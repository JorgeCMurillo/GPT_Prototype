from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

rq = import_module("moonshotGPT.hf_ewok_eval.run_queue")


class _FakeModel:
    hf_device_map = {"model.embed_tokens": "cuda:0"}

    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return None


class _FakeTokenizer:
    pad_token = "<pad>"


class _FakeProcessorWithTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeProcessorNoTokenizer:
    pad_token = None
    eos_token = None
    bos_token_id = 7
    eos_token_id = None
    pad_token_id = None

    def __call__(self, *, text, add_special_tokens, return_tensors, padding, **kwargs):
        del add_special_tokens, padding, kwargs
        if isinstance(text, str):
            values = [11, 12, 13]
            if return_tensors == "pt":
                tensor = rq.torch.tensor([values], dtype=rq.torch.long)
                return {"input_ids": tensor}
            return {"input_ids": values}

        rows = [[21, 22], [31, 32, 33]]
        if return_tensors == "pt":
            max_len = max(len(row) for row in rows)
            padded = [row + [0] * (max_len - len(row)) for row in rows]
            tensor = rq.torch.tensor(padded, dtype=rq.torch.long)
            return {"input_ids": tensor}
        return {"input_ids": rows}


class _FakeEwokModule:
    BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
    EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
    PMI_COMPLETION_CHOICE = "pmi_completion_choice"
    SRC = Path("/tmp/ewok_fast_jsonl.zip")
    ewok_df = [1, 2, 3]

    def evaluate(self, model, tokenizer, *, batch_size, return_per_item, score_reduction, return_all_methods, show_progress):
        del model, tokenizer, return_per_item, score_reduction, return_all_methods, show_progress
        return {
            self.BABYLM_COMPLETION_CHOICE: {
                "domain_scores_full": {"average": [0.61, 0.61]},
                "domain_scores_official": {"average": 0.61},
            },
            self.EWOK_CONTEXT_SENSITIVITY: {
                "domain_scores_full": {"average": [0.72, 0.70]},
                "domain_scores_official": {"average": 0.72},
            },
            self.PMI_COMPLETION_CHOICE: {
                "domain_scores_full": {"average": [0.66, 0.64]},
                "domain_scores_official": {"average": 0.66},
            },
        }, [
            {
                "domain": "agent-properties",
                "row_index": 0,
                "score_reduction": "mean",
                "S11_logp_T1_given_C1": -1.0,
                "S12_logp_T2_given_C1": -2.0,
                "S22_logp_T2_given_C2": -1.5,
                "S21_logp_T1_given_C2": -2.5,
            }
        ]


def _write_yaml(path: Path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_queue_config_sorts_smallest_then_model_id(tmp_path: Path) -> None:
    config_path = tmp_path / "queue.yaml"
    _write_yaml(
        config_path,
        {
            "models": [
                {"model_id": "zeta/model", "param_count_b": 7.0},
                {"model_id": "alpha/model", "param_count_b": 0.5},
                {"model_id": "beta/model", "param_count_b": 7.0},
            ]
        },
    )

    models = rq.load_queue_config(config_path)

    assert [(model.model_id, model.param_count_b) for model in models] == [
        ("alpha/model", 0.5),
        ("beta/model", 7.0),
        ("zeta/model", 7.0),
    ]


def test_temporary_hf_xet_mode_sets_and_restores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    with rq._temporary_hf_xet_mode(True):
        assert rq.os.environ["HF_HUB_DISABLE_XET"] == "1"

    assert "HF_HUB_DISABLE_XET" not in rq.os.environ


def test_parser_disables_xet_by_default_and_enable_flag_opt_out() -> None:
    parser = rq._build_parser()

    default_args = parser.parse_args([])
    enabled_args = parser.parse_args(["--enable-xet"])

    assert default_args.disable_xet is True
    assert enabled_args.disable_xet is False


def test_detect_hardware_prefers_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rq.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rq.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, NVIDIA GeForce RTX 3090, 24268, 23000\n1, NVIDIA GeForce RTX 3090, 24268, 22900\n"
        ),
    )

    hardware = rq.detect_hardware()

    assert hardware["probe_method"] == "nvidia-smi"
    assert hardware["num_gpus"] == 2
    assert hardware["gpus"][0]["index"] == 0
    assert hardware["gpus"][0]["free_memory_bytes"] == 23000 * 1024 * 1024
    assert hardware["gpus"][1]["total_memory_bytes"] == 24268 * 1024 * 1024


def test_detect_hardware_torch_fallback_avoids_mem_get_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rq.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rq, "_detect_hardware_with_nvidia_smi", lambda: None)
    monkeypatch.setattr(rq.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        rq.torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(name=f"GPU-{idx}", total_memory=24 * 1024**3),
    )

    def _boom():
        raise AssertionError("mem_get_info should not be called in the torch fallback")

    monkeypatch.setattr(rq.torch.cuda, "mem_get_info", _boom)

    hardware = rq.detect_hardware()

    assert hardware["probe_method"] == "torch_fallback"
    assert hardware["num_gpus"] == 1
    assert hardware["gpus"][0]["free_memory_bytes"] == 24 * 1024**3


def test_weight_family_prefers_safetensors_and_ignores_other_formats() -> None:
    family, ignore_patterns = rq._weight_family_ignore_patterns(
        [
            "config.json",
            "model-00001-of-00009.safetensors",
            "model.safetensors.index.json",
            "pytorch_model-00001-of-00009.bin",
            "consolidated.00.pt",
        ]
    )

    assert family == "safetensors"
    assert "*.bin" in ignore_patterns
    assert "*.pt" in ignore_patterns
    assert "*.safetensors" not in ignore_patterns


def test_weight_family_falls_back_to_pt_when_no_safetensors_or_bin() -> None:
    family, ignore_patterns = rq._weight_family_ignore_patterns(
        [
            "config.json",
            "consolidated.00.pt",
            "consolidated.01.pt",
        ]
    )

    assert family == "pt"
    assert "*.safetensors" in ignore_patterns
    assert "*.bin" in ignore_patterns


def test_resolve_existing_local_snapshot_falls_back_to_hf_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_root = tmp_path / "hub"
    snapshot_dir = (
        hub_root
        / "models--Qwen--Qwen3-4B"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    refs_dir = hub_root / "models--Qwen--Qwen3-4B" / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123", encoding="utf-8")

    monkeypatch.setattr(rq, "_candidate_hf_hub_roots", lambda: [hub_root])

    spec = rq.ModelSpec(model_id="Qwen/Qwen3-4B", param_count_b=4.0)
    target_dir, model_dir, tokenizer_dir = rq._resolve_existing_local_snapshot(
        spec,
        download_root=tmp_path / "downloads",
    )

    assert target_dir == snapshot_dir.resolve()
    assert model_dir == snapshot_dir.resolve()
    assert tokenizer_dir == snapshot_dir.resolve()


def test_build_load_attempts_prefers_single_then_shard_when_single_gpu_fits() -> None:
    spec = rq.ModelSpec(model_id="test/model", param_count_b=7.0)
    hardware = {
        "cuda_available": True,
        "num_gpus": 2,
        "gpus": [
            {"index": 0, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3},
            {"index": 1, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3},
        ],
    }

    attempts = rq.build_load_attempts(
        spec,
        hardware,
        dtype=rq.torch.float16,
        bitsandbytes_available=False,
    )

    assert [attempt["load_strategy"] for attempt in attempts] == ["single_gpu", "multi_gpu_shard"]


def test_build_load_attempts_uses_shard_and_quantization_for_large_models() -> None:
    spec = rq.ModelSpec(model_id="test/huge", param_count_b=72.0)
    hardware = {
        "cuda_available": True,
        "num_gpus": 4,
        "gpus": [
            {"index": idx, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3}
            for idx in range(4)
        ],
    }

    attempts = rq.build_load_attempts(
        spec,
        hardware,
        dtype=rq.torch.float16,
        bitsandbytes_available=True,
    )

    assert [attempt["load_strategy"] for attempt in attempts] == [
        "multi_gpu_shard",
        "quantized_8bit",
        "quantized_4bit",
    ]


def test_build_load_attempts_returns_no_gpu_strategy_when_model_is_too_large_for_single_gpu() -> None:
    spec = rq.ModelSpec(model_id="test/too-big", param_count_b=72.0)
    hardware = {
        "cuda_available": True,
        "num_gpus": 1,
        "gpus": [{"index": 0, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3}],
    }

    attempts = rq.build_load_attempts(
        spec,
        hardware,
        dtype=rq.torch.float16,
        bitsandbytes_available=False,
    )

    assert attempts == []


def test_evaluate_with_oom_retries_reduces_batch_size() -> None:
    class _OomModule:
        def evaluate(self, model, tokenizer, *, batch_size, **kwargs):
            del model, tokenizer, kwargs
            if batch_size > 1:
                raise RuntimeError("CUDA out of memory")
            return {"ok": True}, [{"row_index": 0}]

    metrics, items, final_batch_size, attempts = rq._evaluate_with_oom_retries(
        _OomModule(),
        model=object(),
        tokenizer=object(),
        start_batch_size=4,
    )

    assert metrics == {"ok": True}
    assert items == [{"row_index": 0}]
    assert final_batch_size == 1
    assert [attempt["batch_size"] for attempt in attempts] == [4, 2, 1]


def test_load_text_tokenizer_for_eval_falls_back_to_processor_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()

    class _BoomTokenizerLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("tokenizer unavailable")

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args, kwargs
            return _FakeProcessorWithTokenizer(tokenizer)

    monkeypatch.setattr(rq, "AutoTokenizer", _BoomTokenizerLoader)
    monkeypatch.setattr(rq, "AutoProcessor", _ProcessorLoader)

    loaded = rq._load_text_tokenizer_for_eval(Path("/tmp/fake"), trust_remote_code=False)

    assert loaded is tokenizer


def test_processor_text_tokenizer_adapter_supports_bare_text_calls() -> None:
    adapter = rq._ProcessorTextTokenizerAdapter(_FakeProcessorNoTokenizer())

    encoded = adapter.encode("plain text", add_special_tokens=False)
    batch = adapter(["a", "b"], add_special_tokens=False, return_tensors="pt", padding=True)

    assert encoded == [11, 12, 13]
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert adapter.bos_token_id == 7


def test_load_model_and_tokenizer_for_attempt_falls_back_to_image_text_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = rq.ModelSpec(model_id="Qwen/Qwen2.5-VL-3B-Instruct", param_count_b=3.0)
    asset_paths = {
        "model_dir": tmp_path,
        "tokenizer_dir": tmp_path,
    }
    attempt = {"load_strategy": "cpu", "device_mode": "cpu", "start_batch_size": 2, "quantization": None}
    calls: list[tuple[str, str | None]] = []

    class _CausalLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("Unrecognized configuration class <class 'Qwen2_5_VLConfig'> for this kind of AutoModel")

    class _ImageTextLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            calls.append(("image_text", kwargs.get("attn_implementation")))
            return _FakeModel()

    monkeypatch.setattr(rq, "AutoModelForCausalLM", _CausalLoader)
    monkeypatch.setattr(rq, "AutoModelForImageTextToText", _ImageTextLoader)
    monkeypatch.setattr(rq, "_load_text_tokenizer_for_eval", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(rq, "_flash_attn_is_broken", lambda: False)

    model, tokenizer = rq._load_model_and_tokenizer_for_attempt(
        spec,
        asset_paths=asset_paths,
        attempt=attempt,
        dtype=rq.torch.float16,
        hardware={"cuda_available": False, "num_gpus": 0, "gpus": []},
    )

    assert isinstance(model, _FakeModel)
    assert isinstance(tokenizer, _FakeTokenizer)
    assert calls == [("image_text", None)]
    assert getattr(model, "_hf_ewok_loader_name") == "AutoModelForImageTextToText"


def test_load_model_and_tokenizer_for_attempt_retries_with_eager_when_flash_attn_is_broken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = rq.ModelSpec(model_id="google/gemma-3-4b-it", param_count_b=4.0)
    asset_paths = {
        "model_dir": tmp_path,
        "tokenizer_dir": tmp_path,
    }
    attempt = {"load_strategy": "cpu", "device_mode": "cpu", "start_batch_size": 2, "quantization": None}
    calls: list[str | None] = []

    class _CausalLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            calls.append(kwargs.get("attn_implementation"))
            if kwargs.get("attn_implementation") != "eager":
                raise RuntimeError(
                    "Failed to import transformers.models.gemma3.modeling_gemma3 because of the following error: "
                    "/path/flash_attn_2_cuda.so: undefined symbol: something"
                )
            return _FakeModel()

    @rq.contextmanager
    def _noop_flash_disable(enabled: bool):
        yield

    monkeypatch.setattr(rq, "AutoModelForCausalLM", _CausalLoader)
    monkeypatch.setattr(rq, "_load_text_tokenizer_for_eval", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(rq, "_flash_attn_is_broken", lambda: True)
    monkeypatch.setattr(rq, "_temporarily_disable_flash_attn_imports", _noop_flash_disable)

    model, tokenizer = rq._load_model_and_tokenizer_for_attempt(
        spec,
        asset_paths=asset_paths,
        attempt=attempt,
        dtype=rq.torch.float16,
        hardware={"cuda_available": False, "num_gpus": 0, "gpus": []},
    )

    assert isinstance(model, _FakeModel)
    assert isinstance(tokenizer, _FakeTokenizer)
    assert calls == [None, "eager"]
    assert getattr(model, "_hf_ewok_attn_implementation") == "eager"


def test_build_ewok_payload_matches_moonshot_field_names() -> None:
    shared_ewok = _FakeEwokModule()
    payload, summary = rq._build_ewok_payload(
        shared_ewok,
        metrics_by_method_mean={
            shared_ewok.BABYLM_COMPLETION_CHOICE: {
                "domain_scores_full": {"average": [0.61, 0.61]},
                "domain_scores_official": {"average": 0.61},
            },
            shared_ewok.EWOK_CONTEXT_SENSITIVITY: {
                "domain_scores_full": {"average": [0.72, 0.70]},
                "domain_scores_official": {"average": 0.72},
            },
            shared_ewok.PMI_COMPLETION_CHOICE: {
                "domain_scores_full": {"domain": [0.66, 0.64], "average": [0.66, 0.64]},
                "domain_scores_official": {"average": 0.66},
            },
        },
        per_item_records=[{"row_index": 0}],
        batch_size=2,
        elapsed_seconds=1.5,
    )

    assert payload["ewok_source"] == str(shared_ewok.SRC)
    assert payload["batch_size"] == 2
    assert payload["mean"]["metrics_by_method"][shared_ewok.BABYLM_COMPLETION_CHOICE]["domain_scores_full"]["average"] == [0.61, 0.61]
    assert summary["babylm_completion_choice_official_mean_average"] == 0.61
    assert summary["ewok_context_sensitivity_official_mean_average"] == 0.72
    assert summary["pmi_completion_choice_official_mean_average"] == 0.66
    assert summary["pmi_completion_choice_symmetric_mean_average"] == 0.64
    assert summary["pmi_completion_choice_full_mean_clean_average"] == 0.65


def test_download_assets_reuses_complete_root_level_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = rq.ModelSpec(model_id="Qwen/Qwen3-4B-Base", param_count_b=4.0)
    target_dir = tmp_path / spec.model_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer1": "model-00001-of-00003.safetensors",
                    "layer2": "model-00002-of-00003.safetensors",
                    "layer3": "model-00003-of-00003.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    for shard_name in (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ):
        (target_dir / shard_name).write_text("weights", encoding="utf-8")

    monkeypatch.setattr(rq, "_list_model_repo_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rq,
        "snapshot_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("snapshot_download should not be called")),
    )

    asset_paths = rq._download_assets(
        spec,
        download_root=tmp_path,
        hf_token=None,
        disable_xet=True,
        max_retries=1,
    )

    assert asset_paths["download_dir"] == target_dir
    assert asset_paths["model_dir"] == target_dir
    assert asset_paths["tokenizer_dir"] == target_dir


def test_run_model_writes_artifacts_and_cleans_download_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = rq.ModelSpec(model_id="test/model", param_count_b=1.0)
    output_root = tmp_path / "results"
    downloads_root = tmp_path / "downloads"
    fake_ewok = _FakeEwokModule()

    def fake_download_assets(spec, *, download_root, hf_token, disable_xet, max_retries):
        del hf_token
        assert disable_xet is True
        assert max_retries == rq.DEFAULT_DOWNLOAD_RETRIES
        download_dir = Path(download_root) / spec.model_slug
        model_dir = download_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "download_dir": download_dir,
            "model_dir": model_dir,
            "tokenizer_dir": model_dir,
            "tokenizer_source": spec.model_id,
            "tokenizer_mode": "model",
            "tokenizer_override_path": None,
        }

    monkeypatch.setattr(rq, "_download_assets", fake_download_assets)
    monkeypatch.setattr(rq, "_bitsandbytes_available", lambda: False)
    monkeypatch.setattr(rq, "_load_model_and_tokenizer_for_attempt", lambda *args, **kwargs: (_FakeModel(), _FakeTokenizer()))

    summary = rq.run_model(
        spec,
        output_root=output_root,
        downloads_root=downloads_root,
        shared_ewok_module=fake_ewok,
        hardware={
            "cuda_available": True,
            "num_gpus": 1,
            "gpus": [{"index": 0, "free_memory_bytes": 80 * 1024**3, "total_memory_bytes": 80 * 1024**3}],
        },
    )

    model_output_dir = output_root / spec.model_slug
    manifest = json.loads((model_output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((model_output_dir / "ewok_metrics.json").read_text(encoding="utf-8"))
    model_log = (model_output_dir / "run.log").read_text(encoding="utf-8")
    item_rows = [
        json.loads(line)
        for line in (model_output_dir / "ewok_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["status"] == "completed"
    assert summary["load_strategy"] == "single_gpu"
    assert summary["batch_size_final"] == 4
    assert manifest["cleanup"]["removed_download_dir"] is True
    assert manifest["download_settings"]["disable_xet"] is True
    assert manifest["download"]["disable_xet"] is True
    assert not (downloads_root / spec.model_slug).exists()
    assert metrics_payload["summary"]["babylm_completion_choice_official_mean_average"] == 0.61
    assert {"S11_logp_T1_given_C1", "S12_logp_T2_given_C1", "S22_logp_T2_given_C2", "S21_logp_T1_given_C2"} <= set(item_rows[0].keys())
    assert "phase=start" in model_log
    assert "phase=download" in model_log
    assert "phase=evaluate" in model_log
    assert "phase=completed" in model_log


def test_run_model_failed_capacity_still_writes_manifest_and_cleans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = rq.ModelSpec(model_id="test/too-big", param_count_b=72.0)
    output_root = tmp_path / "results"
    downloads_root = tmp_path / "downloads"

    def fake_download_assets(spec, *, download_root, hf_token, disable_xet, max_retries):
        del hf_token
        assert disable_xet is True
        assert max_retries == rq.DEFAULT_DOWNLOAD_RETRIES
        download_dir = Path(download_root) / spec.model_slug
        model_dir = download_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "download_dir": download_dir,
            "model_dir": model_dir,
            "tokenizer_dir": model_dir,
            "tokenizer_source": spec.model_id,
            "tokenizer_mode": "model",
            "tokenizer_override_path": None,
        }

    monkeypatch.setattr(rq, "_download_assets", fake_download_assets)
    monkeypatch.setattr(rq, "_bitsandbytes_available", lambda: False)

    summary = rq.run_model(
        spec,
        output_root=output_root,
        downloads_root=downloads_root,
        hardware={
            "cuda_available": True,
            "num_gpus": 1,
            "gpus": [{"index": 0, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3}],
        },
    )

    model_output_dir = output_root / spec.model_slug
    manifest = json.loads((model_output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed_capacity"
    assert manifest["cleanup"]["removed_download_dir"] is True
    assert not (downloads_root / spec.model_slug).exists()
    assert (model_output_dir / "summary.json").exists()


def test_run_model_failed_download_still_cleans_partial_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = rq.ModelSpec(model_id="test/bad-download", param_count_b=13.0)
    output_root = tmp_path / "results"
    downloads_root = tmp_path / "downloads"

    def fake_download_assets(spec, *, download_root, hf_token, disable_xet, max_retries):
        del hf_token, disable_xet, max_retries
        download_dir = Path(download_root) / spec.model_slug
        model_dir = download_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "partial.bin").write_text("partial", encoding="utf-8")
        raise RuntimeError("download blew up")

    monkeypatch.setattr(rq, "_download_assets", fake_download_assets)
    monkeypatch.setattr(rq, "_bitsandbytes_available", lambda: False)

    summary = rq.run_model(
        spec,
        output_root=output_root,
        downloads_root=downloads_root,
        hardware={
            "cuda_available": True,
            "num_gpus": 1,
            "gpus": [{"index": 0, "free_memory_bytes": 24 * 1024**3, "total_memory_bytes": 24 * 1024**3}],
        },
    )

    manifest = json.loads((output_root / spec.model_slug / "manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed_download"
    assert manifest["cleanup"]["attempted"] is True
    assert manifest["cleanup"]["removed_download_dir"] is True
    assert not (downloads_root / spec.model_slug).exists()


def test_run_queue_reuses_completed_models_unless_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "queue.yaml"
    _write_yaml(
        config_path,
        {"models": [{"model_id": "test/model", "param_count_b": 1.0}]},
    )
    spec = rq.ModelSpec(model_id="test/model", param_count_b=1.0)
    output_root = tmp_path / "results"
    model_output_dir = output_root / spec.model_slug
    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "load_strategy": "single_gpu",
                "num_gpus_used": 1,
                "quantization": None,
                "batch_size_final": 4,
                "elapsed_seconds": 12.5,
                "error": None,
                "ewok_babylm_completion_official_mean_avg": 0.61,
                "ewok_context_sensitivity_official_mean_avg": 0.72,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(rq, "run_model", lambda *args, **kwargs: pytest.fail("run_model should not be called"))

    rows = rq.run_queue(
        config_path=config_path,
        output_root=output_root,
        downloads_root=tmp_path / "downloads",
        force=False,
    )

    queue_summary = (output_root / "queue_summary.csv").read_text(encoding="utf-8")
    queue_log = (output_root / "queue_run.log").read_text(encoding="utf-8")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["reused_existing"] is True
    assert "test/model" in queue_summary
    assert "phase=queue_start" in queue_log
    assert "phase=reuse" in queue_log
    assert "phase=queue_end" in queue_log
