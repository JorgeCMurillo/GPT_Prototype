import json
from pathlib import Path

import pytest
import torch

from evaluation.runner import run_core_eval_step


def _append_jsonl(path: str, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _save_metrics(metrics_list, out_path: str) -> None:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics_list, indent=2), encoding="utf-8")


class _FakeAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 2
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def unwrap_model(self, model):
        return model


class _FakeModel:
    def __init__(self) -> None:
        self.mode = "train"

    def eval(self):
        self.mode = "eval"
        return self

    def train(self):
        self.mode = "train"
        return self


class _FakeCoreModule:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = []

    def evaluate_core(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


class _RaisingCoreModule:
    def evaluate_core(self, **kwargs):
        del kwargs
        raise FileNotFoundError("missing eval bundle")


@pytest.mark.parametrize(
    ("core_distributed_env", "expected_distributed"),
    [(None, True), ("0", False)],
)
def test_run_core_eval_step_writes_step_and_jsonl_logs(
    tmp_path,
    monkeypatch,
    core_distributed_env,
    expected_distributed,
) -> None:
    if core_distributed_env is None:
        monkeypatch.delenv("CORE_DISTRIBUTED", raising=False)
    else:
        monkeypatch.setenv("CORE_DISTRIBUTED", core_distributed_env)

    accelerator = _FakeAccelerator()
    model = _FakeModel()
    tokenizer = object()
    step_metrics = []
    scalars_path = tmp_path / "scalars.jsonl"
    core_metrics_path = tmp_path / "core_metrics.jsonl"
    metrics_path = tmp_path / "step_metrics.json"
    release_calls = []
    fake_core = _FakeCoreModule(
        {
            "results": {"task_a": 0.75},
            "centered_results": {"task_a": 0.5},
            "examples_per_task": {"task_a": 4},
            "core_metric": 0.5,
            "num_tasks": 1,
            "max_per_task": 500,
            "bundle_source": "/tmp/eval_bundle",
            "bundle_dir": "/tmp/eval_bundle",
        }
    )

    core_disabled = run_core_eval_step(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        core_eval_module=fake_core,
        core_disabled=False,
        core_bundle_dir="",
        core_local_files_only=False,
        core_max_per_task=500,
        opt_step=2000,
        last_train_loss=1.25,
        loss_val=1.25,
        last_lr=1e-3,
        optimizer=object(),
        tokens_seen_local_total=123,
        scalars_path=str(scalars_path),
        core_metrics_path=str(core_metrics_path),
        step_metrics=step_metrics,
        metrics_path=str(metrics_path),
        append_jsonl_fn=_append_jsonl,
        save_metrics_fn=_save_metrics,
        get_current_lr_fn=lambda optimizer: 9.9,
        release_eval_memory_fn=lambda: release_calls.append("released"),
        final=True,
    )

    assert core_disabled is False
    assert accelerator.wait_calls == 2
    assert release_calls == ["released"]
    assert model.mode == "train"
    assert fake_core.calls
    assert fake_core.calls[0]["distributed"] is expected_distributed

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(saved_metrics) == 1
    assert saved_metrics[0]["final"] is True
    assert saved_metrics[0]["core"]["bundle_source"] == "/tmp/eval_bundle"
    assert saved_metrics[0]["core"]["results"] == {"task_a": 0.75}
    assert saved_metrics[0]["core"]["centered_results"] == {"task_a": 0.5}

    scalar_rows = [
        json.loads(line)
        for line in scalars_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert scalar_rows == [
        {
            "type": "core",
            "step": 2000,
            "timestamp": scalar_rows[0]["timestamp"],
            "num_tasks": 1,
            "max_per_task": 500,
            "bundle_source": "/tmp/eval_bundle",
            "bundle_dir": "/tmp/eval_bundle",
            "core_metric": 0.5,
            "final": True,
        }
    ]

    core_rows = [
        json.loads(line)
        for line in core_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(core_rows) == 1
    assert core_rows[0]["results"] == {"task_a": 0.75}
    assert core_rows[0]["centered_results"] == {"task_a": 0.5}
    assert core_rows[0]["examples_per_task"] == {"task_a": 4}
    assert core_rows[0]["final"] is True


def test_run_core_eval_step_disables_future_runs_after_failure(tmp_path) -> None:
    accelerator = _FakeAccelerator()
    model = _FakeModel()
    step_metrics = []

    core_disabled = run_core_eval_step(
        accelerator=accelerator,
        model=model,
        tokenizer=object(),
        device=torch.device("cpu"),
        core_eval_module=_RaisingCoreModule(),
        core_disabled=False,
        core_bundle_dir=str(tmp_path / "missing_bundle"),
        core_local_files_only=True,
        core_max_per_task=500,
        opt_step=2000,
        last_train_loss=1.25,
        loss_val=1.25,
        last_lr=1e-3,
        optimizer=object(),
        tokens_seen_local_total=123,
        scalars_path=str(tmp_path / "scalars.jsonl"),
        core_metrics_path=str(tmp_path / "core_metrics.jsonl"),
        step_metrics=step_metrics,
        metrics_path=str(tmp_path / "step_metrics.json"),
        append_jsonl_fn=_append_jsonl,
        save_metrics_fn=_save_metrics,
        get_current_lr_fn=lambda optimizer: 9.9,
        release_eval_memory_fn=lambda: None,
    )

    assert core_disabled is True
    assert step_metrics == []
    assert model.mode == "train"
    assert not (tmp_path / "step_metrics.json").exists()
