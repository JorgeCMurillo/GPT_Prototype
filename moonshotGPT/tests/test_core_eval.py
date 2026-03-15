import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.core import evaluate_core, resolve_bos_token_id


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "eval_bundle"
    eval_data_dir = bundle_dir / "eval_data"
    eval_data_dir.mkdir(parents=True)

    (bundle_dir / "core.yaml").write_text(
        """
icl_tasks:
  - label: mc_task
    icl_task_type: multiple_choice
    dataset_uri: mc_task.jsonl
    num_fewshot: [0]
    continuation_delimiter: " "
  - label: schema_task
    icl_task_type: schema
    dataset_uri: schema_task.jsonl
    num_fewshot: [0]
    continuation_delimiter: " "
  - label: lm_task
    icl_task_type: language_modeling
    dataset_uri: lm_task.jsonl
    num_fewshot: [0]
    continuation_delimiter: " "
""".strip(),
        encoding="utf-8",
    )

    with (bundle_dir / "eval_meta_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Eval Task", "Random baseline"])
        writer.writeheader()
        writer.writerow({"Eval Task": "mc_task", "Random baseline": 50})
        writer.writerow({"Eval Task": "schema_task", "Random baseline": 50})
        writer.writerow({"Eval Task": "lm_task", "Random baseline": 0})

    _write_jsonl(
        eval_data_dir / "mc_task.jsonl",
        [
            {"query": "mc_q", "choices": ["mc_good", "mc_bad"], "gold": 0},
            {"query": "mc_q", "choices": ["mc_good", "mc_bad"], "gold": 1},
        ],
    )
    _write_jsonl(
        eval_data_dir / "schema_task.jsonl",
        [
            {
                "context_options": ["schema_good", "schema_bad"],
                "continuation": "schema_cont",
                "gold": 0,
            },
            {
                "context_options": ["schema_good", "schema_bad"],
                "continuation": "schema_cont",
                "gold": 0,
            },
        ],
    )
    _write_jsonl(
        eval_data_dir / "lm_task.jsonl",
        [
            {"context": "lm_ctx", "continuation": "lm_cont", "gold": 0},
            {"context": "lm_ctx", "continuation": "lm_cont", "gold": 0},
        ],
    )
    return bundle_dir


class _FakeTokenizer:
    def __init__(self, *, bos_token_id, eos_token_id):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_max_length = 32
        self._vocab = {
            "mc_q": 1,
            "mc_good": 2,
            "mc_bad": 3,
            "schema_good": 4,
            "schema_bad": 5,
            "schema_cont": 6,
            "lm_ctx": 7,
            "lm_cont": 8,
        }

    @property
    def vocab_size(self) -> int:
        return max(self._vocab.values()) + 1

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    ):
        del add_special_tokens, return_attention_mask, return_token_type_ids
        if isinstance(texts, str):
            texts = [texts]
        encoded = []
        for text in texts:
            toks = [tok for tok in str(text).split() if tok]
            encoded.append([self._vocab[token] for token in toks])
        return {"input_ids": encoded}


class _FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int, next_token_map: dict[int, int]):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=32)
        self._vocab_size = int(vocab_size)
        self._next_token_map = {int(k): int(v) for k, v in next_token_map.items()}

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self._vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                current_token = int(input_ids[batch_idx, token_idx].item())
                next_token = self._next_token_map.get(current_token, 0)
                logits[batch_idx, token_idx, next_token] = 10.0
        return SimpleNamespace(logits=logits)


def _make_components(tmp_path: Path, *, bos_token_id, eos_token_id):
    bundle_dir = _make_bundle(tmp_path)
    tokenizer = _FakeTokenizer(bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    model = _FakeModel(
        tokenizer.vocab_size,
        next_token_map={
            tokenizer._vocab["mc_q"]: tokenizer._vocab["mc_good"],
            tokenizer._vocab["schema_good"]: tokenizer._vocab["schema_cont"],
            tokenizer._vocab["lm_ctx"]: tokenizer._vocab["lm_cont"],
        },
    )
    return bundle_dir, tokenizer, model


def test_evaluate_core_computes_centered_metric_and_bundle_metadata(tmp_path) -> None:
    bundle_dir, tokenizer, model = _make_components(tmp_path, bos_token_id=0, eos_token_id=0)

    results = evaluate_core(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        bundle_dir=bundle_dir,
        local_files_only=True,
        distributed=False,
    )

    assert results["results"] == {
        "mc_task": 0.5,
        "schema_task": 1.0,
        "lm_task": 1.0,
    }
    assert results["centered_results"] == {
        "mc_task": 0.0,
        "schema_task": 1.0,
        "lm_task": 1.0,
    }
    assert abs(results["core_metric"] - (2.0 / 3.0)) < 1e-6
    assert results["examples_per_task"] == {
        "mc_task": 2,
        "schema_task": 2,
        "lm_task": 2,
    }
    assert results["bundle_source"] == str(bundle_dir)
    assert results["bundle_dir"] == str(bundle_dir)


def test_evaluate_core_subsampling_is_deterministic(tmp_path) -> None:
    bundle_dir, tokenizer, model = _make_components(tmp_path, bos_token_id=0, eos_token_id=0)

    first = evaluate_core(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        bundle_dir=bundle_dir,
        local_files_only=True,
        max_per_task=1,
        distributed=False,
    )
    second = evaluate_core(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        bundle_dir=bundle_dir,
        local_files_only=True,
        max_per_task=1,
        distributed=False,
    )

    assert first == second
    assert first["examples_per_task"] == {
        "mc_task": 1,
        "schema_task": 1,
        "lm_task": 1,
    }


def test_resolve_bos_token_id_falls_back_to_eos_and_eval_still_runs(tmp_path) -> None:
    bundle_dir, tokenizer, model = _make_components(tmp_path, bos_token_id=None, eos_token_id=0)

    assert resolve_bos_token_id(tokenizer) == 0

    results = evaluate_core(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        bundle_dir=bundle_dir,
        local_files_only=True,
        max_per_task=1,
        distributed=False,
    )

    assert results["num_tasks"] == 3
    assert results["bundle_source"] == str(bundle_dir)
    assert torch.isfinite(torch.tensor(results["core_metric"]))
