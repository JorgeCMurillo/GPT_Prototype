import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_gpt2_finewebedu_bin as train_mod


def _write_u16(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype=np.uint16).tofile(path)


def _make_tiny_shards(tmp_path: Path) -> Path:
    data_dir = tmp_path / "tiny_data"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"tokenizer": "fake-gpt2", "dtype": "uint16"}),
        encoding="utf-8",
    )
    _write_u16(data_dir / "train_000000.bin", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    _write_u16(data_dir / "val_000000.bin", [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    return data_dir


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)
        self._vocab_size = int(vocab_size)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "tiny-test-gpt", "vocab_size": self._vocab_size}, indent=2),
            encoding="utf-8",
        )
        (target / "model.safetensors").write_bytes(b"tiny-test-model")


class _FakeTokenizer:
    vocab_size = 32
    eos_token = "<|eos|>"
    eos_token_id = 31
    pad_token = "<|eos|>"
    pad_token_id = 31

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "FakeTokenizer",
                    "eos_token": self.eos_token,
                    "eos_token_id": self.eos_token_id,
                    "pad_token": self.pad_token,
                    "pad_token_id": self.pad_token_id,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def test_optimizer_disables_fused_until_params_are_on_accelerator_device() -> None:
    model = _TinyLM(vocab_size=32, hidden_size=8)
    optimizer, _, _, use_fused = train_mod.build_llmc_style_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        device=torch.device("cuda"),
    )
    assert use_fused is False
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        assert optimizer.defaults.get("fused", False) is False


def test_top_level_trainer_smoke_skips_final_ewok(tmp_path, monkeypatch) -> None:
    data_dir = _make_tiny_shards(tmp_path)
    experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(
        train_mod.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        train_mod.AutoModelForCausalLM,
        "from_config",
        lambda config, **kwargs: _TinyLM(config.vocab_size, hidden_size=config.n_embd),
    )
    monkeypatch.setattr(
        train_mod,
        "run_final_ewok_eval_main_process",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("final EWoK should be skipped in this smoke test")
        ),
    )
    monkeypatch.setattr(
        train_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    train_mod.main(
        seed=123,
        micro_batch_size=1,
        total_batch_tokens=8,
        max_train_steps=1,
        data_dir=str(data_dir),
        experiments_dir=str(experiments_dir),
        seq_len=8,
        vocab_size=32,
        n_embd=8,
        n_head=2,
        n_layer=1,
        num_workers=0,
        shuffle_blocks=False,
        grad_clip=0.0,
        learning_rate=1e-3,
        warmup_iters=0,
        learning_rate_decay_frac=0.0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        eval_every=0,
        hellaswag_every=0,
        hellaswag_batch_size=1,
        hellaswag_max_examples=1,
        hellaswag_local_files_only=True,
        blimp_every=0,
        blimp_batch_size=1,
        ewok_every=0,
        ewok_batch_size=1,
        save_every=0,
        exposure_every=0,
        push_to_hub=False,
        skip_final_ewok=True,
        mixed_precision="no",
        rho_ref_loss_dir="",
        rho_keep_frac=1.0,
        rho_warmup_steps=0,
        rho_mode="delta",
        rho_ref_loss_cap=0.0,
        init_from_ckpt="",
        resume_from_run="",
        max_shards=1,
    )

    run_dirs = sorted(experiments_dir.glob("babygpt_fineweb_bin_*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    final_ckpt = run_dir / "ckpt_final_step0000001"
    assert final_ckpt.is_dir()
    assert (final_ckpt / "config.json").exists()
    assert (final_ckpt / "model.safetensors").exists()
    assert (final_ckpt / "optimizer.pt").exists()
    assert (final_ckpt / "tokenizer_config.json").exists()
    assert not (run_dir / "ewok_items.jsonl").exists()
