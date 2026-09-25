from types import SimpleNamespace

import torch

from moonshotGPT.evaluation.ewok import per_token_log_likelihood, resolve_bos_token_id


class _FakeTokenizer:
    bos_token_id = None
    eos_token_id = 7
    pad_token_id = None

    def __call__(self, texts, *, add_special_tokens=False, return_tensors=None, padding=False):
        del add_special_tokens, return_tensors, padding
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            length = max(1, len(str(text).split()))
            rows.append([11] * length)
        max_len = max(len(row) for row in rows)
        padded = [row + [0] * (max_len - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return SimpleNamespace(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.last_input_ids = None

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        self.last_input_ids = input_ids.detach().cpu()
        batch_size, seq_len = input_ids.shape
        vocab_size = 32
        logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32, device=input_ids.device)
        return {"logits": logits}


def test_resolve_bos_token_id_falls_back_to_eos_then_pad() -> None:
    tokenizer = _FakeTokenizer()
    assert resolve_bos_token_id(tokenizer) == 7

    tokenizer.eos_token_id = None
    tokenizer.pad_token_id = 9
    assert resolve_bos_token_id(tokenizer) == 9


def test_per_token_log_likelihood_uses_fallback_bos_token_id() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    token_logprobs, attention_mask = per_token_log_likelihood(
        model,
        tokenizer,
        ["alpha beta"],
        device=torch.device("cpu"),
    )

    assert model.last_input_ids is not None
    assert int(model.last_input_ids[0, 0].item()) == 7
    assert tuple(token_logprobs.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)
