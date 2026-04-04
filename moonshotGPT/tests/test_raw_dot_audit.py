from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from evaluation.ewok import BABYLM_COMPLETION_CHOICE
from research.bos_aligned_proto.analysis.attribution.trackstar.raw_dot_audit import (
    _candidate_gradient_modules,
    _dot_product,
    collect_candidate_grads,
    collect_mean_query_grads,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.one_step_sanity import (
    compute_query_loss_mean,
    measure_one_step_delta,
)
from research.bos_aligned_proto.analysis.attribution.common.ewok_targets import (
    EWOKTargetBundle,
    EWOKTargetItem,
)


class _ToyTokenizer:
    def __init__(self) -> None:
        self.bos_token_id = 0
        self.vocab = {
            "A": 1,
            "B": 2,
            "X": 3,
            "Y": 4,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(self.vocab[piece]) for piece in str(text).split() if piece]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del add_special_tokens, return_attention_mask
        if return_tensors != "pt":
            raise ValueError("Toy tokenizer only supports return_tensors='pt'")
        ids = self.encode(text)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class _ToyLinearLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(vocab_size, vocab_size, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.proj.in_features).to(dtype=self.proj.weight.dtype)
        logits = self.proj(one_hot)
        return SimpleNamespace(logits=logits)


def _make_toy_bundle() -> EWOKTargetBundle:
    item = EWOKTargetItem(
        target_id="toy-target",
        domain="toy-domain",
        row_index=0,
        score_view=BABYLM_COMPLETION_CHOICE,
        concept_a="A",
        concept_b="B",
        context1="A",
        context2="B",
        target1="X",
        target2="Y",
        context_type_raw="direct",
        context_type="direct",
        context_diff_raw="variable_swap",
        context_diff="variable swap",
        target_diff_raw="variable_swap",
        target_diff="variable swap",
    )
    return EWOKTargetBundle(
        items=(item,),
        groups={"overall": ("toy-target",)},
        source_path=Path("toy-ewok.jsonl"),
        score_view=BABYLM_COMPLETION_CHOICE,
        score_reduction="mean",
    )


def test_raw_dot_audit_matches_helpful_vs_harmful_single_example_signs() -> None:
    tokenizer = _ToyTokenizer()
    bundle = _make_toy_bundle()
    model = _ToyLinearLM(vocab_size=5)
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    modules = _candidate_gradient_modules(model)

    baseline_loss = compute_query_loss_mean(
        model,
        tokenizer,
        bundle,
        batch_size=1,
        temperature=1.0,
    )
    query_grads = collect_mean_query_grads(
        model,
        tokenizer,
        bundle,
        modules=modules,
        temperature=1.0,
        batch_size=1,
        show_progress=False,
    )

    model.load_state_dict(base_state, strict=True)
    helpful_grads, _ = collect_candidate_grads(
        model,
        modules=modules,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        labels=torch.tensor([[3]], dtype=torch.long),
        device=torch.device("cpu"),
    )
    model.load_state_dict(base_state, strict=True)
    harmful_grads, _ = collect_candidate_grads(
        model,
        modules=modules,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        labels=torch.tensor([[4]], dtype=torch.long),
        device=torch.device("cpu"),
    )

    helpful_dot = _dot_product(query_grads, helpful_grads)
    harmful_dot = _dot_product(query_grads, harmful_grads)

    helpful_delta = measure_one_step_delta(
        model,
        base_state=base_state,
        tokenizer=tokenizer,
        bundle=bundle,
        baseline_query_loss=baseline_loss,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        labels=torch.tensor([[3]], dtype=torch.long),
        update_lr=0.5,
        target_batch_size=1,
        temperature=1.0,
        device=torch.device("cpu"),
    )["delta_q"]
    harmful_delta = measure_one_step_delta(
        model,
        base_state=base_state,
        tokenizer=tokenizer,
        bundle=bundle,
        baseline_query_loss=baseline_loss,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        labels=torch.tensor([[4]], dtype=torch.long),
        update_lr=0.5,
        target_batch_size=1,
        temperature=1.0,
        device=torch.device("cpu"),
    )["delta_q"]

    assert helpful_dot > 0.0
    assert harmful_dot < 0.0
    assert helpful_dot > harmful_dot
    assert helpful_delta < 0.0
    assert harmful_delta > 0.0
    assert helpful_delta < harmful_delta
