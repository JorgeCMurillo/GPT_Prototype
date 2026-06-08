import copy

import torch
from torch import nn

from training_utils.muon_polar_express import (
    MuonWithAuxAdamPE,
    build_muon_pe_param_groups,
    polar_express_orthogonalize,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(4, 12)
        self.attn.c_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(4, 16)
        self.mlp.c_proj = nn.Linear(16, 4)
        self.ln_1 = nn.LayerNorm(4)


class _TinyGPTLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(32, 4)
        self.transformer.wpe = nn.Embedding(16, 4)
        self.transformer.h = nn.ModuleList([_Block()])
        self.lm_head = nn.Linear(4, 32, bias=False)


class _TinyLlamaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16, bias=False)
        self.mlp.up_proj = nn.Linear(8, 16, bias=False)
        self.mlp.down_proj = nn.Linear(16, 8, bias=False)
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)


class _TinyLlamaLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([_TinyLlamaBlock(), _TinyLlamaBlock()])
        self.model.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)


def test_muon_param_groups_exclude_embeddings_and_split_gpt2_qkv() -> None:
    model = _TinyGPTLike()
    grouped = build_muon_pe_param_groups(
        model,
        aux_lr=6e-4,
        aux_weight_decay=0.1,
        aux_betas=(0.9, 0.95),
        muon_lr=0.02,
        muon_weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_nesterov=True,
        split_qkv=True,
    )

    muon_ids = {id(param) for group in grouped.param_groups if group["use_muon"] for param in group["params"]}

    assert id(model.transformer.wte.weight) not in muon_ids
    assert id(model.transformer.wpe.weight) not in muon_ids
    assert id(model.lm_head.weight) not in muon_ids
    assert id(model.transformer.h[0].attn.c_attn.weight) in muon_ids
    assert id(model.transformer.h[0].mlp.c_fc.weight) in muon_ids
    assert grouped.qkv_split_tensors == 1


def test_muon_param_groups_handle_llama_style_projections() -> None:
    model = _TinyLlamaLike()
    grouped = build_muon_pe_param_groups(
        model,
        aux_lr=6e-4,
        aux_weight_decay=0.1,
        aux_betas=(0.9, 0.95),
        muon_lr=0.02,
        muon_weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_nesterov=True,
        split_qkv=True,
    )
    muon_ids = {id(param) for group in grouped.param_groups if group["use_muon"] for param in group["params"]}
    first_block = model.model.layers[0]

    assert id(model.model.embed_tokens.weight) not in muon_ids
    assert id(model.lm_head.weight) not in muon_ids
    assert id(first_block.self_attn.q_proj.weight) in muon_ids
    assert id(first_block.self_attn.k_proj.weight) in muon_ids
    assert id(first_block.self_attn.v_proj.weight) in muon_ids
    assert id(first_block.self_attn.o_proj.weight) in muon_ids
    assert id(first_block.mlp.gate_proj.weight) in muon_ids
    assert id(first_block.mlp.up_proj.weight) in muon_ids
    assert id(first_block.mlp.down_proj.weight) in muon_ids
    assert grouped.qkv_split_tensors == 0


def test_polar_express_and_optimizer_step_are_finite() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(4, 8)
    orth = polar_express_orthogonalize(matrix, steps=2)
    assert orth.shape == matrix.shape
    assert torch.isfinite(orth).all()

    model = _TinyGPTLike()
    grouped = build_muon_pe_param_groups(
        model,
        aux_lr=6e-4,
        aux_weight_decay=0.1,
        aux_betas=(0.9, 0.95),
        muon_lr=0.02,
        muon_weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=2,
        muon_nesterov=True,
        split_qkv=True,
    )
    optimizer = MuonWithAuxAdamPE(grouped.param_groups, qkv_split_dims=grouped.qkv_split_dims)
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    before = model.transformer.h[0].mlp.c_fc.weight.detach().clone()
    optimizer.step()
    after = model.transformer.h[0].mlp.c_fc.weight.detach()

    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)


def test_batched_same_shape_muon_matches_scalar_llama_style_step() -> None:
    torch.manual_seed(1234)
    scalar_model = _TinyLlamaLike()
    batched_model = copy.deepcopy(scalar_model)

    for scalar_param, batched_param in zip(
        scalar_model.parameters(),
        batched_model.parameters(),
        strict=True,
    ):
        grad = torch.randn_like(scalar_param)
        scalar_param.grad = grad.clone()
        batched_param.grad = grad.clone()

    scalar_groups = build_muon_pe_param_groups(
        scalar_model,
        aux_lr=6e-4,
        aux_weight_decay=0.1,
        aux_betas=(0.9, 0.95),
        muon_lr=0.02,
        muon_weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=2,
        muon_nesterov=True,
        split_qkv=True,
    )
    batched_groups = build_muon_pe_param_groups(
        batched_model,
        aux_lr=6e-4,
        aux_weight_decay=0.1,
        aux_betas=(0.9, 0.95),
        muon_lr=0.02,
        muon_weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=2,
        muon_nesterov=True,
        split_qkv=True,
    )
    scalar_optimizer = MuonWithAuxAdamPE(
        scalar_groups.param_groups,
        qkv_split_dims=scalar_groups.qkv_split_dims,
        batch_muon_updates=False,
    )
    batched_optimizer = MuonWithAuxAdamPE(
        batched_groups.param_groups,
        qkv_split_dims=batched_groups.qkv_split_dims,
        batch_muon_updates=True,
    )

    scalar_optimizer.step()
    batched_optimizer.step()

    for scalar_param, batched_param in zip(
        scalar_model.parameters(),
        batched_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            batched_param,
            scalar_param,
            rtol=2e-3,
            atol=2e-3,
        )
