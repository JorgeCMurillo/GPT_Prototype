"""Muon optimizer with Polar Express orthogonalization.

This is a small, dependency-free implementation for GPT-style pretraining
experiments. Muon is applied only to hidden matrix weights; embeddings, heads,
normalization gains, and biases stay on AdamW.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch


POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)


@dataclass(frozen=True)
class MuonParameterGroups:
    param_groups: list[dict]
    muon_tensors: int
    aux_tensors: int
    qkv_split_tensors: int
    qkv_split_dims: dict[int, tuple[int, tuple[int, ...]]]


def _is_embedding_or_head(name: str) -> bool:
    parts = set(name.split("."))
    if {"wte", "wpe", "lm_head", "embed_tokens", "embed_in", "embed_out"} & parts:
        return True
    lowered = name.lower()
    return "embedding" in lowered or "embed" in lowered and "mlp" not in lowered


def _is_muon_eligible(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    if _is_embedding_or_head(name):
        return False
    return True


def _maybe_gpt2_qkv_split(name: str, param: torch.nn.Parameter) -> tuple[int, tuple[int, ...]] | None:
    if not name.endswith("attn.c_attn.weight"):
        return None
    if param.ndim != 2:
        return None
    if param.shape[-1] % 3 == 0:
        width = int(param.shape[-1] // 3)
        return -1, (width, width, width)
    if param.shape[0] % 3 == 0:
        width = int(param.shape[0] // 3)
        return 0, (width, width, width)
    return None


def build_muon_pe_param_groups(
    model: torch.nn.Module,
    *,
    aux_lr: float,
    aux_weight_decay: float,
    aux_betas: tuple[float, float],
    muon_lr: float,
    muon_weight_decay: float,
    muon_momentum: float,
    muon_ns_steps: int,
    muon_nesterov: bool,
    split_qkv: bool = True,
) -> MuonParameterGroups:
    muon_params: list[torch.nn.Parameter] = []
    aux_decay_params: list[torch.nn.Parameter] = []
    aux_nodecay_params: list[torch.nn.Parameter] = []
    qkv_split_dims: dict[int, tuple[int, tuple[int, ...]]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_eligible(name, param):
            muon_params.append(param)
            if split_qkv:
                split = _maybe_gpt2_qkv_split(name, param)
                if split is not None:
                    qkv_split_dims[id(param)] = split
        elif param.ndim >= 2:
            aux_decay_params.append(param)
        else:
            aux_nodecay_params.append(param)

    param_groups: list[dict] = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": float(muon_lr),
                "base_lr": float(muon_lr),
                "momentum": float(muon_momentum),
                "weight_decay": float(muon_weight_decay),
                "ns_steps": int(muon_ns_steps),
                "nesterov": bool(muon_nesterov),
            }
        )
    if aux_decay_params:
        param_groups.append(
            {
                "params": aux_decay_params,
                "use_muon": False,
                "lr": float(aux_lr),
                "base_lr": float(aux_lr),
                "betas": aux_betas,
                "eps": 1e-8,
                "weight_decay": float(aux_weight_decay),
            }
        )
    if aux_nodecay_params:
        param_groups.append(
            {
                "params": aux_nodecay_params,
                "use_muon": False,
                "lr": float(aux_lr),
                "base_lr": float(aux_lr),
                "betas": aux_betas,
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        )

    return MuonParameterGroups(
        param_groups=param_groups,
        muon_tensors=len(muon_params),
        aux_tensors=len(aux_decay_params) + len(aux_nodecay_params),
        qkv_split_tensors=len(qkv_split_dims),
        qkv_split_dims=qkv_split_dims,
    )


def _iter_polar_coeffs(steps: int) -> Iterable[tuple[float, float, float]]:
    if steps <= 0:
        return
    for idx in range(steps):
        yield POLAR_EXPRESS_COEFFS[min(idx, len(POLAR_EXPRESS_COEFFS) - 1)]


def polar_express_orthogonalize(x: torch.Tensor, *, steps: int, eps: float = 1e-7) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Muon/Polar Express requires a tensor with ndim >= 2.")

    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    transposed = x.size(-2) > x.size(-1)
    work = x
    if transposed:
        work = work.mT

    out_dtype = x.dtype
    work = work.to(torch.bfloat16)
    work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    for a, b, c in _iter_polar_coeffs(int(steps)):
        gram = work @ work.mT
        update = b * gram + c * (gram @ gram)
        work = a * work + update @ work

    if transposed:
        work = work.mT
    return work.to(out_dtype).reshape(original_shape)


def _polar_express_orthogonalize_batched(
    x: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batched Polar Express for a stack of same-oriented matrices."""

    if x.ndim != 3:
        raise ValueError("Batched Muon/Polar Express requires a 3D tensor.")

    transposed = x.size(-2) > x.size(-1)
    work = x.mT if transposed else x

    out_dtype = x.dtype
    work = work.to(torch.bfloat16)
    work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    for a, b, c in _iter_polar_coeffs(int(steps)):
        gram = work @ work.mT
        update = b * gram + c * (gram @ gram)
        work = a * work + update @ work

    if transposed:
        work = work.mT
    return work.to(out_dtype)


def _aspect_ratio_scale(x: torch.Tensor) -> float:
    rows = int(x.size(-2))
    cols = int(x.size(-1))
    return max(1.0, rows / max(cols, 1)) ** 0.5


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    ns_steps: int,
    nesterov: bool,
    qkv_split: tuple[int, tuple[int, ...]] | None = None,
) -> torch.Tensor:
    momentum.lerp_(grad, 1.0 - float(beta))
    update = torch.lerp(grad, momentum, float(beta)) if nesterov else momentum

    if qkv_split is not None:
        dim, split_sizes = qkv_split
        pieces = []
        for piece in torch.split(update, split_sizes, dim=dim):
            orth = polar_express_orthogonalize(piece, steps=ns_steps)
            pieces.append(orth * _aspect_ratio_scale(piece))
        return torch.cat(pieces, dim=dim)

    orth = polar_express_orthogonalize(update, steps=ns_steps)
    return orth * _aspect_ratio_scale(update)


def _same_shape_batch_key(
    param: torch.nn.Parameter,
    grad: torch.Tensor,
) -> tuple[torch.device, torch.dtype, torch.dtype, tuple[int, ...]]:
    # Exact-shape batching is intentional. The 2026-06 Llama benchmark showed
    # transpose-compatible grouping did not improve speed and used more memory.
    return param.device, param.dtype, grad.dtype, tuple(param.shape)


class MuonWithAuxAdamPE(torch.optim.Optimizer):
    """Muon + auxiliary AdamW optimizer using Polar Express for Muon updates."""

    def __init__(
        self,
        param_groups: list[dict],
        *,
        qkv_split_dims: dict[int, tuple[int, tuple[int, ...]]] | None = None,
        batch_muon_updates: bool = True,
    ):
        if not param_groups:
            raise ValueError("MuonWithAuxAdamPE requires at least one parameter group.")
        self.qkv_split_dims = dict(qkv_split_dims or {})
        self.batch_muon_updates = bool(batch_muon_updates)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    @torch.no_grad()
    def _step_muon_group(self, group: dict) -> None:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        beta = float(group.get("momentum", 0.95))
        ns_steps = int(group.get("ns_steps", 5))
        nesterov = bool(group.get("nesterov", True))

        batches: dict[
            tuple[torch.device, torch.dtype, torch.dtype, tuple[int, ...]],
            list[torch.nn.Parameter],
        ] = defaultdict(list)
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonWithAuxAdamPE does not support sparse gradients.")
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            qkv_split = self.qkv_split_dims.get(id(param))
            if (
                self.batch_muon_updates
                and qkv_split is None
                and param.ndim == 2
                and grad.ndim == 2
            ):
                batches[_same_shape_batch_key(param, grad)].append(param)
            else:
                self._step_one_muon_param(
                    param,
                    lr=lr,
                    weight_decay=weight_decay,
                    beta=beta,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                    qkv_split=qkv_split,
                )

        for params in batches.values():
            if len(params) == 1:
                self._step_one_muon_param(
                    params[0],
                    lr=lr,
                    weight_decay=weight_decay,
                    beta=beta,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                    qkv_split=None,
                )
            else:
                self._step_batched_muon_params(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    beta=beta,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                )

    def _step_one_muon_param(
        self,
        param: torch.nn.Parameter,
        *,
        lr: float,
        weight_decay: float,
        beta: float,
        ns_steps: int,
        nesterov: bool,
        qkv_split: tuple[int, tuple[int, ...]] | None,
    ) -> None:
        grad = param.grad
        if grad is None:
            return
        state = self.state[param]
        update = muon_update(
            grad,
            state["momentum_buffer"],
            beta=beta,
            ns_steps=ns_steps,
            nesterov=nesterov,
            qkv_split=qkv_split,
        )
        if weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        param.add_(update.reshape_as(param), alpha=-lr)

    def _step_batched_muon_params(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        beta: float,
        ns_steps: int,
        nesterov: bool,
    ) -> None:
        grads = [param.grad for param in params]
        if any(grad is None for grad in grads):
            raise RuntimeError("Batched Muon received a parameter without a gradient.")

        checked_grads = [grad for grad in grads if grad is not None]
        grad_stack = torch.stack(checked_grads)
        momentum_stack = torch.stack(
            [self.state[param]["momentum_buffer"] for param in params]
        )
        momentum_stack.lerp_(grad_stack, 1.0 - float(beta))
        for param, momentum in zip(params, momentum_stack, strict=True):
            self.state[param]["momentum_buffer"].copy_(momentum)

        update_stack = (
            torch.lerp(grad_stack, momentum_stack, float(beta))
            if nesterov
            else momentum_stack
        )
        update_stack = _polar_express_orthogonalize_batched(
            update_stack,
            steps=ns_steps,
        )
        update_stack = update_stack * _aspect_ratio_scale(update_stack[0])
        if update_stack.dtype != params[0].dtype:
            update_stack = update_stack.to(params[0].dtype)

        updates = list(update_stack.unbind(0))

        if weight_decay:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        torch._foreach_add_(params, updates, alpha=-lr)

    @torch.no_grad()
    def _step_adamw_group(self, group: dict) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonWithAuxAdamPE does not support sparse gradients.")
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = int(state["step"])

            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))
