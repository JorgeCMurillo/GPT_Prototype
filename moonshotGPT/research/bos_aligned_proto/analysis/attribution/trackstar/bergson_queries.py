"""EWoK query helpers for the Bergson attribution backend.

The Bergson backend keeps the repo's existing EWoK target bundle and paired
softplus objective. This module owns target subsetting, reduction-group
construction, diagnostics, and manual query-gradient extraction for the custom
query loss.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Sequence

import torch
from transformers.pytorch_utils import Conv1D as HFConv1D

from ..common.ewok_targets import (
    EWOKTargetBundle,
    TargetDiagnostics,
    iter_target_batches,
    score_target_batch,
    score_target_bundle,
)


QUERY_GRADIENT_REDUCTIONS = ("item", "per_domain", "overall")


def _resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define bos_token_id, eos_token_id, or pad_token_id")


def build_query_groups(
    bundle: EWOKTargetBundle,
    reduction: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if reduction not in QUERY_GRADIENT_REDUCTIONS:
        raise ValueError(
            f"Unsupported reduction={reduction!r}; expected one of {QUERY_GRADIENT_REDUCTIONS!r}"
        )
    if reduction == "item":
        return tuple((item.target_id, (idx,)) for idx, item in enumerate(bundle.items))
    if reduction == "per_domain":
        grouped: OrderedDict[str, list[int]] = OrderedDict()
        for idx, item in enumerate(bundle.items):
            grouped.setdefault(item.domain, []).append(idx)
        return tuple(
            (f"domain:{domain}", tuple(indices))
            for domain, indices in grouped.items()
        )
    return (("overall", tuple(range(len(bundle.items)))),)


def subset_target_bundle(bundle: EWOKTargetBundle, target_indices: Sequence[int]) -> EWOKTargetBundle:
    selected = tuple(bundle.items[int(idx)] for idx in target_indices)
    selected_ids = {item.target_id for item in selected}
    filtered_groups = {
        name: tuple(target_id for target_id in target_ids if target_id in selected_ids)
        for name, target_ids in bundle.groups.items()
    }
    filtered_groups = {name: ids for name, ids in filtered_groups.items() if ids}
    return EWOKTargetBundle(
        items=selected,
        groups=filtered_groups,
        source_path=bundle.source_path,
        score_view=bundle.score_view,
        score_reduction=bundle.score_reduction,
    )


def score_bundle_diagnostics(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
) -> tuple[TargetDiagnostics, ...]:
    if not bundle.items:
        return ()
    return score_target_bundle(
        model,
        tokenizer,
        bundle,
        batch_size=batch_size,
        temperature=temperature,
    )


def _select_gradient_modules(
    model: torch.nn.Module,
    module_names: Sequence[str] | None,
) -> OrderedDict[str, torch.nn.Module]:
    available = OrderedDict(
        (
            name,
            module,
        )
        for name, module in model.named_modules()
        if name
        and hasattr(module, "weight")
        and isinstance(getattr(module, "weight"), torch.Tensor)
        and getattr(module, "weight").ndim == 2
    )
    if not available:
        raise ValueError("Model does not expose any 2D weight modules for query gradients")
    if not module_names:
        return available

    base_model_prefix = getattr(model, "base_model_prefix", "")
    relative_available = {
        name[len(base_model_prefix) + 1 :]: module
        for name, module in available.items()
        if base_model_prefix and name.startswith(f"{base_model_prefix}.")
    }

    selected: OrderedDict[str, torch.nn.Module] = OrderedDict()
    missing = []
    for name in module_names:
        requested = str(name)
        module = available.get(requested)
        if module is None:
            module = relative_available.get(requested)
        if module is None and base_model_prefix and requested.startswith(f"{base_model_prefix}."):
            module = available.get(requested[len(base_model_prefix) + 1 :])
        if module is None:
            missing.append(requested)
            continue
        selected[requested] = module
    if missing:
        raise ValueError(f"Requested query-gradient modules were not found on the model: {missing}")
    return selected


def _normalize_module_weight_grad(module: torch.nn.Module, grad: torch.Tensor) -> torch.Tensor:
    """Match the module-specific gradient layout Bergson stores in its index.

    For standard `nn.Linear`, the weight gradient is already `[out, in]`, which
    matches Bergson's collector layout. Hugging Face `Conv1D` stores weights as
    `[in, out]`, while Bergson's collector materializes gradients as `[out, in]`,
    so we transpose those modules here before flattening or projecting.
    """

    if not isinstance(grad, torch.Tensor) or grad.ndim != 2:
        raise RuntimeError(f"Expected a 2D weight gradient, got {type(grad)!r} with shape {getattr(grad, 'shape', None)}")
    if isinstance(module, HFConv1D):
        return grad.mT
    return grad


def _create_projection_matrix(
    identifier: str,
    rows: int,
    cols: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    projection_type: str,
) -> torch.Tensor:
    """Reproduce Bergson's deterministic random projection matrices locally."""

    message = bytes(identifier, "utf-8")
    digest = hashlib.md5(message).digest()
    seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)

    if projection_type == "normal":
        prng = torch.Generator(device).manual_seed(seed)
        matrix = torch.randn(rows, cols, device=device, dtype=dtype, generator=prng)
    elif projection_type == "rademacher":
        prng = torch.Generator(device).manual_seed(seed)
        matrix = torch.randint(0, 2, (rows, cols), device=device, generator=prng, dtype=torch.int64)
        matrix = matrix.to(dtype=dtype).mul_(2).add_(-1)
    else:
        raise ValueError(f"Unsupported Bergson projection_type: {projection_type!r}")

    matrix /= matrix.norm(dim=1, keepdim=True).clamp_min_(1e-12)
    return matrix


def _project_query_grad(
    module_name: str,
    grad: torch.Tensor,
    *,
    projection_dim: int | None,
    projection_type: str,
) -> torch.Tensor:
    """Project one raw query gradient into the same feature space as the index."""

    if projection_dim is None:
        return grad.reshape(-1)

    out_dim, in_dim = int(grad.shape[0]), int(grad.shape[1])
    left = _create_projection_matrix(
        f"{module_name}/left",
        projection_dim,
        out_dim,
        dtype=grad.dtype,
        device=grad.device,
        projection_type=projection_type,
    )
    right = _create_projection_matrix(
        f"{module_name}/right",
        projection_dim,
        in_dim,
        dtype=grad.dtype,
        device=grad.device,
        projection_type=projection_type,
    )
    projected = left @ grad @ right.mT
    return projected.reshape(-1)


def collect_query_module_grads(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
    module_names: Sequence[str] | None = None,
    reduction: str = "item",
    projection_dim: int | None = None,
    projection_type: str = "rademacher",
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    if not bundle.items:
        return (), {}

    group_specs = build_query_groups(bundle, reduction)
    modules = _select_gradient_modules(model, module_names)
    bos_token_id = _resolve_bos_token_id(tokenizer)

    per_item_grads: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    model.eval()
    for prepared in iter_target_batches(bundle, tokenizer, 1):
        model.zero_grad(set_to_none=True)
        scores = score_target_batch(
            model,
            prepared.batch,
            score_view=bundle.score_view,
            score_reduction=bundle.score_reduction,
            temperature=temperature,
            bos_token_id=bos_token_id,
        )
        loss = scores["softplus_loss"].sum()
        loss.backward()
        for name, module in modules.items():
            grad = getattr(module, "weight").grad
            if grad is None:
                raise RuntimeError(f"Missing gradient for query module {name!r}")
            normalized = _normalize_module_weight_grad(module, grad.detach())
            projected = _project_query_grad(
                name,
                normalized,
                projection_dim=projection_dim,
                projection_type=projection_type,
            )
            per_item_grads[name].append(projected.cpu().to(dtype=torch.float32))

    grouped_grads: dict[str, torch.Tensor] = {}
    for name, grads in per_item_grads.items():
        stacked = torch.stack(grads, dim=0)
        grouped = [stacked[list(indices)].mean(dim=0) for _, indices in group_specs]
        grouped_grads[name] = torch.stack(grouped, dim=0)

    return tuple(group_id for group_id, _ in group_specs), grouped_grads


__all__ = [
    "QUERY_GRADIENT_REDUCTIONS",
    "build_query_groups",
    "collect_query_module_grads",
    "score_bundle_diagnostics",
    "subset_target_bundle",
]
