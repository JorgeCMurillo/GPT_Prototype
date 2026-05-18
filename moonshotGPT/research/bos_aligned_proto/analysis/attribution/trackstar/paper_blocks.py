"""Paper-faithful pooled-block projection helpers for TrackStar.

This module implements the TrackStar Appendix A.1.2 layout for the current
24-layer GPT-2-style checkpoints used in this repo:

- 8 contiguous layer blocks of 3 layers each
- attention and MLP gradients pooled separately
- two-sided random projection applied after pooling
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch

from .bergson_queries import _apply_weight_normalizer, _create_projection_matrix
from ..common.ewok_targets import EWOKTargetBundle, iter_target_batches, score_target_batch


_MODULE_NAME_RE = re.compile(r"^h\.(\d+)\.(attn|mlp)\.(c_attn|c_proj|c_fc)$")
_EXPECTED_LAYER_COUNT = 24
_BLOCK_COUNT = 8
_LAYERS_PER_BLOCK = 3
_ATTN_MODULE_ORDER = ("c_attn", "c_proj")
_MLP_MODULE_ORDER = ("c_fc", "c_proj")
_EXPECTED_MODULE_ORDER = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
PAPER_BLOCK_LAYOUT = "paper_blocks"
MODULE_LAYOUT = "module"


@dataclass(frozen=True)
class PaperBlockMember:
    module_name: str
    layer_index: int
    family: str
    submodule_name: str
    out_dim: int
    in_dim: int
    row_offset: int
    col_offset: int
    block_name: str


@dataclass(frozen=True)
class PaperBlockSpec:
    name: str
    block_index: int
    family: str
    layer_start: int
    layer_end: int
    row_dim: int
    col_dim: int
    side_dim: int
    feature_dim: int
    members: tuple[PaperBlockMember, ...]


@dataclass(frozen=True)
class PaperBlockLayoutSpec:
    layout_name: str
    side_dim: int
    feature_dim: int
    blocks: tuple[PaperBlockSpec, ...]
    module_order: tuple[str, ...]

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    @property
    def block_feature_sizes(self) -> dict[str, int]:
        return {block.name: int(block.feature_dim) for block in self.blocks}

    @property
    def module_to_block(self) -> dict[str, str]:
        return {
            member.module_name: member.block_name
            for block in self.blocks
            for member in block.members
        }

    @property
    def block_lookup(self) -> dict[str, PaperBlockSpec]:
        return {block.name: block for block in self.blocks}

    @property
    def member_lookup(self) -> dict[str, PaperBlockMember]:
        return {
            member.module_name: member
            for block in self.blocks
            for member in block.members
        }


def validate_paper_block_features(feature_dim: int) -> int:
    side_dim = int(math.isqrt(int(feature_dim)))
    if side_dim * side_dim != int(feature_dim):
        raise ValueError(
            "paper_block_features must be a perfect square so TrackStar's two-sided "
            f"projection can produce a square pooled block. Got {feature_dim}."
        )
    if side_dim <= 0:
        raise ValueError("paper_block_features must be > 0")
    return side_dim


def _parse_module_name(module_name: str) -> tuple[int, str, str]:
    match = _MODULE_NAME_RE.fullmatch(str(module_name))
    if match is None:
        raise ValueError(
            "paper_blocks mode currently only supports GPT-2-style module names "
            f"like 'h.7.attn.c_attn'. Got {module_name!r}."
        )
    layer_index = int(match.group(1))
    family = str(match.group(2))
    submodule_name = str(match.group(3))
    return layer_index, family, submodule_name


def build_gpt2_paper_block_layout(
    module_shapes: Mapping[str, Sequence[int] | torch.Size],
    *,
    feature_dim: int,
) -> PaperBlockLayoutSpec:
    side_dim = validate_paper_block_features(int(feature_dim))

    parsed: dict[str, tuple[int, str, str, int, int]] = {}
    by_layer: dict[int, dict[str, tuple[int, int]]] = {}
    for module_name, shape in module_shapes.items():
        out_dim, in_dim = (int(shape[0]), int(shape[1]))
        layer_index, family, submodule_name = _parse_module_name(str(module_name))
        parsed[str(module_name)] = (layer_index, family, submodule_name, out_dim, in_dim)
        layer = by_layer.setdefault(layer_index, {})
        layer[f"{family}.{submodule_name}"] = (out_dim, in_dim)

    discovered_layers = sorted(by_layer)
    expected_layers = list(range(_EXPECTED_LAYER_COUNT))
    if discovered_layers != expected_layers:
        raise ValueError(
            "paper_blocks mode expects a 24-layer GPT-2-style model with layers 0..23. "
            f"Found layers {discovered_layers!r}."
        )

    for layer_index in expected_layers:
        observed = tuple(sorted(by_layer[layer_index]))
        if observed != tuple(sorted(_EXPECTED_MODULE_ORDER)):
            raise ValueError(
                "paper_blocks mode expects each layer to expose exactly "
                f"{_EXPECTED_MODULE_ORDER!r}. Layer {layer_index} had {observed!r}."
            )

    blocks: list[PaperBlockSpec] = []
    ordered_modules: list[str] = []
    for block_index in range(_BLOCK_COUNT):
        layer_start = block_index * _LAYERS_PER_BLOCK
        layer_end = layer_start + _LAYERS_PER_BLOCK - 1
        for family, module_order in (("attn", _ATTN_MODULE_ORDER), ("mlp", _MLP_MODULE_ORDER)):
            members: list[PaperBlockMember] = []
            row_offset = 0
            col_offset = 0
            block_name = f"block_{block_index:02d}_{family}"
            for layer_index in range(layer_start, layer_end + 1):
                for submodule_name in module_order:
                    module_name = f"h.{layer_index}.{family}.{submodule_name}"
                    _, _, _, out_dim, in_dim = parsed[module_name]
                    members.append(
                        PaperBlockMember(
                            module_name=module_name,
                            layer_index=layer_index,
                            family=family,
                            submodule_name=submodule_name,
                            out_dim=out_dim,
                            in_dim=in_dim,
                            row_offset=row_offset,
                            col_offset=col_offset,
                            block_name=block_name,
                        )
                    )
                    ordered_modules.append(module_name)
                    row_offset += out_dim
                    col_offset += in_dim
            blocks.append(
                PaperBlockSpec(
                    name=block_name,
                    block_index=block_index,
                    family=family,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    row_dim=row_offset,
                    col_dim=col_offset,
                    side_dim=side_dim,
                    feature_dim=int(feature_dim),
                    members=tuple(members),
                )
            )

    layout = PaperBlockLayoutSpec(
        layout_name=PAPER_BLOCK_LAYOUT,
        side_dim=side_dim,
        feature_dim=int(feature_dim),
        blocks=tuple(blocks),
        module_order=tuple(ordered_modules),
    )
    _validate_layout(layout)
    return layout


def _validate_layout(layout: PaperBlockLayoutSpec) -> None:
    expected_blocks = _BLOCK_COUNT * 2
    if len(layout.blocks) != expected_blocks:
        raise AssertionError(f"Expected {expected_blocks} pooled blocks, got {len(layout.blocks)}")
    member_names = [member.module_name for block in layout.blocks for member in block.members]
    if len(member_names) != len(set(member_names)):
        raise AssertionError("Each GPT-2 module must appear exactly once in the pooled-block layout")


def paper_block_layout_metadata(layout: PaperBlockLayoutSpec) -> dict[str, Any]:
    return {
        "layout_name": layout.layout_name,
        "side_dim": int(layout.side_dim),
        "feature_dim": int(layout.feature_dim),
        "blocks": [
            {
                "name": block.name,
                "block_index": int(block.block_index),
                "family": block.family,
                "layer_start": int(block.layer_start),
                "layer_end": int(block.layer_end),
                "row_dim": int(block.row_dim),
                "col_dim": int(block.col_dim),
                "member_names": [member.module_name for member in block.members],
            }
            for block in layout.blocks
        ],
    }


def project_paper_block_mapping(
    module_grads: Mapping[str, torch.Tensor],
    *,
    layout: PaperBlockLayoutSpec,
    projection_type: str = "rademacher",
) -> dict[str, torch.Tensor]:
    projected: dict[str, torch.Tensor] = {}
    for block in layout.blocks:
        block_features: torch.Tensor | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for member in block.members:
            grad = module_grads.get(member.module_name)
            if grad is None:
                raise KeyError(
                    f"Module gradients are missing {member.module_name!r} required by pooled block {block.name!r}"
                )
            value = grad.detach()
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim != 3:
                raise ValueError(
                    f"Expected a [N, O, I] tensor for {member.module_name!r}, got {tuple(value.shape)}"
                )
            if int(value.shape[1]) != member.out_dim or int(value.shape[2]) != member.in_dim:
                raise ValueError(
                    f"Gradient shape mismatch for {member.module_name!r}: got {tuple(value.shape)}, "
                    f"expected [N, {member.out_dim}, {member.in_dim}]"
                )
            if block_features is None:
                device = value.device
                dtype = value.dtype
                block_features = torch.zeros(
                    (int(value.shape[0]), block.side_dim, block.side_dim),
                    device=device,
                    dtype=dtype,
                )
                left = _create_projection_matrix(
                    f"{block.name}/left",
                    block.side_dim,
                    block.row_dim,
                    dtype=dtype,
                    device=device,
                    projection_type=projection_type,
                )
                right = _create_projection_matrix(
                    f"{block.name}/right",
                    block.side_dim,
                    block.col_dim,
                    dtype=dtype,
                    device=device,
                    projection_type=projection_type,
                )
            assert block_features is not None
            left_slice = left[:, member.row_offset : member.row_offset + member.out_dim]
            right_slice = right[:, member.col_offset : member.col_offset + member.in_dim]
            # Efficient block-diagonal two-sided projection:
            #   L diag(G_i) R^T = sum_i L_i G_i R_i^T
            tmp = torch.einsum("noi,bi->nob", value, right_slice)
            block_features = block_features + torch.einsum("ao,nob->nab", left_slice, tmp)
        if block_features is None:
            raise RuntimeError(f"Pooled block {block.name!r} has no members")
        projected[block.name] = block_features.reshape(int(block_features.shape[0]), -1)
    return projected


def project_single_paper_block_member(
    member: PaperBlockMember,
    value: torch.Tensor,
    *,
    block: PaperBlockSpec,
    projection_type: str = "rademacher",
) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(
            f"Expected a [N, O, I] tensor for {member.module_name!r}, got {tuple(value.shape)}"
        )
    if int(value.shape[1]) != member.out_dim or int(value.shape[2]) != member.in_dim:
        raise ValueError(
            f"Gradient shape mismatch for {member.module_name!r}: got {tuple(value.shape)}, "
            f"expected [N, {member.out_dim}, {member.in_dim}]"
        )
    left = _create_projection_matrix(
        f"{block.name}/left",
        block.side_dim,
        block.row_dim,
        dtype=value.dtype,
        device=value.device,
        projection_type=projection_type,
    )
    right = _create_projection_matrix(
        f"{block.name}/right",
        block.side_dim,
        block.col_dim,
        dtype=value.dtype,
        device=value.device,
        projection_type=projection_type,
    )
    left_slice = left[:, member.row_offset : member.row_offset + member.out_dim]
    right_slice = right[:, member.col_offset : member.col_offset + member.in_dim]
    tmp = torch.einsum("noi,bi->nob", value, right_slice)
    block_features = torch.einsum("ao,nob->nab", left_slice, tmp)
    return block_features.reshape(int(block_features.shape[0]), -1)


def apply_weight_normalizers_and_project_paper_blocks(
    raw_module_grads: Mapping[str, torch.Tensor],
    *,
    layout: PaperBlockLayoutSpec,
    weight_normalizers: Mapping[str, Any] | None,
    projection_type: str = "rademacher",
) -> dict[str, torch.Tensor]:
    corrected: dict[str, torch.Tensor] = {}
    for module_name, grad in raw_module_grads.items():
        corrected_grad = _apply_weight_normalizer(
            module_name,
            grad,
            weight_normalizers=weight_normalizers,
        )
        corrected[module_name] = corrected_grad
    return project_paper_block_mapping(
        corrected,
        layout=layout,
        projection_type=projection_type,
    )


def reshape_flattened_module_batch(
    flat_grads: torch.Tensor,
    *,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    if flat_grads.ndim != 2:
        raise ValueError(f"Expected [N, O*I] flattened grads, got {tuple(flat_grads.shape)}")
    expected_dim = int(out_dim) * int(in_dim)
    if int(flat_grads.shape[1]) != expected_dim:
        raise ValueError(
            f"Flattened module batch has width {int(flat_grads.shape[1])}, expected {expected_dim}"
        )
    return flat_grads.reshape(int(flat_grads.shape[0]), int(out_dim), int(in_dim))


def collect_paper_block_candidate_index(
    *,
    model: torch.nn.Module,
    data,
    processor,
    cfg,
    preprocess_cfg,
    layout: PaperBlockLayoutSpec,
    target_modules: set[str] | None = None,
    batches: list[list[int]] | None = None,
):
    """Collect candidate gradients using paper-faithful pooled-block projection.

    This reuses Bergson's hook-based per-sample gradient extraction while
    keeping projection disabled in the processor, then pools and projects
    gradients into the TrackStar paper's layer blocks inside a custom collector.
    """

    from bergson.builder import Builder
    from bergson.collector.collector import CollectorComputer
    from bergson.collector.gradient_collectors import GradientCollector
    from bergson.process_preconditioners import process_preconditioners
    from bergson.utils.utils import get_gradient_dtype
    from datasets import Dataset

    @dataclass(kw_only=True)
    class _PaperBlockGradientCollector(GradientCollector):
        layout: PaperBlockLayoutSpec

        def setup(self) -> None:  # type: ignore[override]
            assert isinstance(
                self.model.device, torch.device
            ), "Model device is not set correctly"

            self.attribute_tokens = self.cfg.attribute_tokens
            if self.cfg.attribute_tokens:
                raise ValueError("paper_blocks mode does not support attribute_tokens")

            self.save_dtype = get_gradient_dtype(self.model)
            self.lo = torch.finfo(self.save_dtype).min
            self.hi = torch.finfo(self.save_dtype).max
            self.per_doc_losses = torch.full(
                (len(self.data),),
                device=self.model.device,
                dtype=torch.float32,
                fill_value=0.0,
            )
            self.save_index = self.scorer is None and not self.cfg.skip_index
            if self.save_index:
                self.builder = Builder(
                    self.data,
                    self.layout.block_feature_sizes,
                    self.save_dtype,
                    self.preprocess_cfg,
                    attribute_tokens=False,
                    path=self.cfg.partial_run_path,
                )
            else:
                self.builder = None

        def shapes(self) -> Mapping[str, torch.Size]:  # type: ignore[override]
            return {
                block.name: torch.Size((block.side_dim, block.side_dim))
                for block in self.layout.blocks
            }

        def backward_hook(self, module: torch.nn.Module, g: torch.Tensor) -> None:  # type: ignore[override]
            module_name = str(module._name)  # type: ignore[attr-defined]
            member = self.layout.member_lookup.get(module_name)
            if member is None:
                raise KeyError(f"paper_blocks collector received unexpected module {module_name!r}")

            flat_batch = self._compute_gradient(module, g)
            matrix_batch = reshape_flattened_module_batch(
                flat_batch,
                out_dim=member.out_dim,
                in_dim=member.in_dim,
            )
            projected = project_single_paper_block_member(
                member,
                matrix_batch,
                block=self.layout.block_lookup[member.block_name],
                projection_type=self.processor.projection_type,
            )
            projected = projected.to(dtype=self.save_dtype)
            existing = self.mod_grads.get(member.block_name)
            if existing is None:
                self.mod_grads[member.block_name] = projected
            else:
                self.mod_grads[member.block_name] = existing + projected

        def process_batch(self, indices: list[int], **kwargs) -> None:  # type: ignore[override]
            losses = kwargs.get("losses")
            assert losses is not None, "losses must be provided in kwargs"

            if not self.cfg.skip_preconditioners:
                for block_name, block_features in self.mod_grads.items():
                    block_features = block_features.float()
                    if block_name in self.processor.preconditioners:
                        self.processor.preconditioners[block_name].addmm_(
                            block_features.mT,
                            block_features,
                        )
                    else:
                        self.processor.preconditioners[block_name] = block_features.mT @ block_features

            if self.builder:
                self.builder(indices, self.mod_grads)
            if self.scorer:
                self.scorer(indices, self.mod_grads)
            self.mod_grads.clear()
            self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)

        def teardown(self) -> None:  # type: ignore[override]
            import torch.distributed as dist
            from datasets import Dataset as HFDataset, Value

            if dist.is_initialized():
                dist.reduce(self.per_doc_losses, dst=0)

            grad_sizes = {name: int(math.prod(shape)) for name, shape in self.shapes().items()}
            if self.processor.preconditioners:
                process_preconditioners(
                    self.processor,
                    self.processor.preconditioners,
                    len(self.data),
                    grad_sizes,
                    self.rank,
                )

            if self.builder:
                self.builder.teardown()

            if self.rank == 0:
                if self.preprocess_cfg.aggregation != "none":
                    self.data = HFDataset.from_list(
                        [{"query_index": i} for i in range(self.builder.grad_buffer.shape[0])]
                    )
                else:
                    if self.cfg.drop_columns:
                        self.data = self.data.remove_columns(["input_ids"])
                    losses_np = self.per_doc_losses.cpu().numpy()
                    try:
                        self.data = self.data.add_column(
                            "loss",
                            losses_np,
                            feature=Value("float32"),
                            new_fingerprint="loss",
                        )
                    except TypeError as exc:
                        if "feature" not in str(exc):
                            raise
                        self.data = self.data.add_column(
                            "loss",
                            losses_np,
                            new_fingerprint="loss",
                        )
                self.data.save_to_disk(str(self.cfg.partial_run_path / "data.hf"))
                self.processor.save(self.cfg.partial_run_path)

    if getattr(processor, "projection_dim", None) is not None:
        raise ValueError("paper_blocks collection expects processor.projection_dim to be None")

    raw_target_modules = set(layout.module_order) if target_modules is None else set(target_modules)
    collector = _PaperBlockGradientCollector(
        model=model.base_model,  # type: ignore[arg-type]
        cfg=cfg,
        processor=processor,
        target_modules=raw_target_modules,
        data=data if isinstance(data, Dataset) else data,
        scorer=None,
        preprocess_cfg=preprocess_cfg,
        attention_cfgs={},
        filter_modules=cfg.filter_modules,
        layout=layout,
    )
    computer = CollectorComputer(
        model=model,  # type: ignore[arg-type]
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="TrackStar paper-block gradients")


def collect_query_paper_block_grads(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
    layout: PaperBlockLayoutSpec,
    reduction: str,
    weight_normalizers: Mapping[str, Any] | None = None,
    projection_type: str = "rademacher",
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    from .bergson_queries import _build_tqdm, _resolve_bos_token_id, build_query_groups, _select_gradient_modules
    from .raw_dot_audit import _normalize_module_weight_grad

    group_specs = build_query_groups(bundle, reduction)
    modules = _select_gradient_modules(model, layout.module_order)
    bos_token_id = _resolve_bos_token_id(tokenizer)

    per_item_grads: dict[str, list[torch.Tensor]] = {name: [] for name in layout.block_names}
    progress = _build_tqdm(
        enabled=show_progress,
        total=len(bundle.items),
        desc=progress_desc or "TrackStar paper-block query gradients",
        unit="target",
        leave=False,
    )
    model.eval()
    try:
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
            scores["softplus_loss"].sum().backward()
            raw_module_grads: dict[str, torch.Tensor] = {}
            for name, module in modules.items():
                grad = getattr(module, "weight").grad
                if grad is None:
                    raise RuntimeError(f"Missing gradient for query module {name!r}")
                raw_module_grads[name] = _normalize_module_weight_grad(module, grad.detach()).to(dtype=torch.float32)
            block_features = apply_weight_normalizers_and_project_paper_blocks(
                raw_module_grads,
                layout=layout,
                weight_normalizers=weight_normalizers,
                projection_type=projection_type,
            )
            for block_name, feature in block_features.items():
                per_item_grads[block_name].append(feature.squeeze(0).detach().cpu().to(dtype=torch.float32))
            if progress is not None:
                progress.update(len(prepared.items))
    finally:
        if progress is not None:
            progress.close()

    grouped_grads: dict[str, torch.Tensor] = {}
    for name, grads in per_item_grads.items():
        stacked = torch.stack(grads, dim=0)
        grouped = [stacked[list(indices)].mean(dim=0) for _, indices in group_specs]
        grouped_grads[name] = torch.stack(grouped, dim=0)

    return tuple(group_id for group_id, _ in group_specs), grouped_grads


__all__ = [
    "MODULE_LAYOUT",
    "PAPER_BLOCK_LAYOUT",
    "PaperBlockLayoutSpec",
    "PaperBlockMember",
    "PaperBlockSpec",
    "apply_weight_normalizers_and_project_paper_blocks",
    "build_gpt2_paper_block_layout",
    "collect_query_paper_block_grads",
    "paper_block_layout_metadata",
    "project_paper_block_mapping",
    "project_single_paper_block_member",
    "reshape_flattened_module_batch",
    "validate_paper_block_features",
]
