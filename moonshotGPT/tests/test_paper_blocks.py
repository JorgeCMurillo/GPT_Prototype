import hashlib
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D as HFConv1D

from research.bos_aligned_proto.analysis.attribution.common.checkpoints import CheckpointRef
from research.bos_aligned_proto.analysis.attribution.trackstar.backend import _module_gradient_shape
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_datasets import (
    build_candidate_index_fingerprint,
    load_flat_gradient_index,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_queries import (
    _create_projection_matrix,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.paper_blocks import (
    PaperBlockLayoutSpec,
    PaperBlockMember,
    PaperBlockSpec,
    build_gpt2_paper_block_layout,
    collect_paper_block_candidate_index,
    project_paper_block_mapping,
    validate_paper_block_features,
)


def _bergson_rademacher_projection(identifier: str, rows: int, cols: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.md5(bytes(identifier, "utf-8")).digest(), byteorder="big") % (2**63 - 1)
    numpy_rng = np.random.Generator(np.random.PCG64(seed))
    random_bytes = numpy_rng.bytes((rows * cols + 7) // 8)
    random_bytes = np.frombuffer(random_bytes, dtype=np.uint8)
    bits = np.unpackbits(random_bytes)[: rows * cols].reshape((rows, cols))
    matrix = torch.from_numpy(bits).to(dtype=torch.float32)
    matrix = matrix.add_(-0.5).mul_(2)
    return matrix / matrix.norm(dim=1, keepdim=True)


def _toy_gpt2_module_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(24):
        shapes[f"h.{layer}.attn.c_attn"] = (6, 4)
        shapes[f"h.{layer}.attn.c_proj"] = (4, 6)
        shapes[f"h.{layer}.mlp.c_fc"] = (8, 4)
        shapes[f"h.{layer}.mlp.c_proj"] = (4, 8)
    return shapes


def _random_toy_gpt2_grads(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn((1, out_dim, in_dim), generator=generator, dtype=torch.float32)
        for name, (out_dim, in_dim) in _toy_gpt2_module_shapes().items()
    }


def _module_raw_dot(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    return sum(float(torch.sum(left[name] * right[name]).item()) for name in left)


def _paper_block_projected_dot(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    side_dim: int,
    rescale: bool,
) -> float:
    layout = build_gpt2_paper_block_layout(
        _toy_gpt2_module_shapes(),
        feature_dim=int(side_dim) * int(side_dim),
    )
    left_projected = project_paper_block_mapping(left, layout=layout, projection_type="rademacher")
    right_projected = project_paper_block_mapping(right, layout=layout, projection_type="rademacher")
    dot = 0.0
    for block in layout.blocks:
        value = float(torch.sum(left_projected[block.name] * right_projected[block.name]).item())
        if rescale:
            value *= float(block.row_dim * block.col_dim) / float(side_dim * side_dim)
        dot += value
    return dot


def _paper_block_projection_relative_rmse(*, side_dim: int, pairs: int = 8) -> float:
    squared_errors: list[float] = []
    raw_values: list[float] = []
    for pair_idx in range(int(pairs)):
        left = _random_toy_gpt2_grads(1000 + 2 * pair_idx)
        right = _random_toy_gpt2_grads(1001 + 2 * pair_idx)
        raw_dot = _module_raw_dot(left, right)
        projected_dot = _paper_block_projected_dot(
            left,
            right,
            side_dim=side_dim,
            rescale=True,
        )
        squared_errors.append((projected_dot - raw_dot) ** 2)
        raw_values.append(raw_dot**2)
    rmse = float(np.sqrt(np.mean(squared_errors)))
    raw_rms = float(np.sqrt(np.mean(raw_values)))
    return rmse / raw_rms


class _TinyGPT2Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_attn = torch.nn.Linear(4, 6, bias=False)
        self.c_proj = torch.nn.Linear(6, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.tanh(self.c_attn(hidden)))


class _TinyGPT2MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = torch.nn.Linear(4, 8, bias=False)
        self.c_proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.tanh(self.c_fc(hidden)))


class _TinyGPT2Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyGPT2Attention()
        self.mlp = _TinyGPT2MLP()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(hidden)
        hidden = hidden + self.mlp(hidden)
        return hidden


class _TinyGPT2CausalLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 11) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, 4)
        self.h = torch.nn.ModuleList(_TinyGPT2Block() for _ in range(24))
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)

    @property
    def base_model(self) -> "_TinyGPT2CausalLM":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.wte(input_ids)
        for block in self.h:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _manual_ce_paper_block_features(
    model: _TinyGPT2CausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    layout: PaperBlockLayoutSpec,
) -> dict[str, np.ndarray]:
    model.zero_grad(set_to_none=True)
    logits = model(input_ids).logits[:, :-1]
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.flatten(),
        reduction="none",
    ).reshape_as(shifted_labels)
    losses = losses.sum(dim=1) / (shifted_labels != -100).sum(dim=1, dtype=model.dtype)
    losses.sum().backward()

    module_lookup = dict(model.named_modules())
    raw_module_grads: dict[str, torch.Tensor] = {}
    for module_name in layout.module_order:
        module = module_lookup[module_name]
        grad = module.weight.grad
        assert grad is not None
        raw_module_grads[module_name] = grad.detach().clone()

    model.zero_grad(set_to_none=True)
    projected = project_paper_block_mapping(
        raw_module_grads,
        layout=layout,
        projection_type="rademacher",
    )
    return {
        block_name: feature.detach().cpu().numpy().astype(np.float32)
        for block_name, feature in projected.items()
    }


def test_rademacher_projection_matches_bergson_pcg64_layout() -> None:
    projected = _create_projection_matrix(
        "h.0.attn.c_attn/left",
        4,
        6,
        dtype=torch.float32,
        device=torch.device("cpu"),
        projection_type="rademacher",
    )

    expected = _bergson_rademacher_projection("h.0.attn.c_attn/left", 4, 6)

    torch.testing.assert_close(projected, expected)


def test_build_gpt2_paper_block_layout_assigns_all_96_modules_once() -> None:
    layout = build_gpt2_paper_block_layout(
        _toy_gpt2_module_shapes(),
        feature_dim=4096,
    )

    assert layout.layout_name == "paper_blocks"
    assert layout.side_dim == 64
    assert len(layout.blocks) == 16
    assert len(layout.module_order) == 96
    assert len(set(layout.module_order)) == 96

    first_block = layout.blocks[0]
    assert first_block.name == "block_00_attn"
    assert first_block.layer_start == 0
    assert first_block.layer_end == 2
    assert tuple(member.module_name for member in first_block.members) == (
        "h.0.attn.c_attn",
        "h.0.attn.c_proj",
        "h.1.attn.c_attn",
        "h.1.attn.c_proj",
        "h.2.attn.c_attn",
        "h.2.attn.c_proj",
    )

    last_block = layout.blocks[-1]
    assert last_block.name == "block_07_mlp"
    assert last_block.layer_start == 21
    assert last_block.layer_end == 23


def test_project_paper_block_mapping_matches_explicit_block_diagonal_projection() -> None:
    member_a = PaperBlockMember(
        module_name="h.0.attn.c_attn",
        layer_index=0,
        family="attn",
        submodule_name="c_attn",
        out_dim=2,
        in_dim=3,
        row_offset=0,
        col_offset=0,
        block_name="block_00_attn",
    )
    member_b = PaperBlockMember(
        module_name="h.0.attn.c_proj",
        layer_index=0,
        family="attn",
        submodule_name="c_proj",
        out_dim=1,
        in_dim=2,
        row_offset=2,
        col_offset=3,
        block_name="block_00_attn",
    )
    block = PaperBlockSpec(
        name="block_00_attn",
        block_index=0,
        family="attn",
        layer_start=0,
        layer_end=0,
        row_dim=3,
        col_dim=5,
        side_dim=4,
        feature_dim=16,
        members=(member_a, member_b),
    )
    layout = PaperBlockLayoutSpec(
        layout_name="paper_blocks",
        side_dim=4,
        feature_dim=16,
        blocks=(block,),
        module_order=(member_a.module_name, member_b.module_name),
    )
    grads = {
        member_a.module_name: torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32),
        member_b.module_name: torch.tensor([[[7.0, 8.0]]], dtype=torch.float32),
    }

    projected = project_paper_block_mapping(
        grads,
        layout=layout,
        projection_type="rademacher",
    )[block.name]

    explicit_block = torch.zeros((1, block.row_dim, block.col_dim), dtype=torch.float32)
    explicit_block[:, 0:2, 0:3] = grads[member_a.module_name]
    explicit_block[:, 2:3, 3:5] = grads[member_b.module_name]
    left = _create_projection_matrix(
        f"{block.name}/left",
        block.side_dim,
        block.row_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
        projection_type="rademacher",
    )
    right = _create_projection_matrix(
        f"{block.name}/right",
        block.side_dim,
        block.col_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
        projection_type="rademacher",
    )
    explicit = (left @ explicit_block.squeeze(0) @ right.mT).reshape(1, -1)

    assert torch.allclose(projected, explicit, atol=1e-6, rtol=1e-6)


def test_paper_block_projected_dot_rescaling_corrects_row_normalized_scale() -> None:
    grads = _random_toy_gpt2_grads(123)
    raw_norm = _module_raw_dot(grads, grads)

    unscaled = _paper_block_projected_dot(
        grads,
        grads,
        side_dim=8,
        rescale=False,
    )
    rescaled = _paper_block_projected_dot(
        grads,
        grads,
        side_dim=8,
        rescale=True,
    )

    assert abs(rescaled - raw_norm) / raw_norm < 0.10
    assert abs(rescaled - raw_norm) < abs(unscaled - raw_norm)


def test_paper_block_synthetic_projection_error_improves_with_side_rank() -> None:
    low_rank_error = _paper_block_projection_relative_rmse(side_dim=8)
    high_rank_error = _paper_block_projection_relative_rmse(side_dim=64)

    assert high_rank_error < 0.5 * low_rank_error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="paper-block collector uses Bergson's CUDA Builder")
def test_paper_block_candidate_collector_matches_manual_projection(tmp_path: Path) -> None:
    bergson = pytest.importorskip("bergson")
    bergson_config = pytest.importorskip("bergson.config")

    torch.manual_seed(1234)
    model = _TinyGPT2CausalLM().to("cuda").float()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    layout = build_gpt2_paper_block_layout(
        {name: tuple(module.weight.shape) for name, module in dict(model.named_modules()).items() if name in _toy_gpt2_module_shapes()},
        feature_dim=64,
    )
    expected = _manual_ce_paper_block_features(
        model,
        input_ids,
        labels,
        layout=layout,
    )

    data = Dataset.from_list(
        [
            {
                "input_ids": input_ids.squeeze(0).detach().cpu().tolist(),
                "labels": labels.squeeze(0).detach().cpu().tolist(),
            }
        ]
    )
    cfg = bergson_config.IndexConfig(
        run_path=str(tmp_path / "paper_block_parity"),
        skip_preconditioners=True,
        token_batch_size=32,
    )
    collect_paper_block_candidate_index(
        model=model,
        data=data,
        processor=bergson.GradientProcessor(projection_dim=None, projection_type="rademacher"),
        cfg=cfg,
        preprocess_cfg=bergson_config.PreprocessConfig(aggregation="none"),
        layout=layout,
    )

    loaded = load_flat_gradient_index(cfg.partial_run_path)
    assert set(loaded) == set(layout.block_names)
    for block_name in layout.block_names:
        np.testing.assert_allclose(
            loaded[block_name].astype(np.float32),
            expected[block_name],
            atol=1e-5,
            rtol=1e-5,
        )


def test_candidate_index_fingerprint_separates_module_vs_paper_block_layouts() -> None:
    checkpoint = CheckpointRef(
        step=16000,
        path=Path("/tmp/ckpt_periodic_step0016000"),
        kind="periodic",
    )
    candidate_ids = (1, 2, 3)
    module_fp = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_ids=candidate_ids,
        projection_dim=16,
        use_fast_jl=True,
        adam_second_moment_correction=True,
        projection_layout="module",
        paper_block_features=0,
        paper_block_side=0,
    )
    paper_fp = build_candidate_index_fingerprint(
        checkpoint=checkpoint,
        candidate_ids=candidate_ids,
        projection_dim=4096,
        use_fast_jl=True,
        adam_second_moment_correction=True,
        projection_layout="paper_blocks",
        paper_block_features=4096,
        paper_block_side=64,
    )
    assert module_fp != paper_fp


def test_validate_paper_block_features_requires_square_dim() -> None:
    assert validate_paper_block_features(4096) == 64
    try:
        validate_paper_block_features(3000)
    except ValueError as exc:
        assert "perfect square" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected validate_paper_block_features to reject non-square feature dims")


def test_module_gradient_shape_uses_logical_conv1d_dimensions() -> None:
    linear = torch.nn.Linear(4, 8, bias=False)
    assert _module_gradient_shape(linear) == (8, 4)

    conv1d = HFConv1D(3072, 1024)
    assert tuple(conv1d.weight.shape) == (1024, 3072)
    assert _module_gradient_shape(conv1d) == (3072, 1024)
