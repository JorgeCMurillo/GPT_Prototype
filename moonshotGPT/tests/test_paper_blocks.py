from pathlib import Path

import torch

from research.bos_aligned_proto.analysis.attribution.common.checkpoints import CheckpointRef
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_datasets import (
    build_candidate_index_fingerprint,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.bergson_queries import (
    _create_projection_matrix,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.paper_blocks import (
    PaperBlockLayoutSpec,
    PaperBlockMember,
    PaperBlockSpec,
    build_gpt2_paper_block_layout,
    project_paper_block_mapping,
    validate_paper_block_features,
)


def _toy_gpt2_module_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(24):
        shapes[f"h.{layer}.attn.c_attn"] = (6, 4)
        shapes[f"h.{layer}.attn.c_proj"] = (4, 6)
        shapes[f"h.{layer}.mlp.c_fc"] = (8, 4)
        shapes[f"h.{layer}.mlp.c_proj"] = (4, 8)
    return shapes


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
