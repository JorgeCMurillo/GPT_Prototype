import torch

from research.bos_aligned_proto.analysis.attribution.trackstar.score_path_audit import (
    _apply_weight_normalizers_to_grads,
    _project_grad_mapping,
    _project_grad_mapping_with_layout,
    _squeeze_feature_mapping,
)


class _ToyNormalizer:
    def normalize_weight(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * 2.0


def test_score_path_audit_helpers_apply_normalizers_project_and_squeeze() -> None:
    grads = {
        "toy.module": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    }
    corrected = _apply_weight_normalizers_to_grads(
        grads,
        weight_normalizers={"toy.module": _ToyNormalizer()},
    )
    assert torch.equal(corrected["toy.module"], grads["toy.module"] * 2.0)

    projected = _project_grad_mapping(
        grads,
        projection_dim=None,
    )
    assert projected["toy.module"].ndim == 1
    assert torch.equal(projected["toy.module"], grads["toy.module"].reshape(-1))

    squeezed = _squeeze_feature_mapping(
        {"toy.module": projected["toy.module"].unsqueeze(0)}
    )
    assert squeezed["toy.module"].ndim == 1
    assert torch.equal(squeezed["toy.module"], projected["toy.module"])


def test_score_path_audit_projects_paper_block_layout() -> None:
    grads = {}
    for layer in range(24):
        grads[f"h.{layer}.attn.c_attn"] = torch.ones((2, 3), dtype=torch.float32)
        grads[f"h.{layer}.attn.c_proj"] = torch.ones((3, 2), dtype=torch.float32)
        grads[f"h.{layer}.mlp.c_fc"] = torch.ones((4, 2), dtype=torch.float32)
        grads[f"h.{layer}.mlp.c_proj"] = torch.ones((2, 4), dtype=torch.float32)

    projected = _project_grad_mapping_with_layout(
        grads,
        projection_dim=None,
        projection_layout="paper_blocks",
        paper_block_features=16,
    )

    assert len(projected) == 16
    assert all(value.ndim == 1 for value in projected.values())
    assert all(int(value.shape[0]) == 16 for value in projected.values())
