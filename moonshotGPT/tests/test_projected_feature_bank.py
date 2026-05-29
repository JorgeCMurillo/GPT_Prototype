import numpy as np
import pytest

from research.bos_aligned_proto.analysis.attribution.trackstar.projected_feature_bank import (
    _candidate_metadata_from_manifest,
    _load_candidate_ids_path,
    _normalize_feature_metrics,
    _sample_random_candidate_ids,
    _score_feature_matrix,
)


class _Ref:
    def __init__(self, idx: int) -> None:
        self.global_example_id = idx
        self.candidate_kind = "document_aligned_row"
        self.shard_path = f"/tmp/train_{idx // 10:06d}.bin"
        self.local_example_idx = idx
        self.token_offset_start = idx * 1025
        self.token_offset_end = idx * 1025 + 1025
        self.document_token_offset_start = idx * 1025
        self.document_token_offset_end = idx * 1025 + 2048


class _Manifest:
    candidate_kind = "document_aligned_row"

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def example_ref(self, idx: int) -> _Ref:
        return _Ref(int(idx))


def test_score_feature_matrix_matches_dot_and_cosine_with_chunks() -> None:
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float16,
    )
    candidate_norms = np.linalg.norm(candidates.astype(np.float64), axis=1).astype(np.float32)
    queries = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    dot_scores = _score_feature_matrix(
        candidate_features=candidates,
        candidate_norms=candidate_norms,
        query_features=queries,
        cosine=False,
        chunk_size=2,
        show_progress=False,
        desc="test dot",
    )

    expected_dot = queries @ candidates.astype(np.float32).T
    np.testing.assert_allclose(dot_scores, expected_dot, rtol=1e-6, atol=1e-6)

    cosine_scores = _score_feature_matrix(
        candidate_features=candidates,
        candidate_norms=candidate_norms,
        query_features=queries,
        cosine=True,
        chunk_size=2,
        show_progress=False,
        desc="test cosine",
    )

    query_norms = np.linalg.norm(queries.astype(np.float64), axis=1)
    expected_cosine = expected_dot / (query_norms[:, None] * candidate_norms[None, :])
    np.testing.assert_allclose(cosine_scores, expected_cosine, rtol=1e-6, atol=1e-6)


def test_normalize_feature_metrics_expands_all_and_deduplicates() -> None:
    assert _normalize_feature_metrics(("all", "projected_raw_dot")) == (
        "projected_raw_dot",
        "projected_raw_cosine",
        "projected_adam_dot",
        "projected_adam_cosine",
        "trackstar_no_hessian",
    )


def test_normalize_feature_metrics_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        _normalize_feature_metrics(("not_a_metric",))


def test_random_candidate_ids_are_deterministic_sorted_and_validated() -> None:
    manifest = _Manifest(100)

    first = _sample_random_candidate_ids(manifest=manifest, count=10, seed=123)
    second = _sample_random_candidate_ids(manifest=manifest, count=10, seed=123)

    assert first == second
    assert first == tuple(sorted(first))
    assert len(first) == 10
    assert len(set(first)) == 10
    assert min(first) >= 0
    assert max(first) < len(manifest)


def test_candidate_metadata_from_manifest_uses_document_offsets() -> None:
    rows = _candidate_metadata_from_manifest(_Manifest(10), (3,), selection_score=None)

    assert rows == [
        {
            "candidate_id": 3,
            "candidate_kind": "document_aligned_row",
            "shard_path": "/tmp/train_000000.bin",
            "local_example_idx": 3,
            "token_offset_start": 3075,
            "token_offset_end": 4100,
            "row_id": 3,
            "local_row_idx": 3,
            "document_token_offset_start": 3075,
            "document_token_offset_end": 5123,
            "selection_score": None,
        }
    ]


def test_load_candidate_ids_path_accepts_npy_and_text(tmp_path) -> None:
    npy_path = tmp_path / "ids.npy"
    np.save(npy_path, np.asarray([5, 2, 9], dtype=np.int64))
    assert _load_candidate_ids_path(npy_path) == (5, 2, 9)

    text_path = tmp_path / "ids.txt"
    text_path.write_text("7\n11\n\n13\n", encoding="utf-8")
    assert _load_candidate_ids_path(text_path) == (7, 11, 13)
