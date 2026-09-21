"""Unit tests for Reciprocal Rank Fusion (RRF) algorithm."""

import pytest
from src.indexing.models import SearchResult
from src.retrieval.rrf import compute_rrf


def test_rrf_scoring_math():
    """Verify that RRF score matches sum(1 / (k + rank_m))."""
    k = 60
    # doc-A: rank 1 in dense, rank 2 in sparse
    # doc-B: rank 2 in dense, rank 1 in sparse
    # doc-C: rank 3 in dense only
    dense_results = [
        SearchResult(chunk_id="doc-A", content="A", score=0.9, rank=1, retrieval_type="dense"),
        SearchResult(chunk_id="doc-B", content="B", score=0.8, rank=2, retrieval_type="dense"),
        SearchResult(chunk_id="doc-C", content="C", score=0.7, rank=3, retrieval_type="dense"),
    ]
    sparse_results = [
        SearchResult(chunk_id="doc-B", content="B", score=10.0, rank=1, retrieval_type="sparse"),
        SearchResult(chunk_id="doc-A", content="A", score=5.0, rank=2, retrieval_type="sparse"),
    ]

    fused = compute_rrf(dense_results, sparse_results, k=k, top_n=3)

    assert len(fused) == 3

    # Expected scores:
    # doc-A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639344 + 0.01612903 = 0.03252247
    # doc-B: 1/(60+2) + 1/(60+1) = same! Tied at top.
    # doc-C: 1/(60+3) = 1/63 ≈ 0.01587301
    score_a = 1.0 / 61 + 1.0 / 62
    score_c = 1.0 / 63

    # doc-C must be rank 3
    assert fused[2].chunk_id == "doc-C"
    assert pytest.approx(fused[2].rrf_score, rel=1e-5) == score_c
    assert fused[2].dense_rank == 3
    assert fused[2].sparse_rank is None

    # doc-A and doc-B must have higher scores than doc-C
    assert fused[0].rrf_score > fused[2].rrf_score
    assert fused[1].rrf_score > fused[2].rrf_score
    assert pytest.approx(fused[0].rrf_score, rel=1e-5) == score_a


def test_rrf_truncation_top_n():
    """Verify top_n truncates candidate pool accurately."""
    dense = [
        SearchResult(chunk_id=f"c{i}", content=f"content {i}", score=1.0 - i * 0.05, rank=i, retrieval_type="dense")
        for i in range(1, 11)
    ]
    sparse = []

    fused = compute_rrf(dense, sparse, k=60, top_n=4)
    assert len(fused) == 4
    assert [f.chunk_id for f in fused] == ["c1", "c2", "c3", "c4"]
    assert [f.fused_rank for f in fused] == [1, 2, 3, 4]


def test_rrf_empty_inputs():
    """Verify behavior on empty lists."""
    fused = compute_rrf([], [], k=60, top_n=10)
    assert fused == []
