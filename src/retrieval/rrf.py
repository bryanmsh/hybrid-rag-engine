"""Reciprocal Rank Fusion (RRF) algorithm for fusing dense and sparse candidate rankings."""

from typing import Any
from src.indexing.models import SearchResult
from src.retrieval.models import FusedResult


def compute_rrf(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_n: int = 20,
) -> list[FusedResult]:
    """Merge ranked lists from dense vector search and sparse BM25 search using RRF.

    Formula:
        RRF_score(d) = sum_{m in {dense, sparse}} (1 / (k + rank_m(d)))

    Args:
        dense_results: Ordered list of SearchResult from dense vector retrieval.
        sparse_results: Ordered list of SearchResult from sparse BM25 retrieval.
        k: Smoothing constant (default: 60, mitigates high-rank dominance).
        top_n: Number of fused candidates to return.

    Returns:
        List of FusedResult sorted descending by RRF score.
    """
    candidates: dict[str, dict[str, Any]] = {}

    # 1. Accumulate dense retrieval contributions
    for item in dense_results:
        cid = item.chunk_id
        contribution = 1.0 / (k + item.rank)
        if cid not in candidates:
            candidates[cid] = {
                "chunk_id": cid,
                "content": item.content,
                "metadata": dict(item.metadata),
                "rrf_score": contribution,
                "dense_rank": item.rank,
                "dense_score": item.score,
                "sparse_rank": None,
                "sparse_score": None,
            }
        else:
            candidates[cid]["rrf_score"] += contribution
            candidates[cid]["dense_rank"] = item.rank
            candidates[cid]["dense_score"] = item.score

    # 2. Accumulate sparse retrieval contributions
    for item in sparse_results:
        cid = item.chunk_id
        contribution = 1.0 / (k + item.rank)
        if cid not in candidates:
            candidates[cid] = {
                "chunk_id": cid,
                "content": item.content,
                "metadata": dict(item.metadata),
                "rrf_score": contribution,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": item.rank,
                "sparse_score": item.score,
            }
        else:
            candidates[cid]["rrf_score"] += contribution
            candidates[cid]["sparse_rank"] = item.rank
            candidates[cid]["sparse_score"] = item.score

    # 3. Sort candidates descending by RRF score
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # 4. Truncate to top_n and assign final fused ranks
    fused_results: list[FusedResult] = []
    for rank, cand in enumerate(sorted_candidates[:top_n], start=1):
        fused_results.append(
            FusedResult(
                chunk_id=cand["chunk_id"],
                content=cand["content"],
                rrf_score=round(cand["rrf_score"], 8),
                fused_rank=rank,
                dense_rank=cand["dense_rank"],
                dense_score=cand["dense_score"],
                sparse_rank=cand["sparse_rank"],
                sparse_score=cand["sparse_score"],
                metadata=cand["metadata"],
            )
        )

    return fused_results
