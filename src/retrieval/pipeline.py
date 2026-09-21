"""Unified retrieval pipeline combining hybrid search and cross-encoder reranking."""

import time
from typing import Any
from src.config.settings import get_settings
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.models import RerankResult
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import FusedResult


class SearchPipeline:
    """End-to-end inference pipeline: Dense + Sparse -> RRF (Top-20) -> Cross-Encoder (Top-4)."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        settings = get_settings()
        self.default_top_n = settings.RETRIEVAL_TOP_N
        self.default_top_k = settings.RERANK_TOP_K

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ) -> dict[str, Any]:
        """Execute search with configurable hybrid retrieval and reranking stages.

        Args:
            query: User search query.
            top_k: Number of final passages to return (default: 4).
            use_hybrid: If True, merges Dense + BM25 via RRF. If False, runs dense only.
            use_reranker: If True, reranks candidates using CrossEncoder down to top_k.

        Returns:
            Dictionary containing final results, stage latencies, and candidate counts.
        """
        start_total = time.perf_counter()
        target_k = top_k or self.default_top_k

        # 1. Retrieval Phase
        if use_hybrid:
            candidates, retrieval_metrics = self.retriever.retrieve(
                query=query,
                top_n=self.default_top_n,
            )
            dense_ms = retrieval_metrics.dense_latency_ms
            sparse_ms = retrieval_metrics.sparse_latency_ms
            rrf_ms = retrieval_metrics.rrf_latency_ms
        else:
            # Naive baseline: dense retrieval only
            dense_results, dense_ms = self.retriever.retrieve_dense(
                query=query,
                top_k=self.default_top_n,
            )
            candidates = [
                FusedResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    rrf_score=r.score,
                    fused_rank=r.rank,
                    dense_rank=r.rank,
                    dense_score=r.score,
                    metadata=r.metadata,
                )
                for r in dense_results
            ]
            sparse_ms = 0.0
            rrf_ms = 0.0

        # 2. Reranking Phase
        if use_reranker and candidates:
            final_passages, rerank_metrics = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=target_k,
            )
            rerank_ms = rerank_metrics.rerank_latency_ms
        else:
            # Pass through top_k directly without reranking
            final_passages = candidates[:target_k]
            rerank_ms = 0.0

        total_ms = (time.perf_counter() - start_total) * 1000

        return {
            "query": query,
            "results": [p.to_dict() if hasattr(p, "to_dict") else p for p in final_passages],
            "total_candidates_retrieved": len(candidates),
            "final_results_count": len(final_passages),
            "latencies": {
                "dense_retrieval_ms": dense_ms,
                "sparse_retrieval_ms": sparse_ms,
                "rrf_fusion_ms": rrf_ms,
                "rerank_ms": rerank_ms,
                "total_pipeline_ms": round(total_ms, 2),
            },
            "configuration": {
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "top_k": target_k,
            },
        }
