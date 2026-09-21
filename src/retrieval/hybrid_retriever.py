"""Hybrid retriever coordinating dual dense vector search and sparse BM25 search."""

from pathlib import Path
import time
from typing import Any
from src.config.settings import get_settings
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embeddings import EmbeddingService, get_embedding_service
from src.indexing.models import SearchResult
from src.indexing.vector_store.base import BaseVectorStore
from src.indexing.vector_store.factory import get_vector_store
from src.retrieval.models import FusedResult, RetrievalMetrics
from src.retrieval.rrf import compute_rrf


class HybridRetriever:
    """Coordinates parallel retrieval over dense and sparse indices, merging via RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        bm25_indexer: BM25Indexer | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()

        if bm25_indexer is not None:
            self.bm25_indexer = bm25_indexer
        else:
            self.bm25_indexer = BM25Indexer()
            # Attempt to load cached index from disk if available
            self.bm25_indexer.load(verify_source_file=settings.CHUNKS_FILE)

        self.default_top_n = settings.RETRIEVAL_TOP_N
        self.default_k = settings.RRF_K

    def retrieve_dense(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Perform dense semantic vector retrieval only.

        Args:
            query: User search query string.
            top_k: Number of candidates to retrieve.

        Returns:
            Tuple of (search_results, latency_ms).
        """
        start = time.perf_counter()
        k = top_k or self.default_top_n
        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.search(query_vector=query_vector, top_k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, round(latency_ms, 2)

    def retrieve_sparse(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Perform sparse lexical BM25 retrieval only.

        Args:
            query: User search query string.
            top_k: Number of candidates to retrieve.

        Returns:
            Tuple of (search_results, latency_ms).
        """
        start = time.perf_counter()
        k = top_k or self.default_top_n
        results = self.bm25_indexer.search(query=query, top_k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, round(latency_ms, 2)

    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        k: int | None = None,
    ) -> tuple[list[FusedResult], RetrievalMetrics]:
        """Execute full hybrid retrieval pipeline: Dense + Sparse fused via RRF.

        Args:
            query: User query string.
            top_n: Number of candidates to retrieve from each system and final fused list.
            k: RRF smoothing constant (default: settings.RRF_K = 60).

        Returns:
            Tuple of (fused_results, retrieval_metrics).
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        target_depth = top_n or self.default_top_n
        rrf_k = k or self.default_k

        start_total = time.perf_counter()

        # 1. Dense Semantic Retrieval
        dense_results, dense_ms = self.retrieve_dense(query, top_k=target_depth)

        # 2. Sparse Lexical BM25 Retrieval
        sparse_results, sparse_ms = self.retrieve_sparse(query, top_k=target_depth)

        # 3. Reciprocal Rank Fusion
        start_rrf = time.perf_counter()
        fused_candidates = compute_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=rrf_k,
            top_n=target_depth,
        )
        rrf_ms = (time.perf_counter() - start_rrf) * 1000

        total_ms = (time.perf_counter() - start_total) * 1000

        metrics = RetrievalMetrics(
            dense_latency_ms=dense_ms,
            sparse_latency_ms=sparse_ms,
            rrf_latency_ms=round(rrf_ms, 2),
            total_retrieval_ms=round(total_ms, 2),
            dense_candidates_count=len(dense_results),
            sparse_candidates_count=len(sparse_results),
            fused_candidates_count=len(fused_candidates),
        )

        return fused_candidates, metrics
