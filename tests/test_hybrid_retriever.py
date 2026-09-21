"""Unit and integration tests for HybridRetriever."""

from pathlib import Path
import pytest
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embeddings import EmbeddingService
from src.indexing.vector_store.chroma_store import ChromaVectorStore
from src.ingestion.models import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def populated_retriever(tmp_path: Path):
    """Fixture providing a retriever populated with known test documents."""
    chroma_dir = tmp_path / "chroma_test"
    vector_store = ChromaVectorStore(persist_dir=chroma_dir, collection_name="test_hybrid")
    bm25 = BM25Indexer()
    embedder = EmbeddingService()

    chunks = [
        Chunk(
            chunk_id="chunk-k8s",
            document_id="k8s.md",
            page_number=1,
            chunk_index=0,
            content="Kubernetes orchestrates containerized workloads and handles pod autoscaling.",
            metadata={"category": "devops"},
        ),
        Chunk(
            chunk_id="chunk-db",
            document_id="db.txt",
            page_number=2,
            chunk_index=0,
            content="PostgreSQL executes relational SQL queries and manages B-Tree index trees.",
            metadata={"category": "database"},
        ),
        Chunk(
            chunk_id="chunk-ml",
            document_id="ml.txt",
            page_number=1,
            chunk_index=0,
            content="Transformer attention mechanisms compute weighted representations across tokens.",
            metadata={"category": "ai"},
        ),
    ]

    embeddings = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
    bm25.build(chunks=chunks)

    return HybridRetriever(
        vector_store=vector_store,
        bm25_indexer=bm25,
        embedding_service=embedder,
    )


def test_hybrid_retrieve_returns_fused_candidates(populated_retriever: HybridRetriever):
    """Hybrid retrieval must merge dense and sparse matches and calculate metrics."""
    fused, metrics = populated_retriever.retrieve(
        query="Kubernetes pod autoscaling",
        top_n=2,
    )

    assert len(fused) <= 2
    assert len(fused) > 0

    # Top candidate should be the k8s chunk
    assert fused[0].chunk_id == "chunk-k8s"
    assert fused[0].fused_rank == 1
    assert fused[0].rrf_score > 0

    # Verify metrics
    assert metrics.dense_candidates_count > 0
    assert metrics.sparse_candidates_count > 0
    assert metrics.fused_candidates_count == len(fused)
    assert metrics.total_retrieval_ms >= 0


def test_retrieve_dense_only(populated_retriever: HybridRetriever):
    """retrieve_dense must return dense-only results and latency."""
    results, latency = populated_retriever.retrieve_dense("SQL database queries", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-db"
    assert results[0].retrieval_type == "dense"
    assert latency >= 0


def test_retrieve_sparse_only(populated_retriever: HybridRetriever):
    """retrieve_sparse must return sparse-only BM25 results."""
    results, latency = populated_retriever.retrieve_sparse("Transformer attention", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-ml"
    assert results[0].retrieval_type == "sparse"
    assert latency >= 0
