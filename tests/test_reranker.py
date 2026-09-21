"""Unit and pipeline integration tests for CrossEncoderReranker and SearchPipeline."""

from pathlib import Path
import pytest
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embeddings import EmbeddingService
from src.indexing.models import SearchResult
from src.indexing.vector_store.chroma_store import ChromaVectorStore
from src.ingestion.models import Chunk
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import FusedResult
from src.retrieval.pipeline import SearchPipeline


def test_cross_encoder_rerank_candidates():
    """Verify that CrossEncoder scores and reorders candidates with rank delta."""
    reranker = CrossEncoderReranker()

    # Passages with deliberate varying degrees of relevance
    candidates = [
        FusedResult(
            chunk_id="irrelevant",
            content="Baking chocolate chip cookies requires flour, sugar, and baking powder.",
            rrf_score=0.03,
            fused_rank=1,  # Hypothetical candidate that was ranked 1 before
        ),
        FusedResult(
            chunk_id="relevant",
            content="Reciprocal Rank Fusion is an information retrieval technique that merges ranked lists.",
            rrf_score=0.02,
            fused_rank=2,  # Candidate that was ranked 2 before
        ),
    ]

    query = "What is Reciprocal Rank Fusion used for?"
    reranked, metrics = reranker.rerank(query=query, candidates=candidates, top_k=2)

    assert len(reranked) == 2
    # The relevant passage must be promoted to rank 1!
    assert reranked[0].chunk_id == "relevant"
    assert reranked[0].rerank_rank == 1
    assert reranked[0].pre_rerank_rank == 2
    assert reranked[0].rank_delta == +1  # Promoted from 2 to 1 (2 - 1 = +1)

    # The cookie passage must be demoted to rank 2
    assert reranked[1].chunk_id == "irrelevant"
    assert reranked[1].rerank_rank == 2
    assert reranked[1].pre_rerank_rank == 1
    assert reranked[1].rank_delta == -1  # Demoted from 1 to 2 (1 - 2 = -1)

    assert metrics.candidate_count == 2
    assert metrics.output_count == 2
    assert metrics.rerank_latency_ms >= 0


def test_search_pipeline_end_to_end(tmp_path: Path):
    """Verify full end-to-end pipeline execution from query to reranked passages."""
    chroma_dir = tmp_path / "pipeline_chroma"
    vector_store = ChromaVectorStore(persist_dir=chroma_dir, collection_name="test_pipe")
    bm25 = BM25Indexer()
    embedder = EmbeddingService()

    chunks = [
        Chunk(
            chunk_id="p1",
            document_id="doc1.txt",
            page_number=1,
            chunk_index=0,
            content="Cross-encoders perform joint attention across both query and document tokens.",
            metadata={"source": "nlp"},
        ),
        Chunk(
            chunk_id="p2",
            document_id="doc2.txt",
            page_number=1,
            chunk_index=0,
            content="Python garbage collection relies primarily on reference counting and cycle detection.",
            metadata={"source": "runtime"},
        ),
    ]

    embeddings = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
    bm25.build(chunks=chunks)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_indexer=bm25,
        embedding_service=embedder,
    )
    reranker = CrossEncoderReranker()
    pipeline = SearchPipeline(retriever=retriever, reranker=reranker)

    output = pipeline.search(
        query="How does cross-encoder joint attention work?",
        top_k=1,
        use_hybrid=True,
        use_reranker=True,
    )

    assert output["query"] == "How does cross-encoder joint attention work?"
    assert output["final_results_count"] == 1
    assert output["results"][0]["chunk_id"] == "p1"

    # Check that latency telemetry dictionary contains all expected keys
    latencies = output["latencies"]
    for k in ["dense_retrieval_ms", "sparse_retrieval_ms", "rrf_fusion_ms", "rerank_ms", "total_pipeline_ms"]:
        assert k in latencies
        assert latencies[k] >= 0
