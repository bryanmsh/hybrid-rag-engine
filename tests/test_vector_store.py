"""Unit tests for ChromaVectorStore and QdrantVectorStore implementations."""

from pathlib import Path
import pytest
from src.indexing.vector_store.chroma_store import ChromaVectorStore
from src.indexing.vector_store.qdrant_store import QdrantVectorStore
from src.ingestion.models import Chunk


def test_chroma_vector_store_crud(tmp_path: Path):
    """Verify ChromaVectorStore insertion, search, and count."""
    persist_dir = tmp_path / "test_chroma_db"
    store = ChromaVectorStore(persist_dir=persist_dir, collection_name="test_collection")

    chunks = [
        Chunk(
            chunk_id="chunk-alpha",
            document_id="doc1.txt",
            page_number=1,
            chunk_index=0,
            content="FastAPI asynchronous request processing pipeline.",
            metadata={"env": "prod"},
        ),
        Chunk(
            chunk_id="chunk-beta",
            document_id="doc2.txt",
            page_number=2,
            chunk_index=1,
            content="Deep residual learning for image recognition models.",
            metadata={"env": "dev"},
        ),
    ]

    # Two 384-dimensional synthetic orthogonal unit vectors for deterministic testing
    vec_a = [1.0] + [0.0] * 383
    vec_b = [0.0, 1.0] + [0.0] * 382

    store.add_chunks(chunks=chunks, embeddings=[vec_a, vec_b])
    assert store.count() == 2

    # Query closest to vec_a
    results = store.search(query_vector=vec_a, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-alpha"
    assert results[0].retrieval_type == "dense"
    assert results[0].metadata["document_id"] == "doc1.txt"
    assert results[0].metadata["env"] == "prod"
    assert results[0].score >= 0.99


def test_qdrant_vector_store_in_memory():
    """Verify QdrantVectorStore with an in-memory client."""
    store = QdrantVectorStore(location=":memory:", collection_name="test_qdrant")

    chunks = [
        Chunk(
            chunk_id="550e8400-e29b-41d4-a716-446655440000",
            document_id="spec.md",
            page_number=1,
            chunk_index=0,
            content="Reciprocal rank fusion scoring with k=60.",
            metadata={"domain": "search"},
        )
    ]

    vec = [1.0] + [0.0] * 383
    store.add_chunks(chunks=chunks, embeddings=[vec])
    assert store.count() == 1

    results = store.search(query_vector=vec, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "550e8400-e29b-41d4-a716-446655440000"
    assert results[0].content == "Reciprocal rank fusion scoring with k=60."
    assert results[0].score >= 0.99
