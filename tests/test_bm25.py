"""Unit tests for BM25Indexer, tokenization, persistence, and checksum validation."""

from pathlib import Path
import pytest
from src.indexing.bm25_indexer import BM25Indexer
from src.ingestion.models import Chunk


def test_bm25_tokenization():
    """Verify alphanumeric tokenization and lowercase normalization."""
    raw = "RRF is an algorithm with k=60 and UUID5-based IDs!"
    tokens = BM25Indexer.tokenize(raw)
    assert "rrf" in tokens
    assert "algorithm" in tokens
    assert "60" in tokens
    assert "uuid5" in tokens
    assert "ids" in tokens


def test_bm25_search_scoring():
    """Exact keyword queries must rank the matching chunk first."""
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id="doc1.md",
            page_number=1,
            chunk_index=0,
            content="Kubernetes pod autoscaling using Prometheus metrics.",
            metadata={"source": "k8s"},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id="doc2.md",
            page_number=1,
            chunk_index=0,
            content="Relational database query optimization with PostgreSQL indexes.",
            metadata={"source": "postgres"},
        ),
    ]

    indexer = BM25Indexer()
    indexer.build(chunks)

    results = indexer.search("PostgreSQL indexes", top_k=2)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-2"
    assert results[0].retrieval_type == "sparse"
    assert results[0].metadata["source"] == "postgres"


def test_bm25_save_and_load(tmp_path: Path):
    """BM25 index should serialize to disk and restore identically."""
    cache_file = tmp_path / "bm25_test.pkl"
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="doc.txt",
            page_number=1,
            chunk_index=0,
            content="Cross-encoder joint self-attention mechanism.",
            metadata={},
        )
    ]

    indexer1 = BM25Indexer(index_file=cache_file)
    indexer1.build(chunks)
    indexer1.save()

    assert cache_file.exists()

    indexer2 = BM25Indexer(index_file=cache_file)
    loaded = indexer2.load()
    assert loaded is True
    assert indexer2.count() == 1

    results = indexer2.search("self-attention")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_bm25_checksum_invalidation(tmp_path: Path):
    """If source chunks file is modified, index load must detect checksum invalidation."""
    source_file = tmp_path / "chunks.jsonl"
    source_file.write_text('{"chunk_id": "c1", "content": "Original"}', encoding="utf-8")

    cache_file = tmp_path / "bm25_cache.pkl"
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="doc.txt",
            page_number=1,
            chunk_index=0,
            content="Original content",
            metadata={},
        )
    ]

    indexer = BM25Indexer(index_file=cache_file)
    indexer.build(chunks, source_file=source_file)
    indexer.save()

    # Verify initial valid load
    assert indexer.load(verify_source_file=source_file) is True

    # Mutate source file
    source_file.write_text('{"chunk_id": "c1", "content": "Modified!"}', encoding="utf-8")

    # Invalidation check must fail
    assert indexer.load(verify_source_file=source_file) is False
