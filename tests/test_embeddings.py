"""Unit tests for dense EmbeddingService using sentence-transformers."""

import numpy as np
import pytest
from src.indexing.embeddings import EmbeddingService
from src.ingestion.models import Chunk


def test_embedding_dimension_and_normalization():
    """Verify that embeddings are 384-dimensional and unit-normalized."""
    service = EmbeddingService()
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="doc1.txt",
            page_number=1,
            chunk_index=0,
            content="Testing semantic representation in vector space.",
            metadata={},
        )
    ]

    vectors = service.embed_chunks(chunks)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384

    # Check L2 unit norm: ||v||_2 ≈ 1.0
    vec_np = np.array(vectors[0])
    norm = np.linalg.norm(vec_np)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_query():
    """Verify query vector generation."""
    service = EmbeddingService()
    vec = service.embed_query("Sample query for nearest neighbors")
    assert len(vec) == 384
    norm = np.linalg.norm(np.array(vec))
    assert pytest.approx(norm, rel=1e-3) == 1.0
