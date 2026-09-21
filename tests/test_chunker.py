"""Unit tests for RecursiveBoundaryChunker and chunk models."""

import pytest
from src.ingestion.chunker import RecursiveBoundaryChunker
from src.ingestion.models import Chunk, Document, generate_chunk_id


def test_deterministic_chunk_id_consistency():
    """Verify that identical inputs produce identical UUID5s across runs."""
    doc_id = "doc_test_123.md"
    chunk_index = 0
    content = "Deterministic content test for UUID5 hashing."

    id1 = generate_chunk_id(doc_id, chunk_index, content)
    id2 = generate_chunk_id(doc_id, chunk_index, content)
    assert id1 == id2
    assert len(id1) == 36  # Standard UUID length


def test_deterministic_chunk_id_uniqueness():
    """Verify that varying index or content alters the UUID5."""
    doc_id = "doc_test_123.md"
    id1 = generate_chunk_id(doc_id, 0, "Chunk zero content")
    id2 = generate_chunk_id(doc_id, 1, "Chunk zero content")
    id3 = generate_chunk_id(doc_id, 0, "Chunk modified content")

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_chunker_small_text_no_split():
    """Short text below chunk_size should not be fragmented."""
    chunker = RecursiveBoundaryChunker(chunk_size=50, chunk_overlap=10)
    text = "Short single paragraph within limits."
    chunks = chunker.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_paragraph_boundary_split():
    """Text exceeding chunk_size should prioritize splitting on paragraph breaks."""
    chunker = RecursiveBoundaryChunker(chunk_size=20, chunk_overlap=5)
    para1 = "This is paragraph one with enough words to fill the size budget."
    para2 = "This is paragraph two that should cleanly split into its own piece."
    text = f"{para1}\n\n{para2}"

    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    assert any("paragraph one" in c for c in chunks)
    assert any("paragraph two" in c for c in chunks)


def test_chunker_overlap_retention():
    """Consecutive chunks must retain overlapping context from preceding chunks."""
    chunker = RecursiveBoundaryChunker(chunk_size=15, chunk_overlap=5)
    text = (
        "Alpha beta gamma delta. "
        "Epsilon zeta eta theta. "
        "Iota kappa lambda mu. "
        "Nu xi omicron pi."
    )
    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    # Check that adjacent chunks have words in common
    for i in range(len(chunks) - 1):
        words_first = set(chunks[i].split())
        words_second = set(chunks[i + 1].split())
        overlap = words_first.intersection(words_second)
        assert len(overlap) > 0, f"Expected overlap between chunk {i} and {i+1}"


def test_chunk_document_provenance():
    """Verify that chunk_document correctly binds Document metadata and indexes."""
    chunker = RecursiveBoundaryChunker(chunk_size=10, chunk_overlap=2)
    doc = Document(
        content="First part of document.\n\nSecond part of document with more details.",
        document_id="spec_v1.pdf",
        page_number=3,
        metadata={"author": "Staff Engineer"},
    )

    chunks, next_idx = chunker.chunk_document(
        document=doc,
        file_path="/path/to/spec_v1.pdf",
        start_index=0,
    )

    assert len(chunks) >= 2
    assert next_idx == len(chunks)
    assert chunks[0].document_id == "spec_v1.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata["author"] == "Staff Engineer"
    assert chunks[0].metadata["file_path"] == "/path/to/spec_v1.pdf"
