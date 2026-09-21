"""CLI entrypoint and orchestrator to build dense and sparse indices from chunks.jsonl."""

import argparse
from pathlib import Path
import sys
import time
from typing import Any
from src.config.settings import get_settings
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embeddings import get_embedding_service
from src.indexing.vector_store.factory import get_vector_store
from src.ingestion.pipeline import IngestionPipeline


def build_indices(
    chunks_file: Path | str | None = None,
    provider: str | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Load chunks, compute dense embeddings, and build both vector store and BM25 indices.

    Args:
        chunks_file: Path to chunks.jsonl file.
        provider: Vector store provider ('chroma' or 'qdrant').
        force_rebuild: Whether to clear existing indices before building.

    Returns:
        Summary dict containing counts and timings.
    """
    settings = get_settings()
    file_path = Path(chunks_file or settings.CHUNKS_FILE)

    if not file_path.exists():
        print(f"[Warning] Chunks file '{file_path}' not found. Running ingestion first...")
        pipeline = IngestionPipeline()
        pipeline.run()

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find or generate chunks file at {file_path}")

    chunks = IngestionPipeline.load_chunks_from_file(file_path)
    if not chunks:
        print("[Warning] No chunks found in chunks.jsonl. Nothing to index.")
        return {
            "status": "empty",
            "chunks_count": 0,
            "elapsed_ms": 0.0,
        }

    start_total = time.perf_counter()

    # 1. Dense Vector Index
    print(f"\n🧠 Generating Dense Embeddings for {len(chunks)} chunks...")
    start_dense = time.perf_counter()
    embedder = get_embedding_service()
    embeddings = embedder.embed_chunks(chunks, show_progress_bar=True)
    dense_embed_ms = (time.perf_counter() - start_dense) * 1000

    vector_store = get_vector_store(provider=provider)
    if force_rebuild:
        print("  - Clearing existing vector collection...")
        vector_store.delete_collection()

    print(f"  - Inserting into {provider or settings.VECTOR_STORE_PROVIDER} vector store...")
    start_store = time.perf_counter()
    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
    dense_store_ms = (time.perf_counter() - start_store) * 1000

    # 2. Sparse Lexical BM25 Index
    print(f"\n🔍 Building Sparse BM25 Index for {len(chunks)} chunks...")
    start_bm25 = time.perf_counter()
    bm25_indexer = BM25Indexer()
    bm25_indexer.build(chunks=chunks, source_file=file_path)
    bm25_indexer.save()
    bm25_ms = (time.perf_counter() - start_bm25) * 1000

    total_ms = (time.perf_counter() - start_total) * 1000

    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "vector_store_count": vector_store.count(),
        "bm25_count": bm25_indexer.count(),
        "bm25_index_file": str(settings.BM25_INDEX_FILE),
        "dense_embed_ms": round(dense_embed_ms, 2),
        "dense_store_ms": round(dense_store_ms, 2),
        "bm25_build_ms": round(bm25_ms, 2),
        "total_elapsed_ms": round(total_ms, 2),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build dense and sparse indices for Hybrid RAG.")
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=None,
        help="Path to chunks.jsonl (defaults to data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["chroma", "qdrant"],
        help="Vector store provider (defaults to settings.VECTOR_STORE_PROVIDER)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Drop existing collections and rebuild from scratch",
    )

    args = parser.parse_args()

    print("==================================================")
    print("🏗️  Building Hybrid RAG Indices (Dense + Sparse)")
    print(f"Provider: {args.provider or 'default (chroma)'}")
    print(f"Force Rebuild: {args.force_rebuild}")
    print("==================================================")

    result = build_indices(
        chunks_file=args.chunks_file,
        provider=args.provider,
        force_rebuild=args.force_rebuild,
    )

    print("\n✅ Indexing Completed Successfully!")
    print(f"Chunks Indexed     : {result['chunks_indexed']}")
    print(f"Vector Store Count : {result['vector_store_count']}")
    print(f"BM25 Store Count   : {result['bm25_count']}")
    print(f"Dense Embedding    : {result['dense_embed_ms']} ms")
    print(f"Dense Insertion    : {result['dense_store_ms']} ms")
    print(f"BM25 Index & Save  : {result['bm25_build_ms']} ms")
    print(f"Total Time         : {result['total_elapsed_ms']} ms")
    print("==================================================")


if __name__ == "__main__":
    main()
