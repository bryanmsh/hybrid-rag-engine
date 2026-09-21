"""ChromaDB implementation of BaseVectorStore."""

from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config.settings import get_settings
from src.indexing.models import SearchResult
from src.indexing.vector_store.base import BaseVectorStore
from src.ingestion.models import Chunk


class ChromaVectorStore(BaseVectorStore):
    """Local embedded vector store using ChromaDB with persistent HNSW index."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "rag_documents",
    ):
        settings = get_settings()
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._init_collection()

    def _init_collection(self) -> None:
        """Create or get Chroma collection with cosine space and model metadata."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.model_name,
                "embedding_dim": self.dimension,
            },
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any], doc_id: str, page_num: int, chunk_idx: int) -> dict[str, Any]:
        """Ensure all metadata fields are primitive types acceptable by ChromaDB."""
        sanitized: dict[str, str | int | float | bool] = {
            "document_id": doc_id,
            "page_number": page_num,
            "chunk_index": chunk_idx,
        }
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks and embeddings into ChromaDB collection."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but got {len(embeddings)} embeddings."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            self._sanitize_metadata(
                metadata=chunk.metadata,
                doc_id=chunk.document_id,
                page_num=chunk.page_number,
                chunk_idx=chunk.chunk_index,
            )
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector similarity search in ChromaDB.

        Args:
            query_vector: Normalized 384-dimensional query vector.
            top_k: Max candidate count to retrieve.

        Returns:
            Sorted list of SearchResult with cosine similarity scores.
        """
        if self.count() == 0:
            return []

        # Bound top_k to existing collection count
        effective_k = min(top_k, self.count())
        if effective_k <= 0:
            return []

        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for rank, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(ids, docs, metadatas, distances), start=1
        ):
            # For cosine space, Chroma distance = 1 - cosine_similarity
            # Score bounded to [0.0, 1.0]
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=doc_text,
                    score=round(similarity_score, 6),
                    rank=rank,
                    metadata=meta or {},
                    retrieval_type="dense",
                )
            )

        return results

    def count(self) -> int:
        """Count total vectors in collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete collection and reset."""
        self.client.delete_collection(name=self.collection_name)
        self._init_collection()

    def get_metadata(self) -> dict[str, Any]:
        """Return collection metadata."""
        return self.collection.metadata or {}
