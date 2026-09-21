"""Abstract base class interface for dense vector stores."""

from abc import ABC, abstractmethod
from typing import Any
from src.indexing.models import SearchResult
from src.ingestion.models import Chunk


class BaseVectorStore(ABC):
    """Abstract contract for vector database adapters (ChromaDB, Qdrant, etc.)."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or upsert chunks along with their dense vector embeddings.

        Args:
            chunks: List of Chunk metadata and content objects.
            embeddings: Parallel list of dense float vectors.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector similarity search.

        Args:
            query_vector: Dense query embedding.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of SearchResult objects ordered by descending similarity score.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors indexed in the collection."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Purge all vectors and collection data."""
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return collection metadata, including embedding model name and dimensions."""
        pass
