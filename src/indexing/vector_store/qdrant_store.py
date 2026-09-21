"""Qdrant implementation of BaseVectorStore for distributed or isolated vector storage."""

from typing import Any
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from src.config.settings import get_settings
from src.indexing.models import SearchResult
from src.indexing.vector_store.base import BaseVectorStore
from src.ingestion.models import Chunk


class QdrantVectorStore(BaseVectorStore):
    """Qdrant client adapter implementing BaseVectorStore."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        location: str | None = None,
    ):
        settings = get_settings()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model_name = settings.EMBEDDING_MODEL_NAME

        if location:
            # Allows in-memory or local path for testing/embedded use
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(host=self.host, port=self.port)

        self._init_collection()

    def _init_collection(self) -> None:
        """Create Qdrant collection if not already existing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunk vectors into Qdrant collection."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but got {len(embeddings)} embeddings."
            )

        points: list[PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            # Qdrant accepts UUID or uint as point id
            # Our chunk_id is already a valid UUID5 string
            point_id = str(uuid.UUID(chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Perform cosine similarity vector search in Qdrant."""
        if self.count() == 0:
            return []

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[SearchResult] = []
        for rank, point in enumerate(search_result, start=1):
            payload = point.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    content=payload.get("content", ""),
                    score=round(float(point.score), 6),
                    rank=rank,
                    metadata=payload.get("metadata", {}),
                    retrieval_type="dense",
                )
            )

        return results

    def count(self) -> int:
        """Return point count in collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count or 0

    def delete_collection(self) -> None:
        """Drop collection from Qdrant."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._init_collection()

    def get_metadata(self) -> dict[str, Any]:
        """Return collection config details."""
        return {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "model_name": self.model_name,
        }
