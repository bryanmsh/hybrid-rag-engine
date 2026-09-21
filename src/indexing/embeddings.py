"""Dense embedding service using sentence-transformers."""

from typing import Any
import numpy as np
from src.config.settings import get_settings
from src.ingestion.models import Chunk


class EmbeddingService:
    """Manages dense semantic embedding generation via SentenceTransformer."""

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-loaded SentenceTransformer instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Compute normalized dense embeddings for a list of strings in batches.

        Args:
            texts: List of input strings.
            batch_size: Override default batch size.
            normalize: Whether to L2-normalize vectors (so cosine similarity = dot product).
            show_progress_bar: Whether to display tqdm progress bar.

        Returns:
            List of float vectors.
        """
        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=effective_batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        # Validate dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Extract text content from Chunk models and generate dense embeddings."""
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress_bar,
        )

    def embed_query(
        self,
        query: str,
        normalize: bool = True,
    ) -> list[float]:
        """Generate a single normalized dense embedding for a search query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        vectors = self.embed_texts(
            texts=[query],
            batch_size=1,
            normalize=normalize,
            show_progress_bar=False,
        )
        return vectors[0]


_embedding_service_instance: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Retrieve singleton instance of EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
