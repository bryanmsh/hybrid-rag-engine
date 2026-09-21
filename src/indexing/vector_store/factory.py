"""Factory method for instantiating vector store providers."""

from typing import Any
from src.config.settings import get_settings
from src.indexing.vector_store.base import BaseVectorStore
from src.indexing.vector_store.chroma_store import ChromaVectorStore
from src.indexing.vector_store.qdrant_store import QdrantVectorStore


def get_vector_store(
    provider: str | None = None,
    **kwargs: Any,
) -> BaseVectorStore:
    """Instantiate and return the configured BaseVectorStore provider.

    Args:
        provider: "chroma" or "qdrant" (defaults to settings.VECTOR_STORE_PROVIDER).
        **kwargs: Additional parameters passed to the store constructor.

    Returns:
        Concrete instance of BaseVectorStore.
    """
    settings = get_settings()
    store_provider = (provider or settings.VECTOR_STORE_PROVIDER).lower()

    if store_provider == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_provider == "qdrant":
        return QdrantVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unsupported vector store provider '{store_provider}'. Choose 'chroma' or 'qdrant'."
        )
