"""Application configuration and settings."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Hybrid RAG Engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    CHUNKS_FILE: Path = DATA_PROCESSED_DIR / "chunks.jsonl"
    BM25_INDEX_FILE: Path = DATA_PROCESSED_DIR / "bm25_index.pkl"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    EVAL_DATASET_FILE: Path = BASE_DIR / "eval" / "eval_dataset.jsonl"
    EVAL_RESULTS_DIR: Path = BASE_DIR / "eval" / "results"

    # Ingestion & Chunking
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64

    # Embedding & Indexing
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 64
    VECTOR_STORE_PROVIDER: Literal["chroma", "qdrant"] = "chroma"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag_documents"

    # Retrieval & RRF
    RETRIEVAL_TOP_N: int = 20
    RRF_K: int = 60

    # Cross-Encoder Reranking
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 4

    # LLM & Generation
    LLM_PROVIDER: Literal["openai", "anthropic", "ollama", "mock"] = "mock"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Retrieve singleton instance of Settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
