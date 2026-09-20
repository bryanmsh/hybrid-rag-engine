"""Data models for document ingestion and chunk representation."""

from datetime import datetime, timezone
import hashlib
from typing import Any
import uuid
from pydantic import BaseModel, Field


# Deterministic namespace for chunk UUID generation
CHUNK_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Generate a deterministic UUID5 for a chunk based on its document provenance and content hash."""
    content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    unique_key = f"{document_id}:{chunk_index}:{content_digest}"
    return str(uuid.uuid5(CHUNK_NAMESPACE, unique_key))


class Document(BaseModel):
    """Raw parsed document with page or section granularity."""

    content: str
    document_id: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Metadata detailing file provenance and timestamp."""

    file_path: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Normalized chunk schema matching PRD Section 4.1."""

    chunk_id: str
    document_id: str
    page_number: int | None = None
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document_part(
        cls,
        document: Document,
        chunk_index: int,
        content: str,
        file_path: str,
        created_at: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        """Factory method to construct a validated chunk with deterministic UUID5."""
        chunk_id = generate_chunk_id(document.document_id, chunk_index, content)
        meta = {
            "file_path": file_path,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            **(document.metadata or {}),
            **(extra_metadata or {}),
        }
        return cls(
            chunk_id=chunk_id,
            document_id=document.document_id,
            page_number=document.page_number,
            chunk_index=chunk_index,
            content=content.strip(),
            metadata=meta,
        )
