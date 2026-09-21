"""Data models for indexing and retrieval operations."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Normalized search result returned by any retrieval system (dense, sparse, hybrid)."""

    chunk_id: str
    content: str
    score: float
    rank: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_type: Literal["dense", "sparse", "hybrid"] = "dense"

    def to_dict(self) -> dict[str, Any]:
        """Serialize search result to dictionary."""
        return self.model_dump()
