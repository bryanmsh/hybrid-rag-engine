"""Data models for hybrid retrieval, rank fusion, and retrieval telemetry."""

from typing import Any
from pydantic import BaseModel, Field


class FusedResult(BaseModel):
    """Represents a candidate passage fused from multiple retrieval systems via RRF."""

    chunk_id: str
    content: str
    rrf_score: float
    fused_rank: int = 1
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fused result to dictionary."""
        return self.model_dump()


class RetrievalMetrics(BaseModel):
    """Execution latency telemetry for the retrieval stage."""

    dense_latency_ms: float = 0.0
    sparse_latency_ms: float = 0.0
    rrf_latency_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    fused_candidates_count: int = 0
