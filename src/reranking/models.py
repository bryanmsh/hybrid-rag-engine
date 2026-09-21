"""Data models for cross-encoder reranking and latency telemetry."""

from typing import Any
from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Represents a candidate passage evaluated and scored by a Cross-Encoder."""

    chunk_id: str
    content: str
    rerank_score: float
    rerank_rank: int = 1
    pre_rerank_rank: int = 1
    rank_delta: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize rerank result to dictionary."""
        return self.model_dump()


class RerankMetrics(BaseModel):
    """Execution telemetry for the cross-encoder reranking stage."""

    rerank_latency_ms: float = 0.0
    candidate_count: int = 0
    output_count: int = 0
