"""Abstract base class interface for candidate rerankers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union
from src.indexing.models import SearchResult
from src.reranking.models import RerankMetrics, RerankResult

if TYPE_CHECKING:
    from src.retrieval.models import FusedResult
    CandidateType = Union[SearchResult, FusedResult]
else:
    CandidateType = Any


class BaseReranker(ABC):
    """Abstract contract for reranking candidate passages using cross-attention models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[CandidateType],
        top_k: int = 4,
    ) -> tuple[list[RerankResult], RerankMetrics]:
        """Score candidate passages against the query using cross-attention.

        Args:
            query: User search query string.
            candidates: List of SearchResult or FusedResult candidate passages.
            top_k: Number of highest-scoring passages to return.

        Returns:
            Tuple of (reranked_results, rerank_metrics).
        """
        pass
