"""Cross-encoder reranking implementation using joint attention."""

import time
from typing import Union
import numpy as np
from src.config.settings import get_settings
from src.indexing.models import SearchResult
from src.reranking.base import BaseReranker, CandidateType
from src.reranking.models import RerankMetrics, RerankResult
from src.retrieval.models import FusedResult


class CrossEncoderReranker(BaseReranker):
    """Deep cross-attention reranker scoring (query, candidate) pairs jointly."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.default_top_k = settings.RERANK_TOP_K
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-loaded CrossEncoder instance."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[CandidateType],
        top_k: int | None = None,
    ) -> tuple[list[RerankResult], RerankMetrics]:
        """Score candidate passages using joint query-passage attention.

        Args:
            query: Input user query.
            candidates: Candidate passages from RRF or vector/BM25 retrieval.
            top_k: Number of highest-ranked passages to retain (default: 4).

        Returns:
            Tuple of (reranked_results, rerank_metrics).
        """
        k = top_k or self.default_top_k

        if not candidates:
            return [], RerankMetrics(
                rerank_latency_ms=0.0,
                candidate_count=0,
                output_count=0,
            )

        start_time = time.perf_counter()

        # Build cross-attention pairs: (query, passage)
        pairs = [(query, cand.content) for cand in candidates]

        # Predict cross-encoder logits
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Handle scalar output for single candidate
        if isinstance(scores, (float, np.floating)):
            scores = np.array([scores])

        # Pair candidates with their cross-encoder scores and incoming ranks
        scored_candidates: list[dict] = []
        for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
            pre_rank = getattr(cand, "fused_rank", getattr(cand, "rank", idx))
            scored_candidates.append(
                {
                    "candidate": cand,
                    "score": float(score),
                    "pre_rank": pre_rank,
                }
            )

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Truncate to top_k and compute rank deltas
        results: list[RerankResult] = []
        for new_rank, item in enumerate(scored_candidates[:k], start=1):
            cand = item["candidate"]
            score = item["score"]
            pre_rank = item["pre_rank"]
            rank_delta = pre_rank - new_rank  # Positive = promoted, Negative = demoted

            results.append(
                RerankResult(
                    chunk_id=cand.chunk_id,
                    content=cand.content,
                    rerank_score=round(score, 6),
                    rerank_rank=new_rank,
                    pre_rerank_rank=pre_rank,
                    rank_delta=rank_delta,
                    metadata=dict(cand.metadata),
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metrics = RerankMetrics(
            rerank_latency_ms=round(elapsed_ms, 2),
            candidate_count=len(candidates),
            output_count=len(results),
        )

        return results, metrics
