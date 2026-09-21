"""Sparse lexical indexing and retrieval engine using BM25Okapi."""

import hashlib
import math
from pathlib import Path
import pickle
import re
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
from src.config.settings import get_settings
from src.indexing.models import SearchResult
from src.ingestion.models import Chunk


class BM25Indexer:
    """Manages sparse lexical BM25 index construction, caching, and querying."""

    def __init__(self, index_file: Path | str | None = None):
        settings = get_settings()
        self.index_file = Path(index_file or settings.BM25_INDEX_FILE)
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self.corpus_tokens: list[list[str]] = []
        self.source_checksum: str | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Extract alphanumeric tokens in lowercase, preserving numbers and words.

        Args:
            text: Input string.

        Returns:
            List of normalized string tokens.
        """
        if not text:
            return []
        # Matches alphanumeric tokens including underscores and numbers (e.g. "uuid5", "bm25", "top_k")
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def compute_checksum(file_path: Path | str) -> str:
        """Calculate SHA-256 hash of a file for index invalidation checks."""
        path = Path(file_path)
        if not path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def build(
        self,
        chunks: list[Chunk],
        source_file: Path | str | None = None,
    ) -> None:
        """Build BM25Okapi index from a list of Chunk models.

        Args:
            chunks: List of Chunk objects.
            source_file: Optional path to chunks.jsonl to record source checksum.
        """
        if not chunks:
            self.bm25 = None
            self.chunks = []
            self.corpus_tokens = []
            return

        self.chunks = chunks
        self.corpus_tokens = [self.tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Standard Robertson + 1 (Lucene) formula to guarantee strictly positive IDF:
        # IDF(w) = ln(1 + (N - n + 0.5) / (n + 0.5))
        # rank_bm25 defaults to the 1994 Robertson formula without +1, which produces 0.0 or negative
        # IDFs whenever a term appears in >= 50% of documents or when corpus size <= 2.
        doc_freqs: dict[str, int] = {}
        for doc in self.corpus_tokens:
            for token in set(doc):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        n_docs = len(self.corpus_tokens)
        for word, freq in doc_freqs.items():
            self.bm25.idf[word] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        if source_file:
            self.source_checksum = self.compute_checksum(source_file)
        else:
            self.source_checksum = None

    def save(self, path: Path | str | None = None) -> None:
        """Serialize BM25 index and chunk references to disk.

        Args:
            path: Destination pickle path. Defaults to settings.BM25_INDEX_FILE.
        """
        if self.bm25 is None or not self.chunks:
            raise ValueError("Cannot save an empty BM25 index. Call build() first.")

        target_path = Path(path or self.index_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "bm25": self.bm25,
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "corpus_tokens": self.corpus_tokens,
            "source_checksum": self.source_checksum,
        }

        with open(target_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(
        self,
        path: Path | str | None = None,
        verify_source_file: Path | str | None = None,
    ) -> bool:
        """Load cached BM25 index from disk, optionally verifying source checksum.

        Args:
            path: Source pickle path. Defaults to settings.BM25_INDEX_FILE.
            verify_source_file: If provided, validates cached checksum against this file.

        Returns:
            True if loaded successfully and valid; False if file missing or checksum invalidated.
        """
        target_path = Path(path or self.index_file)
        if not target_path.exists():
            return False

        try:
            with open(target_path, "rb") as f:
                payload: dict[str, Any] = pickle.load(f)

            cached_checksum = payload.get("source_checksum")
            if verify_source_file:
                current_checksum = self.compute_checksum(verify_source_file)
                if current_checksum and cached_checksum != current_checksum:
                    # Checksum mismatch: underlying chunks.jsonl changed!
                    return False

            self.bm25 = payload["bm25"]
            self.chunks = [Chunk(**d) for d in payload["chunks"]]
            self.corpus_tokens = payload.get("corpus_tokens", [])
            self.source_checksum = cached_checksum
            return True
        except Exception:
            return False

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Query the BM25 index and return the highest-scoring chunks.

        Args:
            query: User search query.
            top_k: Number of candidate chunks to return.

        Returns:
            List of SearchResult objects sorted descending by lexical score.
        """
        if self.bm25 is None or not self.chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores: np.ndarray = self.bm25.get_scores(tokens)

        # Filter out chunks with 0 score (no matching tokens at all)
        nonzero_indices = np.where(scores > 0)[0]
        if len(nonzero_indices) == 0:
            return []

        # Sort candidate indices by score descending
        sorted_indices = nonzero_indices[np.argsort(-scores[nonzero_indices])]
        top_indices = sorted_indices[:top_k]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=round(float(scores[idx]), 6),
                    rank=rank,
                    metadata={
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                    retrieval_type="sparse",
                )
            )

        return results

    def count(self) -> int:
        """Return total chunks indexed in BM25."""
        return len(self.chunks)
