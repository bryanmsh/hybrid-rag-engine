"""Recursive boundary-aware text chunker prioritizing semantic structures."""

import re
from typing import Callable
from src.ingestion.models import Chunk, Document


class RecursiveBoundaryChunker:
    """Splits documents hierarchically preserving paragraphs, headings, and sentences."""

    DEFAULT_SEPARATORS = [
        "\n\n",       # Paragraph breaks
        "\n# ",       # H1 Headings
        "\n## ",      # H2 Headings
        "\n### ",     # H3 Headings
        "\n",         # Line breaks
        ". ",         # Sentence terminator
        "? ",         # Question terminator
        "! ",         # Exclamation terminator
        "; ",         # Clause separator
        ", ",         # Phrase separator
        " ",          # Word boundary
        "",           # Character fallback
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        length_function: Callable[[str], int] | None = None,
        separators: list[str] | None = None,
    ):
        """Initialize the recursive chunker.

        Args:
            chunk_size: Target maximum size for each chunk (in units of length_function).
            chunk_overlap: Number of overlapping units between consecutive chunks.
            length_function: Function to compute length (default: word count estimation ~tokens).
            separators: Hierarchical list of string delimiters.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default length function estimates tokens via whitespace tokenization
        self.length_function = length_function or self._default_token_estimator
        self.separators = separators or self.DEFAULT_SEPARATORS

    @staticmethod
    def _default_token_estimator(text: str) -> int:
        """Estimate token count based on whitespace and punctuation splits."""
        if not text:
            return 0
        # Average English word is ~1.3 tokens; whitespace split * 1.25 approximates subword token count
        words = len(text.split())
        return max(1, int(words * 1.25))

    def split_text(self, text: str) -> list[str]:
        """Recursively split text string into chunks within chunk_size with chunk_overlap."""
        text = text.strip()
        if not text:
            return []

        if self.length_function(text) <= self.chunk_size:
            return [text]

        raw_pieces = self._split_recursive(text, self.separators)
        return self._merge_pieces_with_overlap(raw_pieces)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively break text down using the first separator that appears."""
        final_pieces: list[str] = []

        # Find the first separator present in the text
        separator = ""
        next_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                next_separators = []
                break
            if sep in text:
                separator = sep
                next_separators = separators[i + 1:]
                break

        if separator:
            splits = text.split(separator)
        else:
            # Character-level fallback
            splits = list(text)

        for split in splits:
            if not split.strip():
                continue
            if self.length_function(split) <= self.chunk_size:
                final_pieces.append(split)
            else:
                if next_separators:
                    sub_pieces = self._split_recursive(split, next_separators)
                    final_pieces.extend(sub_pieces)
                else:
                    final_pieces.append(split)

        return final_pieces

    def _merge_pieces_with_overlap(self, pieces: list[str]) -> list[str]:
        """Combine small text pieces into chunks honoring chunk_size and chunk_overlap."""
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for piece in pieces:
            piece_len = self.length_function(piece)

            if current_length + piece_len > self.chunk_size and current_chunk:
                merged_text = " ".join(current_chunk).strip()
                if merged_text:
                    chunks.append(merged_text)

                # Keep overlap pieces from the tail of current_chunk
                overlap_chunk: list[str] = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    p_l = self.length_function(p)
                    if overlap_len + p_l <= self.chunk_overlap:
                        overlap_chunk.insert(0, p)
                        overlap_len += p_l
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            final_text = " ".join(current_chunk).strip()
            if final_text:
                chunks.append(final_text)

        return chunks

    def chunk_document(
        self,
        document: Document,
        file_path: str,
        start_index: int = 0,
    ) -> tuple[list[Chunk], int]:
        """Split a Document into normalized Chunk models with provenance tracking.

        Args:
            document: Raw parsed Document instance.
            file_path: Originating file path.
            start_index: Starting chunk index offset for document continuity.

        Returns:
            Tuple of (list of Chunk instances, next chunk index).
        """
        text_chunks = self.split_text(document.content)
        chunks: list[Chunk] = []
        curr_index = start_index

        for text in text_chunks:
            chunk = Chunk.from_document_part(
                document=document,
                chunk_index=curr_index,
                content=text,
                file_path=file_path,
            )
            chunks.append(chunk)
            curr_index += 1

        return chunks, curr_index
