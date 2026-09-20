"""Document ingestion and processing package."""

from src.ingestion.models import Chunk, Document, generate_chunk_id
from src.ingestion.chunker import RecursiveBoundaryChunker
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders import get_loader_for_file, BaseLoader, TextLoader, MarkdownLoader, PDFLoader

__all__ = [
    "Chunk",
    "Document",
    "generate_chunk_id",
    "RecursiveBoundaryChunker",
    "IngestionPipeline",
    "get_loader_for_file",
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
]
