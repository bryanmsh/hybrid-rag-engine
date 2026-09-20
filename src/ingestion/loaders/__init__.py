"""Document loaders package."""

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.text_loader import TextLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.factory import get_loader_for_file, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "get_loader_for_file",
    "SUPPORTED_EXTENSIONS",
]
