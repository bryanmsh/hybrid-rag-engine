"""Factory for resolving document loaders based on file extension."""

from pathlib import Path
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.text_loader import TextLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pdf": PDFLoader,
}


def get_loader_for_file(file_path: Path | str) -> BaseLoader:
    """Return an instantiated loader for the specified file extension.

    Args:
        file_path: Path to the target file.

    Returns:
        Instance of BaseLoader appropriate for the file format.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    loader_cls = SUPPORTED_EXTENSIONS.get(suffix)
    if not loader_cls:
        valid_exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise ValueError(
            f"Unsupported file format '{suffix}' for file: {path.name}. "
            f"Supported formats: {valid_exts}"
        )

    return loader_cls()
