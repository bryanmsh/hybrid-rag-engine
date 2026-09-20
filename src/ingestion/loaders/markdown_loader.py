"""Markdown document loader."""

from pathlib import Path
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import Document


class MarkdownLoader(BaseLoader):
    """Loads Markdown (.md, .markdown) files preserving structural headings."""

    def load(self, file_path: Path) -> list[Document]:
        """Load markdown file into Document instances.

        If markdown contains distinct top-level chapters or sections, it can
        structure them cleanly; defaults to full document with section preservation.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        doc_id = path.name
        return [
            Document(
                content=content,
                document_id=doc_id,
                page_number=1,
                metadata={
                    "file_path": str(path.resolve()),
                    "file_type": "markdown",
                    "file_size_bytes": path.stat().st_size,
                },
            )
        ]
