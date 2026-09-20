"""Plain text document loader."""

from pathlib import Path
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import Document


class TextLoader(BaseLoader):
    """Loads plain text (.txt) files."""

    def load(self, file_path: Path) -> list[Document]:
        """Load text file into a single Document."""
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
                    "file_type": "txt",
                    "file_size_bytes": path.stat().st_size,
                },
            )
        ]
