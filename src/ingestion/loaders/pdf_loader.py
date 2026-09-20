"""PDF document loader with page-level granularity."""

from pathlib import Path
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import Document


class PDFLoader(BaseLoader):
    """Loads PDF (.pdf) files and extracts content page-by-page."""

    def load(self, file_path: Path) -> list[Document]:
        """Extract text from PDF pages using pypdf.

        Each page is represented as a separate Document object with accurate page_number.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            import pypdf
        except ImportError as exc:
            raise ImportError(
                "pypdf is required to process PDF files. Please install pypdf."
            ) from exc

        documents: list[Document] = []
        doc_id = path.name

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                if not cleaned_text:
                    continue  # Skip empty or image-only scanned pages

                documents.append(
                    Document(
                        content=cleaned_text,
                        document_id=doc_id,
                        page_number=page_idx + 1,
                        metadata={
                            "file_path": str(path.resolve()),
                            "file_type": "pdf",
                            "page_number": page_idx + 1,
                            "total_pages": total_pages,
                        },
                    )
                )

        return documents
