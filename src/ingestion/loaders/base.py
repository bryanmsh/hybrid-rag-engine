"""Abstract base class for document loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from src.ingestion.models import Document


class BaseLoader(ABC):
    """Abstract interface for all document file format loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """Parse the given file path into one or more Document instances.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of Document objects representing parsed sections/pages.
        """
        pass
