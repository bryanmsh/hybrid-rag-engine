"""Unit tests for document loaders and loader factory."""

import pytest
from pathlib import Path
from src.ingestion.loaders import (
    TextLoader,
    MarkdownLoader,
    get_loader_for_file,
    SUPPORTED_EXTENSIONS,
)


def test_text_loader_reads_file(tmp_path: Path):
    """TextLoader should parse plain text files and populate metadata."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello world, this is a plain text file.", encoding="utf-8")

    loader = TextLoader()
    docs = loader.load(sample_file)

    assert len(docs) == 1
    assert docs[0].content == "Hello world, this is a plain text file."
    assert docs[0].document_id == "sample.txt"
    assert docs[0].page_number == 1
    assert docs[0].metadata["file_type"] == "txt"


def test_markdown_loader_reads_file(tmp_path: Path):
    """MarkdownLoader should parse markdown files correctly."""
    sample_file = tmp_path / "architecture.md"
    sample_file.write_text(
        "# Architecture Spec\n\n## Section 1\nDetails here.",
        encoding="utf-8",
    )

    loader = MarkdownLoader()
    docs = loader.load(sample_file)

    assert len(docs) == 1
    assert "Architecture Spec" in docs[0].content
    assert docs[0].document_id == "architecture.md"
    assert docs[0].metadata["file_type"] == "markdown"


def test_loader_factory_resolution():
    """get_loader_for_file should return correct loader type for known extensions."""
    txt_loader = get_loader_for_file(Path("data/test.txt"))
    assert isinstance(txt_loader, TextLoader)

    md_loader = get_loader_for_file(Path("data/test.md"))
    assert isinstance(md_loader, MarkdownLoader)


def test_loader_factory_unsupported_extension():
    """get_loader_for_file should raise ValueError on unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        get_loader_for_file(Path("data/spreadsheet.xlsx"))
