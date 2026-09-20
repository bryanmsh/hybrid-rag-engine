"""Orchestrator pipeline for document discovery, ingestion, chunking, and persistence."""

import json
from pathlib import Path
import time
from typing import Any
from src.config.settings import get_settings
from src.ingestion.chunker import RecursiveBoundaryChunker
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, get_loader_for_file
from src.ingestion.models import Chunk


class IngestionPipeline:
    """Manages the lifecycle of discovering, parsing, chunking, and saving documents."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        self.chunker = RecursiveBoundaryChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_file(self, file_path: Path) -> list[Chunk]:
        """Load and chunk a single file.

        Args:
            file_path: Path to target file.

        Returns:
            List of generated Chunk objects.
        """
        path = Path(file_path)
        loader = get_loader_for_file(path)
        documents = loader.load(path)

        all_chunks: list[Chunk] = []
        chunk_index_counter = 0

        for doc in documents:
            chunks, next_idx = self.chunker.chunk_document(
                document=doc,
                file_path=str(path.resolve()),
                start_index=chunk_index_counter,
            )
            all_chunks.extend(chunks)
            chunk_index_counter = next_idx

        return all_chunks

    def run(
        self,
        source_dir: Path | str | None = None,
        output_file: Path | str | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Scan directory, chunk all supported documents, and persist to JSONL.

        Args:
            source_dir: Directory containing raw documents.
            output_file: Target JSONL path for saving chunks.
            force_reindex: If False and output_file exists, skips or appends as needed.

        Returns:
            Execution summary dict.
        """
        settings = get_settings()
        raw_dir = Path(source_dir or settings.DATA_RAW_DIR)
        out_path = Path(output_file or settings.CHUNKS_FILE)

        start_time = time.perf_counter()

        if not raw_dir.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Discover matching files
        discovered_files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS.keys():
            discovered_files.extend(raw_dir.glob(f"**/*{ext}"))

        discovered_files = sorted(list(set(discovered_files)))

        total_chunks: list[Chunk] = []
        processed_file_count = 0

        for file_path in discovered_files:
            try:
                chunks = self.process_file(file_path)
                total_chunks.extend(chunks)
                processed_file_count += 1
            except Exception as err:
                print(f"[Warning] Failed to process {file_path.name}: {err}")

        # Persist to data/processed/chunks.jsonl
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in total_chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "status": "success",
            "files_found": len(discovered_files),
            "files_processed": processed_file_count,
            "chunks_created": len(total_chunks),
            "output_file": str(out_path.resolve()),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    @staticmethod
    def load_chunks_from_file(file_path: Path | str) -> list[Chunk]:
        """Read and validate serialized chunks from a JSONL file."""
        path = Path(file_path)
        if not path.exists():
            return []

        chunks: list[Chunk] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    chunks.append(Chunk(**data))

        return chunks
