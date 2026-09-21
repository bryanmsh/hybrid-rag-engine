"""CLI entrypoint to run document ingestion directly from the terminal."""

import argparse
import sys
from pathlib import Path
from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ingest and chunk raw documents.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to raw documents directory (defaults to data/raw)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output chunks.jsonl file (defaults to data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (tokens/words)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing all documents",
    )

    args = parser.parse_args()
    settings = get_settings()

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size or settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=args.chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP,
    )

    print("==================================================")
    print("🚀 Running Document Ingestion Pipeline")
    print(f"Source Directory: {args.source_dir or settings.DATA_RAW_DIR}")
    print(f"Output Target   : {args.output_file or settings.CHUNKS_FILE}")
    print(f"Chunk Size/Over : {pipeline.chunk_size} / {pipeline.chunk_overlap}")
    print("==================================================")

    result = pipeline.run(
        source_dir=args.source_dir,
        output_file=args.output_file,
        force_reindex=args.force_reindex,
    )

    print("\n✅ Ingestion Completed Successfully!")
    print(f"Files Found     : {result['files_found']}")
    print(f"Files Processed : {result['files_processed']}")
    print(f"Chunks Created  : {result['chunks_created']}")
    print(f"Elapsed Time    : {result['elapsed_ms']} ms")
    print(f"Output File     : {result['output_file']}")
    print("==================================================")


if __name__ == "__main__":
    main()
