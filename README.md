# Modular Hybrid RAG Engine with Cross-Encoder Reranking

A modular search and retrieval engine built in Python featuring dense semantic vector search, sparse BM25 retrieval, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.

---

## 🏛️ System Architecture

```
[ INGESTION PIPELINE ]
Source Documents (.pdf, .md, .txt)
       │
       ▼
Recursive Boundary Chunker (512 tokens / 64 overlap, metadata & provenance tracking)
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                                                   ▼
[ Dense Vector Index ]                                           [ Sparse Lexical Index ]
sentence-transformers (all-MiniLM-L6-v2)                          rank_bm25 (BM25Okapi)
ChromaDB / Qdrant                                                 Disk-cached Token Index

===================================================================================================

[ INFERENCE PIPELINE ]
User Query
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                                                   ▼
Dense Retrieval (Top-20)                                          Sparse Retrieval (Top-20)
       │                                                                   │
       └─────────────────────────────────┬─────────────────────────────────┘
                                         ▼
                      [ Reciprocal Rank Fusion (RRF) ]
                          Score: sum(1 / (60 + rank_i))
                                         │
                                         ▼ (Top-20 Fused Candidates)
                      [ Cross-Encoder Stage (ms-marco-MiniLM-L-6-v2) ]
                          Joint Query-Passage Attention Scoring
                                         │
                                         ▼ (Top-4 Reranked Passages)
                      [ Generation Engine (Prompt Assembler) ]
                          Grounded Citations + Strict Boundary Checks
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                      # Raw input documents (.pdf, .md, .txt)
│   └── processed/                # Normalized chunks.jsonl and cached indexes
├── src/
│   ├── config/                   # Centralized Pydantic application settings
│   ├── ingestion/                # PDF, Markdown, TXT parsers & recursive chunker
│   ├── indexing/                 # ChromaDB / Qdrant vector store & BM25 indexer
│   ├── retrieval/                # Dense, sparse, and Reciprocal Rank Fusion (RRF)
│   ├── reranking/                # Cross-Encoder scoring and candidate pruning
│   ├── generation/               # Grounded prompt synthesis & LLM client abstraction
│   ├── evaluation/               # Automated Ragas evaluation harness
│   └── api/                      # FastAPI application schemas and endpoints
├── eval/
│   ├── eval_dataset.jsonl        # Labeled benchmark test pairs
│   └── results/                  # Benchmark runs comparing Naive vs Hybrid
├── dashboard/                    # Streamlit metrics visualizer
├── tests/                        # Pytest unit and integration test suite
└── README.md
```

---

## 🚀 Quickstart: Step 1 Ingestion

### 1. Set Up Environment
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# or source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Documents
Place your raw `.pdf`, `.md`, or `.txt` files in `data/raw/`, then run:
```bash
python -m src.ingestion.run_ingest
```

This will:
- Discover all supported files.
- Parse pages and headings with document provenance.
- Hierarchically split content into overlapping chunks with deterministic UUID5 hashes.
- Persist structured records to `data/processed/chunks.jsonl`.

### 3. Run Unit Tests
```bash
pytest tests/ -v
```
