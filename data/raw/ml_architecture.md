# Modern AI Systems Architecture: Hybrid Retrieval & Reranking

## 1. Overview of Hybrid RAG
Standard Retrieval-Augmented Generation (RAG) relies on bi-encoder dense vector similarity search to supply external context to a Large Language Model. While dense vectors excel at capturing abstract conceptual and semantic similarity, they exhibit noticeable failure modes on exact keyword queries, rare nomenclature, part numbers, and alphanumeric identifiers.

To overcome these limitations, hybrid retrieval fuses dense semantic retrieval with sparse lexical retrieval (BM25) using Reciprocal Rank Fusion (RRF).

## 2. Sparse Lexical Search (BM25)
BM25 (Best Matching 25) is a probabilistic bag-of-words retrieval function that ranks documents based on term frequency (TF) and inverse document frequency (IDF). Unlike dense bi-encoders, BM25 operates on exact token matches, making it robust against out-of-vocabulary terms and keyword-specific queries.

The standard BM25Okapi scoring formula calculates:
- Term frequency adjusted by document length normalization (parameter b = 0.75).
- Term saturation dampening (parameter k1 = 1.5).

## 3. Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion merges candidate lists from distinct retrieval algorithms without requiring calibration or normalization of their raw score distributions. The RRF score for document d is calculated as:
RRF(d) = sum(1 / (k + rank(d)))
where k is a smoothing constant typically set to 60.

## 4. Cross-Encoder Reranking
In the final retrieval stage, candidate passages identified by RRF are evaluated by a deep cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2). The cross-encoder performs joint multi-head self-attention across the concatenated query and candidate passage, computing non-linear token-to-token interactions. The top 4 candidates are passed to the generator prompt assembler.
