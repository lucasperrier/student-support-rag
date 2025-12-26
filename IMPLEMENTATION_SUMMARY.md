# ESILV Smart Assistant — Implementation Summary

**Date**: December 2025  
**Scope**: RAG ingestion pipeline + retrieval agent, and how they integrate with the FastAPI/Streamlit/React layers.

---

## Overview

The repository implements a document-grounded assistant for ESILV. At a high level:

1. Documents are placed in `data/raw/` (or uploaded via the API/UI).
2. The ingestion pipeline converts documents into cleaned, chunked text.
3. A sentence-transformers embedder creates dense vectors for each chunk.
4. A FAISS index + sidecar metadata JSON are saved to disk.
5. At runtime, the retrieval agent searches the index and generates answers grounded in retrieved chunks (LLM via Ollama when available).

---

## Components

### 1) Document loader (`ingestion/loader.py`)
**Responsibility**: Read supported formats and extract text + metadata.

- Inputs: PDF / TXT / HTML (and potentially other formats depending on installed libs)
- Output: `Document` objects containing:
  - `text`
  - `metadata` (filename, type, etc.)
  - `source` (path)

Primary entry point used by the pipeline:
```python
from ingestion.loader import load_documents_from_directory
docs = load_documents_from_directory("data/raw")
```

---

### 2) Text cleaning (`ingestion/text_cleaning.py`)
**Responsibility**: Normalize extraction artifacts so embeddings and retrieval stay stable.

Typical operations:
- whitespace normalization,
- unicode normalization,
- newline cleanup,
- (optionally) removal of PDF page markers / hyphenation cleanup when present.

Primary entry point:
```python
from ingestion.text_cleaning import clean_documents
clean_texts = clean_documents(docs)
```

---

### 3) Chunking (`ingestion/chunker.py`)
**Responsibility**: Split cleaned text into overlapping chunks suitable for retrieval.

- Goal: preserve semantic coherence while respecting prompt limits.
- Defaults should remain consistent across indexing and retrieval evaluation.

Primary entry point:
```python
from ingestion.chunker import chunk_documents
chunks = chunk_documents(clean_texts, metadata_list)
```

---

### 4) Embeddings (`ingestion/embedder.py`)
**Responsibility**: Map chunks and queries into a shared embedding space.

- Backend: sentence-transformers
- Default model intent: `all-MiniLM-L6-v2` (384-d)
- Vectors are L2-normalized so cosine similarity is compatible with inner product search.

Primary entry points:
```python
from ingestion.embedder import embed_chunks, embed_query
chunk_embeddings = embed_chunks(chunks)
query_embedding = embed_query("When are exams scheduled?")
```

---

### 5) Vector store (`ingestion/vector_store.py`)
**Responsibility**: Persist and query the FAISS index.

- Index type: `IndexFlatIP` (exact search)
- Metadata sidecar JSON keeps chunk text + provenance aligned by position (`i` in FAISS == `i` in metadata).

Public integration point (used across layers):
```python
from ingestion.vector_store import search
results = search("What is the internship duration?", top_k=5)
# returns: List[Tuple[text, metadata]]
```

---

### 6) Ingestion pipeline (`ingestion/pipeline.py`)
**Responsibility**: Orchestrate end-to-end indexing and write artifacts to disk.

Canonical CLI intent:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

Expected outputs:
- `data/vector_db/index.faiss`
- `data/vector_db/index_metadata.json`

---

### 7) Retrieval agent (`agents/retrieval_agent.py`)
**Responsibility**: Answer questions grounded in indexed documents.

Runtime pipeline:
1. embed user query,
2. search vector store (top-k),
3. filter by score threshold (grounding safeguard),
4. build prompt context from retrieved chunks,
5. generate answer (Ollama via `BaseAgent.generate_response`) when available,
6. return answer + renderable sources.

Public integration point (keep stable):
```python
agent.answer(query: str) -> str
```

And orchestrator-facing response shape is dict-like, typically:
- `answer: str`
- optional `sources: list`
- optional `action: str`

---

## Integration points (serving layer)

- **FastAPI** (`backend/main.py`): exposes `/api/chat`, `/api/upload`, `/api/admin`, `/api/admin/reindex`, etc.
- **Streamlit demo** (`app/main.py`): optional UI that starts an embedded API server.
- **React frontend** (`frontend/`): calls the backend through `frontend/src/api.ts`.

---

## Notes / known constraints

- Tests should not require a live Ollama server (mock LLM calls when needed).
- Keep index path usage consistent across ingestion, retrieval, and the backend’s reindex trigger.
- `data/vector_index.json` is a stub index used for admin/upload display and **must not** be confused with the FAISS retrieval index.