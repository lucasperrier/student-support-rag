# ESILV Smart Assistant

AI-powered, **document-grounded** assistant for answering questions about ESILV (programs, rules, calendars, internships, IT charters, procedures, etc.).  
The project uses **Retrieval-Augmented Generation (RAG)** with a **FAISS** vector index (default) and a **multi-agent** orchestration layer.

This repo contains:
- a **FastAPI backend** (`backend/`) for chat, uploads, admin and reindexing,
- a **React/Vite frontend** (`frontend/`) for a modern UI,
- an optional **Streamlit demo app** (`app/`) that can bootstrap an embedded API server,
- the shared technical core: **ingestion + agents**.

---

## Key ideas

### RAG (Retrieval-Augmented Generation)
Instead of answering from general model knowledge, the assistant:
1. **retrieves** the most relevant chunks from the indexed ESILV documents (FAISS),
2. **generates** an answer using only that context (grounded prompt).

### Multi-agent orchestration
Requests are routed by an `Orchestrator` to specialized agents:
- `RetrievalAgent`: information queries grounded in documents (RAG)
- `FormAgent`: lead collection (name/email/interest) → `data/leads.json`
- `FAQAgent`: deterministic answers for a few known questions

---

## Project structure

```
esilv_smart_assistant/
├── agents/                 # orchestrator + retrieval/form/faq agents
├── ingestion/              # loader → cleaning → chunking → embedding → vector store
├── backend/                # FastAPI server (production-oriented)
├── app/                    # Streamlit demo app (starts embedded server)
├── frontend/               # React/Vite UI
├── data/                   # raw docs, FAISS index, leads, sqlite db, etc. (local artifacts)
├── tests/                  # pytest suites
└── docs/                   # diagrams/specs (optional)
```

---

## Storage layout (under `data/`)

After indexing, retrieval uses the FAISS artifacts:
- `data/vector_db/index.faiss`
- `data/vector_db/index_metadata.json`

Other local runtime artifacts:
- `data/raw/` — uploaded or manually added source documents
- `data/processed/` — optional intermediate outputs
- `data/leads.json` — captured leads (FormAgent)
- `data/students.db` — SQLite DB (admin/student endpoints)
- `data/vector_index.json` — *demo/stub index* used for admin upload display only (not used for retrieval)

---

## Requirements

- Python 3.x
- Recommended: create a virtualenv
- Optional (for generation): **Ollama** running locally (the agents call Ollama via the `ollama` Python package)

---

## Quick start (backend + frontend)

### 1) Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build the vector index (RAG ingestion)
Indexes everything in `data/raw/` into `data/vector_db/index(.faiss + _metadata.json)`:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### 3) Run the backend API (FastAPI)
From repo root:
```bash
uvicorn backend.main:app --reload --port 8001
```

Backend will be available at:
- `http://127.0.0.1:8001`

### 4) Run the frontend (React/Vite)
In another terminal:
```bash
cd frontend
npm install
npm run dev
```

Frontend will use `VITE_API_URL` from `frontend/.env` (default: `http://127.0.0.1:8001`).

---

## Streamlit demo (optional)

This starts Streamlit and an embedded FastAPI server (in a background thread):

```bash
streamlit run app/main.py
```

Notes:
- The Streamlit demo includes **Chat / Upload / Admin** tabs.
- Uploads are saved into `data/raw/`.
- Admin views show uploaded docs using a *stub index* (`data/vector_index.json`).
- Real retrieval still relies on the FAISS index in `data/vector_db/index.faiss`.

---

## Backend API endpoints (FastAPI)

Core endpoints (see `backend/main.py`):

- `POST /api/chat`  
  Body: `{ "message": "..." }`  
  Response: `{ "answer": "...", "action"?: "...", "sources"?: [...] }`

- `POST /api/upload`  
  Multipart form: `file`  
  Saves file to `data/raw/`

- `GET /api/admin`  
  Returns `{ leads, uploads }` (uploads are from the stub index)

- `POST /api/admin/lead`  
  Form fields: `name`, `email`, `interest`  
  Appends to `data/leads.json`

- `POST /api/admin/ingest_url`  
  Body: `{ "url": "https://..." }`  
  Fetches the page and saves a `.txt` into `data/raw/`

- `POST /api/admin/reindex`  
  Runs `python -m ingestion.pipeline ...` to rebuild the FAISS index from `data/raw/`

Student registry (SQLite, demo/admin):
- `GET /api/admin/students`
- `POST /api/admin/students`

---

## Ingestion pipeline (details)

Pipeline stages (see `ingestion/pipeline.py`):
1. `loader.py`: read PDF/HTML/TXT and extract text
2. `text_cleaning.py`: normalize whitespace, fix PDF artifacts, remove page markers, etc.
3. `chunker.py`: recursive chunking with overlap
4. `embedder.py`: sentence-transformers embeddings (default `all-MiniLM-L6-v2`, 384-d)
5. `vector_store.py`: FAISS `IndexFlatIP` + sidecar metadata JSON

Run:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

---

## Agents

- `agents/orchestrator.py`: routes message to agent (retrieval vs form vs FAQ)
- `agents/retrieval_agent.py`: RAG retrieval + grounded generation
- `agents/form_agent.py`: collects name/email/interest, persists leads
- `agents/faq_agent.py`: curated FAQ answers

Public integration points (keep stable):
- `ingestion.vector_store.search(query: str, top_k: int = 5) -> List[Tuple[str, dict]]`
- `RetrievalAgent.answer(query: str) -> str`

---

## Testing

Run unit tests:
```bash
pytest -q
```

Guideline: tests should not require a live Ollama daemon; mock LLM calls when needed.

---

## Troubleshooting

### “Vector store not found … run ingestion.pipeline”
You need to ingest documents first:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### Ollama / LLM errors
If the `ollama` Python package is missing or Ollama is not running, generation may fail or return an error string. The retrieval step can still run if the index exists, but answer generation depends on the LLM.

---

## License

Add a `LICENSE` file to specify the repository license.