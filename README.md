# ESILV Smart Assistant

AI-powered, **document-grounded** assistant for answering questions about ESILV (programs, rules, calendars, internships, IT charters, procedures, etc.).  
The project uses **Retrieval-Augmented Generation (RAG)** with a **FAISS** vector index (default) and a **multi-agent** orchestration layer.

This repo contains:
- a **FastAPI backend** (`backend/`) for chat, uploads, admin and reindexing,
- a **React/Vite frontend** (`frontend/`) for a modern UI,
- an optional **Streamlit demo app** (`app/`) that can bootstrap an embedded API server,
- the shared technical core: **ingestion + agents**.

---

## What you can do (features)

### Chat (RAG + agents)
- Ask questions about ESILV and get **document-grounded** answers (RAG).
- Responses can include **sources** (filenames) when available.
- Requests are routed by an **Orchestrator**:
  - `RetrievalAgent`: information queries grounded in documents (RAG)
  - `FormAgent`: lead collection (name/email/interest) → `data/leads.json`
  - `FAQAgent`: deterministic answers for a few known questions

### Upload documents
- Upload PDFs/HTML/TXT via the UI or API.
- Files are saved under `data/raw/`.

### Reindex (build/rebuild FAISS)
- Rebuild the vector index from `data/raw/` via CLI or admin endpoint.
- Outputs are stored under `data/vector_db/`.

### Admin / monitoring
- View uploaded docs (from a lightweight stub index for display).
- View captured leads (`data/leads.json`).
- Optional student registry endpoints backed by SQLite (`data/students.db`).

### Web ingestion (URL → raw text file)
- Provide a URL → the backend fetches the content and saves a timestamped `.txt` into `data/raw/`.

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
- `data/vector_index.json` — **stub/demo index** used for admin upload display only (not used for retrieval)

---

## Requirements

### Local (no Docker)
- Python 3.x
- Node.js + npm (for the React frontend)
- Optional (for generation): **Ollama** running locally (agents call Ollama via the `ollama` Python package)

### Docker
- Docker + Docker Compose (recommended) or Docker only

---

## Quick start (no Docker)

### 1) Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Optional) Start Ollama for generation
If you want full answers (not just retrieval), start Ollama in a separate terminal:
```bash
ollama serve
```

If needed, pull the model (default: `llama2`):
```bash
ollama pull llama2
```

You can override the model via:
```bash
export OLLAMA_MODEL=llama2
```

### 3) Add documents to index
Put documents in:
```bash
data/raw/
```

### 4) Build the vector index (RAG ingestion)
Indexes everything in `data/raw/` into `data/vector_db/index(.faiss + _metadata.json)`:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### 5) Run the backend API (FastAPI)
From repo root:
```bash
uvicorn backend.main:app --reload --port 8001
```

Backend will be available at:
- `http://127.0.0.1:8001`

### 6) Run the frontend (React/Vite)
In another terminal:
```bash
cd frontend
npm install
npm run dev
```

Frontend will use `VITE_API_URL` from `frontend/.env` (default: `http://127.0.0.1:8001`).

### 7) Run tests
```bash
pytest -q
```

> Guideline: unit tests should not require a live Ollama daemon; mock LLM calls when needed.

---

## Streamlit demo (optional, no Docker)

This starts Streamlit and an embedded FastAPI server (in a background thread):

```bash
streamlit run app/main.py
```

Notes:
- The Streamlit demo includes **Chat / Upload / Admin** tabs.
- Uploads are saved into `data/raw/`.
- Admin views show uploaded docs using a **stub index** (`data/vector_index.json`).
- Real retrieval still relies on the FAISS index in `data/vector_db/index.faiss`.

---

## Quick start (Docker)

> This repository includes Docker support for backend + frontend. Data is persisted by mounting `./data` into the containers.

### 1) Build and start
From repo root:
```bash
docker compose up --build
```

Expected services:
- frontend: `http://localhost:8080`
- backend: `http://localhost:8001`

### 2) Indexing with Docker
You have two common options:

**Option A — run ingestion on the host (recommended for dev):**
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

**Option B — run ingestion inside the backend container (if Python tooling is available there):**
```bash
docker compose exec backend python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### 3) Ollama with Docker
Generation requires Ollama.

This repo supports a single-command Docker stack (frontend + backend + Ollama):

```bash
docker compose up -d --build
```

Then open:
- Frontend: http://localhost:8080
- Backend API: http://localhost:8001

#### First run: pull a model into the Ollama container
The Ollama model cache is stored in the Docker named volume `ollama` (separate from any models you may have pulled on your host). On first run you must pull a model once:

```bash
docker compose exec ollama ollama pull llama2:latest
```

After that, generation should work without re-downloading unless you delete the `ollama` volume.

If Ollama is not available, the system may still retrieve context, but answer generation can fail or return an error message depending on agent behavior.

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
  Runs:
  `python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index`

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

## Troubleshooting

### “Vector store not found … run ingestion.pipeline”
You need to ingest documents first:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### Ollama / LLM errors
If the `ollama` Python package is missing or Ollama is not running, generation may fail or return an error string. Retrieval can still work if the index exists, but answer generation depends on the LLM.

### Uploads appear in Admin but chat retrieval doesn’t use them
Admin uses `data/vector_index.json` (stub listing). Retrieval uses the FAISS index. Rebuild the FAISS index:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

---

## License

Add a `LICENSE` file to specify the repository license.