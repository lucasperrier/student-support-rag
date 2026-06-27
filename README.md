# ESILV Smart Assistant

A **document-grounded RAG assistant** that answers questions about a school's programs,
rules, calendars, and internships from its own documents — with **source attribution**
and a **multi-agent** router.

[![tests](https://github.com/lucasperrier/student-support-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/lucasperrier/student-support-rag/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> **Stack:** PDF/HTML/TXT ingestion → recursive chunking → sentence-transformer embeddings
> → FAISS retrieval → agent router → FastAPI + React, with Docker Compose, pytest, and a
> lightweight retrieval evaluation.

---

## What it does

- **Grounded chat (RAG):** answers are built from retrieved document chunks and return the
  **source filenames** they came from.
- **Multi-agent routing:** an `Orchestrator` routes each message to a `RetrievalAgent`
  (RAG), a `FormAgent` (lead capture → `data/leads.json`), or a `FAQAgent` (curated answers).
- **Ingestion pipeline:** load PDF/HTML/TXT → clean → chunk → embed → FAISS index.
- **Upload & reindex:** add documents via the UI/API, then rebuild the index.
- **Retrieval evaluation:** measure top-1 / top-3 source recall and latency on a question set.

See it end-to-end in the **[demo walkthrough](docs/demo.md)** (real inputs and outputs).

---

## Architecture

```
                      ┌─────────────────────── ingestion (offline) ───────────────────────┐
  data/raw/*.pdf,html,txt → loader → text cleaning → chunker → embedder → FAISS vector store
                                                                                  │ index.faiss
                                                                                  ▼ + metadata
  user ──▶ React UI ──▶ FastAPI /api/chat ──▶ Orchestrator (router)
                                                  ├─▶ RetrievalAgent ─▶ FAISS search ─▶ LLM (Ollama) ─▶ grounded answer + sources
                                                  ├─▶ FormAgent ─▶ leads.json
                                                  └─▶ FAQAgent ─▶ curated answer
```

| Layer        | Path           | Tech                                              |
|--------------|----------------|---------------------------------------------------|
| Ingestion    | `ingestion/`   | pypdf, BeautifulSoup, sentence-transformers, FAISS |
| Agents       | `agents/`      | rule-first router + RAG, optional LLM classifier  |
| Backend      | `backend/`     | FastAPI, SQLModel (SQLite)                         |
| Frontend     | `frontend/`    | React 19, Vite, TypeScript                         |
| Demo app     | `app/`         | Streamlit (optional)                              |
| LLM          | —              | Ollama (`llama2` by default), local & optional    |

**Retrieval is decoupled from generation:** if Ollama isn't running you still get the
correct sources, just not the generated prose.

---

## Quick start (Docker — one command)

Brings up the frontend, backend, and an Ollama container:

```bash
docker compose up --build
```

| Service  | URL                                                |
|----------|----------------------------------------------------|
| Frontend | http://localhost:8080                              |
| Backend  | http://localhost:8001 (docs: http://localhost:8001/docs) |

Two one-time steps for full **generated** answers (retrieval works without them):

```bash
# 1) pull a model into the Ollama container
docker compose exec ollama ollama pull llama2

# 2) build the index from the bundled sample documents
docker compose exec backend python -m ingestion.pipeline --data-dir data/sample_docs --output data/vector_db/index
```

Then ask a question in the UI, or:

```bash
curl -s -X POST http://localhost:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What engineering majors does ESILV offer?"}'
```

## Quick start (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                         # optional: tweak OLLAMA_MODEL etc.

# Build the index from the bundled sample documents
python -m ingestion.pipeline --data-dir data/sample_docs --output data/vector_db/index

# Run the API (http://127.0.0.1:8001)
uvicorn backend.main:app --reload --port 8001

# In another terminal, run the React UI (http://127.0.0.1:5173)
cd frontend && npm install && npm run dev
```

Optional Streamlit demo: `streamlit run app/main.py`.
Optional generation: install [Ollama](https://ollama.com), run `ollama serve`, `ollama pull llama2`.

---

## Tests

```bash
$ pytest -q
........................................................................ [100%]
72 passed in ~15s
```

72 tests cover the ingestion pipeline (32), the RAG/retrieval layer (26), and the agents
(14). Unit tests do **not** require a running Ollama daemon. CI runs `pytest -q` on every
push and pull request (see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

---

## Retrieval evaluation

The repo includes a small retrieval eval over a labelled question set
([`eval/questions.json`](eval/questions.json)):

```bash
python -m eval.retrieval_eval          # builds the index from data/sample_docs if missing
```

```
Retrieval evaluation — 17 questions, top_k=3

Top-1 source hit rate : 17/17 = 100.0%
Top-3 source hit rate : 17/17 = 100.0%
Avg query latency     : ~7 ms      (steady-state; varies run-to-run)
p95 query latency     : ~7 ms
```

A perfect score is expected on this small curated corpus; the harness is built to surface
misses on larger or noisier sets (`--data-dir data/raw`, `--rebuild`, `--top-k`).

### What is and isn't evaluated

- **Evaluated:** retrieval quality — whether the correct **source document** is returned
  (top-1 / top-3 source recall) and query latency.
- **Not evaluated / not guaranteed:** the factual correctness of the LLM-generated answer
  text, and citation faithfulness to a medical/legal standard. Generation depends on the
  local LLM and is out of scope for the automated eval.

---

## API (FastAPI)

| Method & path                 | Purpose                                            |
|-------------------------------|----------------------------------------------------|
| `POST /api/chat`              | `{ "message": "..." }` → `{ answer, sources, action, routed_agent }` |
| `POST /api/upload`            | Multipart `file` → saved to `data/raw/`            |
| `POST /api/admin/reindex`     | Rebuild the FAISS index from `data/raw/`           |
| `POST /api/admin/ingest_url`  | `{ "url": "..." }` → fetch page → `.txt` in `data/raw/` |
| `GET  /api/admin`             | Leads + uploaded-doc list                          |
| `POST /api/admin/lead`        | Append a lead to `data/leads.json`                 |
| `GET/POST /api/admin/students`| SQLite student registry                            |

Interactive docs at `/docs` when the backend is running.

---

## Project structure

```
esilv_smart_assistant/
├── agents/           # orchestrator + retrieval / form / faq agents
├── ingestion/        # loader → cleaning → chunking → embedding → FAISS vector store
├── backend/          # FastAPI server
├── frontend/         # React + Vite + TypeScript UI
├── app/              # optional Streamlit demo
├── eval/             # retrieval evaluation (questions.json + retrieval_eval.py)
├── tests/            # pytest suites (72 tests)
├── data/
│   ├── sample_docs/  # bundled non-sensitive demo corpus (committed)
│   ├── raw/          # your documents (gitignored)
│   └── vector_db/    # generated FAISS index (gitignored)
└── docs/             # demo walkthrough + report / pitch deck
```

---

## Limitations

This is a portfolio / development project, **not** a production-hardened system:

- **Local & dev-oriented:** no authentication; CORS is permissive (`allow_origins=["*"]`).
- **Generation needs a local LLM:** answers require a running Ollama; without it you get
  retrieval + sources only.
- **Indexing is not automatic:** uploaded documents become searchable only after a reindex.
- **No PHI/PII handling or compliance guarantees;** don't load sensitive documents.
- **Answer factuality is not verified** — see *What is and isn't evaluated*.
- The bundled `data/sample_docs/` are **synthetic** demo documents, not official ESILV files.

---

## Documentation

- [Demo walkthrough](docs/demo.md) — ingest, ask, and evaluate with real outputs
- [Contributor guide](AGENTS.md) — structure, commands, conventions
- [Project report (PDF)](docs/EsilvSmartAssistantReport.pdf) · [Pitch deck (PDF)](docs/ESILV_Smart_assistant_PitchDeck.pdf)

## License

[MIT](LICENSE)
