# Repository Guidelines

These notes are meant for contributors (and coding assistants) to make changes consistent with the current codebase.

---

## Project Structure & Module Organization

- `backend/` — FastAPI API server (chat, upload, admin, reindex, students DB)
- `frontend/` — React/Vite UI (Chat / Upload / Admin)
- `agents/` — multi-agent orchestration (retrieval / form / FAQ)
- `ingestion/` — loader → cleaning → chunking → embedding → vector store (FAISS)
- `tests/` — pytest suites
- `data/` — local artifacts (raw docs, indexes, leads, sqlite db)

### Stable integration points (do not break)
- `ingestion.vector_store.search(query: str, top_k: int = 5) -> List[Tuple[str, dict]]`
- `agents.retrieval_agent.RetrievalAgent.answer(query: str) -> str`

---

## Build, Test, and Development Commands (Linux)

### Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build / rebuild the FAISS index
The pipeline reads `data/raw/` and writes the FAISS artifacts to the output prefix:
```bash
python -m ingestion.pipeline --data-dir data/raw --output data/vector_db/index
```

### Run the FastAPI backend
```bash
uvicorn backend.main:app --reload --port 8001
```

### Run the React frontend
```bash
cd frontend
npm install
npm run dev
```

### Tests
```bash
pytest -q
```

---

## Coding Style & Naming Conventions

- Python 3, 4-space indentation, type hints where practical.
- Keep modules small and role-focused (`agents/` vs `ingestion/` vs `backend/`).
- Functions: `snake_case`; classes: `PascalCase`; constants: `ALL_CAPS`.
- Prefer docstrings for non-obvious modules/functions; keep comments brief and purposeful.
- When changing return shapes for API endpoints or agent results, update the README/API docs accordingly.

---

## Testing Guidelines

- Framework: `pytest`.
- Prefer unit tests with tiny in-memory fixtures (short strings) over large real documents.
- Tests must **not require a live Ollama daemon**:
  - mock LLM calls (`BaseAgent.generate_response`) where needed,
  - keep retrieval/search tests focused on deterministic behavior.
- Name tests `test_*`. Add them next to their feature domain:
  - ingestion: `tests/test_ingestion.py`
  - RAG/agents: `tests/test_rag.py`, `tests/test_agents.py`

---

## Commit & Pull Request Guidelines

- Commit messages: imperative and scoped (e.g., “Fix pipeline output flag”, “Unify vector index path”).
- PRs should include:
  - short behavior summary,
  - affected modules (`agents`, `ingestion`, `backend`, `frontend`),
  - test command + result (`pytest -q`),
  - screenshots/GIFs for UI changes (when applicable).

---

## Agent & RAG Tips

- Retrieval agent:
  - build answers strictly from retrieved chunks,
  - return lightweight source metadata (filename/page) when available.
- Ingestion:
  - keep the vector index path configurable and consistent end-to-end,
  - don’t commit large indexes or raw corpora; keep artifacts under `data/` and ensure `.gitignore` covers them.