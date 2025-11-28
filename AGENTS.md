# Repository Guidelines

## Project Structure & Module Organization
- Core app: `app/` (Streamlit UI + FastAPI bootstrap in `main.py`), `agents/` (orchestrator, retrieval, form, FAQ), `ingestion/` (loader → cleaning → chunking → embedding → vector store), `tests/` (pytest suites), `docs/` (diagrams/specs), `data/` (raw/processed/vector_db/leads).
- Person A integration points: `ingestion/vector_store.py`, `ingestion/pipeline.py`, `agents/retrieval_agent.py`. Person B UI/orchestration is already scaffolded; avoid breaking their public functions (`vector_store.search`, `RetrievalAgent.answer`).

## Build, Test, and Development Commands
- Setup env: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Ingest sample docs: `python -m ingestion.pipeline --data-dir data/raw --out data/vector_db`.
- Run app locally: `streamlit run app/main.py` (starts Streamlit and the embedded FastAPI in a thread).
- Tests: `pytest -q`.

## Coding Style & Naming Conventions
- Python 3, 4-space indentation, type hints where practical. Keep modules small and role-focused (agents vs ingestion vs UI).
- Function names: snake_case; classes: PascalCase; configs/constants: ALL_CAPS in `app/config.py`.
- Keep public APIs stable (`vector_store.search(query, top_k) -> List[(text, meta)]`, `RetrievalAgent.answer(query) -> str`).
- Prefer docstrings for modules/functions that aren’t obvious; keep comments brief and purposeful.

## Testing Guidelines
- Framework: pytest. Add unit tests near the feature area (e.g., ingestion behaviors in `tests/test_ingestion.py`, RAG search/answer in `tests/test_rag.py`).
- Name tests `test_*`. Include fixtures for tiny text snippets rather than large files. Validate both retrieval ordering and graceful fallbacks when no hits are found.

## Commit & Pull Request Guidelines
- Commit messages: imperative and scoped (e.g., “Add FAISS-backed vector store”, “Improve form agent extraction”). Keep focused commits.
- PR expectations: short summary of behavior change, mention affected modules (`agents`, `ingestion`, `app`), note breaking API changes (avoid if possible), and include test command/results (`pytest -q`). Add screenshots/GIFs if UI changes.

## Agent & RAG Tips
- Retrieval agent: build answers strictly from search results; include light source metadata (filename/page) in outputs when available.
- Ingestion: keep vector DB path configurable; avoid committing large indexes or data dumps—use `data/` for local artifacts, ensure `.gitignore` covers them.

codex resume 019ab676-58a4-7360-8a1f-9883b705cf5c