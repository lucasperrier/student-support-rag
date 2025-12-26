# Testing Summary — ESILV Smart Assistant

This document summarizes the test suites present in `tests/` and the intent behind them.

> Note: Keep test claims (counts, coverage) aligned with what actually exists in `tests/`.
> If you add/remove tests, update this file.

---

## How to run

```bash
pytest -q
```

Common variants:
```bash
pytest -q tests/test_ingestion.py
pytest -q tests/test_rag.py
pytest -q tests/test_agents.py
```

---

## Testing principles used in this repo

- Prefer small fixtures (short strings) and temporary folders via pytest.
- Avoid requiring large real PDFs for unit tests.
- Unit tests should not require a live **Ollama** daemon:
  - mock `BaseAgent.generate_response` where generation is involved,
  - focus retrieval tests on deterministic vector search and context building.

---

## What should be covered

### Ingestion pipeline
- Loader: supported formats, missing files, directory scanning.
- Cleaning: whitespace/unicode normalization and regression cases.
- Chunker: overlap, boundaries, metadata preservation.
- Embedder: shape, normalization, repeatability.
- Vector store: save/load, top-k ordering, min_score behavior.
- Pipeline: end-to-end run in a temp workspace with tiny docs.

### RAG / Agents
- RetrievalAgent: retrieval behavior, empty index handling, response structure.
- Orchestrator routing: deterministic routing for obvious cases; graceful fallback if classifier fails.
- FormAgent: email extraction/validation and lead persistence (prefer temp `leads.json`).
- FAQAgent: normalization/tokenization and mapping behavior.

---

## Keeping this file accurate

Instead of hardcoding “N tests” and “100% coverage”, document:
- which files exist (`tests/test_ingestion.py`, `tests/test_rag.py`, ...),
- what they verify,
- constraints (no Ollama required for unit tests).

If you want exact numbers, generate them from your local run and paste the output of:
```bash
pytest -q --disable-warnings
```