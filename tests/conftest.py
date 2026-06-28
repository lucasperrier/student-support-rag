"""Shared pytest fixtures.

The production FAISS index (``data/vector_db/index``) is gitignored, so it is absent
in a fresh clone and in CI. Several retrieval tests load it, so build it once per test
session from the bundled sample documents when it is missing.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_index():
    index = Path("data/vector_db/index")
    if not index.with_suffix(".faiss").exists():
        from ingestion.pipeline import run_pipeline

        index.parent.mkdir(parents=True, exist_ok=True)
        run_pipeline(data_dir="data/sample_docs", output_path=str(index), backend="faiss")
    yield
