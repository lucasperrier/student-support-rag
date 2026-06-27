"""
Lightweight retrieval evaluation for the ESILV Smart Assistant.

WHAT THIS MEASURES
------------------
This script evaluates *retrieval quality only* — i.e. whether the FAISS vector store
returns the correct source document(s) for a set of known questions. It reports:

  - top-1 source hit rate: the correct source is the single best result
  - top-3 source hit rate: the correct source is among the top 3 results
  - average / p95 query latency

WHAT THIS DOES NOT MEASURE
--------------------------
It does NOT evaluate the factual correctness of generated answers, nor citation
faithfulness. Answer generation depends on a local LLM (Ollama) and is out of scope
here. See README "What is and isn't evaluated".

USAGE
-----
    python -m eval.retrieval_eval                # build index from sample docs if missing, then evaluate
    python -m eval.retrieval_eval --rebuild      # force-rebuild the index first
    python -m eval.retrieval_eval --top-k 5      # change the top-k cutoff
    python -m eval.retrieval_eval --data-dir data/raw   # evaluate against a different corpus
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from ingestion.pipeline import run_pipeline
from ingestion.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "questions.json"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "data" / "sample_docs"
DEFAULT_INDEX = REPO_ROOT / "data" / "vector_db" / "index"


def ensure_index(index_path: Path, data_dir: Path, rebuild: bool) -> None:
    """Build the FAISS index from `data_dir` if it is missing or a rebuild is requested."""
    faiss_file = index_path.with_suffix(".faiss")
    if faiss_file.exists() and not rebuild:
        return
    print(f"Building index from {data_dir} -> {index_path} ...")
    stats = run_pipeline(data_dir=str(data_dir), output_path=str(index_path), backend="faiss")
    if "error" in stats:
        raise SystemExit(f"Index build failed: {stats['error']}")
    print(f"Indexed {stats.get('total_docs', '?')} documents "
          f"({stats.get('total_chunks', '?')} chunks).\n")


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def evaluate(questions_path: Path, index_path: Path, top_k: int) -> int:
    spec = json.loads(questions_path.read_text())
    questions = spec["questions"]

    store = VectorStore.load(str(index_path), backend="faiss")

    # Warm-up: the first search lazily loads the embedding model. Run one untimed
    # query so reported latencies reflect steady-state retrieval, not one-time init.
    if questions:
        store.search(questions[0]["question"], top_k=top_k, min_score=0.0)

    top1_hits = 0
    topk_hits = 0
    latencies_ms: List[float] = []
    rows = []

    for item in questions:
        query = item["question"]
        expected = set(item["expected_sources"])

        start = time.perf_counter()
        results = store.search(query, top_k=top_k, min_score=0.0)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        retrieved = [meta.get("filename", "?") for _, meta, _ in results]
        top1 = retrieved[0] if retrieved else "—"

        hit1 = bool(expected & set(retrieved[:1]))
        hitk = bool(expected & set(retrieved[:top_k]))
        top1_hits += int(hit1)
        topk_hits += int(hitk)

        rows.append((query, top1, hit1, hitk))

    n = len(questions)
    print(f"Retrieval evaluation — {n} questions, top_k={top_k}, corpus={index_path}\n")
    print(f"{'#':>2}  {'hit@1':^5}  {'hit@3':^5}  top-1 source            question")
    print("-" * 100)
    for i, (query, top1, hit1, hitk) in enumerate(rows, 1):
        m1 = "✓" if hit1 else "·"
        mk = "✓" if hitk else "✗"
        print(f"{i:>2}    {m1:^5}  {mk:^5}  {top1:<22}  {query[:50]}")

    print("-" * 100)
    print(f"\nTop-1 source hit rate : {top1_hits}/{n} = {top1_hits / n:.1%}")
    print(f"Top-{top_k} source hit rate : {topk_hits}/{n} = {topk_hits / n:.1%}")
    print(f"Avg query latency     : {sum(latencies_ms) / n:.1f} ms")
    print(f"p95 query latency     : {percentile(latencies_ms, 95):.1f} ms")
    print("\nNote: retrieval/source recall only. Answer factuality is NOT evaluated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval source recall.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SAMPLE_DIR,
                        help="Corpus to index when building (default: data/sample_docs)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the index")
    args = parser.parse_args()

    ensure_index(args.index, args.data_dir, args.rebuild)
    return evaluate(args.questions, args.index, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
