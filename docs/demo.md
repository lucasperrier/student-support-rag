# Demo Walkthrough

This walkthrough takes you from a clean clone to grounded answers using the bundled
**sample document set** ([`data/sample_docs/`](../data/sample_docs/)) — no private school
files required. All outputs below are real, captured from this repo.

> Generation uses a local LLM via **Ollama** (`llama2`). Retrieval works without it; if
> Ollama is not running you still get the correct **sources**, just not the generated answer.

---

## 1. Ingest / reindex the sample documents

Build the FAISS index from the six sample documents:

```bash
python -m ingestion.pipeline --data-dir data/sample_docs --output data/vector_db/index
```

Output:

```
✓ Processed 6 documents in 2.58s
✓ Created 14 chunks
✓ Generated 14 embeddings
✓ Vector store saved to data/vector_db/index
```

You can also add a document at runtime and rebuild from the API:

```bash
curl -F "file=@/path/to/your.pdf" http://127.0.0.1:8001/api/upload   # saves to data/raw/
curl -X POST http://127.0.0.1:8001/api/admin/reindex                 # rebuilds the FAISS index
```

> Note: uploads land in `data/raw/` and become searchable only **after** a reindex.

---

## 2. Ask questions (real outputs)

Each call is `POST /api/chat` with `{"message": "..."}`. The orchestrator routes the
query, then the retrieval agent answers from the indexed documents.

### Q1 — "What engineering majors does ESILV offer?"

```json
{
  "answer": "Based on the provided context, ESILV offers four engineering majors:\n\n1. Data & Artificial Intelligence\n2. Computer Science & Digital Technologies\n3. Energy & Smart Cities\n4. Industrial Engineering & Numerical Mechanics.",
  "sources": ["academic_calendar.txt", "campus_life.html", "internships.md", "programs.md"],
  "action": "answer",
  "routed_agent": "retrieval_agent"
}
```

Top retrieved chunk: `programs.md` (similarity ≈ 0.77). ✅ grounded in the correct source.

### Q2 — "How long is the end-of-studies internship?"

```json
{
  "answer": "Based on the provided context, the end-of-studies internship lasts for four to six months. According to [3], final-year students complete a four- to six-month end-of-studies internship that often leads to a first job.",
  "sources": ["academic_calendar.txt", "internships.md"],
  "action": "answer",
  "routed_agent": "retrieval_agent"
}
```

Top retrieved chunk: `academic_calendar.txt` (similarity ≈ 0.69), with `internships.md` close behind. ✅

### Q3 — Lead capture (routes to the FormAgent, not retrieval)

Message: `My name is Jane Doe and my email is jane@example.com, I'm interested in the AI major`

```json
{
  "answer": "Thanks, Jane Doe! I saved your contact info. Interest: the AI major",
  "sources": [],
  "action": "lead_saved",
  "routed_agent": "form_agent"
}
```

The orchestrator detected an email + name and routed to the FormAgent, which extracted
the fields and appended a record to `data/leads.json`.

---

## 3. Retrieval evaluation

Measure source-recall over the bundled question set
([`eval/questions.json`](../eval/questions.json)):

```bash
python -m eval.retrieval_eval
```

Latest result on the sample corpus:

```
Retrieval evaluation — 17 questions, top_k=3

Top-1 source hit rate : 17/17 = 100.0%
Top-3 source hit rate : 17/17 = 100.0%
Avg query latency     : ~7 ms      (steady-state; varies run-to-run)
p95 query latency     : ~7 ms

Note: retrieval/source recall only. Answer factuality is NOT evaluated.
```

This is a small, curated corpus, so a perfect score is expected; the harness is designed
to surface misses on larger or noisier document sets. See the README section
[*What is and isn't evaluated*](../README.md#what-is-and-isnt-evaluated).
