<!-- ...existing code... -->

# ESILV Smart Assistant

Overview
--------
The ESILV Smart Assistant is an AI-powered chatbot for answering questions about ESILV programs, admissions, courses and academic information. It uses Retrieval-Augmented Generation (RAG) and multi-agent orchestration. Features include document ingestion, context-aware answers, structured form-filling, and a Streamlit front-end for chat, uploads and admin visualization.

Project structure
-----------------
project-esilv-assistant/
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── chat.py                  # Chat interface logic
│   ├── admin.py                 # Admin dashboard (collected leads)
│   ├── uploader.py              # Document upload → triggers reindexing
│   ├── config.py                # App-wide configuration
│
├── agents/
│   ├── orchestrator.py          # Chooses which agent handles each query
│   ├── retrieval_agent.py       # RAG agent — queries vector DB
│   ├── form_agent.py            # Collects name, email, interest
│   ├── faq_agent.py             # Optional: static FAQ answering
│   ├── utils.py                 # Helper functions for agent messaging
│
├── ingestion/
│   ├── loader.py                # Load PDFs, HTML, text, etc.
│   ├── text_cleaning.py         # Clean extracted text
│   ├── chunker.py               # Split documents into chunks
│   ├── embedder.py              # Embedding generation (Ollama or API)
│   ├── vector_store.py          # FAISS/Chroma wrapper
│   ├── pipeline.py              # Full ingestion → vector DB pipeline
│
├── ui/
│   ├── components.py            # Reusable Streamlit UI components
│
├── notebooks/
│   ├── evaluation.ipynb         # RAG tests and retrieval quality
│   ├── ingestion_tests.ipynb    # PDF/text extraction debugging
│   ├── retrieval_tests.ipynb    # Vector DB search debugging
│
├── data/
│   ├── raw/                     # Raw ESILV PDFs, HTML, docs
│   ├── processed/               # Cleaned text chunks
│   ├── vector_db/               # Embedding index files
│
├── docs/
│   ├── architecture-diagram.png
│   ├── agent-design.md
│   ├── system-overview.md
│
├── tests/
│   ├── test_agents.py
│   ├── test_ingestion.py
│   ├── test_rag.py
│
├── requirements.txt
├── README.md
└── .gitignore

To-Do (high-level)
------------------
Phase 1 — Setup
- Create repository & project structure
- Install dependencies
- Configure Ollama / Google AI access
- Create configuration file

Ingestion pipeline
- Implement loaders for PDF/HTML/text
- Implement text cleaning
- Implement chunking strategy
- Implement embedding generation
- Implement vector DB (FAISS/Chroma)
- Build ingestion pipeline and re-indexing
- Test ingestion with sample ESILV documents

Phase 3 — RAG / Retrieval
- Implement search function
- Implement retrieval agent
- Implement context-aware answer generation
- Add optional citations
- Evaluate retrieval quality (notebook)

Phase 4 — Agents & Orchestration
- Implement orchestrator agent
- Implement form-filling agent
- Implement FAQ agent
- Implement agent-to-agent communication
- Implement intent classification and routing
- Integrate retrieval agent

Phase 5 — Streamlit app
- Build chat UI and admin dashboard
- Build document upload interface and connect ingestion pipeline
- Store collected user info (JSON/CSV)
- Streamline demo and user flows

Phase 6 — Evaluation & Delivery
- Real query testing
- Latency & hallucination benchmarks
- Final report, slides and demo video

Work split (Person A / Person B)
--------------------------------
Person A — ingestion & retrieval_agent.py
Responsibilities
- Build ingestion pipeline (loader, cleaning, chunking)
- Create embeddings
- Implement vector DB and search
- Implement retrieval agent and RAG answer generation
- Evaluate retrieval quality and deliver notebooks & diagrams

Deliverables
- ingestion/pipeline.py
- ingestion/vector_store.py
- agents/retrieval_agent.py
- notebooks/evaluation.ipynb

API contract (stable functions)
- ingestion/vector_store.py
  def search(query: str, top_k: int = 5) -> List[Tuple[str, dict]]
    # returns list of (chunk_text, metadata)
- agents/retrieval_agent.py
  def answer(query: str) -> str
    # returns grounded answer using RAG

Person B — agents, UI, orchestration
Responsibilities
- Build orchestrator and form agent
- Build Streamlit UI (app/)
- Connect UI to ingestion & retrieval API
- Implement user data storage and admin views

Deliverables
- app/main.py, app/chat.py, app/admin.py, app/uploader.py
- agents/orchestrator.py, agents/form_agent.py
- Streamlit demo and diagrams

Integration notes
-----------------
- Keep the Person A API stable and minimal so Person B can call it directly from the orchestrator:
  if intent == "information_query":
      return retrieval_agent.answer(user_message)
- Use the vector_store.search(...) function as the standard retrieval primitive.

Quick start (development)
-------------------------
1. Create virtualenv and install deps:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Configure credentials in app/config.py (Ollama / Google AI / vector DB path).

3. Ingest sample documents:
   python -m ingestion.pipeline --data-dir data/raw --out data/vector_db

4. Run Streamlit app:
   streamlit run app/main.py

Testing
-------
- Unit tests under tests/ — run with pytest:
  pytest -q

Notes
-----
- This README summarizes the original project brief and corrects typos from the brief.
- Keep the retrieval API stable during parallel work to simplify integration between collaborators.

License
-------
Specify project license in LICENSE file.
