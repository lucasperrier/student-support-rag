# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESILV Smart Assistant is an AI-powered chatbot for answering questions about ESILV programs, admissions, and academic information. It uses Retrieval-Augmented Generation (RAG) with multi-agent orchestration. The system combines a Streamlit frontend with a FastAPI backend embedded in the same process.

## Working Context - Person A

**IMPORTANT**: The user is **Person A**, responsible for ingestion pipeline and retrieval agent.

### Development Guidelines:

1. **Focus on Person A's Work**:
   - Ingestion pipeline (`ingestion/` directory)
   - Retrieval agent (`agents/retrieval_agent.py`)
   - Vector store implementation
   - Evaluation notebooks

2. **Person B's Code**:
   - Do NOT modify Person B's code (orchestrator, form agent, FAQ agent, UI) unless absolutely necessary
   - **Person B's files**: `agents/orchestrator.py`, `agents/form_agent.py`, `agents/faq_agent.py`, `app/*`, `ui/*`
   - If modification is required, document it in `CODE_INTERFERENCE.md` with:
     - What was changed
     - Why it was necessary
     - Impact on Person B's work
   - Check `CODE_INTERFERENCE.md` before starting work to see if any Person B code was modified

3. **Code Documentation Standards**:
   - Add comprehensive file header comments explaining:
     - Purpose of the file
     - Expected inputs
     - Outputs/API that Person B will consume
   - Comment all non-trivial logic clearly
   - Explain reasoning behind implementation choices

   **File Header Template**:
   ```python
   """
   [Module Name]

   PURPOSE:
   [What this file does and why it exists]

   INPUTS:
   - [What data/files this module expects]
   - [Configuration requirements]
   - [Dependencies]

   OUTPUTS:
   - [What this module produces]
   - [Files written]
   - [API functions that Person B will use]

   PERSON B INTEGRATION:
   [Key functions/classes Person B should use and how to use them]

   EXAMPLE USAGE:
   [Simple code example showing how to use this module]
   """
   ```

4. **Educational Approach**:
   - Explain every move and reasoning before implementing
   - Help Person A understand the architectural decisions
   - Provide context for why certain patterns are used

### Person A's Responsibilities & Files:

**Ingestion Pipeline** (`ingestion/` directory):
- `loader.py` - Load PDFs, HTML, text files
- `text_cleaning.py` - Preprocess and clean extracted text
- `chunker.py` - Split documents into retrieval chunks
- `embedder.py` - Generate embeddings (Ollama/sentence-transformers)
- `vector_store.py` - FAISS/Chroma wrapper with `search()` function
- `pipeline.py` - Orchestrate full ingestion flow

**Retrieval Agent**:
- `agents/retrieval_agent.py` - Implement RAG answer generation

**Evaluation**:
- `notebooks/evaluation.ipynb` - RAG quality testing
- `notebooks/retrieval_tests.ipynb` - Vector search debugging
- `notebooks/ingestion_tests.ipynb` - PDF/text extraction testing

**Critical API Contracts** (must be stable for Person B):
```python
# ingestion/vector_store.py
def search(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    """Returns list of (chunk_text, metadata_dict)"""

# agents/retrieval_agent.py
def answer(query: str) -> str:
    """Returns grounded RAG answer"""
```

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit app (also starts FastAPI server in background thread)
streamlit run app/main.py

# The FastAPI server runs at http://127.0.0.1:8000
# The Streamlit UI runs at the default Streamlit port
```

### Testing
```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/test_agents.py -v
pytest tests/test_ingestion.py -v
pytest tests/test_rag.py -v
```

### Document Ingestion
```bash
# Ingest documents (when Person A implements pipeline.py)
python -m ingestion.pipeline --data-dir data/raw --out data/vector_db
```

## Architecture

### Multi-Agent System

The application uses an **orchestrator pattern** for routing queries to specialized agents:

1. **Orchestrator** (`agents/orchestrator.py`):
   - Entry point for all queries
   - Uses Ollama LLM for intent classification
   - Routes to appropriate agent based on query type
   - Maintains registry of available agents

2. **Retrieval Agent** (`agents/retrieval_agent.py`):
   - Handles information queries about ESILV
   - Uses RAG pattern: retrieves relevant chunks from vector DB, generates grounded answers
   - **Currently a stub** - Person A responsible for implementation
   - Must implement `answer(query: str) -> str` method

3. **Form Agent** (`agents/form_agent.py`):
   - Collects lead information (name, email, interest)
   - Extracts structured data from user messages
   - Persists leads to `data/leads.json`
   - Prompts for missing required fields

4. **FAQ Agent** (`agents/faq_agent.py`):
   - Handles static frequently asked questions
   - Quick responses without vector DB lookup

### BaseAgent Pattern

All agents inherit from `BaseAgent` (`agents/utils.py`) which provides:
- Memory management (persisted to JSON)
- Tool execution framework
- LLM integration via Ollama
- Abstract `process(query, context)` method that all agents must implement
- Abstract `get_system_prompt()` for agent-specific instructions

### Application Architecture

**app/main.py** is the entry point and does several things:
1. Starts FastAPI server in a background daemon thread
2. Defines REST API endpoints (`/api/chat`, `/api/upload`, `/api/admin`)
3. Instantiates agent system (orchestrator + agents)
4. Renders Streamlit multi-tab UI (Chat, Upload, Admin)

**Key insight**: The FastAPI server runs embedded in the same process as Streamlit using threading, not as a separate service.

### Data Flow

```
User Query (Streamlit UI)
  ↓
POST /api/chat
  ↓
Orchestrator.process()
  ↓
Orchestrator.classify_query() → determines agent
  ↓
[RetrievalAgent | FormAgent | FAQAgent].process()
  ↓
Response returned to UI
```

### Ingestion Pipeline (Person A's Responsibility)

The ingestion pipeline is **not yet fully implemented**. Expected architecture:

1. **Loader** (`ingestion/loader.py`): Load PDFs, HTML, text files
2. **Text Cleaning** (`ingestion/text_cleaning.py`): Preprocess and clean extracted text
3. **Chunker** (`ingestion/chunker.py`): Split documents into retrieval chunks
4. **Embedder** (`ingestion/embedder.py`): Generate embeddings (Ollama or sentence-transformers)
5. **Vector Store** (`ingestion/vector_store.py`): FAISS/ChromaDB wrapper with `search(query, top_k)` function
6. **Pipeline** (`ingestion/pipeline.py`): Orchestrates the full ingestion flow

### Critical API Contracts

These interfaces must remain stable for Person A/Person B integration:

```python
# ingestion/vector_store.py
def search(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    """Returns list of (chunk_text, metadata_dict)"""

# agents/retrieval_agent.py
def answer(query: str) -> str:
    """Returns grounded RAG answer"""
```

## Configuration

**app/config.py** contains global configuration:
- `API_HOST` and `API_PORT`: FastAPI server location (default: 127.0.0.1:8000)
- `DATA_DIR`: Root data directory
- `UPLOAD_DIR`: Where uploaded documents are saved

For LLM configuration, agents use Ollama directly via the `ollama` Python package. The default model is `llama2`.

## Data Storage

- `data/raw/`: Original uploaded documents
- `data/processed/`: Cleaned text chunks (Person A's output)
- `data/vector_db/`: Vector embeddings and index files
- `data/leads.json`: Collected user lead information
- `data/vector_index.json`: Simple demo index used by FastAPI endpoints (stub implementation)

## Development Workflow

### Person A (Ingestion & Retrieval)
- Implement ingestion pipeline modules
- Implement vector store search
- Implement retrieval agent's `answer()` method
- Evaluation notebooks in `notebooks/`

### Person B (Agents, UI, Orchestration)
- Implement orchestrator routing logic
- Implement form agent
- Build Streamlit UI components
- Connect UI to ingestion pipeline

### Parallel Development
- Person B can develop using stub implementations
- The orchestrator already instantiates `RetrievalAgent` even when stub
- Integration point: When Person A implements `vector_store.search()` and `retrieval_agent.answer()`, they will be automatically called

## LLM Usage

The codebase uses **Ollama** for local LLM inference:
- Agents call `self.generate_response(prompt, context, model)` from BaseAgent
- Default model: `llama2`
- Timeout: 120 seconds
- Used for: intent classification, answer generation, form field extraction

## UI Structure

Streamlit UI has three tabs:
1. **Chat** (`ui/components.py`, `app/chat.py`): Main conversational interface
2. **Upload** (`app/uploader.py`): Document upload for re-indexing
3. **Admin** (`app/admin.py`): View collected leads and uploaded documents

## Important Notes

- The FastAPI server starts automatically when `app/main.py` runs - do not start it separately
- Agents log actions to stdout with `[agent_name] action` format
- The current vector search in FastAPI endpoints is a simple keyword-matching stub; proper RAG requires Person A's implementation
- Memory is persisted per-agent to `data/{agent_name}_memory.json`
