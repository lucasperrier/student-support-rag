from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from agents.form_agent import FormAgent
from agents.faq_agent import FAQAgent
from agents.retrieval_agent import RetrievalAgent


class ChatRequest(BaseModel):
    message: str


def create_app() -> FastAPI:
    app = FastAPI(title="ESILV Smart Assistant API")

    # Keep permissive CORS for dev (React/Streamlit). Tighten for prod later.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Keep same storage layout as current app/main.py
    storage_dir = Path(__file__).parent.parent.joinpath("data")
    raw_dir = storage_dir.joinpath("raw")
    processed_dir = storage_dir.joinpath("processed")
    leads_path = storage_dir.joinpath("leads.json")
    index_path = storage_dir.joinpath("vector_index.json")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not leads_path.exists():
        leads_path.write_text("[]")
    if not index_path.exists():
        index_path.write_text("[]")

    # --- keep the existing tiny demo index helpers (unchanged behavior) ---
    def load_index():
        try:
            return json.loads(index_path.read_text())
        except Exception:
            return []

    def save_index(idx):
        index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2))

    def ingest_text(doc_id: str, text: str, meta: Dict[str, Any]):
        idx = load_index()
        idx.append({"id": doc_id, "text": text, "meta": meta})
        save_index(idx)

    # Instantiate agents (keep current wiring)
    vector_store_path = storage_dir.joinpath("vector_db")
    form_agent = FormAgent(name="form_agent", llm_client=None, leads_path=leads_path)
    faq_agent = FAQAgent(name="faq_agent", llm_client=None)
    retrieval_agent = RetrievalAgent(name="retrieval_agent", llm_client=None)

    agents = [faq_agent, form_agent, retrieval_agent]
    orchestrator = Orchestrator(
        name="orchestrator",
        llm_client=None,
        agents=agents,
        vector_store_path=str(vector_store_path),
    )

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        user_msg = req.message.strip()
        return orchestrator.process(user_msg)

    @app.post("/api/upload")
    async def upload_endpoint(file: UploadFile = File(...)):
        content = await file.read()
        filename = file.filename
        dst = raw_dir.joinpath(filename)
        dst.write_bytes(content)

        try:
            text = content.decode("utf-8")
        except Exception:
            text = f"[binary file saved: {filename}]"

        doc_id = f"doc-{int(time.time() * 1000)}"
        ingest_text(doc_id, text, {"filename": filename})
        return {"status": "ok", "filename": filename, "doc_id": doc_id}

    @app.get("/api/admin")
    async def admin_endpoint():
        leads = json.loads(leads_path.read_text())
        index = load_index()
        uploads = [{"id": it["id"], "meta": it.get("meta", {})} for it in index]
        return {"leads": leads, "uploads": uploads}

    @app.post("/api/admin/lead")
    async def create_lead(name: str = Form(...), email: str = Form(...), interest: str = Form("")):
        leads = json.loads(leads_path.read_text())
        lead = {
            "id": f"lead-{int(time.time() * 1000)}",
            "name": name,
            "email": email,
            "interest": interest,
        }
        leads.append(lead)
        leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2))
        return {"status": "ok", "lead": lead}

    return app


# uvicorn entrypoint: `uvicorn backend.main:app --reload --port 8001`
app = create_app()