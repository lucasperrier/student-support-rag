from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlmodel import Session, select

from backend.db import Student, get_engine, init_db
from backend.web_ingest import fetch_website_to_raw

from agents.orchestrator import Orchestrator
from agents.form_agent import FormAgent
from agents.faq_agent import FAQAgent
from agents.retrieval_agent import RetrievalAgent


class ChatRequest(BaseModel):
    message: str

class StudentCreate(BaseModel):
    student_id: str
    first_name: str
    last_name: str
    email: str
    program: str = ""
    year: int = 0

class WebIngestRequest(BaseModel):
    url: str

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

    # --- Student DB (SQLite file under data/) ---
    db_path = storage_dir.joinpath("students.db")
    engine = get_engine(str(db_path))
    init_db(engine)

    @app.get("/api/admin/students")
    async def list_students() -> List[Student]:
        with Session(engine) as session:
            rows = session.exec(select(Student).order_by(Student.created_at.desc())).all()
            return rows

    @app.post("/api/admin/students")
    async def create_student(payload: StudentCreate) -> Dict[str, Any]:
        student = Student(
            student_id=payload.student_id.strip(),
            first_name=payload.first_name.strip(),
            last_name=payload.last_name.strip(),
            email=payload.email.strip(),
            program=payload.program.strip(),
            year=int(payload.year),
        )
        with Session(engine) as session:
            session.add(student)
            session.commit()
            session.refresh(student)
        return {"status": "ok", "student": student}

    @app.post("/api/admin/ingest_url")
    async def ingest_url(req: WebIngestRequest):
        url = req.url.strip()
        saved_path = fetch_website_to_raw(url, raw_dir)
        return {"status": "ok", "saved_as": saved_path.name}
    
    @app.post("/api/admin/reindex")
    async def reindex():
        """
        Rebuild vector DB from files in data/raw using existing ingestion pipeline.
        Keeps behavior consistent with the CLI: `python -m ingestion.pipeline --data-dir data/raw --out data/vector_db`.
        """
        project_root = Path(__file__).parent.parent
        data_dir = project_root.joinpath("data", "raw")
        out_dir = project_root.joinpath("data", "vector_db")

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "ingestion.pipeline",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out_dir),
        ]
        subprocess.run(cmd, cwd=str(project_root), check=True)

        indexed_files = len([p for p in data_dir.iterdir() if p.is_file()])
        return {"status": "ok", "indexed_files": indexed_files}

    return app


# uvicorn entrypoint: `uvicorn backend.main:app --reload --port 8001`
app = create_app()