"""
Streamlit entry point.

This Streamlit app starts a small FastAPI server in a background thread (once per
Streamlit session) and renders three tabs:
- Chat: talks to /api/chat (agent orchestrator)
- Upload: uploads a file to /api/upload (demo persistence)
- Admin: reads /api/admin (leads + uploaded docs list)

Notes:
- The "real" RAG ingestion + FAISS index lives in ingestion/ and data/vector_db/index(.faiss).
- The /api/upload route stores raw files and also maintains a small JSON "stub index"
  (data/vector_index.json) used only for admin display; retrieval uses the FAISS index.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Ensure project root is importable when running "streamlit run app/main.py"
sys.path.append(str(Path(__file__).parent.parent))

import admin as admin_ui  # noqa: E402
import chat as chat_ui  # noqa: E402
import config  # noqa: E402
import uploader as upload_ui  # noqa: E402

from agents.faq_agent import FAQAgent  # noqa: E402
from agents.form_agent import FormAgent  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from agents.retrieval_agent import RetrievalAgent  # noqa: E402


# =============================================================================
# Paths / storage
# =============================================================================

@dataclass(frozen=True)
class StoragePaths:
    storage_dir: Path
    raw_dir: Path
    processed_dir: Path
    leads_path: Path
    stub_index_path: Path
    vector_store_path: Path  # base path, expects "{vector_store_path}.faiss"


def _get_storage_paths() -> StoragePaths:
    storage_dir = Path(__file__).parent.joinpath("..", "data").resolve()
    raw_dir = storage_dir.joinpath("raw")
    processed_dir = storage_dir.joinpath("processed")
    leads_path = storage_dir.joinpath("leads.json")
    stub_index_path = storage_dir.joinpath("vector_index.json")

    vector_store_dir = storage_dir.joinpath("vector_db")
    vector_store_path = vector_store_dir.joinpath("index")

    return StoragePaths(
        storage_dir=storage_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        leads_path=leads_path,
        stub_index_path=stub_index_path,
        vector_store_path=vector_store_path,
    )


def _ensure_storage(paths: StoragePaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    if not paths.leads_path.exists():
        paths.leads_path.write_text("[]", encoding="utf-8")

    if not paths.stub_index_path.exists():
        paths.stub_index_path.write_text("[]", encoding="utf-8")


# =============================================================================
# Demo stub index (used for upload/admin only)
# =============================================================================

def _load_stub_index(stub_index_path: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads(stub_index_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_stub_index(stub_index_path: Path, idx: List[Dict[str, Any]]) -> None:
    stub_index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


def _stub_ingest_text(stub_index_path: Path, doc_id: str, text: str, meta: Dict[str, Any]) -> None:
    idx = _load_stub_index(stub_index_path)
    idx.append({"id": doc_id, "text": text, "meta": meta})
    _save_stub_index(stub_index_path, idx)


# =============================================================================
# Agents
# =============================================================================

def _create_orchestrator(paths: StoragePaths) -> Orchestrator:
    """
    Wire up the orchestrator and agents.

    RetrievalAgent is configured to use the FAISS index at:
        {paths.vector_store_path}.faiss
    which matches the ingestion pipeline defaults (data/vector_db/index.faiss).
    """
    form_agent = FormAgent(name="form_agent", llm_client=None, leads_path=paths.leads_path)
    faq_agent = FAQAgent(name="faq_agent", llm_client=None)
    retrieval_agent = RetrievalAgent(
        name="retrieval_agent",
        llm_client=None,
        vector_store_path=str(paths.vector_store_path),
    )

    agents = [faq_agent, form_agent, retrieval_agent]
    return Orchestrator(
        name="orchestrator",
        llm_client=None,
        agents=agents,
        vector_store_path=str(paths.vector_store_path),  # accepted; currently unused by orchestrator
    )


# =============================================================================
# FastAPI bootstrap (runs in background thread)
# =============================================================================

def _build_fastapi_app(paths: StoragePaths):
    # Import here so importing app/main.py doesn't require FastAPI unless you run the app.
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="ESILV Smart Assistant API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    orchestrator = _create_orchestrator(paths)

    class ChatRequest(BaseModel):
        message: str

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        user_msg = req.message.strip()
        return orchestrator.process(user_msg)

    @app.post("/api/upload")
    async def upload_endpoint(file: UploadFile = File(...)):
        content = await file.read()
        filename = file.filename
        dst = paths.raw_dir.joinpath(filename)
        dst.write_bytes(content)

        try:
            text = content.decode("utf-8")
        except Exception:
            text = f"[binary file saved: {filename}]"

        doc_id = f"doc-{int(time.time() * 1000)}"
        _stub_ingest_text(paths.stub_index_path, doc_id, text, {"filename": filename})
        return {"status": "ok", "filename": filename, "doc_id": doc_id}

    @app.get("/api/admin")
    async def admin_endpoint():
        leads = json.loads(paths.leads_path.read_text(encoding="utf-8"))
        index = _load_stub_index(paths.stub_index_path)
        uploads = [{"id": it["id"], "meta": it.get("meta", {})} for it in index]
        return {"leads": leads, "uploads": uploads}

    @app.post("/api/admin/lead")
    async def create_lead(name: str = Form(...), email: str = Form(...), interest: str = Form("")):
        leads = json.loads(paths.leads_path.read_text(encoding="utf-8"))
        lead = {
            "id": f"lead-{int(time.time() * 1000)}",
            "name": name,
            "email": email,
            "interest": interest,
        }
        leads.append(lead)
        paths.leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "lead": lead}

    return app


def _start_api_server() -> None:
    try:
        import uvicorn

        paths = _get_storage_paths()
        _ensure_storage(paths)

        app = _build_fastapi_app(paths)
        uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="warning")
    except Exception as e:
        print(f"Error starting FastAPI server: {e}")
        import traceback
        traceback.print_exc()


def _ensure_api_started() -> None:
    """
    Start the FastAPI server once per Streamlit session.

    This uses Streamlit's session_state to avoid re-launching the server on reruns.
    """
    if st.session_state.get("api_started"):
        return

    st.session_state["api_started"] = True
    t = threading.Thread(target=_start_api_server, daemon=True)
    t.start()
    time.sleep(0.7)  # allow uvicorn to bind


# =============================================================================
# Streamlit UI
# =============================================================================

_ensure_api_started()

st.set_page_config(page_title="ESILV Smart Assistant", layout="wide")
st.title("ESILV Smart Assistant (Streamlit + FastAPI demo)")

tabs = st.tabs(["Chat", "Upload", "Admin"])
with tabs[0]:
    chat_ui.render_chat(api_url=f"{config.API_URL}/api/chat", create_lead_url=f"{config.API_URL}/api/admin/lead")
with tabs[1]:
    upload_ui.render_uploader(upload_url=f"{config.API_URL}/api/upload")
with tabs[2]:
    admin_ui.render_admin(admin_url=f"{config.API_URL}/api/admin")