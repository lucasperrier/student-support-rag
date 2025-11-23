# Entry point for Streamlit app
import threading
import time
import os
import json
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add project root
import streamlit as st
import requests
from app import chat as chat_ui  # Now absolute
from app import uploader as upload_ui
from app import admin as admin_ui
from app import config

from agents.orchestrator import Orchestrator
from agents.form_agent import FormAgent
from agents.retrieval_agent import RetrievalAgent  # Stub
from agents.faq_agent import FAQAgent


# Start FastAPI server in background (only once)
if "api_started" not in st.session_state:
    st.session_state["api_started"] = True

    def _start_api():
        try:
            # ...existing code (the entire _start_api function body)...


            # import here to avoid importing FastAPI/uvicorn during static analysis
            from fastapi import FastAPI, UploadFile, File, Form
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
            from typing import List, Dict, Any
            import uvicorn

            app = FastAPI(title="ESILV Smart Assistant API")

            # allow Streamlit frontend to call API
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # simple in-memory stores (persist to disk for demo)
            storage_dir = Path(__file__).parent.joinpath("..", "data")
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

            # --- very small demo retrieval / RAG pipeline (stubbed) ---
            # index: list of {"id":str, "text":str, "meta":{...}}
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

            def simple_search(query: str, top_k: int = 3):
                idx = load_index()
                q_words = set(query.lower().split())
                scored = []
                for item in idx:
                    words = set(item["text"].lower().split())
                    score = len(q_words & words)
                    scored.append((score, item))
                scored.sort(reverse=True, key=lambda x: x[0])
                results = [it for s, it in scored if s > 0][:top_k]
                # if nothing matches, return top-k most recent
                if not results:
                    results = [it for s, it in scored][:top_k]
                return results

            # Instantiate agents (use stubs for Person A)
            vector_store_path = storage_dir.joinpath("vector_db")  # Person A configures this
            # retrieval_agent = RetrievalAgent(name="retrieval_agent", llm_client=None, vector_store_path=str(vector_store_path))
            form_agent = FormAgent(name="form_agent", llm_client=None, leads_path=leads_path)
            faq_agent = FAQAgent(name="faq_agent", llm_client=None)  # Removed leads_path; uses default FAQs
            # agents = [faq_agent, form_agent]
            agents = [faq_agent, form_agent]



            # orchestrator: very simple rule-based
            class ChatRequest(BaseModel):
                message: str

            orchestrator = Orchestrator(name="orchestrator", llm_client=None, agents=agents, vector_store_path=str(vector_store_path))  # Added vector_store_path and agents list

            @app.post("/api/chat")
            async def chat_endpoint(req: ChatRequest):
                user_msg = req.message.strip()
                response = orchestrator.process(user_msg)
                return response

            # @app.post("/api/chat")
            # async def chat_endpoint(req: ChatRequest):
            #     user_msg = req.message.strip()
            #     # form agent trigger: look for intent words
            #     form_triggers = ["name", "email", "contact", "apply", "admission"]
            #     if any(t in user_msg.lower() for t in form_triggers):
            #         # ask for structured info (form agent)
            #         resp = {
            #             "answer": "I can help with admissions and collect contact details. Please provide your full name and email.",
            #             "sources": [],
            #             "action": "collect_lead",
            #         }
            #         return resp

                # else use retrieval agent (RAG)
                retrieved = simple_search(user_msg, top_k=3)
                if not retrieved:
                    answer = "Sorry — I couldn't find relevant documents. Try rephrasing or upload documents."
                    return {"answer": answer, "sources": [], "action": "answer"}
                # build a grounded answer (simulated LLM)
                snippets = []
                sources = []
                for r in retrieved:
                    snippets.append(r["text"][:400])
                    sources.append({"id": r["id"], "meta": r.get("meta", {})})
                answer = (
                    "Simulated RAG answer: based on the following documents I found:\n\n"
                    + "\n\n---\n\n".join(snippets)
                    + "\n\nIf you need a concise summary ask me to summarize."
                )
                return {"answer": answer, "sources": sources, "action": "answer"}

            @app.post("/api/upload")
            async def upload_endpoint(file: UploadFile = File(...)):
                # save file and do a trivial ingestion (store content or filename)
                content = await file.read()
                filename = file.filename
                dst = raw_dir.joinpath(filename)
                dst.write_bytes(content)
                # attempt to decode text for simple ingestion
                try:
                    text = content.decode("utf-8")
                except Exception:
                    text = f"[binary file saved: {filename}]"
                doc_id = f"doc-{int(time.time()*1000)}"
                ingest_text(doc_id, text, {"filename": filename})
                return {"status": "ok", "filename": filename, "doc_id": doc_id}
            # @app.post("/api/upload")
            # async def upload_endpoint(file: UploadFile = File(...)):
            #     # ...existing code (save file)...
            #     # After saving, trigger ingestion
            #     pipeline.ingest_file(dst, doc_id)  # Assume pipeline has an ingest_file method; adjust based on implementation
            #     return {"status": "ok", "filename": filename, "doc_id": doc_id}

            @app.get("/api/admin")
            async def admin_endpoint():
                leads = json.loads(leads_path.read_text())
                index = load_index()
                uploads = [{"id": it["id"], "meta": it.get("meta", {})} for it in index]
                return {"leads": leads, "uploads": uploads}

            @app.post("/api/admin/lead")
            async def create_lead(name: str = Form(...), email: str = Form(...), interest: str = Form("")):
                leads = json.loads(leads_path.read_text())
                lead = {"id": f"lead-{int(time.time()*1000)}", "name": name, "email": email, "interest": interest}
                leads.append(lead)
                leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2))
                return {"status": "ok", "lead": lead}

            # Run uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

        except Exception as e:
            print(f"Error starting FastAPI server: {e}")
            import traceback
            traceback.print_exc()
        
    t = threading.Thread(target=_start_api, daemon=True)
    t.start()
    # basic wait for server to be usable
    time.sleep(0.7)

# Streamlit UI — top-level layout
st.set_page_config(page_title="ESILV Smart Assistant", layout="wide")
st.title("ESILV Smart Assistant (Streamlit + FastAPI demo)")

tabs = st.tabs(["Chat", "Upload", "Admin"])
with tabs[0]:
    chat_ui.render_chat(api_url=f"{config.API_URL}/api/chat", create_lead_url=f"{config.API_URL}/api/admin/lead")
with tabs[1]:
    upload_ui.render_uploader(upload_url=f"{config.API_URL}/api/upload")
with tabs[2]:
    admin_ui.render_admin(admin_url=f"{config.API_URL}/api/admin")