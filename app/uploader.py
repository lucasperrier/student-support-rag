# File upload interface
import streamlit as st
import requests
from pathlib import Path

def render_uploader(upload_url: str):
    st.header("Upload documents (PDF / TXT / HTML)")
    st.info("Uploaded files are ingested into a simple demo index used by the RAG pipeline.")
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            if st.button(f"Upload '{f.name}'"):
                try:
                    files = {"file": (f.name, f.getvalue(), f.type)}
                    r = requests.post(upload_url, files=files, timeout=30)
                    if r.ok:
                        data = r.json()
                        st.success(f"Uploaded: {data.get('filename')} (id: {data.get('doc_id')})")
                    else:
                        st.error("Upload failed")
                except Exception as e:
                    st.error(f"Upload error: {e}")

    st.markdown("---")
    st.write("Tip: for better results upload ESILV PDFs or text documents that describe programs, admissions and courses.")