from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader


def url_to_filename(url: str) -> str:
    p = urlparse(url)
    safe_host = (p.netloc or "site").replace(":", "_")
    safe_path = (p.path or "index").strip("/").replace("/", "_")
    if not safe_path:
        safe_path = "index"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"web_{safe_host}_{safe_path}_{ts}.txt"


def fetch_website_to_raw(url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)

    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()  # list[Document]
    text = "\n\n".join(d.page_content for d in docs).strip()

    out_path = raw_dir / url_to_filename(url)
    out_path.write_text(text, encoding="utf-8")
    return out_path