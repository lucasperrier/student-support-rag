import { useMemo, useState } from "react";
import { ChatPage } from "./pages/ChatPage";
import { UploadPage } from "./pages/UploadPage";
import { AdminPage } from "./pages/AdminPage";

import esilvLogo from "./assets/esilv-logo.png";

type Tab = "chat" | "upload" | "admin";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");

  const header = useMemo(() => {
    if (tab === "chat") {
      return { title: "Chat", desc: "Ask about ESILV programs, rules, dates, internships, mobility, and student life." };
    }
    if (tab === "upload") {
      return { title: "Upload documents", desc: "Add PDFs/TXT/HTML so answers can use your files as context." };
    }
    return { title: "Admin", desc: "View uploaded documents and collected contacts (demo)." };
  }, [tab]);

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <img src={esilvLogo} alt="ESILV logo" />
          <div>
            <div className="brand-title">ESILV Smart Assistant</div>
            <div className="brand-sub">RAG chatbot for ESILV info</div>
          </div>
        </div>

        <div className="nav" style={{ marginTop: 14 }}>
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
            Chat
          </button>
          <button className={tab === "upload" ? "active" : ""} onClick={() => setTab("upload")}>
            Upload
          </button>
          <button className={tab === "admin" ? "active" : ""} onClick={() => setTab("admin")}>
            Admin
          </button>
        </div>

        <div className="sidebar-help">
          <div style={{ fontWeight: 650, marginBottom: 6, color: "var(--text)" }}>How to use</div>
          <div>1) Ask your question in Chat.</div>
          <div>2) If answers are missing, upload documents.</div>
          <div>3) The assistant may ask for contact details.</div>
        </div>
      </aside>

      <main className="main">
        <div className="page-header">
          <div>
            <div className="page-title">{header.title}</div>
            <div className="page-desc">{header.desc}</div>
          </div>
        </div>

        {tab === "chat" ? <ChatPage /> : null}
        {tab === "upload" ? <UploadPage /> : null}
        {tab === "admin" ? <AdminPage /> : null}
      </main>
    </div>
  );
}