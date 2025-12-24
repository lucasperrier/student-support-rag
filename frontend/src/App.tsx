import { useState } from "react";
import { ChatPage } from "./pages/ChatPage";
import { UploadPage } from "./pages/UploadPage";
import { AdminPage } from "./pages/AdminPage";

type Tab = "chat" | "upload" | "admin";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div>
      <div style={{ display: "flex", gap: 8, padding: 12, borderBottom: "1px solid #eee" }}>
        <button onClick={() => setTab("chat")}>Chat</button>
        <button onClick={() => setTab("upload")}>Upload</button>
        <button onClick={() => setTab("admin")}>Admin</button>
      </div>

      {tab === "chat" ? <ChatPage /> : null}
      {tab === "upload" ? <UploadPage /> : null}
      {tab === "admin" ? <AdminPage /> : null}
    </div>
  );
}