import { useEffect, useMemo, useRef, useState } from "react";
import { createLead, sendChat } from "../api";

type Message = { role: "user" | "assistant"; text: string };

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text:
        "Ask me anything about ESILV.\n\nTip: If I don’t have enough information, upload the relevant PDF in the Upload tab.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const [action, setAction] = useState<string | null>(null);
  const [leadName, setLeadName] = useState("");
  const [leadEmail, setLeadEmail] = useState("");
  const [leadInterest, setLeadInterest] = useState("");
  const [leadStatus, setLeadStatus] = useState<string>("");

  const endRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !isSending, [input, isSending]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isSending]);

  async function onSend() {
    const text = input.trim();
    if (!text || isSending) return;

    setIsSending(true);
    setAction(null);
    setLeadStatus("");

    setMessages((prev) => [...prev, { role: "user", text }]);
    setInput("");

    try {
      const resp = await sendChat(text);
      setMessages((prev) => [...prev, { role: "assistant", text: resp.answer ?? "No answer returned." }]);
      setAction(resp.action ?? null);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Service error: ${String(e)}\n\nCheck the backend is running on 127.0.0.1:8001.` },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function onSubmitLead() {
    const name = leadName.trim();
    const email = leadEmail.trim();
    if (!name || !email) return;

    setLeadStatus("");
    try {
      await createLead({ name, email, interest: leadInterest.trim() });
      setLeadStatus("Thanks — contact saved.");
    } catch (e) {
      setLeadStatus(`Failed to save contact: ${String(e)}`);
    }
  }

  return (
    <div className="card chat-wrap">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg-row ${m.role}`}>
            <div className="bubble">
              <div className="bubble-meta">{m.role === "user" ? "You" : "Assistant"}</div>
              {m.text}
            </div>
          </div>
        ))}

        {isSending ? (
          <div className="msg-row assistant">
            <div className="bubble">
              <div className="bubble-meta">Assistant</div>
              Thinking…
            </div>
          </div>
        ) : null}

        <div ref={endRef} />
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          onKeyDown={(e) => {
            if (e.key === "Enter") onSend();
          }}
        />
        <button className="btn" onClick={onSend} disabled={!canSend}>
          Send
        </button>
      </div>

      {action === "collect_lead" ? (
        <div className="notice" style={{ margin: 12 }}>
          <div style={{ fontWeight: 650, marginBottom: 8, color: "var(--text)" }}>Contact details</div>

          <div style={{ display: "grid", gap: 8, maxWidth: 520 }}>
            <input placeholder="Full name" value={leadName} onChange={(e) => setLeadName(e.target.value)} />
            <input placeholder="Email" value={leadEmail} onChange={(e) => setLeadEmail(e.target.value)} />
            <input
              placeholder="Interest (optional)"
              value={leadInterest}
              onChange={(e) => setLeadInterest(e.target.value)}
            />

            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn" onClick={onSubmitLead} disabled={!leadName.trim() || !leadEmail.trim()}>
                Submit
              </button>
              <button className="btn secondary" onClick={() => setAction(null)}>
                Not now
              </button>
            </div>

            {leadStatus ? <div className={leadStatus.startsWith("Failed") ? "error" : ""}>{leadStatus}</div> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}