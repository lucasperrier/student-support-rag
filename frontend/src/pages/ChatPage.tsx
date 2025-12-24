import { useMemo, useState } from "react";
import { createLead, sendChat } from "../api";

type Message = { role: "user" | "assistant"; text: string };

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Hello — ask me about ESILV programs, admissions, courses or upload documents for context." },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const [action, setAction] = useState<string | null>(null);
  const [leadName, setLeadName] = useState("");
  const [leadEmail, setLeadEmail] = useState("");
  const [leadInterest, setLeadInterest] = useState("");
  const [leadStatus, setLeadStatus] = useState("");

  const canSend = useMemo(() => input.trim().length > 0 && !isSending, [input, isSending]);

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
      setMessages((prev) => [...prev, { role: "assistant", text: `Service error: ${String(e)}` }]);
    } finally {
      setIsSending(false);
    }
  }

  async function onSubmitLead() {
    setLeadStatus("");
    try {
      await createLead({ name: leadName.trim(), email: leadEmail.trim(), interest: leadInterest.trim() });
      setLeadStatus("Thanks — contact saved.");
    } catch (e) {
      setLeadStatus(`Failed to save contact: ${String(e)}`);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <h2>Chat</h2>

      <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12, minHeight: 360 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ margin: "10px 0" }}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong> {m.text}
          </div>
        ))}
        {isSending ? <div style={{ opacity: 0.7 }}>Assistant is thinking…</div> : null}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Your question…"
          style={{ flex: 1, padding: 10 }}
          onKeyDown={(e) => {
            if (e.key === "Enter") onSend();
          }}
        />
        <button onClick={onSend} disabled={!canSend} style={{ padding: "10px 14px" }}>
          Send
        </button>
      </div>

      {action === "collect_lead" ? (
        <div style={{ marginTop: 16, border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
          <h3>Contact details</h3>
          <div style={{ display: "grid", gap: 8, maxWidth: 520 }}>
            <input placeholder="Full name" value={leadName} onChange={(e) => setLeadName(e.target.value)} />
            <input placeholder="Email" value={leadEmail} onChange={(e) => setLeadEmail(e.target.value)} />
            <input placeholder="Interest (optional)" value={leadInterest} onChange={(e) => setLeadInterest(e.target.value)} />
            <button onClick={onSubmitLead} disabled={!leadName.trim() || !leadEmail.trim()}>
              Submit
            </button>
            {leadStatus ? <div>{leadStatus}</div> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}