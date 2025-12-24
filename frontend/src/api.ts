export type ChatResponse = {
  answer: string;
  action?: string;
  sources?: unknown[];
};

const API_URL = import.meta.env.VITE_API_URL as string;

async function assertOk(res: Response) {
  if (res.ok) return;
  const text = await res.text().catch(() => "");
  throw new Error(`${res.status} ${res.statusText}${text ? ` - ${text}` : ""}`);
}

export async function sendChat(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  await assertOk(res);
  return res.json();
}

export async function uploadDocument(file: File): Promise<{ status: string; filename: string; doc_id: string }> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: form,
  });
  await assertOk(res);
  return res.json();
}

export async function fetchAdmin(): Promise<{ leads: unknown[]; uploads: unknown[] }> {
  const res = await fetch(`${API_URL}/api/admin`);
  await assertOk(res);
  return res.json();
}

export async function createLead(payload: { name: string; email: string; interest?: string }) {
  const body = new URLSearchParams();
  body.set("name", payload.name);
  body.set("email", payload.email);
  body.set("interest", payload.interest ?? "");

  const res = await fetch(`${API_URL}/api/admin/lead`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
    body,
  });
  await assertOk(res);
  return res.json();
}