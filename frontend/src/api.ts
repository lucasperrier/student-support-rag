export type ChatResponse = {
  answer: string;
  action?: string;
  sources?: unknown[];
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001";

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

export type Student = {
  id: number;
  student_id: string;
  first_name: string;
  last_name: string;
  email: string;
  program: string;
  year: number;
  created_at: string;
};

export async function fetchStudents(): Promise<Student[]> {
  const res = await fetch(`${API_URL}/api/admin/students`);
  await assertOk(res);
  return res.json();
}

export async function ingestUrl(url: string): Promise<{ status: string; saved_as: string }> {
  const res = await fetch(`${API_URL}/api/admin/ingest_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  await assertOk(res);
  return res.json();
}

/**
 * Triggers a backend re-index operation (POST /api/admin/reindex).
 */
export async function reindexNow(): Promise<{ status: string; indexed_files: number }> {
  const res = await fetch(`${API_URL}/api/admin/reindex`, { method: "POST" });
  await assertOk(res);
  return res.json();
}

export async function createStudent(payload: {
  student_id: string;
  first_name: string;
  last_name: string;
  email: string;
  program?: string;
  year?: number;
}): Promise<{ status: string; student: Student }> {
  const res = await fetch(`${API_URL}/api/admin/students`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      student_id: payload.student_id,
      first_name: payload.first_name,
      last_name: payload.last_name,
      email: payload.email,
      program: payload.program ?? "",
      year: payload.year ?? 0,
    }),
  });
  await assertOk(res);
  return res.json();
}