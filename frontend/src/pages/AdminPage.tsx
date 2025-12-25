import { useEffect, useMemo, useState } from "react";
import { createStudent, fetchAdmin, fetchStudents, ingestUrl, reindexNow } from "../api";
import type { Student } from "../api";

type AdminData = { leads: unknown[]; uploads: unknown[] };

export function AdminPage() {
  const [admin, setAdmin] = useState<AdminData | null>(null);
  const [students, setStudents] = useState<Student[] | null>(null);
  const [err, setErr] = useState<string>("");

  const [url, setUrl] = useState("");
  const [ingestStatus, setIngestStatus] = useState<string>("");
  const [reindexStatus, setReindexStatus] = useState<string>("");
  const [busy, setBusy] = useState(false);

  // --- Student creation form state ---
  const [studentId, setStudentId] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [studentEmail, setStudentEmail] = useState("");
  const [program, setProgram] = useState("");
  const [year, setYear] = useState<string>("0");
  const [studentCreateStatus, setStudentCreateStatus] = useState<string>("");

  const canCreateStudent = useMemo(() => {
    return (
      studentId.trim().length > 0 &&
      firstName.trim().length > 0 &&
      lastName.trim().length > 0 &&
      studentEmail.trim().length > 0 &&
      !busy
    );
  }, [studentId, firstName, lastName, studentEmail, busy]);

  async function refreshAll() {
    setErr("");
    try {
      const [a, s] = await Promise.all([fetchAdmin(), fetchStudents()]);
      setAdmin(a);
      setStudents(s);
    } catch (e) {
      setErr(String(e));
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const studentCount = useMemo(() => (students ? students.length : 0), [students]);

  async function onIngestUrl() {
    const clean = url.trim();
    if (!clean) return;

    setBusy(true);
    setIngestStatus("");
    try {
      const resp = await ingestUrl(clean);
      setIngestStatus(`Saved: ${resp.saved_as}. (Now click “Re-index now” to include it in retrieval.)`);
      setUrl("");
    } catch (e) {
      setIngestStatus(`Error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function onReindex() {
    setBusy(true);
    setReindexStatus("");
    try {
      const resp = await reindexNow();
      setReindexStatus(`Re-index complete. Indexed files: ${resp.indexed_files}`);
      await refreshAll();
    } catch (e) {
      setReindexStatus(`Error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function onCreateStudent() {
    if (!canCreateStudent) return;

    setBusy(true);
    setStudentCreateStatus("");

    try {
      const payload = {
        student_id: studentId.trim(),
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        email: studentEmail.trim(),
        program: program.trim(),
        year: Number.isFinite(Number(year)) ? Number(year) : 0,
      };

      await createStudent(payload);

      setStudentCreateStatus("Student created.");
      setStudentId("");
      setFirstName("");
      setLastName("");
      setStudentEmail("");
      setProgram("");
      setYear("0");

      // Refresh students list
      const s = await fetchStudents();
      setStudents(s);
    } catch (e) {
      setStudentCreateStatus(`Error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card" style={{ padding: 14 }}>
      {err ? (
        <div className="error" style={{ marginBottom: 10 }}>
          {err}
        </div>
      ) : null}

      <div style={{ display: "grid", gap: 14 }}>
        {/* Actions */}
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ fontWeight: 750, fontSize: 16 }}>Actions</div>

          <div className="notice">
            <div style={{ fontWeight: 650, marginBottom: 6, color: "var(--text)" }}>Add ESILV web page to knowledge</div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input
                style={{ flex: 1 }}
                placeholder="https://www.esilv.fr/..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
              <button className="btn" onClick={onIngestUrl} disabled={busy || !url.trim()}>
                Ingest URL
              </button>
            </div>
            {ingestStatus ? <div style={{ marginTop: 8 }}>{ingestStatus}</div> : null}
          </div>

          <div className="notice">
            <div style={{ display: "flex", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <div style={{ fontWeight: 650, color: "var(--text)" }}>Re-index now</div>
                <div style={{ color: "var(--muted)", fontSize: 13 }}>
                  Runs ingestion on <code>data/raw/</code> and updates the vector database used by RAG.
                </div>
              </div>
              <button className="btn" onClick={onReindex} disabled={busy}>
                Re-index now
              </button>
            </div>
            {reindexStatus ? <div style={{ marginTop: 8 }}>{reindexStatus}</div> : null}
          </div>
        </div>
        {/* Students */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ fontWeight: 750, fontSize: 16 }}>Student registry</div>
            <div style={{ color: "var(--muted)", fontSize: 13 }}>
              Total: {students ? studentCount : "Loading…"}
            </div>
          </div>

          {/* Student creation form */}
          <div className="notice" style={{ marginTop: 10 }}>
            <div style={{ fontWeight: 650, marginBottom: 8, color: "var(--text)" }}>Create a student</div>

            <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
              <input
                placeholder="Student ID (e.g. ESILV2025-001)"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
              />
              <input placeholder="Email" value={studentEmail} onChange={(e) => setStudentEmail(e.target.value)} />

              <input placeholder="First name" value={firstName} onChange={(e) => setFirstName(e.target.value)} />
              <input placeholder="Last name" value={lastName} onChange={(e) => setLastName(e.target.value)} />

              <input placeholder="Program (optional)" value={program} onChange={(e) => setProgram(e.target.value)} />
              <input
                placeholder="Year (number)"
                value={year}
                onChange={(e) => setYear(e.target.value)}
              />
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 10, alignItems: "center" }}>
              <button className="btn" onClick={onCreateStudent} disabled={!canCreateStudent}>
                Create student
              </button>
              {studentCreateStatus ? (
                <div className={studentCreateStatus.startsWith("Error:") ? "error" : ""}>{studentCreateStatus}</div>
              ) : null}
            </div>
          </div>

          {/* Students table */}
          {!students ? (
            <div className="notice" style={{ marginTop: 10 }}>
              Loading students…
            </div>
          ) : students.length === 0 ? (
            <div className="notice" style={{ marginTop: 10 }}>
              No students in the database yet.
            </div>
          ) : (
            <div style={{ overflow: "auto", marginTop: 10, border: "1px solid var(--border)", borderRadius: 12 }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", background: "rgba(255,255,255,0.04)" }}>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Student ID</th>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Name</th>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Email</th>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Program</th>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Year</th>
                    <th style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {students.map((s) => (
                    <tr key={s.id}>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>{s.student_id}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>
                        {s.first_name} {s.last_name}
                      </td>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>{s.email}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>{s.program}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>{s.year}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid var(--border)" }}>
                        {String(s.created_at).slice(0, 19).replace("T", " ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Existing admin data (leads/uploads) */}
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ fontWeight: 750, fontSize: 16 }}>Leads</div>
          <pre style={{ margin: 0, padding: 12, borderRadius: 12, border: "1px solid var(--border)", overflow: "auto" }}>
            {admin ? JSON.stringify(admin.leads, null, 2) : "Loading…"}
          </pre>

          <div style={{ fontWeight: 750, fontSize: 16 }}>Uploads (demo index)</div>
          <pre style={{ margin: 0, padding: 12, borderRadius: 12, border: "1px solid var(--border)", overflow: "auto" }}>
            {admin ? JSON.stringify(admin.uploads, null, 2) : "Loading…"}
          </pre>
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn secondary" onClick={refreshAll} disabled={busy}>
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}