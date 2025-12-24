import { useEffect, useState } from "react";
import { fetchAdmin } from "../api";

export function AdminPage() {
  const [data, setData] = useState<{ leads: unknown[]; uploads: unknown[] } | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetchAdmin()
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <h2>Admin</h2>
      {err ? <div style={{ color: "crimson" }}>{err}</div> : null}
      {!data && !err ? <div>Loading...</div> : null}
      {data ? (
        <>
          <h3>Leads</h3>
          <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 10, overflow: "auto" }}>
            {JSON.stringify(data.leads, null, 2)}
          </pre>
          <h3>Uploads</h3>
          <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 10, overflow: "auto" }}>
            {JSON.stringify(data.uploads, null, 2)}
          </pre>
        </>
      ) : null}
    </div>
  );
}