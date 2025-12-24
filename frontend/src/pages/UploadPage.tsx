import { useState } from "react";
import { uploadDocument } from "../api";

export function UploadPage() {
  const [status, setStatus] = useState<string>("");

  async function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus("Uploading...");
    try {
      const resp = await uploadDocument(file);
      setStatus(`Uploaded: ${resp.filename} (id: ${resp.doc_id})`);
    } catch (err) {
      setStatus(`Upload error: ${String(err)}`);
    } finally {
      e.target.value = "";
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <h2>Upload</h2>
      <p>Upload PDF / TXT / HTML. The backend stores it under <code>data/raw/</code>.</p>
      <input type="file" onChange={onPickFile} />
      {status ? <div style={{ marginTop: 12 }}>{status}</div> : null}
    </div>
  );
}