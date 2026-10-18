// Small pieces every page uses.

import { useEffect, useState } from "react";
import { get } from "./api";

export function Problems({ problems }: { problems: string[] }) {
  if (problems.length === 0) return null;
  return (
    <ul className="problems">
      {problems.map((p) => (
        <li key={p}>{p}</li>
      ))}
    </ul>
  );
}

// The name of the patch a change of a source's files becomes.
export function PatchName({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <label className="inline">
      Patch name{" "}
      <input value={value} pattern="[A-Za-z0-9._\-]+" required onChange={(e) => onChange(e.target.value)}
        title="letters, digits, '.', '_' and '-': the file becomes NNNN-<name>.patch" />
    </label>
  );
}

export type SourceSummary = { id: string; type: string | null; decision: string | null; problems: string[] };

// A choice among the sources (only the empty ones, written for WPT, when onlyEmpty).
export function SourcePicker({ onlyEmpty, value, onChange }: { onlyEmpty: boolean; value: string; onChange: (id: string) => void }) {
  const [rows, setRows] = useState<SourceSummary[]>([]);
  useEffect(() => {
    get<SourceSummary[]>("/api/sources").then((a) => a.ok && setRows(a.data));
  }, []);
  const shown = rows.filter((r) => !onlyEmpty || r.type === "empty");
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="source">
      <option value="">{shown.length ? "choose a source" : "no source yet"}</option>
      {shown.map((r) => (
        <option key={r.id} value={r.id}>{r.id} ({r.decision})</option>
      ))}
    </select>
  );
}

// git's status of the data files: what is changed and not committed yet.
export function GitStatus() {
  const [status, setStatus] = useState<{ git: boolean; files: { path: string; state: string }[] } | null>(null);
  useEffect(() => {
    const load = () => get<typeof status>("/api/status").then((a) => a.ok && setStatus(a.data));
    load();
    window.addEventListener("wpt-saved", load);
    return () => window.removeEventListener("wpt-saved", load);
  }, []);
  if (!status) return null;
  if (!status.git) return <span className="hint">not a git working tree</span>;
  return (
    <details className="status">
      <summary>{status.files.length ? `${status.files.length} changed data files, not committed` : "data committed"}</summary>
      <ul>
        {status.files.map((f) => (
          <li key={f.path}><code>{f.state}</code> {f.path}</li>
        ))}
      </ul>
      <p className="hint">Commit them with git (signed off; a decision signed too).</p>
    </details>
  );
}
