import { useEffect, useMemo, useState } from "react";
import { get, part, put } from "../api";
import { Problems } from "../parts";
import { SchemaForm, type Schema } from "../schema-form";
import { Files } from "./files";
import type { Page } from "./index";

type Summary = {
  id: string;
  address: string | null;
  schema: number | null;
  type: string | null;
  decision: string | null;
  reason: string | null;
  licenses: string[];
  problems: string[];
};
type Detail = { id: string; document: Record<string, unknown> | null; version: string; problems: string[] };

const DECISIONS = ["refused", "admitted", "retired", "all"];

function List() {
  const [rows, setRows] = useState<Summary[] | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [decision, setDecision] = useState("refused");
  const [search, setSearch] = useState("");

  useEffect(() => {
    get<Summary[]>("/api/sources").then((a) => (a.ok ? setRows(a.data) : setProblems(a.problems)));
  }, []);

  const shown = useMemo(
    () =>
      (rows ?? []).filter(
        (r) =>
          (decision === "all" || r.decision === decision) &&
          (search === "" || `${r.id} ${r.address} ${r.licenses.join(" ")}`.toLowerCase().includes(search.toLowerCase())),
      ),
    [rows, decision, search],
  );

  return (
    <section>
      <h1>Sources</h1>
      <p><a href="#/new-source">New source</a>: write or paste your own text.</p>
      <Problems problems={problems} />
      <div className="filters">
        <label>
          Decision{" "}
          <select value={decision} onChange={(e) => setDecision(e.target.value)}>
            {DECISIONS.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <input placeholder="Search id, address, license" value={search} onChange={(e) => setSearch(e.target.value)} />
        <span className="hint">{rows ? `${shown.length} of ${rows.length}` : "loading"}</span>
      </div>
      <div className="table">
        <table>
          <thead>
            <tr><th>Source</th><th>Decision</th><th>Licenses</th><th>Problems</th><th>Address</th></tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.id}>
                <td><a href={`#/sources/${encodeURIComponent(r.id)}`}>{r.id}</a></td>
                <td><span className={`pill ${r.decision ?? ""}`}>{r.decision ?? "?"}</span> {r.reason}</td>
                <td>{r.licenses.join(", ")}</td>
                <td className={r.problems.length ? "bad" : "good"}>{r.problems.length || "none"}</td>
                <td>{r.address && /^https?:/.test(r.address) ? <a href={r.address} target="_blank" rel="noreferrer">{r.address}</a> : r.address}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Editor({ id }: { id: string }) {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [stale, setStale] = useState(false);

  const load = () => {
    setStale(false);
    setMessage("");
    get<Detail>(`/api/sources/${encodeURIComponent(id)}`).then((a) => {
      if (!a.ok) return setProblems(a.problems);
      setDetail(a.data);
      setDraft(a.data.document);
      setProblems(a.data.problems);
    });
  };
  useEffect(() => {
    get<Schema>("/api/schemas/source").then((a) => a.ok && setSchema(a.data));
    load();
  }, [id]);

  const changed = detail !== null && JSON.stringify(draft) !== JSON.stringify(detail.document);

  const save = async () => {
    if (!detail) return;
    const a = await put<{ version: string }>(`/api/sources/${encodeURIComponent(id)}`, { document: draft, version: detail.version });
    if (a.ok) {
      setDetail({ ...detail, document: draft, version: a.data.version, problems: [] });
      setProblems([]);
      setMessage("Saved: source.json passes the schema and the source rules.");
    } else {
      setProblems(a.problems);
      setStale(a.status === 409);
      setMessage(a.status === 409 ? "Not saved: the file changed on disk." : "Not saved.");
    }
  };

  const mirrors = (draft?.mirrors as { address?: string }[] | undefined) ?? [];
  return (
    <section>
      <p><a href="#/sources">All sources</a></p>
      <h1>{id}</h1>
      <p className="filters">
        <a href={`#/sources/${part(id)}/files`}>Files</a>
        <a href={`#/conversations/${part(id)}`}>Conversations</a>
        <a href={`#/tasks/${part(id)}`}>Tasks</a>
      </p>
      <p className="links">
        {[draft?.address as string | undefined, ...mirrors.map((m) => m.address)]
          .filter((a): a is string => !!a && /^https?:/.test(a))
          .map((a) => (
            <a key={a} href={a} target="_blank" rel="noreferrer">{a}</a>
          ))}
      </p>
      <Problems problems={problems} />
      {schema && draft && (
        <form onSubmit={(e) => { e.preventDefault(); save(); }}>
          <SchemaForm value={draft} schema={schema} root={schema} name="source"
            onChange={(v) => setDraft(v as Record<string, unknown>)} />
          <div className="actions">
            <button type="submit" disabled={!changed}>Save</button>
            <button type="button" disabled={!changed} onClick={() => setDraft(detail!.document)}>Undo changes</button>
            {stale && <button type="button" onClick={load}>Reload from disk</button>}
            <span role="status">{message}</span>
          </div>
        </form>
      )}
      {detail && detail.document === null && <p>source.json is not readable JSON; fix it in an editor.</p>}
    </section>
  );
}

function SourcesPage({ rest }: { rest: string[] }) {
  if (rest[0] && rest[1] === "files") return <Files id={rest[0]} path={rest[2]} />;
  return rest[0] ? <Editor id={rest[0]} /> : <List />;
}

export const sources: Page = { path: "sources", title: "Sources", Component: SourcesPage };
