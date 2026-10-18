// A source's final files (after its cuts and patches), and an editor for
// one of them. A save never writes a patch's text: the server unpacks the
// source, puts the file in, and records the difference as the next patch,
// kept only if the source then passes the rules of `wpt-dataset check --fetch`.

import { useEffect, useState } from "react";
import { get, part, put, saying } from "../api";
import { PatchName, Problems } from "../parts";

type FileInfo = { mode: string; size: number; format: string | null; tags: string[] };
type Listing = { version: string; files: Record<string, FileInfo> };

export function Files({ id, path }: { id: string; path?: string }) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [newPath, setNewPath] = useState("");

  const load = () =>
    get<Listing>(`/api/sources/${part(id)}/files`).then((a) => (a.ok ? setListing(a.data) : setProblems(a.problems)));
  useEffect(() => {
    load();
  }, [id]);

  if (path !== undefined)
    return <FileEditor id={id} path={path} known={listing?.files[path] !== undefined} />;
  return (
    <section>
      <p><a href={`#/sources/${part(id)}`}>{id}</a></p>
      <h1>Files of {id}</h1>
      <Problems problems={problems} />
      {listing && (
        <div className="table">
          <table>
            <thead><tr><th>Path</th><th>Format</th><th>Tags</th><th>Bytes</th></tr></thead>
            <tbody>
              {Object.entries(listing.files).map(([p, f]) => (
                <tr key={p}>
                  <td><a href={`#/sources/${part(id)}/files/${part(p)}`}>{p}</a></td>
                  <td>{f.format === "chat" ? <a href={`#/conversations/${part(id)}/${part(p)}`}>chat</a> : f.format ?? "none"}</td>
                  <td>{f.tags.join(", ")}</td>
                  <td>{f.size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <form className="filters" onSubmit={(e) => { e.preventDefault(); if (newPath) window.location.hash = `#/sources/${part(id)}/files/${part(newPath)}`; }}>
        <input placeholder="new file, such as texts/letter.md" value={newPath} onChange={(e) => setNewPath(e.target.value)} />
        <button type="submit" disabled={!newPath}>Add a file</button>
        <span className="hint">Its path needs a format entry in source.json.</span>
      </form>
    </section>
  );
}

function FileEditor({ id, path, known }: { id: string; path: string; known: boolean }) {
  const [version, setVersion] = useState<string | null>(null);
  const [original, setOriginal] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [binary, setBinary] = useState(false);
  const [patch, setPatch] = useState("edit");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");

  const load = () => {
    setMessage("");
    get<{ version: string; text?: string }>(`/api/sources/${part(id)}/files/${part(path)}`).then((a) => {
      if (a.ok) {
        setVersion(a.data.version);
        setBinary(a.data.text === undefined);
        setOriginal(a.data.text ?? null);
        setText(a.data.text ?? "");
        setProblems([]);
      } else if (a.status === 404) {
        get<Listing>(`/api/sources/${part(id)}/files`).then((l) => l.ok && setVersion(l.data.version));
        setOriginal(null);
      } else setProblems(a.problems);
    });
  };
  useEffect(load, [id, path, known]);

  const save = async (content: string | null) => {
    if (version === null) return;
    const a = await put<{ version: string }>(`/api/sources/${part(id)}/files/${part(path)}`, { text: content, patch, version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, content === null ? "Removed, as the next patch." : "Saved as the next patch."));
    if (a.ok) {
      setVersion(a.data.version);
      setOriginal(content);
      if (content === null) window.location.hash = `#/sources/${part(id)}/files`;
    }
  };

  return (
    <section>
      <p><a href={`#/sources/${part(id)}/files`}>Files of {id}</a></p>
      <h1>{path}</h1>
      <Problems problems={problems} />
      {binary ? <p>This file is not UTF-8 text; it cannot be edited here.</p> : (
        <form onSubmit={(e) => { e.preventDefault(); save(text); }}>
          <textarea className="page" value={text} rows={24} aria-label={path} onChange={(e) => setText(e.target.value)} />
          <div className="actions">
            <PatchName value={patch} onChange={setPatch} />
            <button type="submit" disabled={text === original || version === null}>{original === null ? "Add" : "Save"}</button>
            {original !== null && <button type="button" onClick={() => save(null)}>Remove the file</button>}
            <button type="button" onClick={load}>Reload</button>
            <span role="status">{message}</span>
          </div>
        </form>
      )}
    </section>
  );
}
