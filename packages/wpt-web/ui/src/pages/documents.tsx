// The data files outside the sources: recipes, the Dockerfiles of the
// images of agentic tasks, and the signer lists. A JSON kind is edited with
// a form built from its JSON schema; a text kind as text. Generated files
// (tokenizer locks, image locks and license lists) are shown read-only: only
// the command that makes them writes them. Every save goes through the rules
// of `wpt-dataset check` and refuses a file changed on disk since it was read.

import { useEffect, useState } from "react";
import { get, part, put, saying } from "../api";
import { Problems } from "../parts";
import { SchemaForm, initial, type Schema } from "../schema-form";
import type { Page } from "./index";

type Kind = { name: string; form: string; description: string; schema: string | null; made_by: string | null; single: string | null; names: string[] };
type Found = { text: string; document: unknown; version: string; problems: string[] };

function Listing({ page, kinds }: { page: string; kinds: Kind[] }) {
  const [names, setNames] = useState<Record<string, string>>({});
  return (
    <>
      {kinds.map((kind) => (
        <div key={kind.name}>
          <h2>{kind.name} <span className="hint">{kind.description}{kind.made_by ? `, written by ${kind.made_by}` : ""}</span></h2>
          <ul>
            {kind.names.map((n) => <li key={n}><a href={`#/${page}/${kind.name}/${part(n)}`}>{n}</a></li>)}
            {kind.names.length === 0 && <li className="hint">none yet</li>}
          </ul>
          {!kind.made_by && (kind.single ? kind.names.length === 0 : true) && (
            <form className="filters" onSubmit={(e) => {
              e.preventDefault();
              const name = kind.single ?? names[kind.name];
              if (name) window.location.hash = `#/${page}/${kind.name}/${part(name)}`;
            }}>
              {!kind.single && <input placeholder="new name, lowercase a-z 0-9 . _ -" aria-label={`new ${kind.name}`}
                value={names[kind.name] ?? ""} onChange={(e) => setNames({ ...names, [kind.name]: e.target.value })} />}
              <button type="submit">New {kind.name}</button>
            </form>
          )}
        </div>
      ))}
    </>
  );
}

function Editor({ page, kind, name }: { page: string; kind: Kind; name: string }) {
  const [found, setFound] = useState<Found | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [document, setDocument] = useState<unknown>(null);
  const [text, setText] = useState("");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const json = kind.form === "json";
  const generated = kind.made_by !== null;

  const load = () => {
    get<Found>(`/api/documents/${kind.name}/${part(name)}`).then((a) => {
      if (a.ok) {
        setFound(a.data);
        setDocument(a.data.document);
        setText(a.data.text);
        setProblems(a.data.problems);
      } else if (a.status === 404 && !generated) {
        setFound({ text: "", document: null, version: "", problems: [] });
        setDocument(schema ? initial(schema, schema) : {});
        setText("");
        setProblems([]);
      } else setProblems(a.problems);
    });
  };
  useEffect(() => {
    if (json && kind.schema) get<Schema>(`/api/schemas/${kind.name}`).then((a) => a.ok && setSchema(a.data));
  }, [kind.name]);
  useEffect(load, [kind.name, name, schema === null]);

  const save = async () => {
    if (!found) return;
    const a = await put<{ version: string }>(`/api/documents/${kind.name}/${part(name)}`,
      json ? { document, version: found.version } : { text, version: found.version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, "Saved."));
    if (a.ok) load();
  };

  const changed = found !== null && (json ? JSON.stringify(document) !== JSON.stringify(found.document) : text !== found.text);
  return (
    <section>
      <p><a href={`#/${page}`}>{kind.name}</a></p>
      <h1>{name} {found?.version === "" && <span className="hint">(new)</span>}</h1>
      <p className="hint">{kind.description}{generated ? `. Written only by ${kind.made_by}; read-only here.` : ""}</p>
      <Problems problems={problems} />
      {found && (generated ? <pre className="page">{found.text}</pre>
        : json && schema ? <SchemaForm value={document} schema={schema} root={schema} name={name} onChange={setDocument} />
        : <textarea className="page" value={text} rows={24} aria-label={name} onChange={(e) => setText(e.target.value)} />)}
      {!generated && (
        <div className="actions">
          <button type="button" disabled={!changed} onClick={save}>{found?.version === "" ? "Create" : "Save"}</button>
          <button type="button" onClick={() => { setMessage(""); load(); }}>Reload</button>
          <span role="status">{message}</span>
        </div>
      )}
    </section>
  );
}

function documentsPage(path: string, title: string, intro: string, names: string[]): Page {
  function Component({ rest }: { rest: string[] }) {
    const [kinds, setKinds] = useState<Kind[] | null>(null);
    const [problems, setProblems] = useState<string[]>([]);
    useEffect(() => {
      get<Kind[]>("/api/kinds").then((a) => (a.ok ? setKinds(a.data.filter((k) => names.includes(k.name))) : setProblems(a.problems)));
    }, [rest.join("/")]);
    const kind = kinds?.find((k) => k.name === rest[0]);
    if (kind && rest[1]) return <Editor page={path} kind={kind} name={rest[1]} />;
    return (
      <section>
        <h1>{title}</h1>
        <p className="hint">{intro}</p>
        <Problems problems={problems} />
        {kinds && <Listing page={path} kinds={kinds} />}
      </section>
    );
  }
  return { path, title, Component };
}

export const recipes = documentsPage("recipes", "Recipes",
  "Model, mixture and tokenizer recipes (docs/model.md), and the locks that build-tokenizer writes.",
  ["model", "mixture", "tokenizer", "lock"]);
export const images = documentsPage("images", "Images",
  "The container images of agentic tasks: each Dockerfile, and the lock and license list that lock-image writes.",
  ["dockerfile", "image-lock", "image-licenses"]);
export const signers = documentsPage("signers", "Signers",
  "The keys allowed to sign a commit that changes a decision (docs/licenses.md).",
  ["allowed-signers", "allowed-gpg-keys"]);
