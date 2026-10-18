// The task editor: the task folders of an empty source (docs/post-training.md,
// "Tasks"): task.json, edited with a form built from its JSON schema, the
// check that decides whether an attempt solved it, and the starting files.
// A save writes the folder as exactly these files, as the next patch (only
// the files that changed, each keeping its mode); a symbolic link is kept as
// it is, and a task at the source's root is changed with unpack and patch.
// The folder's tags (grpo) say which steps read it.

import { useEffect, useState } from "react";
import { get, part, put, saying } from "../api";
import { PatchName, Problems, SourcePicker } from "../parts";
import { SchemaForm, initial, type Schema } from "../schema-form";
import type { Page } from "./index";

type Task = { task: unknown; check: string; files: Record<string, string>; kept?: Record<string, string> };
type Listing = { version: string; tasks: Record<string, Task> };

function Editor({ id, folder }: { id: string; folder: string }) {
  const [version, setVersion] = useState<string | null>(null);
  const [saved, setSaved] = useState<Task | null>(null);
  const [task, setTask] = useState<Task>({ task: {}, check: "", files: {} });
  const [schema, setSchema] = useState<Schema | null>(null);
  const [patch, setPatch] = useState("tasks");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [newFile, setNewFile] = useState("");

  const load = () =>
    get<Listing>(`/api/sources/${part(id)}/tasks`).then((a) => {
      if (!a.ok) return setProblems(a.problems);
      setVersion(a.data.version);
      const found = a.data.tasks[folder] ?? null;
      setSaved(found);
      setTask(found ?? { task: schema ? initial(schema, schema) : {}, check: "", files: {} });
    });
  useEffect(() => { get<Schema>("/api/schemas/task").then((a) => a.ok && setSchema(a.data)); }, []);
  useEffect(() => { load(); }, [id, folder, schema === null]);

  const save = async (remove = false) => {
    if (version === null) return;
    const a = await put<{ version: string }>(`/api/sources/${part(id)}/tasks/${part(folder)}`,
      { task: remove ? null : task.task, check: task.check, files: task.files, patch, version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, remove ? "Removed, as the next patch." : "Saved as the next patch."));
    if (!a.ok) return;
    if (remove) window.location.hash = `#/tasks/${part(id)}`;
    else load();
  };

  const files = Object.entries(task.files);
  const setFiles = (f: Record<string, string>) => setTask({ ...task, files: f });
  return (
    <section>
      <p><a href={`#/tasks/${part(id)}`}>Tasks of {id}</a></p>
      <h1>{folder} {saved === null && <span className="hint">(new)</span>}</h1>
      <Problems problems={problems} />
      <h2>task.json</h2>
      {schema ? <SchemaForm value={task.task} schema={schema} root={schema} name="task" onChange={(v) => setTask({ ...task, task: v })} />
        : <p className="hint">loading the task schema</p>}
      <h2>check</h2>
      <p className="hint">The program that decides whether an attempt solved the task (docs/post-training.md).</p>
      <textarea className="page" value={task.check} rows={10} aria-label="check" onChange={(e) => setTask({ ...task, check: e.target.value })} />
      <h2>Starting files</h2>
      {files.map(([path, text]) => (
        <div className="block" key={path}>
          <div className="filters">
            <code>files/{path}</code>
            <button type="button" className="quiet" onClick={() => { const { [path]: _, ...rest } = task.files; setFiles(rest); }}>remove file</button>
          </div>
          <textarea value={text} rows={Math.min(20, text.split("\n").length + 2)} aria-label={path}
            onChange={(e) => setFiles({ ...task.files, [path]: e.target.value })} />
        </div>
      ))}
      {Object.entries(task.kept ?? {}).map(([p, why]) => (
        <p className="hint" key={p}><code>{p}</code>: {why}, kept as it is</p>
      ))}
      <form className="filters" onSubmit={(e) => { e.preventDefault(); if (newFile && !(newFile in task.files)) { setFiles({ ...task.files, [newFile]: "" }); setNewFile(""); } }}>
        <input value={newFile} placeholder="path under files/, such as src/calc.py" aria-label="new starting file" onChange={(e) => setNewFile(e.target.value)} />
        <button type="submit" disabled={!newFile}>+ file</button>
      </form>
      <p className="hint">
        Which steps read this task comes from the tags of its files in source.json (grpo); <a href={`#/sources/${part(id)}`}>edit them there</a>.
      </p>
      <div className="actions">
        <PatchName value={patch} onChange={setPatch} />
        <button type="button" disabled={version === null || JSON.stringify(task) === JSON.stringify(saved)} onClick={() => save()}>Save</button>
        {saved !== null && <button type="button" onClick={() => save(true)}>Remove the task</button>}
        <button type="button" onClick={() => { setMessage(""); load(); }}>Reload</button>
        <span role="status">{message}</span>
      </div>
    </section>
  );
}

function TaskList({ id }: { id: string }) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [folder, setFolder] = useState("tasks/");
  useEffect(() => {
    if (id) get<Listing>(`/api/sources/${part(id)}/tasks`).then((a) => (a.ok ? setListing(a.data) : setProblems(a.problems)));
  }, [id]);
  return (
    <section>
      <h1>Tasks</h1>
      <p className="filters">
        <SourcePicker onlyEmpty value={id} onChange={(s) => (window.location.hash = `#/tasks/${part(s)}`)} />
        <a href="#/new-source">New source</a>
      </p>
      <Problems problems={problems} />
      {id && listing && (
        <>
          <ul>
            {Object.keys(listing.tasks).map((f) => (
              <li key={f}>{f ? <a href={`#/tasks/${part(id)}/${part(f)}`}>{f}</a>
                : <span className="hint">a task at the source's root: change it with unpack and patch</span>}</li>
            ))}
            {Object.keys(listing.tasks).length === 0 && <li className="hint">no task yet</li>}
          </ul>
          <form className="filters" onSubmit={(e) => { e.preventDefault(); window.location.hash = `#/tasks/${part(id)}/${part(folder)}`; }}>
            <input value={folder} aria-label="new task folder" onChange={(e) => setFolder(e.target.value)} />
            <button type="submit" disabled={!folder || folder.endsWith("/")}>New task</button>
          </form>
        </>
      )}
    </section>
  );
}

function TasksPage({ rest }: { rest: string[] }) {
  const [id, folder] = rest;
  if (id && folder) return <Editor id={id} folder={folder} />;
  return <TaskList id={id ?? ""} />;
}

export const tasks: Page = { path: "tasks", title: "Tasks", Component: TasksPage };
