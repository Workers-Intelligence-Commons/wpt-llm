// The probe page: the questions the probes test asks a model, and the
// patterns that flag a reply as alarming (docs/evaluation.md, "Probes"), as
// lines of a probe file of an empty source. Each probe is saved alone, as
// the next patch; the service layer checks that every pattern compiles as a
// Python regular expression. Probes are a measure only: no training step
// reads them.

import { useEffect, useState } from "react";
import { get, part, put, saying } from "../api";
import { PatchName, Problems, SourcePicker } from "../parts";
import type { Page } from "./index";

type Probe = { question: string; patterns: string[] };

const FILE = "probes/probes.jsonl"; // the probe file the page opens unless the person names another
const EMPTY: Probe = { question: "", patterns: [""] };

function ProbeForm({ probe, onChange }: { probe: Probe; onChange: (p: Probe) => void }) {
  const setPattern = (n: number, value: string) =>
    onChange({ ...probe, patterns: probe.patterns.map((p, i) => (i === n ? value : p)) });
  return (
    <>
      <label>Question
        <textarea value={probe.question} rows={3} aria-label="question" onChange={(e) => onChange({ ...probe, question: e.target.value })} />
      </label>
      <p className="hint">A reply is flagged when any pattern matches somewhere in it (Python's re.search); write flags in the pattern, such as (?i) for any case.</p>
      {probe.patterns.map((p, n) => (
        <div className="filters" key={n}>
          <input value={p} aria-label={`pattern ${n + 1}`} onChange={(e) => setPattern(n, e.target.value)} />
          <button type="button" className="quiet" disabled={probe.patterns.length === 1}
            onClick={() => onChange({ ...probe, patterns: probe.patterns.filter((_, i) => i !== n) })}>remove pattern</button>
        </div>
      ))}
      <button type="button" onClick={() => onChange({ ...probe, patterns: [...probe.patterns, ""] })}>+ pattern</button>
    </>
  );
}

function Editor({ id, path }: { id: string; path: string }) {
  const [version, setVersion] = useState<string | null>(null);
  const [saved, setSaved] = useState<Probe[]>([]);
  const [probes, setProbes] = useState<Probe[]>([]);
  const [added, setAdded] = useState<Probe>(EMPTY);
  const [patch, setPatch] = useState("probes");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");

  const load = () =>
    get<{ version: string; probes: Probe[] }>(`/api/sources/${part(id)}/probes/${part(path)}`).then((a) => {
      if (!a.ok) return setProblems(a.problems);
      setVersion(a.data.version);
      setSaved(a.data.probes);
      setProbes(a.data.probes);
    });
  useEffect(() => { load(); }, [id, path]);

  // line: from 1, or "new"; probe null removes the line
  const save = async (line: number | "new", probe: Probe | null, done: string) => {
    if (version === null) return;
    const a = await put<{ version: string }>(`/api/sources/${part(id)}/probes/${part(path)}/${line}`, { probe, patch, version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, done));
    if (!a.ok) return;
    if (line === "new") setAdded(EMPTY);
    load();
  };

  return (
    <section>
      <p><a href={`#/probes/${part(id)}`}>Probe files</a></p>
      <h1>{path}</h1>
      <p className="hint">Probes of {id}: the probes test asks each question many times and counts how often a reply matches one of its patterns.</p>
      <Problems problems={problems} />
      {probes.map((p, n) => (
        <div className="block" key={n}>
          <h3>Line {n + 1}</h3>
          <ProbeForm probe={p} onChange={(changed) => setProbes(probes.map((q, i) => (i === n ? changed : q)))} />
          <div className="filters">
            <button type="button" disabled={version === null || JSON.stringify(p) === JSON.stringify(saved[n])}
              onClick={() => save(n + 1, p, "Saved as the next patch.")}>Save</button>
            <button type="button" onClick={() => save(n + 1, null, "Removed, as the next patch.")}>Remove the probe</button>
          </div>
        </div>
      ))}
      {probes.length === 0 && <p className="hint">no probe yet</p>}
      <div className="block">
        <h3>New probe</h3>
        <ProbeForm probe={added} onChange={setAdded} />
        <div className="filters">
          <button type="button" disabled={version === null || !added.question}
            onClick={() => save("new", added, "Added, as the next patch.")}>Add</button>
        </div>
      </div>
      <div className="actions">
        <PatchName value={patch} onChange={setPatch} />
        <button type="button" onClick={() => { setMessage(""); load(); }}>Reload</button>
        <span role="status">{message}</span>
      </div>
    </section>
  );
}

function FileChoice({ id }: { id: string }) {
  const [file, setFile] = useState(FILE);
  return (
    <section>
      <h1>Probes</h1>
      <p className="hint">Questions the probes test asks a model, and the patterns that flag an alarming reply: a measure only, never a training signal.</p>
      <div className="filters">
        <SourcePicker onlyEmpty value={id} onChange={(s) => (window.location.hash = `#/probes/${part(s)}`)} />
        <a href="#/new-source">New source</a>
      </div>
      {id && (
        <form className="filters" onSubmit={(e) => { e.preventDefault(); window.location.hash = `#/probes/${part(id)}/${part(file)}`; }}>
          <label className="inline">Probe file <input value={file} aria-label="probe file" onChange={(e) => setFile(e.target.value)} /></label>
          <button type="submit" disabled={!file || file.endsWith("/")}>Open</button>
        </form>
      )}
    </section>
  );
}

function ProbesPage({ rest }: { rest: string[] }) {
  const [id, path] = rest;
  if (id && path) return <Editor id={id} path={path} />;
  return <FileChoice id={id ?? ""} />;
}

export const probes: Page = { path: "probes", title: "Probes", Component: ProbesPage };
