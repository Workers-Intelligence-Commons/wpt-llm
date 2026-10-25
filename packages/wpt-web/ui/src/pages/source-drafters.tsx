// The source drafter page: the built-in source drafters, each with its
// settings, and a run started as a separate process of wpt-web (`wpt-dataset
// source-drafter`), which saves every draft through the service layer: a new
// source under review, an existing one updated in place. Each run's progress
// comes from its record in the store: how many candidates, the count of
// drafted, skipped, needs-a-person, stopped and failed outcomes, and each
// outcome's details. A run started here stops when wpt-web stops and starts
// again, where it left off, when wpt-web starts next. A run that ended
// without a record shows its exit code and the end of its error output.

import { useEffect, useState } from "react";
import { get, part, post } from "../api";
import { Problems } from "../parts";
import type { Page } from "./index";

type SourceDrafter = { name: string; file: string; summary: string; settings: Record<string, string | number | boolean> };
type Candidates = { candidates: number; chosen: number; done_before: number; to_draft: number };
type Outcome = { candidate: string; outcome: string; source: string | null; change: string | null; reason: string; notes: unknown };
type Run = {
  folder: string; inputs: Record<string, unknown>; candidates: Candidates | null; counts: Record<string, number>;
  end: { status: string; exit_code: number; error: string | null } | null; running: boolean;
  arguments?: string[]; outcomes?: Outcome[];
};
type NotStarted = { started_by: string; exit_code: number; error: string };
type Runs = { runs: Run[]; not_started: NotStarted[] };

const KINDS = ["drafted", "skipped", "needs-a-person", "stopped", "failed"];
const NAME_KEY = "wpt-web source drafter";
const POLL_MS = 3000;

function remembered(): string {
  try { return localStorage.getItem(NAME_KEY) ?? ""; } catch { return ""; }
}

function state(run: Run): string {
  if (run.running) return "running here";
  if (!run.end) return "no end recorded (stopped, or running elsewhere)";
  if (run.end.exit_code === 130) return "stopped";
  return `${run.end.status}${run.end.exit_code ? ` (exit code ${run.end.exit_code})` : ""}`;
}

function Progress({ run }: { run: Run }) {
  const c = run.candidates;
  const done = KINDS.reduce((sum, k) => sum + (run.counts[k] ?? 0), 0);
  return (
    <span>
      {c ? `${done} of ${c.to_draft} to draft (${c.chosen} chosen of ${c.candidates}, ${c.done_before} done before)` : "listing candidates"}
      {"; "}
      {KINDS.map((k) => `${run.counts[k] ?? 0} ${k}`).join(", ")}
    </span>
  );
}

function Start({ sourceDrafter, onStarted }: { sourceDrafter: SourceDrafter; onStarted: () => void }) {
  const [settings, setSettings] = useState<Record<string, string>>(
    Object.fromEntries(Object.entries(sourceDrafter.settings).map(([k, v]) => [k, String(v)])));
  const [by, setBy] = useState(remembered);
  const [only, setOnly] = useState("");
  const [again, setAgain] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const start = async () => {
    try { localStorage.setItem(NAME_KEY, by); } catch { /* a browser that keeps nothing */ }
    const a = await post<{ started_by: string }>(`/api/source-drafters/${part(sourceDrafter.name)}/runs`, {
      by, settings, only: only.split(/\s+/).filter(Boolean), again,
    });
    setProblems(a.ok ? [] : a.problems);
    setMessage(a.ok ? `Started (${a.data.started_by}).` : "Not started.");
    if (a.ok) onStarted();
  };
  return (
    <div className="block">
      <h2>{sourceDrafter.name} <span className="hint">{sourceDrafter.file}</span></h2>
      <p className="hint">{sourceDrafter.summary}</p>
      <div className="filters">
        {Object.keys(sourceDrafter.settings).map((k) => (
          <label className="inline" key={k}>{k}{" "}
            <input value={settings[k]} onChange={(e) => setSettings({ ...settings, [k]: e.target.value })} />
          </label>
        ))}
        <label className="inline">Drafted by <input value={by} required onChange={(e) => setBy(e.target.value)} /></label>
        <label className="inline"><input type="checkbox" checked={again} onChange={(e) => setAgain(e.target.checked)} /> again, even those done before</label>
      </div>
      <textarea rows={2} value={only} aria-label="only these candidates" placeholder="only these candidates, by name or source id (empty: every one)"
        onChange={(e) => setOnly(e.target.value)} />
      <Problems problems={problems} />
      <div className="filters">
        <button type="button" disabled={!by} onClick={start}>Start a run</button>
        <span role="status">{message}</span>
      </div>
    </div>
  );
}

function RunView({ folder }: { folder: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [shown, setShown] = useState("needs-a-person");
  useEffect(() => {
    const load = () => get<Run>(`/api/source-drafter-runs/${part(folder)}`).then((a) => (a.ok ? setRun(a.data) : setProblems(a.problems)));
    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => window.clearInterval(timer);
  }, [folder]);
  const rows = (run?.outcomes ?? []).filter((o) => o.outcome === shown);
  return (
    <section>
      <p><a href="#/source-drafters">Source drafters</a></p>
      <h1>{folder}</h1>
      <Problems problems={problems} />
      {run && (
        <>
          <p>{String(run.inputs.source_drafter)}: {state(run)}. <Progress run={run} /></p>
          {run.end?.error && <p className="bad">{run.end.error}</p>}
          <p className="hint"><code>wpt-dataset {(run.arguments ?? []).join(" ")}</code></p>
          <div className="filters">
            {KINDS.map((k) => (
              <button type="button" key={k} aria-pressed={shown === k} onClick={() => setShown(k)}>{k} ({run.counts[k] ?? 0})</button>
            ))}
          </div>
          <div className="table">
            <table>
              <thead><tr><th>candidate</th><th>source</th><th>change</th><th>reason</th><th>notes</th></tr></thead>
              <tbody>
                {rows.map((o, n) => (
                  <tr key={n}>
                    <td>{o.candidate}</td>
                    <td>{o.source ? <a href={`#/sources/${part(o.source)}`}>{o.source}</a> : ""}</td>
                    <td>{o.change ?? ""}</td>
                    <td>{o.reason}</td>
                    <td>{o.notes === null || o.notes === undefined ? "" : <pre className="page">{JSON.stringify(o.notes, null, 2)}</pre>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {rows.length === 0 && <p className="hint">none</p>}
          </div>
        </>
      )}
    </section>
  );
}

function SourceDrafters({ rest }: { rest: string[] }) {
  const [drafters, setDrafters] = useState<SourceDrafter[]>([]);
  const [runs, setRuns] = useState<Runs>({ runs: [], not_started: [] });
  const [problems, setProblems] = useState<string[]>([]);
  const loadRuns = () => get<Runs>("/api/source-drafter-runs").then((a) => (a.ok ? setRuns(a.data) : setProblems(a.problems)));
  useEffect(() => {
    if (rest[0]) return;
    get<SourceDrafter[]>("/api/source-drafters").then((a) => (a.ok ? setDrafters(a.data) : setProblems(a.problems)));
    loadRuns();
    const timer = window.setInterval(loadRuns, POLL_MS);
    return () => window.clearInterval(timer);
  }, [rest[0]]);
  if (rest[0]) return <RunView folder={rest[0]} />;
  return (
    <section>
      <h1>Source drafters</h1>
      <p className="hint">Programs that write source.json drafts from what a site publishes: a new source starts under review; one that exists is updated in place, and goes back under review when its license entries change; one with patches is left for a person.</p>
      <Problems problems={problems} />
      {drafters.map((d) => <Start key={d.name} sourceDrafter={d} onStarted={loadRuns} />)}
      <h2>Runs</h2>
      <div className="table">
        <table>
          <thead><tr><th>record</th><th>source drafter</th><th>state</th><th>progress</th></tr></thead>
          <tbody>
            {runs.runs.map((r) => (
              <tr key={r.folder}>
                <td><a href={`#/source-drafters/${part(r.folder)}`}>{r.folder}</a></td>
                <td>{String(r.inputs.source_drafter)}</td>
                <td>{state(r)}</td>
                <td><Progress run={r} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.runs.length === 0 && <p className="hint">no run in the store yet</p>}
      </div>
      {runs.not_started.length > 0 && (
        <>
          <h2>Ended without a record</h2>
          <div className="table">
            <table>
              <thead><tr><th>started by</th><th>exit code</th><th>error</th></tr></thead>
              <tbody>
                {runs.not_started.map((r) => (
                  <tr key={r.started_by}>
                    <td>{r.started_by}</td>
                    <td>{r.exit_code}</td>
                    <td><pre className="page">{r.error}</pre></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

export const sourceDrafters: Page = { path: "source-drafters", title: "Source drafters", Component: SourceDrafters };
