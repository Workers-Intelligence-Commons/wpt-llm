// The jobs page: start a `wpt` subcommand as a job of the server and follow
// it: the lines it prints (stdout and stderr) and the run record it writes in
// the store. The jobs are the server's; a stopped server forgets them, and
// their records stay in the store. The routes are POST /api/jobs/<job> (one
// per job, listed by /api/openapi.json) and GET /api/jobs[/<id>[/lines/<n>|/records]].

import { useEffect, useRef, useState } from "react";
import { get, part, post } from "../api";
import { Problems } from "../parts";
import type { Page } from "./index";

type Job = { id: string; command: string; started: string; ended: string | null; state: string; exit_code: number | null };
type Printed = { stream: "stdout" | "stderr"; text: string };
type Lines = { first: number; next: number; lines: Printed[] };
type RunRecord = { folder: string; start: Record<string, unknown>; ids: Record<string, unknown> | null; end: Record<string, unknown> | null };

const EVERY = 1000; // milliseconds between two looks at a running job

function Starter({ onStarted }: { onStarted: (job: Job) => void }) {
  const [names, setNames] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [args, setArgs] = useState("");
  const [problems, setProblems] = useState<string[]>([]);

  useEffect(() => {
    get<{ paths: Record<string, { post?: unknown }> }>("/api/openapi.json").then((a) => {
      if (!a.ok) return setProblems(a.problems);
      const found = Object.entries(a.data.paths)
        .filter(([path, methods]) => path.startsWith("/api/jobs/") && methods.post)
        .map(([path]) => path.slice("/api/jobs/".length));
      setNames(found);
      setName(found[0] ?? "");
    });
  }, []);

  const start = async () => {
    const lines = args.split("\n").filter((line) => line !== "");
    const a = await post<Job>(`/api/jobs/${name}`, { args: lines });
    if (!a.ok) return setProblems(a.problems);
    setProblems([]);
    onStarted(a.data);
  };

  return (
    <form onSubmit={(e) => { e.preventDefault(); start(); }}>
      <div className="filters">
        <label>
          Job{" "}
          <select value={name} onChange={(e) => setName(e.target.value)}>
            {names.map((n) => (
              <option key={n} value={n}>{n.replace("/", " ")}</option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={name === ""}>Start</button>
      </div>
      <label>Arguments, one per line
        <textarea value={args} rows={3} aria-label="arguments" onChange={(e) => setArgs(e.target.value)} />
      </label>
      <Problems problems={problems} />
    </form>
  );
}

function List() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const load = () => get<Job[]>("/api/jobs").then((a) => (a.ok ? setJobs(a.data) : setProblems(a.problems)));
  useEffect(() => {
    load();
    const timer = setInterval(load, 2 * EVERY);
    return () => clearInterval(timer);
  }, []);
  return (
    <section>
      <h1>Jobs</h1>
      <p className="hint">A job is a `wpt` subcommand the server runs. A stopped server forgets its jobs; their run records stay in the store.</p>
      <Starter onStarted={(job) => { window.location.hash = `#/jobs/${part(job.id)}`; }} />
      <Problems problems={problems} />
      <div className="table">
        <table>
          <thead>
            <tr><th>Job</th><th>Command</th><th>State</th><th>Started</th></tr>
          </thead>
          <tbody>
            {[...(jobs ?? [])].reverse().map((j) => (
              <tr key={j.id}>
                <td><a href={`#/jobs/${part(j.id)}`}>{j.id}</a></td>
                <td><code>{j.command}</code></td>
                <td className={j.state === "failed" ? "bad" : j.state === "succeeded" ? "good" : undefined}>
                  {j.state}{j.exit_code === null ? "" : ` (${j.exit_code})`}
                </td>
                <td>{j.started}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {jobs && jobs.length === 0 && <p className="hint">No job yet.</p>}
    </section>
  );
}

function Followed({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [lines, setLines] = useState<Printed[]>([]);
  const [records, setRecords] = useState<RunRecord[]>([]);
  const [problems, setProblems] = useState<string[]>([]);
  const next = useRef(0);
  const ended = job !== null && job.state !== "running";

  // the job's lines, from the one after the last seen, until it has ended and all it printed is here
  useEffect(() => {
    next.current = 0;
    setLines([]);
    setRecords([]);
    let stopped = false;
    const look = async () => {
      const j = await get<Job>(`/api/jobs/${part(id)}`);
      if (!j.ok) return setProblems(j.problems);
      const l = await get<Lines>(`/api/jobs/${part(id)}/lines/${next.current}`);
      if (stopped || !l.ok) return;
      next.current = l.data.next;
      setLines((seen) => [...seen, ...l.data.lines]);
      setJob(j.data);
      if (j.data.state === "running") timer = setTimeout(look, EVERY);
    };
    let timer = setTimeout(look, 0);
    return () => { stopped = true; clearTimeout(timer); };
  }, [id]);

  // the run records the job wrote, now and when it has ended
  useEffect(() => {
    get<{ records: RunRecord[] }>(`/api/jobs/${part(id)}/records`).then((a) => (a.ok ? setRecords(a.data.records) : setProblems(a.problems)));
  }, [id, ended]);

  return (
    <section>
      <p><a href="#/jobs">All jobs</a></p>
      <h1>{job ? <code>{job.command}</code> : id}</h1>
      <Problems problems={problems} />
      {job && <p>{job.state}{job.exit_code === null ? "" : ` (exit code ${job.exit_code})`}, started {job.started}{job.ended ? `, ended ${job.ended}` : ""}</p>}
      <pre className="page" aria-label="printed lines">
        {lines.map((l) => (l.stream === "stderr" ? `stderr: ${l.text}` : l.text)).join("\n")}
      </pre>
      <h2>Run record</h2>
      {records.length === 0 && <p className="hint">None in the store (yet): not every job writes one.</p>}
      {records.map((r) => (
        <details key={r.folder}>
          <summary><code>{r.folder}</code> {r.end ? String(r.end.status) : "not ended"}</summary>
          <pre className="page">{JSON.stringify({ start: r.start, ids: r.ids, end: r.end }, null, 2)}</pre>
        </details>
      ))}
    </section>
  );
}

function JobsPage({ rest }: { rest: string[] }) {
  return rest[0] ? <Followed id={rest[0]} /> : <List />;
}

export const jobs: Page = { path: "jobs", title: "Jobs", Component: JobsPage };
