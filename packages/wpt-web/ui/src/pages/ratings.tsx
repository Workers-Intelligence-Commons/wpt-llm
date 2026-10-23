// The rating page: a person's pass or fail on the attempts of a wpt-eval run
// (docs/evaluation.md, "People's ratings"). It lists the attempts the run
// wrote into its record folder in the store (attempts.jsonl), each attempt's
// result and its changes to /work next to its request, and records the
// verdict as a line of a rating file of an empty source, naming the attempt
// (task, record, index, plain), the weights the run measured, who rated and
// the day. Each save becomes the next patch of the source; a person's newer
// rating of an attempt replaces their earlier one (the service layer does
// it). An attempt Docker dropped is shown, not rated. Ratings are a measure
// only: no training step reads them.

import { useEffect, useState } from "react";
import { get, part, put, saying } from "../api";
import { PatchName, Problems, SourcePicker } from "../parts";
import type { Page } from "./index";

type Block = Record<string, unknown> & { type: string };
type Message = { role: string; content: Block[] };
type Attempt = {
  task: string; kind: string; index: number; plain: boolean; dropped: boolean; error?: string; score?: number;
  changes?: string;
  conversation: { messages: Message[]; metadata?: { answer?: { answer: string; confidence: number } | null } };
};
type Run = { weights: Record<string, string>; attempts: Attempt[] };
type Rating = {
  schema: number; task: string; weights: Record<string, string>; record: string; attempt: number; plain: boolean;
  verdict: "pass" | "fail"; by: string; date: string;
};

const SCHEMA = 1;
const NAME_KEY = "wpt-web rater";
const FILE = "ratings/ratings.jsonl"; // the rating file a new rating goes to unless the person names another

// The rater's name as this browser remembers it, "" when it cannot.
function remembered(): string {
  try { return localStorage.getItem(NAME_KEY) ?? ""; } catch { return ""; }
}

// The day as YYYY-MM-DD, on this machine's calendar.
function today(): string {
  const now = new Date();
  return [now.getFullYear(), now.getMonth() + 1, now.getDate()].map((n) => String(n).padStart(2, "0")).join("-");
}

// Whether a rating rates this attempt of this run.
function rates(r: Rating, folder: string, run: Run, a: Attempt): boolean {
  return r.record === folder && r.task === a.task && r.attempt === a.index && r.plain === a.plain
    && JSON.stringify(r.weights) === JSON.stringify(run.weights);
}

function Blocks({ blocks }: { blocks: Block[] }) {
  return (
    <>
      {blocks.map((b, n) => {
        if (b.type === "text" || b.type === "thinking")
          return <pre className="page" key={n}>{b.type === "thinking" ? "(thinking) " : ""}{String(b.text)}</pre>;
        if (b.type === "tool_call")
          return <pre className="page" key={n}>call {String(b.name)} {JSON.stringify(b.arguments, null, 2)}</pre>;
        if (b.type === "tool_result")
          return <div key={n}><span className="hint">result{b.error ? " (failed)" : ""}</span><Blocks blocks={b.content as Block[]} /></div>;
        return <pre className="page" key={n}>{JSON.stringify(b, null, 2)}</pre>;
      })}
    </>
  );
}

function AttemptView({ attempt, mine, others, onRate }: {
  attempt: Attempt; mine: Rating | null; others: number; onRate: (v: "pass" | "fail") => void;
}) {
  const messages = attempt.conversation.messages;
  const request = messages.find((m) => m.role === "user");
  // the model's final reply: its last assistant message before the harness's confidence question, the last user
  // turn after the request (docs/chat-format.md, "The answer")
  const asked = messages.map((m) => m.role).lastIndexOf("user");
  const last = [...(asked > 0 ? messages.slice(0, asked) : messages)].reverse().find((m) => m.role === "assistant");
  const answer = attempt.conversation.metadata?.answer;
  return (
    <div className="block">
      <div className="filters">
        <code>{attempt.task}</code>
        <span className="hint">{attempt.kind}, attempt {attempt.index}{attempt.plain ? ", plain tools" : ""}</span>
        {others > 0 && <span className="hint">{others} other ratings</span>}
        {attempt.dropped && <span className="bad">dropped, Docker failed: {attempt.error}</span>}
      </div>
      <div className="pair">
        <div>
          <h3>Request</h3>
          {request ? <Blocks blocks={request.content} /> : <p className="hint">no request</p>}
        </div>
        <div>
          <h3>Result</h3>
          {answer && <p>Answer: <strong>{answer.answer}</strong> <span className="hint">(confidence {answer.confidence})</span></p>}
          {last ? <Blocks blocks={last.content} /> : <p className="hint">the model wrote nothing</p>}
          <h3>Changes to /work</h3>
          {attempt.changes ? <pre className="page">{attempt.changes}</pre> : <p className="hint">{attempt.dropped ? "none: the attempt never ran" : "no file changed"}</p>}
          <details>
            <summary>Every message ({messages.length})</summary>
            {messages.map((m, n) => (
              <div className={`message ${m.role}`} key={n}><strong>{m.role}</strong><Blocks blocks={m.content} /></div>
            ))}
          </details>
        </div>
      </div>
      {!attempt.dropped && (
        <div className="filters">
          <button type="button" aria-pressed={mine?.verdict === "pass"} onClick={() => onRate("pass")}>Pass</button>
          <button type="button" aria-pressed={mine?.verdict === "fail"} onClick={() => onRate("fail")}>Fail</button>
          {mine && <span className={mine.verdict === "pass" ? "good" : "bad"}>you rated it {mine.verdict} on {mine.date}</span>}
        </div>
      )}
    </div>
  );
}

function Rater({ id, path, folder }: { id: string; path: string; folder: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [version, setVersion] = useState<string | null>(null);
  const [by, setBy] = useState(remembered);
  const [kind, setKind] = useState("");
  const [patch, setPatch] = useState("ratings");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");

  const loadRatings = () =>
    get<{ version: string; ratings: Rating[] }>(`/api/sources/${part(id)}/ratings/${part(path)}`).then((a) => {
      if (!a.ok) return setProblems(a.problems);
      setVersion(a.data.version);
      setRatings(a.data.ratings);
    });
  useEffect(() => {
    get<Run>(`/api/runs/${part(folder)}/attempts`).then((a) => (a.ok ? setRun(a.data) : setProblems(a.problems)));
  }, [folder]);
  useEffect(() => { loadRatings(); }, [id, path]);
  useEffect(() => {
    try { localStorage.setItem(NAME_KEY, by); } catch { /* the name is only remembered when the browser allows it */ }
  }, [by]);

  // Always a new rating: the service layer replaces the person's earlier rating of the same attempt.
  const rate = async (a: Attempt, verdict: "pass" | "fail") => {
    if (version === null || run === null || !by.trim()) return;
    const rating: Rating = { schema: SCHEMA, task: a.task, weights: run.weights, record: folder, attempt: a.index,
      plain: a.plain, verdict, by: by.trim(), date: today() };
    const answer = await put<{ version: string }>(
      `/api/sources/${part(id)}/ratings/${part(path)}/new`, { rating, patch, version });
    setProblems(answer.ok ? [] : answer.problems);
    setMessage(saying(answer, `Rated ${verdict}, as the next patch.`));
    if (answer.ok) loadRatings();
  };

  const kinds = [...new Set((run?.attempts ?? []).map((a) => a.kind))];
  const shown = (run?.attempts ?? []).filter((a) => !kind || a.kind === kind);
  return (
    <section>
      <p><a href={`#/ratings/${part(id)}/${part(path)}`}>Runs</a></p>
      <h1>{folder}</h1>
      <p className="hint">Ratings go to <code>{path}</code> in {id}. Is this what a co-worker would have delivered?</p>
      <Problems problems={problems} />
      <div className="filters">
        <label className="inline">Your name <input value={by} required aria-label="your name" onChange={(e) => setBy(e.target.value)} /></label>
        <select value={kind} aria-label="kind of task" onChange={(e) => setKind(e.target.value)}>
          <option value="">every kind</option>
          {kinds.map((k) => <option key={k}>{k}</option>)}
        </select>
        {!by.trim() && <span className="hint">give your name to rate</span>}
      </div>
      {run && shown.map((a) => {
        const about = ratings.filter((r) => rates(r, folder, run, a));
        return (
          <AttemptView key={`${a.task}/${a.plain}/${a.index}`} attempt={a} onRate={(v) => rate(a, v)}
            mine={about.find((r) => r.by === by.trim()) ?? null} others={about.filter((r) => r.by !== by.trim()).length} />
        );
      })}
      {run && run.attempts.length === 0 && <p className="hint">this run wrote no attempt</p>}
      <div className="actions">
        <PatchName value={patch} onChange={setPatch} />
        <button type="button" onClick={() => { setMessage(""); loadRatings(); }}>Reload</button>
        <span role="status">{message}</span>
      </div>
    </section>
  );
}

function RunList({ id, path }: { id: string; path: string }) {
  const [runs, setRuns] = useState<string[]>([]);
  const [problems, setProblems] = useState<string[]>([]);
  const [file, setFile] = useState(path || FILE);
  useEffect(() => { get<string[]>("/api/runs").then((a) => (a.ok ? setRuns(a.data) : setProblems(a.problems))); }, []);
  const ready = id && file;
  return (
    <section>
      <h1>Ratings</h1>
      <p className="hint">A person's pass or fail on the attempts of a wpt-eval run: a measure only, never a training signal.</p>
      <div className="filters">
        <SourcePicker onlyEmpty value={id} onChange={(s) => (window.location.hash = `#/ratings/${part(s)}/${part(file)}`)} />
        <label className="inline">Rating file <input value={file} aria-label="rating file" onChange={(e) => setFile(e.target.value)} /></label>
        <a href="#/new-source">New source</a>
      </div>
      <Problems problems={problems} />
      <ul>
        {runs.map((r) => (
          <li key={r}>{ready ? <a href={`#/ratings/${part(id)}/${part(file)}/${part(r)}`}>{r}</a> : r}</li>
        ))}
        {runs.length === 0 && <li className="hint">no wpt-eval run with attempts in the store</li>}
      </ul>
      {!ready && <p className="hint">Choose a source and a rating file to rate a run.</p>}
    </section>
  );
}

function RatingsPage({ rest }: { rest: string[] }) {
  const [id, path, folder] = rest;
  if (id && path && folder) return <Rater id={id} path={path} folder={folder} />;
  return <RunList id={id ?? ""} path={path ?? ""} />;
}

export const ratings: Page = { path: "ratings", title: "Ratings", Component: RatingsPage };
