// The conversation editor: the conversations of a chat file of a source
// (docs/chat-format.md), one per line. A conversation holds its tools, its
// messages and their blocks, and its metadata; the file's tags say which
// training step reads it (answer-as-assistant). A WPT model folder on this machine can
// draft the next assistant message: the draft is edited or confirmed as it
// is, and its metadata.draft records the model, its weights and whether the
// person changed it (docs/chat-format.md, "Model drafts"). Each save becomes
// the next patch of the source.

import { useEffect, useState } from "react";
import { get, part, post, put, saying } from "../api";
import { PatchName, Problems, SourcePicker } from "../parts";
import type { Page } from "./index";

type Block = Record<string, unknown> & { type: string };
type Draft = { model: string; weights: Record<string, string>; sha256: string; edited: boolean };
type Message = { role: string; content: Block[]; metadata?: Record<string, unknown> & { draft?: Draft } };
type Tool = { name: string; description: string; parameters: Record<string, unknown> };
type Conversation = { tools?: Tool[]; messages: Message[]; metadata?: Record<string, unknown> };
type Listing = { version: string; conversations: Conversation[] };

const ROLES = ["system", "user", "assistant", "tool"];
const TYPES: Record<string, string[]> = {
  system: ["text", "json", "file"], user: ["text", "json", "file"], tool: ["tool_result"],
  assistant: ["text", "thinking", "tool_call", "json", "file"],
};
const MODEL_KEY = "wpt-web model folder";

// The sha256 of a message's content as the server computes it
// (wpt_dataset.conversations.content_sha256): JSON with sorted keys, no spaces.
// Only for showing whether a draft was changed while the person types: the
// server sets metadata.draft.edited itself when the conversation is saved.
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`).join(",")}}`;
  return JSON.stringify(value);
}

async function sha256(content: Block[]): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical(content)));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Whether a draft message's content is no longer the model's, as the page shows it
// while the person types (the sha256 is computed after each change).
function useEdited(message: Message | null): boolean {
  const draft = message?.metadata?.draft;
  const [edited, setEdited] = useState(draft?.edited ?? false);
  const content = message ? canonical(message.content) : "";
  useEffect(() => {
    if (!message || !draft) return;
    let live = true;
    sha256(message.content).then((h) => live && setEdited(h !== draft.sha256));
    return () => { live = false; };
  }, [content, draft?.sha256]);
  return edited;
}

function newBlock(type: string): Block {
  if (type === "tool_call") return { type, id: `call-${Date.now()}`, name: "", arguments: {} };
  if (type === "tool_result") return { type, call: "", content: [{ type: "text", text: "" }] };
  if (type === "json") return { type, value: null };
  if (type === "file") return { type, sha256: "", media_type: "image/png" };
  return { type, text: "" };
}

// A field holding any JSON value, edited as JSON text.
function JsonField({ label, value, onChange }: { label: string; value: unknown; onChange: (v: unknown) => void }) {
  const [text, setText] = useState(JSON.stringify(value, null, 2));
  const [bad, setBad] = useState(false);
  useEffect(() => setText(JSON.stringify(value, null, 2)), [JSON.stringify(value)]);
  return (
    <label className="field">
      {label} {bad && <span className="bad">not JSON yet</span>}
      <textarea value={text} rows={Math.min(12, text.split("\n").length + 1)} aria-label={label}
        onChange={(e) => {
          setText(e.target.value);
          try { onChange(JSON.parse(e.target.value)); setBad(false); } catch { setBad(true); }
        }} />
    </label>
  );
}

function BlockEditor({ block, role, onChange, onRemove }: { block: Block; role: string; onChange: (b: Block) => void; onRemove: () => void }) {
  const set = (key: string, value: unknown) => onChange({ ...block, [key]: value });
  const text = (key: string) => (
    <textarea value={String(block[key] ?? "")} rows={Math.min(16, String(block[key] ?? "").split("\n").length + 2)}
      aria-label={key} onChange={(e) => set(key, e.target.value)} />
  );
  return (
    <div className="block">
      <div className="filters">
        <select value={block.type} aria-label="block type" onChange={(e) => onChange(newBlock(e.target.value))}>
          {(TYPES[role] ?? []).map((t) => <option key={t}>{t}</option>)}
        </select>
        <button type="button" className="quiet" onClick={onRemove}>remove block</button>
      </div>
      {(block.type === "text" || block.type === "thinking") && text("text")}
      {block.type === "tool_call" && (
        <>
          <input value={String(block.name)} placeholder="tool name" aria-label="tool name" onChange={(e) => set("name", e.target.value)} />
          <input value={String(block.id)} placeholder="call id" aria-label="call id" onChange={(e) => set("id", e.target.value)} />
          <JsonField label="arguments" value={block.arguments} onChange={(v) => set("arguments", v)} />
        </>
      )}
      {block.type === "tool_result" && (
        <>
          <input value={String(block.call)} placeholder="the call id it answers" aria-label="call" onChange={(e) => set("call", e.target.value)} />
          <label className="inline"><input type="checkbox" checked={block.error === true}
            onChange={(e) => { const { error: _, ...rest } = block; onChange(e.target.checked ? { ...rest, error: true } : (rest as Block)); }} /> the call failed</label>
          <JsonField label="content (blocks)" value={block.content} onChange={(v) => set("content", v)} />
        </>
      )}
      {block.type === "json" && <JsonField label="value" value={block.value} onChange={(v) => set("value", v)} />}
      {block.type === "file" && (
        <>
          <input value={String(block.sha256)} placeholder="sha256" aria-label="sha256" onChange={(e) => set("sha256", e.target.value)} />
          <input value={String(block.media_type)} placeholder="media type" aria-label="media type" onChange={(e) => set("media_type", e.target.value)} />
        </>
      )}
    </div>
  );
}

function MessageEditor({ message, onChange, onRemove, onMove }: {
  message: Message; onChange: (m: Message) => void; onRemove?: () => void; onMove?: (by: number) => void;
}) {
  const draft = message.metadata?.draft;
  const edited = useEdited(message);
  const blocks = message.content;
  const change = (content: Block[]) => onChange({ ...message, content });
  return (
    <div className={`message ${message.role}`}>
      <div className="filters">
        <select value={message.role} aria-label="role" disabled={!!draft}
          onChange={(e) => onChange({ ...message, role: e.target.value, content: [newBlock(TYPES[e.target.value][0])] })}>
          {ROLES.map((r) => <option key={r}>{r}</option>)}
        </select>
        {draft && (
          <span className={`pill ${edited ? "refused" : "admitted"}`}
            title={`weights ${Object.entries(draft.weights).map(([f, s]) => `${f} ${s.slice(0, 12)}`).join(", ")}`}>
            model draft from {draft.model}, {edited ? "changed by you" : "unchanged"}
          </span>
        )}
        {onMove && <button type="button" className="quiet" onClick={() => onMove(-1)}>up</button>}
        {onMove && <button type="button" className="quiet" onClick={() => onMove(1)}>down</button>}
        {onRemove && <button type="button" className="quiet" onClick={onRemove}>remove message</button>}
      </div>
      {blocks.map((block, i) => (
        <BlockEditor key={i} block={block} role={message.role}
          onChange={(b) => change(blocks.map((old, j) => (j === i ? b : old)))}
          onRemove={() => change(blocks.filter((_, j) => j !== i))} />
      ))}
      <button type="button" onClick={() => change([...blocks, newBlock(TYPES[message.role][0])])}>+ block</button>
    </div>
  );
}

function Editor({ id, path, line }: { id: string; path: string; line: string }) {
  const [version, setVersion] = useState<string | null>(null);
  const [saved, setSaved] = useState<Conversation | null>(null);
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [patch, setPatch] = useState("conversations");
  const [problems, setProblems] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [model, setModel] = useState(() => { try { return localStorage.getItem(MODEL_KEY) ?? ""; } catch { return ""; } });
  const [draft, setDraft] = useState<Message | null>(null);
  const [asking, setAsking] = useState(false);
  const draftEdited = useEdited(draft);

  const base = `/api/sources/${part(id)}/conversations/${part(path)}`;
  const load = () =>
    get<Listing>(base).then((a) => {
      if (!a.ok) return setProblems(a.problems);
      setVersion(a.data.version);
      const found = line === "new" ? null : a.data.conversations[Number(line) - 1] ?? null;
      setSaved(found);
      setConversation(found ?? { messages: [] });
    });
  useEffect(() => { load(); }, [id, path, line]);

  const messages = conversation.messages;
  const setMessages = (m: Message[]) => setConversation({ ...conversation, messages: m });

  const ask = async () => {
    try { localStorage.setItem(MODEL_KEY, model); } catch { /* the folder is only remembered when the browser allows it */ }
    setAsking(true);
    const a = await post<{ message: Message }>("/api/drafts", { model, conversation });
    setAsking(false);
    if (a.ok) { setDraft(a.data.message); setProblems([]); } else setProblems(a.problems);
  };

  const save = async (remove = false) => {
    if (version === null) return;
    const target = `${base}/${line}`;
    const written = conversation;  // the server sets each draft's `edited`
    const a = await put<{ version: string }>(target, { conversation: remove ? null : written, patch, version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, remove ? "Removed, as the next patch." : "Saved as the next patch."));
    if (!a.ok) return;
    setVersion(a.data.version);
    setConversation(written);
    setSaved(written);
    if (remove || line === "new") {
      const listed = await get<Listing>(base);
      window.location.hash = remove || !listed.ok ? `#/conversations/${part(id)}/${part(path)}`
        : `#/conversations/${part(id)}/${part(path)}/${listed.data.conversations.length}`;
    }
  };

  const tools = conversation.tools ?? [];
  const setTools = (t: Tool[]) => {
    const { tools: _, ...rest } = conversation;
    setConversation(t.length ? { ...rest, tools: t } : rest);
  };
  const changed = JSON.stringify(conversation) !== JSON.stringify(saved);

  return (
    <section>
      <p><a href={`#/conversations/${part(id)}/${part(path)}`}>{path} of {id}</a></p>
      <h1>{line === "new" ? "New conversation" : `Conversation on line ${line}`}</h1>
      <Problems problems={problems} />
      <h2>Tools</h2>
      {tools.map((tool, i) => (
        <div className="block" key={i}>
          <input value={tool.name} placeholder="name" aria-label="tool name" onChange={(e) => setTools(tools.map((t, j) => (j === i ? { ...t, name: e.target.value } : t)))} />
          <textarea value={tool.description} rows={2} placeholder="description" aria-label="tool description"
            onChange={(e) => setTools(tools.map((t, j) => (j === i ? { ...t, description: e.target.value } : t)))} />
          <JsonField label="parameters (JSON Schema)" value={tool.parameters}
            onChange={(v) => setTools(tools.map((t, j) => (j === i ? { ...t, parameters: v as Record<string, unknown> } : t)))} />
          <button type="button" className="quiet" onClick={() => setTools(tools.filter((_, j) => j !== i))}>remove tool</button>
        </div>
      ))}
      <button type="button" onClick={() => setTools([...tools, { name: "", description: "", parameters: { type: "object", properties: {} } }])}>+ tool</button>
      <h2>Messages</h2>
      {messages.map((m, i) => (
        <MessageEditor key={i} message={m}
          onChange={(n) => setMessages(messages.map((old, j) => (j === i ? n : old)))}
          onRemove={() => setMessages(messages.filter((_, j) => j !== i))}
          onMove={(by) => {
            const to = i + by;
            if (to < 0 || to >= messages.length) return;
            const next = [...messages];
            [next[i], next[to]] = [next[to], next[i]];
            setMessages(next);
          }} />
      ))}
      <div className="filters">
        {ROLES.map((r) => (
          <button type="button" key={r} onClick={() => setMessages([...messages, { role: r, content: [newBlock(TYPES[r][0])] }])}>+ {r}</button>
        ))}
      </div>
      <h2>Draft with a model</h2>
      <div className="filters">
        <input value={model} placeholder="a WPT model folder on this machine" aria-label="model folder" onChange={(e) => setModel(e.target.value)} />
        <button type="button" disabled={!model || asking || messages.length === 0} onClick={ask}>{asking ? "Writing..." : "Draft the next reply"}</button>
      </div>
      {draft && (
        <div className="draft">
          <MessageEditor message={draft} onChange={setDraft} />
          <div className="filters">
            <button type="button" onClick={() => { setMessages([...messages, draft]); setDraft(null); }}>
              {draftEdited ? "Confirm with my changes" : "Confirm as it is"}
            </button>
            <button type="button" onClick={() => setDraft(null)}>Discard</button>
          </div>
        </div>
      )}
      <h2>Metadata</h2>
      <JsonField label="the conversation's metadata (never read as tokens)" value={conversation.metadata ?? {}}
        onChange={(v) => {
          const { metadata: _, ...rest } = conversation;
          setConversation(v && typeof v === "object" && Object.keys(v).length ? { ...rest, metadata: v as Record<string, unknown> } : rest);
        }} />
      <div className="actions">
        <PatchName value={patch} onChange={setPatch} />
        <button type="button" disabled={!changed || messages.length === 0 || version === null} onClick={() => save()}>Save</button>
        {line !== "new" && <button type="button" onClick={() => save(true)}>Remove the conversation</button>}
        <button type="button" onClick={() => { setMessage(""); load(); }}>Reload</button>
        <span role="status">{message}</span>
      </div>
    </section>
  );
}

function FileView({ id, path }: { id: string; path: string }) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [source, setSource] = useState<{ version: string; document: { tags?: { tags: string[]; paths: string[] }[] } } | null>(null);
  const [tags, setTags] = useState("");
  const [message, setMessage] = useState("");

  const load = () => {
    get<Listing>(`/api/sources/${part(id)}/conversations/${part(path)}`).then((a) => (a.ok ? setListing(a.data) : setProblems(a.problems)));
    get<typeof source>(`/api/sources/${part(id)}`).then((a) => {
      if (!a.ok || !a.data) return;
      setSource(a.data);
      setTags((a.data.document.tags ?? []).find((e) => e.paths.length === 1 && e.paths[0] === path)?.tags.join(" ") ?? "");
    });
  };
  useEffect(load, [id, path]);

  const saveTags = async () => {
    if (!source) return;
    const a = await put<{ version: string }>(`/api/sources/${part(id)}/tags/${part(path)}`,
      { tags: tags.split(/[\s,]+/).filter(Boolean), version: source.version });
    setProblems(a.ok ? [] : a.problems);
    setMessage(saying(a, "Tags saved in source.json."));
    load();
  };

  const others = (source?.document.tags ?? []).filter((e) => !(e.paths.length === 1 && e.paths[0] === path));
  return (
    <section>
      <p><a href={`#/conversations/${part(id)}`}>Chat files of {id}</a></p>
      <h1>{path}</h1>
      <Problems problems={problems} />
      <div className="filters">
        <label className="inline">This file's tags <input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="answer-as-assistant" /></label>
        <button type="button" onClick={saveTags}>Save tags</button>
        <span className="hint">{others.length ? `other entries: ${others.map((e) => `${e.tags.join(" ")} (${e.paths.join(", ")})`).join("; ")}` : ""}</span>
        <span role="status">{message}</span>
      </div>
      {listing && (
        <ol>
          {listing.conversations.map((c, i) => {
            const drafts = c.messages.filter((m) => m.metadata?.draft);
            return (
              <li key={i}>
                <a href={`#/conversations/${part(id)}/${part(path)}/${i + 1}`}>
                  {c.messages.length} messages ({c.messages.map((m) => m.role).join(", ")})
                </a>
                {drafts.length > 0 && <span className="hint"> {drafts.length} model drafts, {drafts.filter((m) => m.metadata!.draft!.edited).length} changed</span>}
              </li>
            );
          })}
        </ol>
      )}
      <p><a href={`#/conversations/${part(id)}/${part(path)}/new`}>New conversation</a></p>
    </section>
  );
}

function ChatFiles({ id }: { id: string }) {
  const [files, setFiles] = useState<string[]>([]);
  const [entries, setEntries] = useState<string[]>([]);
  const [problems, setProblems] = useState<string[]>([]);
  const [newPath, setNewPath] = useState("answer-as-assistant/conversations.jsonl");
  useEffect(() => {
    if (!id) return;
    get<{ files: Record<string, { format: string | null }> }>(`/api/sources/${part(id)}/files`).then((a) =>
      a.ok ? setFiles(Object.keys(a.data.files).filter((p) => a.data.files[p].format === "chat")) : setProblems(a.problems));
    get<{ document: { formats?: { format: string; paths: string[] }[] } }>(`/api/sources/${part(id)}`).then((a) =>
      a.ok && setEntries((a.data.document.formats ?? []).filter((f) => f.format === "chat").flatMap((f) => f.paths).filter((p) => !/[*?[]/.test(p))));
  }, [id]);
  const shown = [...new Set([...files, ...entries])].sort();
  return (
    <section>
      <h1>Conversations</h1>
      <p className="filters">
        <SourcePicker onlyEmpty value={id} onChange={(s) => (window.location.hash = `#/conversations/${part(s)}`)} />
        <a href="#/new-source">New source</a>
      </p>
      <Problems problems={problems} />
      {id && (
        <>
          <ul>
            {shown.map((p) => <li key={p}><a href={`#/conversations/${part(id)}/${part(p)}`}>{p}</a></li>)}
          </ul>
          <form className="filters" onSubmit={(e) => { e.preventDefault(); window.location.hash = `#/conversations/${part(id)}/${part(newPath)}/new`; }}>
            <input value={newPath} aria-label="new chat file" onChange={(e) => setNewPath(e.target.value)} />
            <button type="submit" disabled={!newPath}>Start a conversation in this file</button>
          </form>
        </>
      )}
    </section>
  );
}

function ConversationsPage({ rest }: { rest: string[] }) {
  const [id, path, line] = rest;
  if (id && path && line) return <Editor id={id} path={path} line={line} />;
  if (id && path) return <FileView id={id} path={path} />;
  return <ChatFiles id={id ?? ""} />;
}

export const conversations: Page = { path: "conversations", title: "Conversations", Component: ConversationsPage };
