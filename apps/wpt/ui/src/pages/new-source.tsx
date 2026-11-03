// "New source": text a person writes for WPT, written or pasted here, becomes
// a new empty source (CONTRIBUTING.md, "Adding your own texts"): its
// source.json, under review ("under-review") until a maintainer decides, and
// its text as the first patch. Or a source for conversations, written next in
// the conversation editor.

import { useState } from "react";
import { part, post } from "../api";
import { PatchName, Problems } from "../parts";
import type { Page } from "./index";

const KINDS = [
  { value: "text", label: "plain text" },
  { value: "markdown", label: "Markdown" },
  { value: "chat", label: "conversations (written in the conversation editor)" },
];

function NewSource() {
  const [origin, setOrigin] = useState("");
  const [item, setItem] = useState("");
  const [name, setName] = useState("");
  const [by, setBy] = useState("");
  const [license, setLicense] = useState("CC0-1.0");
  const [attribution, setAttribution] = useState("");
  const [licenseText, setLicenseText] = useState("");
  const [tags, setTags] = useState("");
  const [format, setFormat] = useState("text");
  const [path, setPath] = useState("text.txt");
  const [text, setText] = useState("");
  const [patch, setPatch] = useState("texts");
  const [problems, setProblems] = useState<string[]>([]);

  const create = async () => {
    const a = await post<{ id: string }>("/api/sources", {
      origin, item: item || null, name, by, license, attribution, license_text: licenseText,
      tags: tags.split(/[\s,]+/).filter(Boolean), files: { [path]: { format, text: format === "chat" ? "" : text } }, patch,
    });
    if (!a.ok) return setProblems(a.problems);
    window.location.hash = format === "chat" ? `#/conversations/${part(a.data.id)}/${part(path)}` : `#/sources/${part(a.data.id)}`;
  };

  const field = (label: string, value: string, set: (v: string) => void, hint = "") => (
    <div className="field">
      <label>{label} {hint && <span className="hint">{hint}</span>}</label>
      <div className="string"><input value={value} aria-label={label} onChange={(e) => set(e.target.value)} /></div>
    </div>
  );

  return (
    <section>
      <h1>New source</h1>
      <p className="hint">
        Text you wrote yourself, for WPT. Committed text stays public in the repository's history for good; it holds no
        other person's personal data and no output of another company's model. A maintainer admits it later, in a signed commit.
      </p>
      <Problems problems={problems} />
      <form className="object" onSubmit={(e) => { e.preventDefault(); create(); }}>
        {field("Origin", origin, setOrigin, "the id's first part, lowercase a-z 0-9 -")}
        {field("Item", item, setItem, "optional, the id's second part")}
        {field("Name", name, setName, "the address becomes empty:<name>")}
        {field("Your name", by, setBy, "who started it")}
        {field("License", license, setLicense, "for your text, such as CC0-1.0 (docs/licenses.md)")}
        {field("Attribution", attribution, setAttribution, "the credit you want")}
        <div className="field">
          <label>License text</label>
          <textarea value={licenseText} rows={4} aria-label="license text" onChange={(e) => setLicenseText(e.target.value)} />
        </div>
        {field("Tags", tags, setTags, "for every file, such as en literature")}
        <div className="field">
          <label>What it holds</label>
          <select value={format} aria-label="format" onChange={(e) => {
            setFormat(e.target.value);
            setPath(e.target.value === "chat" ? "answer-as-assistant/conversations.jsonl" : e.target.value === "markdown" ? "text.md" : "text.txt");
          }}>
            {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
          </select>
        </div>
        {field("File path", path, setPath)}
        {format !== "chat" && (
          <div className="field">
            <label>Your text</label>
            <textarea className="page" value={text} rows={20} aria-label="your text" onChange={(e) => setText(e.target.value)} />
          </div>
        )}
        <div className="actions">
          <PatchName value={patch} onChange={setPatch} />
          <button type="submit" disabled={!origin || !name || !by || (format !== "chat" && !text)}>Create the source</button>
        </div>
      </form>
    </section>
  );
}

export const newSource: Page = { path: "new-source", title: "New source", Component: NewSource };
