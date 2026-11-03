// An editor for any JSON document, built from its JSON schema: every field
// the schema names is shown and editable, so a new field in a shape needs no
// change here. A value that may take one of several shapes (oneOf, anyOf)
// shows the shape it has and lets a person pick another; an object whose
// keys are free (additionalProperties, such as a step's settings) lists its
// keys and takes new ones. It only shapes the input; the server's rules
// decide what is valid (conditions such as "if/then" are left to them).

import { useState } from "react";

export type Schema = {
  type?: string | string[];
  properties?: Record<string, Schema>;
  additionalProperties?: Schema | boolean;
  required?: string[];
  items?: Schema;
  enum?: string[];
  const?: unknown;
  oneOf?: Schema[];
  anyOf?: Schema[];
  $ref?: string;
  description?: string;
};

type Json = unknown;
type Props = { value: Json; schema: Schema; root: Schema; name: string; onChange: (value: Json) => void };

const LONG = new Set(["text", "attribution", "note", "reason"]);

function resolve(schema: Schema, root: Schema): Schema {
  if (!schema.$ref) return schema;
  const found = schema.$ref.replace(/^#\//, "").split("/").reduce<any>((at, key) => at?.[key], root);
  return found ? resolve(found, root) : schema;
}

// The JSON kind of a value, as a schema's type names it.
function kind(value: Json): string {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  if (typeof value === "number") return "number";
  return typeof value;
}

function fits(schema: Schema, value: Json): boolean {
  const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];
  const found = kind(value);
  return types.includes(found) || (found === "number" && types.includes("integer"));
}

// The alternative of a oneOf or anyOf the value has: of those of its kind,
// the first whose required fields it holds, else the first of its kind.
function chosen(choices: Schema[], value: Json, root: Schema): number {
  const resolved = choices.map((choice) => resolve(choice, root));
  const ofKind = resolved.map((s, i) => [s, i] as const).filter(([s]) => fits(s, value));
  const whole = ofKind.find(([s]) => (s.required ?? []).every((key) => value && typeof value === "object" && key in value));
  return (whole ?? ofKind[0] ?? [null, 0])[1];
}

// A value typed in a field that may be a number, a boolean or text.
function scalar(text: string): Json {
  if (text === "true" || text === "false") return text === "true";
  return /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text) ? Number(text) : text;
}

// A new value of that schema: its required fields, each at its own start.
export function initial(schema: Schema, root: Schema): Json {
  const s = resolve(schema, root);
  const choices = s.oneOf ?? s.anyOf;
  if (choices) return initial(choices[0], root);
  if (s.const !== undefined) return s.const;
  if (s.enum) return s.enum[0];
  if (s.type === "object")
    return Object.fromEntries((s.required ?? []).map((key) => [key, initial(s.properties?.[key] ?? {}, root)]));
  if (s.type === "array") return [];
  if (s.type === "integer" || s.type === "number") return 0;
  return "";
}

function isLink(value: Json): value is string {
  return typeof value === "string" && /^https?:\/\/\S+$/.test(value);
}

export function SchemaForm({ value, schema, root, name, onChange }: Props) {
  const s = resolve(schema, root);
  const title = s.description ? <span className="hint">{s.description}</span> : null;

  if (s.const !== undefined) return <code>{JSON.stringify(value)}</code>;

  const choices = s.oneOf ?? s.anyOf;
  if (choices) {
    const at = chosen(choices, value, root);
    return (
      <div className="choice">
        {choices.length > 1 && (
          <select value={at} aria-label={`${name} shape`}
            onChange={(e) => onChange(initial(choices[Number(e.target.value)], root))}>
            {choices.map((choice, i) => {
              const r = resolve(choice, root);
              return <option key={i} value={i}>{r.description ?? String(r.type ?? `shape ${i + 1}`)}</option>;
            })}
          </select>
        )}
        <SchemaForm value={value} schema={choices[at]} root={root} name={name} onChange={onChange} />
      </div>
    );
  }

  if (s.enum)
    return (
      <select value={String(value)} onChange={(e) => onChange(e.target.value)} aria-label={name}>
        {s.enum.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
    );

  if (s.type === "object" && !s.properties && typeof s.additionalProperties === "object")
    return <FreeObject value={value} schema={s} root={root} name={name} onChange={onChange} />;

  if (s.type === "object") {
    const object = (value && typeof value === "object" ? value : {}) as Record<string, Json>;
    const required = new Set(s.required ?? []);
    const keys = Object.keys(s.properties ?? {});
    const missing = keys.filter((key) => !(key in object));
    return (
      <div className="object">
        {title}
        {keys.filter((key) => key in object).map((key) => (
          <div className="field" key={key}>
            <label>
              {key}
              {!required.has(key) && (
                <button type="button" className="quiet" onClick={() => {
                  const { [key]: _, ...rest } = object;
                  onChange(rest);
                }}>remove</button>
              )}
            </label>
            <SchemaForm value={object[key]} schema={s.properties![key]} root={root} name={key}
              onChange={(v) => onChange({ ...object, [key]: v })} />
          </div>
        ))}
        {missing.length > 0 && (
          <div className="add">
            {missing.map((key) => (
              <button type="button" key={key}
                onClick={() => onChange({ ...object, [key]: initial(s.properties![key], root) })}>+ {key}</button>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (s.type === "array") {
    const list = Array.isArray(value) ? value : [];
    return (
      <div className="array">
        {title}
        {list.map((item, i) => (
          <div className="item" key={i}>
            <SchemaForm value={item} schema={s.items ?? {}} root={root} name={`${name} ${i + 1}`}
              onChange={(v) => onChange(list.map((old, j) => (j === i ? v : old)))} />
            <button type="button" className="quiet" onClick={() => onChange(list.filter((_, j) => j !== i))}>
              remove
            </button>
          </div>
        ))}
        <button type="button" onClick={() => onChange([...list, initial(s.items ?? {}, root)])}>+ add</button>
      </div>
    );
  }

  if (s.type === "integer" || s.type === "number")
    return <input type="number" value={Number(value)} aria-label={name}
      onChange={(e) => onChange(s.type === "integer" ? parseInt(e.target.value || "0", 10) : Number(e.target.value))} />;

  if (Array.isArray(s.type))  // a number, a boolean or text, as typed
    return <input value={typeof value === "string" ? value : JSON.stringify(value)} aria-label={name}
      onChange={(e) => onChange(scalar(e.target.value))} />;

  const text = typeof value === "string" ? value : JSON.stringify(value);
  const long = LONG.has(name) || text.includes("\n") || text.length > 90;
  return (
    <div className="string">
      {long ? (
        <textarea value={text} rows={Math.min(20, Math.max(3, text.split("\n").length + 1))} aria-label={name}
          onChange={(e) => onChange(e.target.value)} />
      ) : (
        <input value={text} aria-label={name} onChange={(e) => onChange(e.target.value)} />
      )}
      {isLink(value) && <a href={value} target="_blank" rel="noreferrer">open</a>}
    </div>
  );
}

// An object whose keys are free: each key with its value, removable, and a
// name to add a new key.
function FreeObject({ value, schema, root, name, onChange }: Props) {
  const [key, setKey] = useState("");
  const object = (value && typeof value === "object" && !Array.isArray(value) ? value : {}) as Record<string, Json>;
  const item = schema.additionalProperties as Schema;
  return (
    <div className="object">
      {schema.description ? <span className="hint">{schema.description}</span> : null}
      {Object.keys(object).map((k) => (
        <div className="field" key={k}>
          <label>
            {k}
            <button type="button" className="quiet" onClick={() => {
              const { [k]: _, ...rest } = object;
              onChange(rest);
            }}>remove</button>
          </label>
          <SchemaForm value={object[k]} schema={item} root={root} name={k}
            onChange={(v) => onChange({ ...object, [k]: v })} />
        </div>
      ))}
      <div className="add">
        <input value={key} placeholder="new name" aria-label={`new ${name} name`} onChange={(e) => setKey(e.target.value)} />
        <button type="button" disabled={!key || key in object}
          onClick={() => { onChange({ ...object, [key]: initial(item, root) }); setKey(""); }}>+ add</button>
      </div>
    </div>
  );
}
