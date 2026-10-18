// An editor for any JSON document, built from its JSON schema: every field
// the schema names is shown and editable, so a new field in a shape needs no
// change here. It only shapes the input; the server's rules decide what is
// valid (conditions such as "if/then" are left to them).

export type Schema = {
  type?: string;
  properties?: Record<string, Schema>;
  required?: string[];
  items?: Schema;
  enum?: string[];
  const?: unknown;
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

// A new value of that schema: its required fields, each at its own start.
export function initial(schema: Schema, root: Schema): Json {
  const s = resolve(schema, root);
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

  if (s.enum)
    return (
      <select value={String(value)} onChange={(e) => onChange(e.target.value)} aria-label={name}>
        {s.enum.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
    );

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
