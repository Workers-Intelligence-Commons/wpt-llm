// Calls to the wpt-web server (its routes are described at /api/openapi.json).
// Every answer is JSON; a refusal carries {problems: string[]} with the rules'
// own messages: 404 not found, 409 changed since it was read, 422 refused by
// the rules of `wpt-dataset check`, 503 missing on this machine.

export type Answer<T> = { ok: true; data: T } | { ok: false; status: number; problems: string[] };

async function call<T>(method: string, path: string, body?: unknown): Promise<Answer<T>> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (response.ok && method !== "GET") window.dispatchEvent(new Event("wpt-saved"));
  return response.ok ? { ok: true, data } : { ok: false, status: response.status, problems: data.problems ?? [] };
}

export const get = <T,>(path: string) => call<T>("GET", path);
export const put = <T,>(path: string, body: unknown) => call<T>("PUT", path, body);
export const post = <T,>(path: string, body: unknown) => call<T>("POST", path, body);

// One part of a route: a path such as "answer-as-assistant/talks.jsonl" stays one part ("/" becomes %2F).
export const part = encodeURIComponent;

// What a save answers in words.
export function saying(a: Answer<unknown>, done: string): string {
  if (a.ok) return done;
  if (a.status === 409) return "Not saved: it changed on disk since it was read. Reload it.";
  return "Not saved.";
}
