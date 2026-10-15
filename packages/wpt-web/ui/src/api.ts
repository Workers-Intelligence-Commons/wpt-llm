// Calls to the wpt-web server. Every answer is JSON; a refusal carries
// {problems: string[]} with the rules' own messages.

export type Answer<T> = { ok: true; data: T } | { ok: false; status: number; problems: string[] };

async function call<T>(method: string, path: string, body?: unknown): Promise<Answer<T>> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  return response.ok ? { ok: true, data } : { ok: false, status: response.status, problems: data.problems ?? [] };
}

export const get = <T,>(path: string) => call<T>("GET", path);
export const put = <T,>(path: string, body: unknown) => call<T>("PUT", path, body);
