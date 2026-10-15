import { useEffect, useState } from "react";
import { pages } from "./pages";

// The address after "#/" picks the page: #/sources/<id> gives page "sources"
// the rest ["<id>"]. Links are plain <a href="#/...">.
function location(): string[] {
  return window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}

export function App() {
  const [parts, setParts] = useState(location);
  useEffect(() => {
    const changed = () => setParts(location());
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  const page = pages.find((p) => p.path === parts[0]) ?? pages[0];
  return (
    <>
      <nav>
        <strong>WPT</strong>
        {pages.map((p) => (
          <a key={p.path} href={`#/${p.path}`} aria-current={p === page ? "page" : undefined}>
            {p.title}
          </a>
        ))}
      </nav>
      <main>
        <page.Component rest={parts.slice(1)} />
      </main>
    </>
  );
}
