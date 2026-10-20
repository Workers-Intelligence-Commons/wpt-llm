"""Every training document is Markdown.

A source's `formats` list maps each file to one format (every path matches
exactly one entry, as with licenses), and the build trains on the file's
Markdown, never on the file itself:

- `markdown`: kept as it is.
- `text`: plain text. Blank lines separate paragraphs, each line break
  inside a paragraph becomes a Markdown line break (a backslash at the end
  of the line), spaces at the start and end of lines are removed, and
  characters with a meaning in Markdown are escaped, so the text reads
  exactly as written.
- `code`: one fenced code block whose opening line names the language and
  the file's path, as in ```python path="./src/main.py". In a git or
  Mercurial source the path starts with "./", the repository's root; in
  any other source (an archive, a single file) it is written as it is,
  ```python path="src/main.py", since no repository root exists.
- `xhtml`: web pages written as strict XML (EPUB books), read by lxml's XML
  parser; a file that is not well-formed stops the build. Headings become
  `#` headings, paragraphs paragraphs, lists lists, quotations `>`
  quotations, italics and bold `*` and `**`, a line break a Markdown line
  break, and tables Markdown tables, except dialogue rows (a cell marked as
  the speaker, epub:type z3998:persona), which become "**Speaker:** words".
  The page's head, scripts, styles and images are left out. A note
  reference (epub:type noteref) whose note exists in the source becomes a
  Markdown note, [^12], with the note's text at the end of the document
  that refers to it; a reference whose note is missing is dropped. Notes
  sections themselves (epub:type endnotes or footnotes) are left where they
  were, so a file holding only notes gives no document.

- `chat`: conversations for post-training, one per line, or with `tree`
  one message per record naming its parent (see conversations.py). Each
  conversation is checked, and the file gives no document: only the
  post-training step named by its tags reads it.
- `task`: the files of task folders for post-training (see tasks.py).
  Each folder is checked, and its files give no document: only the
  post-training step named by the tags of its task.json reads it.
- `records`: one document per record whose license enters (see
  records.py), at the path `<file>#<record id>`; its text field is read as
  `text` above, or kept as it is when the entry says `"as": "markdown"`.

A file whose Markdown is empty gives no document.
"""

import posixpath
import re
from collections.abc import Iterator
from typing import NamedTuple
from urllib.parse import unquote, urlparse

from lxml import etree

from . import conversations, records, tasks
from .errors import FormatMapError, NotText
from .manifest import SYMLINK, glob_match_any, license_entries

EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
SKIPPED = {"head", "script", "style", "img", "svg", "math", "audio", "video", "object", "template", "noscript"}
CONTAINERS = {"body", "div", "section", "article", "header", "footer", "main", "aside", "nav", "figure", "hgroup",
              "html"}
PARAGRAPHS = {"p", "dt", "dd", "figcaption", "caption", "address", "summary"}
HEADINGS = {f"h{n}": n for n in range(1, 7)}
NOTES_SECTIONS = {"endnotes", "footnotes", "rearnotes"}
NOTES = {"endnote", "footnote", "rearnote"}
SPACES = re.compile(r"[ \t\r\n\f]+")  # XML white space only: a no-break space or word joiner stays
ALWAYS = re.compile(r"([\\`*_\[\]])")
TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
ENTITY = re.compile(r"&(?=#?[A-Za-z0-9]+;)")
LINE_START = re.compile(r"^([#>+=|~-])")
NUMBERED = re.compile(r"^(\d+)([.)])")
REPOSITORIES = {"git", "mercurial"}  # source types whose paths start at a repository's root


def format_of(source, path: str) -> dict:
    found = [e for e in source.data["formats"] if glob_match_any(path, e["paths"])]
    if len(found) != 1:
        what = "no format entry" if not found else f"{len(found)} format entries"
        raise FormatMapError(f"{source.id}: {path} matches {what}; every path must match exactly 1")
    return found[0]


class Converted(NamedTuple):
    path: str  # the document's path in its source: the file's, or <file>#<record id>
    markdown: bytes
    file: str  # the file it comes from
    record: records.Record | None  # for a records file, the record it comes from


def documents(source, tree, store, repo, skipped: dict | None = None) -> Iterator[Converted]:
    """Each document of a final tree, in file path order; links, chat and
    task files and documents whose Markdown is empty are left out; repo
    holds the images task folders name. skipped, when given, receives
    file -> records.Skipped for each records file.
    Raises FormatMapError when a path matches no format entry or several,
    NotText when an xhtml file is not well-formed, InvalidFile when a chat or
    records file holds a line that is not valid or a task folder is not."""
    files = {path: entry for path, entry in tree.items() if entry.mode != SYMLINK}
    formats = {path: format_of(source, path) for path in files}
    pages = {}  # parsed xhtml files, also read for the notes they hold

    def page(path):
        if path not in pages:
            pages[path] = _parse(source.id, path, store.get_bytes(files[path].sha256))
        return pages[path]

    def note(path, element_id):
        if formats.get(path, {}).get("format") != "xhtml":
            return None
        return next((e for e in page(path).iter() if isinstance(e.tag, str) and e.get("id") == element_id), None)

    for path in sorted(files, key=str.encode):
        entry, kind = formats[path], formats[path]["format"]
        if kind == "records":
            yield from _records(source, path, entry, store.path(files[path].sha256), skipped)
            continue
        if kind == "chat":
            conversations.of_file(source.id, path, entry, store, files[path].sha256)
            continue
        if kind == "task":
            continue  # checked by folder, below
        data = store.get_bytes(files[path].sha256)
        if kind == "markdown":
            markdown = data
        elif kind == "text":
            markdown = text(data.decode("utf-8"))
        elif kind == "code":
            markdown = code(data.decode("utf-8"), entry["language"], path, source.data["type"] in REPOSITORIES)
        else:
            markdown = Page(path, note).render(page(path))
        if markdown.strip():
            yield Converted(path, markdown, path, None)
    tasks.of_source(source, tree, store, repo)


def _records(source, path, entry, file, skipped):
    licenses = {value: license_entry["license"] for license_entry in license_entries(source.data["licenses"], path)
                for value in license_entry.get("values", [])}
    report = records.Skipped({})
    if skipped is not None:
        skipped[path] = report
    for record in records.records(file, path, entry, licenses, report):
        markdown = record.text.encode("utf-8") if entry.get("as") == "markdown" else text(record.text)
        if markdown.strip():
            yield Converted(f"{path}#{record.id}", markdown, path, record)
        else:
            report.empty += 1


# -- text and code ----------------------------------------------------------------

def escape(text: str) -> str:
    """Text that Markdown shows exactly as written, inside a line."""
    return ENTITY.sub(r"\\&", TAG_START.sub(r"\\<", ALWAYS.sub(r"\\\1", text)))


def protect(block: str) -> str:
    """Escape what would start a heading, quotation, list, table or fence."""
    return "\n".join(NUMBERED.sub(r"\1\\\2", LINE_START.sub(r"\\\1", line)) for line in block.split("\n"))


def text(content: str) -> bytes:
    content = content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs, lines = [], []
    for line in content.split("\n") + [""]:
        if line.strip():
            lines.append(protect(escape(line.strip())))
        elif lines:
            paragraphs.append("\\\n".join(lines))
            lines = []
    return ("\n\n".join(paragraphs) + "\n").encode() if paragraphs else b""


def code(content: str, language: str, path: str, from_repository: bool) -> bytes:
    shown = ("./" if from_repository else "") + path.replace("\\", "\\\\").replace('"', '\\"')
    return (fenced(content.removeprefix("\ufeff"), f'{language} path="{shown}"') + "\n").encode()


def fenced(content: str, info: str) -> str:
    """A code block holding content exactly, its fence longer than any run inside."""
    if not content.endswith("\n"):
        content += "\n"
    mark = "~" if "`" in info else "`"  # an info string after backticks cannot hold a backtick
    longest = max((len(run) for run in re.findall(re.escape(mark) + "+", content)), default=0)
    fence = mark * max(3, longest + 1)
    return f"{fence}{info}\n{content}{fence}"


# -- xhtml ------------------------------------------------------------------------

def _parse(source_id, path, data):
    # no DTD, no outside entities, no network: the file alone decides its text
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        return etree.fromstring(data, parser)
    except etree.XMLSyntaxError as error:
        raise NotText(f"{source_id}: {path} is not well-formed XHTML: {error}") from None


def _name(element) -> str | None:
    return etree.QName(element).localname if isinstance(element.tag, str) else None


def _kinds(element) -> set[str]:
    return set((element.get(EPUB_TYPE) or "").split())


class Page:
    """One xhtml file as Markdown, with the notes it refers to at its end."""

    def __init__(self, path, find_note):
        self.path, self.find_note = path, find_note
        self.notes = {}  # label -> note element, in the order first referred to
        self.targets = {}  # (file, id) -> label

    def render(self, root) -> bytes:
        body = self.blocks(root)
        written, definitions = set(), []
        while len(written) < len(self.notes):  # a note may refer to further notes
            for label, element in list(self.notes.items()):
                if label not in written:
                    written.add(label)
                    definitions.append(self.definition(label, element))
        markdown = "\n\n".join(body + definitions)
        return (markdown + "\n").encode() if markdown.strip() else b""

    def definition(self, label, element) -> str:
        blocks = self.blocks(element)
        lines = "\n\n".join(blocks).split("\n")
        return "\n".join([f"[^{label}]: {lines[0]}"] + [f"    {line}" if line else "" for line in lines[1:]])

    # blocks: a list of Markdown blocks, joined later by blank lines
    def blocks(self, element) -> list[str]:
        out, inline = [], [escape_text(element.text)]

        def flush():
            paragraph = clean("".join(inline))
            if paragraph:
                out.append(protect(paragraph))
            inline.clear()

        for child in element:
            name = _name(child)
            if name is None or name in SKIPPED or _kinds(child) & (NOTES_SECTIONS | NOTES) and name != "a":
                pass
            elif name in HEADINGS or name in PARAGRAPHS or name in CONTAINERS or name in (
                    "blockquote", "ul", "ol", "table", "hr", "pre", "dl", "li", "tr"):
                flush()
                out += self.block(child, name)
            else:
                inline.append(self.inline(child))
            inline.append(escape_text(child.tail))
        flush()
        return out

    def block(self, element, name) -> list[str]:
        if name in HEADINGS:
            heading = clean(self.inline_content(element)).replace("\\\n", " ")
            return [f"{'#' * HEADINGS[name]} {heading}"] if heading else []
        if name in PARAGRAPHS:
            paragraph = clean(self.inline_content(element))
            return [protect(paragraph)] if paragraph else []
        if name == "blockquote":
            inner = "\n\n".join(self.blocks(element))
            return ["\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))] if inner else []
        if name in ("ul", "ol"):
            return self.list(element, name)
        if name == "table":
            return self.table(element)
        if name == "hr":
            return ["---"]
        if name == "pre":
            return [fenced("".join(element.itertext()), "")]
        return self.blocks(element)  # containers, dl, a stray li or tr

    def list(self, element, name) -> list[str]:
        items, number = [], int(element.get("start", "1")) if name == "ol" else 0
        for child in element:
            if _name(child) != "li":
                continue
            marker = f"{number}. " if name == "ol" else "- "
            number += 1
            body = "\n\n".join(self.blocks(child)).split("\n")
            indent = " " * len(marker)
            items.append("\n".join([marker + body[0]] + [indent + line if line else "" for line in body[1:]]))
        loose = any("\n\n" in item for item in items)
        return [("\n\n" if loose else "\n").join(items)] if items else []

    def table(self, element) -> list[str]:
        rows = [row for row in element.iter() if _name(row) == "tr"]
        cells = [[cell for cell in row if _name(cell) in ("td", "th")] for row in rows]
        if any("z3998:persona" in _kinds(c) for row in cells for c in row):  # a dialogue
            out = []
            for row in cells:
                speaker = [c for c in row if "z3998:persona" in _kinds(c)]
                words = [b for c in row if c not in speaker for b in self.blocks(c)]
                if speaker:
                    name = clean(self.inline_content(speaker[0])).replace("\\\n", " ")
                    words = [f"**{name}:** {words[0] if words else ''}".rstrip()] + words[1:]
                out += words
            return out
        if not cells:
            return []
        width = max(len(row) for row in cells)
        text = [[clean(" ".join(self.blocks(c))).replace("\\\n", " ").replace("|", "\\|") for c in row]
                + [""] * (width - len(row)) for row in cells]
        header = text.pop(0) if all(_name(c) == "th" for c in cells[0]) else [""] * width
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width] + ["| " + " | ".join(r) + " |" for r in text]
        return ["\n".join(lines)]

    # inline: one line of Markdown text
    def inline_content(self, element) -> str:
        parts = [escape_text(element.text)]
        for child in element:
            parts.append(self.inline(child))
            parts.append(escape_text(child.tail))
        return "".join(parts)

    def inline(self, element) -> str:
        name, kinds = _name(element), _kinds(element)
        if name is None or name in SKIPPED or "backlink" in kinds:
            return ""
        if name == "br":
            return "\\\n"
        if "noteref" in kinds:
            return self.reference(element)
        inner = self.inline_content(element)
        if name in ("i", "em", "cite", "dfn", "var"):
            return wrap(inner, "*")
        if name in ("b", "strong"):
            return wrap(inner, "**")
        if name in ("code", "kbd", "samp"):
            raw = "".join(element.itertext())
            ticks = "`" * (max((len(r) for r in re.findall("`+", raw)), default=0) + 1)
            return f"{ticks} {raw} {ticks}" if raw.startswith("`") or raw.endswith("`") else f"{ticks}{raw}{ticks}"
        if name == "a" and urlparse(element.get("href") or "").scheme in ("http", "https") and inner.strip():
            return f"[{inner}](<{element.get('href')}>)"
        if name in HEADINGS or name in PARAGRAPHS or name in CONTAINERS:  # a block inside a line: keep its words
            return " " + inner + " "
        return inner

    def reference(self, element) -> str:
        href = urlparse(element.get("href") or "")
        if href.scheme or href.netloc or not href.fragment:
            return ""
        file = posixpath.normpath(posixpath.join(posixpath.dirname(self.path), unquote(href.path))) if href.path \
            else self.path
        target = (file, unquote(href.fragment))
        if target not in self.targets:
            found = self.find_note(*target)
            if found is None:
                return ""  # the note is not in the source: drop the reference
            label = re.sub(r"[^A-Za-z0-9_-]", "", "".join(element.itertext())) or re.sub(r"[^A-Za-z0-9_-]", "", target[1])
            while label in self.notes:
                label += "x"
            self.targets[target], self.notes[label] = label, found
        return f"[^{self.targets[target]}]"


def escape_text(text) -> str:
    return escape(SPACES.sub(" ", text)) if text else ""


def clean(markdown: str) -> str:
    """Single spaces, no space at the start or end of a line."""
    lines = (SPACES.sub(" ", line).strip() for line in markdown.split("\\\n"))
    return "\\\n".join(line for line in lines if line)


def wrap(inner: str, mark: str) -> str:
    """*inner*, with spaces kept outside the marks (Markdown needs that)."""
    core = inner.strip()
    if not core:
        return inner
    return inner[:len(inner) - len(inner.lstrip())] + mark + core + mark + inner[len(inner.rstrip()):]
