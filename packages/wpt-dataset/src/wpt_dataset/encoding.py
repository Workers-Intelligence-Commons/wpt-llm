"""The text of a source's file, in the encoding the file itself states.

A file whose encoding is stated in the file itself is text in that
encoding, decoded to Unicode, and whatever the pipeline stores or emits
from it is UTF-8. Every reader of a
source's files as text decodes them here: the check and fill-checksum
(sources.not_text), the Markdown of every format (markdown.py, so the
build and markdown-check too), wpt-web's file views and the evaluations
that read a source's files. The statements read, the first that applies:

- a byte-order mark, in any file: UTF-8, UTF-16 LE or BE, UTF-32 LE or BE;
  the mark stays at the start of the text as U+FEFF, as UTF-8's always did;
- an `html` file, or `code` in html: the HTML standard's prescan of the
  first 1024 bytes, <meta charset="..."> or <meta http-equiv="Content-Type"
  content="...; charset=...">, comments skipped; a label means what the
  Encoding Standard says it means where Python reads it otherwise
  (iso-8859-1, latin1, ascii and windows-1252's other labels are
  windows-1252, and a UTF-16 label is UTF-8, as the prescan says);
- an `html` file (or `code` in html) whose prescan finds none, such as
  Project Gutenberg's older XHTML pages named .html: the XML declaration's
  encoding="...", its label read as the prescan reads one (the HTML
  standard's prescan reads an XML declaration the same way);
- an `xhtml` file, or `code` in xml or svg: the XML declaration's
  encoding="...";
- a `text` file: Project Gutenberg's header line "Character set encoding:
  NAME", before the header's "*** START OF" line; ASCII is Python's ascii,
  so a byte above 127 does not decode;
- `code` in python: the PEP 263 coding line, in the first two lines;
- `code` in css: the @charset "..."; that starts the file.

A file that states no encoding is UTF-8 text, or no text at all, as before:
nothing is guessed. Nothing in `chat` and `rating` files is read as a
statement, since they are JSON, UTF-8 by its standard, nor in a `task`
folder's files, UTF-8 by the task format (docs/post-training.md). A stated
encoding Python does not know, or bytes that do not decode under it, make
the file no text, and NotText names why. A file written in place of
another (wpt-web's saves) is encoded the same way (encode).

Nothing rewrites a file's bytes: its checksum, `unpack` and the patches
keep them as upstream wrote them, and git's diffs carry them as they are.
"""

import codecs
import re

from .errors import NotText
from .manifest import format_entries

BOMS = [(codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be"), (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")]  # UTF-32 LE before UTF-16 LE
UTF_8 = {"chat", "task", "rating"}  # formats whose own standard says UTF-8
SPACE = rb"[\t\n\f\r ]"
COMMENT = re.compile(rb"<!--.*?(?:-->|\Z)", re.S)
META = re.compile(rb"<meta[\t\n\f\r /]([^>]*)", re.I)
ATTRIBUTE = re.compile(rb"([^\t\n\f\r />=]+)(?:" + SPACE + rb"*=" + SPACE + rb"*(\"[^\"]*\"|'[^']*'|[^\t\n\f\r >]+))?")
CHARSET = re.compile(rb"charset" + SPACE + rb"*=" + SPACE + rb"*(?:\"([^\"]*)\"|'([^']*)'|([^\t\n\f\r ;\"']+))", re.I)
WINDOWS_1252 = {"ansi_x3.4-1968", "ascii", "cp1252", "cp819", "csisolatin1", "ibm819", "iso-8859-1", "iso-ir-100",
                "iso8859-1", "iso88591", "iso_8859-1", "iso_8859-1:1987", "l1", "latin1", "us-ascii", "windows-1252",
                "x-cp1252"}  # the Encoding Standard's labels of windows-1252
UTF_16 = {"csunicode", "iso-10646-ucs-2", "ucs-2", "unicode", "unicodefeff", "utf-16", "utf-16le", "unicodefffe",
          "utf-16be"}
XML_DECLARATION = re.compile(rb"<\?xml[\t\n\r ][^>]*?encoding[\t\n\r ]*=[\t\n\r ]*(?:\"([^\"]*)\"|'([^']*)')")
GUTENBERG_START = re.compile(rb"^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG", re.M | re.I)
GUTENBERG_LINE = re.compile(rb"^Character set encoding:[ \t]*([^\r\n]*?)[ \t]*\r?$", re.M)
GUTENBERG_NAMES = {"iso-latin-1": "latin-1"}  # Project Gutenberg's spellings that Python does not know
PEP_263 = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")
COMMENT_LINE = re.compile(rb"^[ \t\f]*(?:#|\r?$)")
CSS_CHARSET = re.compile(rb'@charset "([^"]*)";')


def stated(data: bytes, entry: dict | None) -> tuple[str, str] | None:
    """(the encoding's name as the file states it, the Python codec it
    means), or None when data states none; entry is the file's format
    entry, None when the format map gives none."""
    for mark, codec in BOMS:
        if data.startswith(mark):
            return f"{codec.upper()} byte-order mark", codec
    form = (entry or {}).get("format")
    language = entry.get("language") if form == "code" else None
    if form == "html" or language == "html":
        return _web(_prescan(data[:1024]) or _xml_label(data))
    if language == "css":
        found = CSS_CHARSET.match(data)
        return _web(found[1] if found else None)
    if form == "xhtml" or language in ("xml", "svg"):
        return _named(_xml_label(data))
    if form == "text":
        start = GUTENBERG_START.search(data)
        found = start and GUTENBERG_LINE.search(data, 0, start.start())
        return _named(found and found[1], GUTENBERG_NAMES)
    if language == "python":
        for line in data.split(b"\n", 2)[:2]:
            if found := PEP_263.match(line):
                return _named(found[1])
            if not COMMENT_LINE.match(line):
                break
    return None


def _named(label: bytes | None, spellings=None) -> tuple[str, str] | None:
    if not label:
        return None
    name = label.decode("ascii", errors="replace")
    return name, (spellings or {}).get(name.lower(), name)


def _xml_label(data: bytes) -> bytes | None:
    found = XML_DECLARATION.match(data)
    return found and next(group for group in found.groups() if group is not None)


def _prescan(head: bytes) -> bytes | None:
    """The label of the first <meta> that states an encoding."""
    for meta in META.finditer(COMMENT.sub(b"", head)):
        attributes = {}
        for name, value in ATTRIBUTE.findall(meta[1]):
            attributes.setdefault(name.lower(), value[1:-1] if value[:1] in (b'"', b"'") else value)
        if attributes.get(b"charset"):
            return attributes[b"charset"]
        if attributes.get(b"http-equiv", b"").lower() == b"content-type":
            if found := CHARSET.search(attributes.get(b"content", b"")):
                return next(group for group in found.groups() if group is not None)
    return None


def _web(label: bytes | None) -> tuple[str, str] | None:
    """A label as the Encoding Standard reads it, which the HTML and CSS
    standards use."""
    found = _named(label.strip() if label else None)
    if found is None:
        return None
    key = found[0].lower()
    return found[0], "cp1252" if key in WINDOWS_1252 else "utf-8" if key in UTF_16 else found[1]


def _statement(data: bytes, entry: dict | None) -> tuple[str, str] | None:
    return None if (entry or {}).get("format") in UTF_8 else stated(data, entry)


def decode(data: bytes, entry: dict | None, name: str) -> str:
    """The text of a file, under the encoding it states (UTF-8 when it
    states none); NotText "<name> <why>" when it is not text."""
    found = _statement(data, entry)
    if found is None:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise NotText(f"{name} is not UTF-8 text") from None
    statement, codec = found
    try:
        return data.decode(codec)
    except LookupError:
        raise NotText(f"{name} states the encoding {statement!r}, which Python does not know") from None
    except UnicodeDecodeError as error:
        raise NotText(f"{name} is not text in {statement!r}, the encoding it states (byte {error.start})") from None


def encode(text: str, data: bytes, entry: dict | None, name: str) -> bytes:
    """text as the bytes of a file that replaces data, in the encoding data
    states (UTF-8 when it states none); NotText "<name> <why>" when that
    encoding cannot hold text."""
    statement, codec = _statement(data, entry) or ("UTF-8", "utf-8")
    try:
        return text.encode(codec)
    except UnicodeEncodeError as error:
        raise NotText(f"{name} is written in {statement!r}, which cannot hold {error.object[error.start]!r}") from None


def entry_of(source, path: str) -> dict | None:
    """The format entry of a source's path, None unless exactly one covers it."""
    found = format_entries(source.data["formats"], path)
    return found[0] if len(found) == 1 else None


def source_text(source, path: str, data: bytes) -> str:
    """decode, for one file of a source."""
    return decode(data, entry_of(source, path), f"{source.id}: {path}")
