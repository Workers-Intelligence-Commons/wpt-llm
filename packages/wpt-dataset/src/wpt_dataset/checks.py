"""The checks every command that applies them shares (the build, the task
builder, `check --fetch`, the writes of the service layer), each defined
once here so they cannot disagree. They change nothing.

    text check          whether a file of a source is text: decoded under
                        the encoding it states (encoding.py, "Encodings" in
                        docs/architecture.md); symbolic links, records files
                        and message trees are read otherwise, and are not
                        asked
    pointer-task check  whether a task's passage, in the Markdown of its
                        document, still holds the sha256 its pointer names

The other shared checks live with what they check: the cut match in
erasure.py, the license match in the license check (license_check.py).
"""

import hashlib

from . import encoding, manifest
from .errors import NotText
from .manifest import SYMLINK


def asks_text(source, path: str, entry: manifest.Entry) -> bool:
    """Whether the text check applies to a file of the source: not a
    symbolic link, a records file nor a message tree (read otherwise)."""
    return entry.mode != SYMLINK and not any(
        found["format"] == "records" or (found["format"] == "chat" and "tree" in found)
        for found in manifest.format_entries(source, path))


def not_text(source, path: str, content: bytes) -> str | None:
    """Why a file of the source is not text under the encoding it states,
    or None when it is text."""
    try:
        encoding.decode(content, encoding.entry_of(source, path), path)
    except NotText as error:
        return str(error)
    return None


def passage_holds(markdown: bytes, span, sha256: str) -> bool:
    """Whether the bytes [start, end) of a document's Markdown are still
    the passage whose sha256 a pointer names."""
    start, end = span
    return end <= len(markdown) and hashlib.sha256(markdown[start:end]).hexdigest() == sha256
