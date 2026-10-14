"""Checks the Markdown that markdown.py makes from a source's files.

markdown.py turns `text` and `xhtml` files into Markdown, so a new kind of
input can expose a mistake in it. This reads every such document and
reports each problem:

- a note reference ([^12]) without its definition ([^12]: ...), or a
  definition that nothing refers to;
- an HTML tag left unescaped (outside inline code), which a Markdown reader
  would treat as markup instead of showing it.

`markdown` files are kept as written (HTML inside them is valid Markdown)
and `code` files are one fenced block, so neither is checked.
"""

import re
from dataclasses import dataclass, field

from .markdown import documents, format_of
from .repo import Source
from .sources import source_tree

CHECKED = {"text", "xhtml"}
CODE_SPAN = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)", re.S)
REFERENCE = re.compile(r"(?<!\\)\[\^([^\]\s]+)\](?!:)")
DEFINITION = re.compile(r"^\[\^([^\]\s]+)\]:", re.M)
TAG = re.compile(r"(?<!\\)</?[A-Za-z][A-Za-z0-9-]*(?:\s|/?>)")


@dataclass
class SourceReport:
    source: str
    documents: int = 0
    notes: int = 0
    problems: list[str] = field(default_factory=list)  # "path: what is wrong"


def problems(markdown: str) -> list[str]:
    """What is wrong in one converted document (empty when nothing is)."""
    text = CODE_SPAN.sub("", markdown)
    references, definitions = set(REFERENCE.findall(text)), set(DEFINITION.findall(text))
    found = [f"note [^{label}] has no definition" for label in sorted(references - definitions)]
    found += [f"note [^{label}] is defined but never referred to" for label in sorted(definitions - references)]
    found += [f"unescaped HTML tag {match.group().strip()!r}" for match in TAG.finditer(text)]
    return found


def check_markdown(source: Source, store) -> SourceReport:
    tree, _ = source_tree(source, store)
    report = SourceReport(source.id)
    for path, markdown in documents(source, tree, store):
        if format_of(source, path)["format"] not in CHECKED:
            continue
        text = markdown.decode("utf-8")
        report.documents += 1
        report.notes += len(DEFINITION.findall(CODE_SPAN.sub("", text)))
        report.problems += [f"{path}: {problem}" for problem in problems(text)]
    return report
