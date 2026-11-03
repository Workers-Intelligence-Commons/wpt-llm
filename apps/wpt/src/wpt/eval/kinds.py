"""The kinds of text a held-out document belongs to, by one field of its
source (docs/evaluation.md, "Held-out loss"), which the held-out-loss test
groups its losses by. wpt eval's --kind-by (WPT_KIND_BY) names the field:

    origin    the source's origin, the part of its id before the first '-'
              (gutenberg, github, wikimedia, ...; docs/architecture.md, "The id")
    tag       the document's tags, as the source's tag entries give them now
              (wpt_dataset.tasks.document_tags): a document with several tags
              is in each of their kinds
    format    the document's formats, as the source's format entries give them now

A document none of the field's values covers is in the kind NONE.
"""

import functools

from wpt_dataset import manifest
from wpt_dataset.tasks import document_tags

NONE = "(none)"


def origin(repo, store, sid: str, path: str) -> frozenset[str]:
    return frozenset({sid.split("-", 1)[0]})


def tag(repo, store, sid: str, path: str) -> frozenset[str]:
    return document_tags(repo, store, sid, path)


def format_(repo, store, sid: str, path: str) -> frozenset[str]:
    entries = manifest.source_paths(repo.source(sid)).format_entries(path.split("#", 1)[0])
    return frozenset(entry["format"] for entry in entries)


FIELDS = {"origin": origin, "tag": tag, "format": format_}


class Sources:
    """The repository's sources, each read once (repo.source), so that the
    matcher of a source's entries is compiled once for all its documents."""

    def __init__(self, repo):
        self.source = functools.cache(repo.source)


def kinds_of(field: str, repo, store, document: str) -> frozenset[str]:
    """The kinds of the document <source id>/<path> by field (a key of
    FIELDS); repo: a Repo, or Sources of one."""
    sid, path = document.split("/", 1)
    return FIELDS[field](repo, store, sid, path) or frozenset({NONE})
