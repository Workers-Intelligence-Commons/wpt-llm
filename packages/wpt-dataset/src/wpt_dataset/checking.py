"""Which sources `wpt-dataset check` downloads and then checks as `check
--fetch` does (Repo.check_fetched). It changes nothing: it reads data/
through the service layer, and, for `--changed`, the store, to find the
sources the pointer tasks point into.

    --fetch with ids               those ids
    --fetch alone                  every admitted source
    --changed REV                  the sources changed since REV, and:
                                   every source with task files when data/images/ changed (a task
                                   may name an image that changed or went); otherwise, when a source
                                   changed or went, every task source whose pointer tasks point into
                                   one of them, or that cannot be read (the check then names why)
    neither                        none

A source with no text yet (Source.unwritten) is never fetched: the check
names it (Selection.unwritten), and it is not a failure.
"""

from dataclasses import dataclass

from wpt_store import StoreError
from wpt_utils.programs import MissingTool

from .download import wanted
from .errors import DatasetError
from .repo import Repo, Source
from .sources import source_tree
from .tasks import has_tasks, pointed_sources


@dataclass(frozen=True)
class Selection:
    """ids: the sources to download and check, in order; unwritten: {source
    id: note} of the sources with no text yet, named and left out."""
    ids: list[str]
    unwritten: dict[str, str]


def to_fetch(repo: Repo, open_store, changed: str | None = None, fetch: list[str] | None = None) -> Selection:
    """The sources a check downloads (see the module's docstring); open_store()
    gives the store, asked only when pointer tasks must be read."""
    unwritten = {s.id: s.note for s in repo.sources() if s.unwritten}
    ids = []
    if changed:
        ids, images_changed, removed = repo.changed_since(changed)
        if images_changed:
            ids = sorted(set(ids) | {s.id for s in repo.sources() if has_tasks(s)})
        elif ids or removed:
            store, gone = open_store(), set(ids) | set(removed)
            ids = sorted(set(ids) | {s.id for s in repo.sources() if has_tasks(s) and not s.unwritten
                                    and ((pointed := _points_into(repo, s, store)) is None or pointed & gone)})
    elif fetch is not None:
        ids = wanted(repo, fetch)
    return Selection([sid for sid in ids if sid not in unwritten], unwritten)


def _points_into(repo: Repo, source: Source, store) -> set[str] | None:
    """The sources the pointer tasks of a task source point into; None for
    a task source that cannot be read (also one whose pointers name a
    source that was deleted)."""
    try:
        return pointed_sources(source, source_tree(source, store)[0], store, repo)
    except (DatasetError, StoreError, MissingTool):
        return None
