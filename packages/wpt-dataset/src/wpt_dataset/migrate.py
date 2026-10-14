"""Move data files from one shape to the next, so no file is edited by hand.

Each migration turns a document of shape N into shape N+1. To change a
shape: bump CURRENT_SCHEMA in repo.py, update the schema file, and register
the step here:

    MIGRATIONS["source"][1] = lambda doc: {**doc, "schema": 2, ...}

`wpt-dataset migrate` then rewrites every file that is behind, and checks
the result against the new schema.
"""

import json
from pathlib import Path

from .repo import CURRENT_SCHEMA, Repo, dump_json, load_json

MIGRATIONS: dict[str, dict[int, callable]] = {kind: {} for kind in CURRENT_SCHEMA}


def _source_1_to_2(document):
    """Shape 2 gives every license entry the credit it asks for. Only a
    person can write it, so the step leaves it empty and the check that
    follows names each file to fill in."""
    licenses = [{**entry, "attribution": entry.get("attribution", "")} for entry in document["licenses"]]
    return {**document, "schema": 2, "licenses": licenses}


MIGRATIONS["source"][1] = _source_1_to_2


def _source_2_to_3(document):
    """Shape 3 says how each file becomes Markdown for training (`formats`).
    Only a person knows each file's format, so the step leaves the list
    empty and the check that follows names each file to fill in."""
    return {**document, "schema": 3, "formats": document.get("formats", [])}


MIGRATIONS["source"][2] = _source_2_to_3


def _files(repo: Repo):
    for sid in repo.source_ids():
        yield "source", repo.data / "sources" / sid / "source.json"
    for kind in ("mixture", "tokenizer", "dataset"):
        for name in repo.recipe_names(kind):
            yield kind, repo.recipe_path(kind, name)
    for name in repo.recipe_names("tokenizer"):
        if repo.lock_path(name).exists():
            yield "lock", repo.lock_path(name)


def migrate(repo: Repo, migrations=None, current=None, log=print) -> list[Path]:
    """Rewrite every file whose shape is behind. Returns the files changed."""
    migrations = MIGRATIONS if migrations is None else migrations
    current = CURRENT_SCHEMA if current is None else current
    changed = []
    for kind, path in _files(repo):
        document = json.loads(path.read_text(encoding="utf-8"))
        start = document.get("schema")
        while document.get("schema", current[kind]) < current[kind]:
            step = migrations[kind].get(document["schema"])
            if step is None:
                raise KeyError(f"{path}: no migration for {kind} shape {document['schema']}")
            document = step(document)
        if document.get("schema") != start:
            path.write_text(dump_json(document), encoding="utf-8", newline="\n")
            changed.append(path)
            log(f"{path.relative_to(repo.root)}: shape {start} -> {document['schema']}")
    if current is CURRENT_SCHEMA:  # the real shapes: every file must now validate
        for kind, path in _files(repo):
            load_json(path, kind)
    return changed
