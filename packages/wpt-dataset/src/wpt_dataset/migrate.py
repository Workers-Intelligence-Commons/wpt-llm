"""Move data files from one shape to the next, so no file is edited by hand.

Each migration turns a document of shape N into shape N+1. To change a
shape: bump CURRENT_SCHEMA in repo.py, update the schema file, and register
the step here:

    MIGRATIONS["source"][1] = lambda doc, path: {**doc, "schema": 2, ...}

A step gets the file's path too, for the rare step that reads a file next
to it.

`wpt-dataset migrate` then rewrites every file that is behind, and checks
the result against the new schema.
"""

import json
from pathlib import Path

from .errors import DatasetError, OutdatedSchema
from .repo import CURRENT_SCHEMA, Repo, dump_json, load_json

MIGRATIONS: dict[str, dict[int, callable]] = {kind: {} for kind in CURRENT_SCHEMA}


def _source_1_to_2(document, _path):
    """Shape 2 gives every license entry the credit it asks for. Only a
    person can write it, so the step leaves it empty and the check that
    follows names each file to fill in."""
    licenses = [{**entry, "attribution": entry.get("attribution", "")} for entry in document["licenses"]]
    return {**document, "schema": 2, "licenses": licenses}


MIGRATIONS["source"][1] = _source_1_to_2


def _source_2_to_3(document, _path):
    """Shape 3 says how each file becomes Markdown for training (`formats`).
    Only a person knows each file's format, so the step leaves the list
    empty and the check that follows names each file to fill in."""
    return {**document, "schema": 3, "formats": document.get("formats", [])}


MIGRATIONS["source"][2] = _source_2_to_3


def _license_file(entry, path):
    """The LicenseRef-<name>.txt next to source.json that shape 3 kept for a
    LicenseRef license, or None."""
    for alternative in entry["license"].split(" OR "):
        candidate = path.parent / f"{alternative.partition(' WITH ')[0]}.txt"
        if alternative.startswith("LicenseRef-") and candidate.is_file():
            return candidate
    return None


def _source_3_to_4(document, path):
    """Shape 4 keeps every license's text in its entry (`text`), so that it
    travels into each dataset record and model NOTICE. A LicenseRef text
    moves in from its file; any other text is only on the source's side, so
    the step leaves it empty and the check that follows names each file to
    fill in."""
    licenses = []
    for entry in document["licenses"]:
        found = _license_file(entry, path)
        text = found.read_text(encoding="utf-8") if found else entry.get("text", "")
        licenses.append({**entry, "text": text})
    return {**document, "schema": 4, "licenses": licenses}


MIGRATIONS["source"][3] = _source_3_to_4


def _source_4_to_5(document, _path):
    """Shape 5 adds the huggingface type, the records format and the `values`
    of a license entry; a shape 4 file uses none of them, so only its number
    changes."""
    return {**document, "schema": 5}


MIGRATIONS["source"][4] = _source_4_to_5


def _drop_moved_license_files(path, document):
    """Remove the LicenseRef files whose text the written source.json now
    holds exactly; anything else stays for the check to report."""
    for entry in document["licenses"]:
        found = _license_file(entry, path)
        if found and found.read_text(encoding="utf-8") == entry["text"]:
            found.unlink()


def _tokenizer_1_to_2(document, _path):
    """Shape 2 gives the recipe a vocabulary `version`; existing ones start at 1."""
    return {**document, "schema": 2, "version": 1}


MIGRATIONS["tokenizer"][1] = _tokenizer_1_to_2


def _lock_1_to_2(document, _path):
    """Shape 2 lists the documents the vocabulary was trained on, which a
    shape 1 lock never recorded, so it cannot be converted."""
    raise OutdatedSchema("a shape 1 lock does not list the documents its vocabulary was trained on, "
                         "so it cannot become shape 2: delete it and build again")


MIGRATIONS["lock"][1] = _lock_1_to_2


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
            try:
                document = step(document, path)
            except DatasetError as error:
                raise type(error)(f"{path}: {error}") from None
        if document.get("schema") != start:
            path.write_text(dump_json(document), encoding="utf-8", newline="\n")
            changed.append(path)
            if kind == "source" and start < 4 <= document["schema"]:
                _drop_moved_license_files(path, document)
            log(f"{path.relative_to(repo.root)}: shape {start} -> {document['schema']}")
    if current is CURRENT_SCHEMA:  # the real shapes: every file must now validate
        for kind, path in _files(repo):
            load_json(path, kind)
    return changed
