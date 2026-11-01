"""Probe questions for the probes test (docs/evaluation.md, "Probes").

A file in the `probe` format is JSON Lines: one probe per line, each
matching schemas/probe.schema.json: a question a person wrote, which the
probes test asks a model, and the patterns (Python regular expressions)
that flag a reply to it as alarming. Probes are stored as files of an
`empty` source so the measure is reproducible, and they are a measure
only: such a file is never a pre-training document (markdown.py gives it
no Markdown), no dataset build keeps it and no training step reads it.

parse() refuses a line whose patterns do not compile, so every way of
saving keeps them usable (the check runs it on every patch). read()
gives the probes of every admitted source, for the test.
"""

import json
import re
from typing import NamedTuple

from . import manifest
from .errors import InvalidFile
from .migrate import upgrade
from .repo import Repo, validate

FORMAT = "probe"


class Probe(NamedTuple):
    source: str  # the id of the source holding it
    path: str  # its probe file in the source
    line: int  # its line in the file, from 1
    question: str
    patterns: tuple[str, ...]


def parse(source_id: str, path: str, data: bytes) -> list[dict]:
    """The probes of one probe file; InvalidFile names the line at fault."""
    probes = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        where = f"{source_id}/{path} line {number}"
        if not line.strip():
            raise InvalidFile(f"{where}: empty line; a probe file holds one probe per line")
        try:
            probe = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidFile(f"{where}: {error}") from error
        probe = upgrade(probe, FORMAT, where)
        validate(probe, FORMAT, where)
        for index, pattern in enumerate(probe["patterns"]):
            try:
                re.compile(pattern)
            except re.error as error:
                raise InvalidFile(f"{where}: at patterns/{index}: not a regular expression Python compiles: "
                                  f"{error}") from None
        probes.append(probe)
    return probes


def read(repo: Repo, store) -> tuple[list[Probe], list[str]]:
    """([Probe] of every admitted source, in source id, path and line
    order; [the ids of the admitted sources holding probe files whose
    files the store does not hold]). A source whose license text is empty
    is left out, as every reader leaves it out."""
    from .markdown import format_of  # markdown.py reads this module's parse
    from .sources import in_store, source_tree

    found, missing = [], []
    for source in repo.sources(lambda _: None):
        if not source.admitted or not any(entry["format"] == FORMAT for entry in source.data["formats"]):
            continue
        if not in_store(source, store):
            missing.append(source.id)
            continue
        tree, _ = source_tree(source, store)
        for path in sorted(tree, key=str.encode):
            if format_of(source, path)["format"] != FORMAT:
                continue
            for number, probe in enumerate(parse(source.id, path, manifest.content(store, tree[path])), start=1):
                found.append(Probe(source.id, path, number, probe["question"], tuple(probe["patterns"])))
    return found, missing
