"""The store's index: one SQLite file at the store's root, each session ID
and what it holds, for fast lookup. It says nothing the files do not: every
row comes from a file of the session, and rebuild() (`wpt store
rebuild-index`) makes the index again from the files alone, over a stored
file that is damaged too.

    held    session (its ID), kind (sources, datasets, models or runs), name
            (the source id, the dataset's, the model's or the run's), time
            (UTC, ISO 8601), about (a JSON object: what the file says of
            itself, without what only reading needs)

The file lies at KEY, compressed (wpt_store.blocks). Every read and every
change of it goes through a copy on this machine's disk, in a local folder
the caller gives (the store's cache folder): the stored file is read part
by part into a temporary file there and decompressed into another one,
which SQLite opens. No step holds the index whole in memory. A change (put,
replace) is made in that copy, which is then compressed into a file beside
it and replaces the stored one only if that is unchanged since it was read:
on S3 a conditional upload, in a folder under a lock (wpt_store.kinds,
replace). When another command replaced it meanwhile, it is read again, the
change is made again, and the replace is tried again. SQLite's own journal
is no help here: it does not work across machines.

Each temporary copy is owned through a lock file beside it
(wpt_store.local_files): a copy a crash left is removed before the next
copy is made.

The rows of a run reach the index when it ends. A run that crashed leaves
them out; the next download into its ID puts them in for each source it
finds held, and rebuild puts in every one.

Failures: CorruptFile (the stored file is not an index; nothing
written), and what the store raised: the index is then as it was, or
replaced with these rows.
"""

import json
import random
import secrets
import shutil
import sqlite3
import time
from compression import zstd
from pathlib import Path

from . import blocks
from .errors import CorruptFile, MissingFile
from .kinds import Absent, Changed
from .local_files import LocalFile, unowned

KEY = "index"
SOURCES, DATASETS, MODELS, RUNS = "sources", "datasets", "models", "runs"  # the kinds of things a session holds
SCHEMA = """
CREATE TABLE IF NOT EXISTS held (
    session TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    time TEXT NOT NULL,
    about TEXT NOT NULL,
    PRIMARY KEY (session, kind, name)
);
CREATE INDEX IF NOT EXISTS held_by_name ON held (kind, name);
"""
# what a record's fields the index repeats, by kind
ABOUT = {SOURCES: ("key", "address", "size", "stored", "sha256", "manifest", "got", "code", "error", "from"),
         DATASETS: ("id", "names", "from"), MODELS: ("from",), RUNS: ("command", "from")}
PAUSE = 0.2  # seconds: the longest wait, drawn at random, before reading again after another command's replace
PART = 64 << 20  # bytes of the stored file read at once into its local copy
COPY = ".store-index-"  # the start of the name of a temporary copy of the index on this machine's disk
PACKED = ".zst"  # the end of the name of a local file holding the index compressed
ENDINGS = ("", "-journal", "-wal", "-shm", PACKED)  # what SQLite and a change add to a local file's name


def about(kind: str, record: dict) -> dict:
    """What the index says of a thing of that kind, from its record (a
    source's <source>.json, a dataset's dataset.json, a model's files.json,
    a run's start.json), the same whoever asks: without what only reading
    needs."""
    found = {name: record[name] for name in ABOUT[kind] if name in record}
    if kind == DATASETS and isinstance(record.get("tokenizer", {}).get("vocabulary"), dict):
        found["vocabulary"] = record["tokenizer"]["vocabulary"]["sha256"]
    if kind == MODELS:
        found |= {"files": len(record["files"]), "size": sum(entry["size"] for entry in record["files"].values())}
    return found


def held_row(session: str, kind: str, name: str, record: dict) -> tuple:
    """One thing a session holds, as the index keeps it, from its record
    (about): its time is the record's, or its error's, or its start's."""
    moment = record.get("error", {}).get("time") or record.get("time") or record.get("started") or ""
    return session, kind, name, moment, json.dumps(about(kind, record), ensure_ascii=False, sort_keys=True)


def read(kind, folder, threads=False) -> sqlite3.Connection:
    """The index as it is in the store, as a connection to a copy on this
    machine's disk in folder (empty, with its table, when the store has
    none yet), a Copy: removed when the connection closes, by close() or at
    the end of a `with` block over it. threads: usable from any thread (one
    at a time). CorruptFile when the stored file is not one."""
    local = _copy(kind, folder)[0]
    try:
        connection = _connect(local.path(), threads, factory=Copy)
    except BaseException:
        local.remove()
        raise
    connection.local = local
    return connection


class Copy(sqlite3.Connection):
    """A connection to a temporary copy of the index (read), whose files go
    when it closes. The end of a `with` block over it closes it too, where
    a plain sqlite3 connection's only ends a transaction and stays open."""

    local = None

    def close(self):
        super().close()
        if self.local is not None:
            self.local.remove()
            self.local = None

    def __exit__(self, kind, error, trace):
        try:
            return super().__exit__(kind, error, trace)
        finally:
            self.close()


def put(kind, folder, rows):
    """Put rows (row) into the index in the store, replacing what it holds
    for the same session, kind and name, through a copy in folder."""
    rows = list(rows)
    if rows:
        _change(kind, folder, lambda connection: connection.executemany(INSERT, rows))


def replace(kind, folder, rows):
    """Make rows (row) the whole index, through a copy in folder. What the
    stored file holds is never read, so a damaged one is replaced too."""
    rows = list(rows)
    _change(kind, folder, lambda connection: connection.executemany(INSERT, rows), blank=True)


def rebuild(store) -> int:
    """Make the index again from the files alone: every session's sources,
    datasets, models and runs. How many things it holds."""
    rows = []
    for session in store.sessions():
        for key in store.kind.list(f"{session}/{SOURCES}/", deep=False):
            if (name := _json_name(key)) is not None:
                record = store.get_json(f"{session}/{SOURCES}/{name}.json")
                rows.append(held_row(session, SOURCES, name, record))
        for kind, file in ((DATASETS, "dataset.json"), (MODELS, "files.json")):
            for key in store.kind.list(f"{session}/{kind}/", deep=False):
                name = key.removesuffix("/").rsplit("/", 1)[1]
                if (record := store.get_json(f"{session}/{kind}/{name}/{file}")) is not None:
                    rows.append(held_row(session, kind, name, record))
        for key in store.kind.list(f"{session}/{RUNS}/", deep=False):
            name = key.removesuffix("/").rsplit("/", 1)[1]
            try:
                rows.append(held_row(session, RUNS, name, json.loads(store.read_record(f"{session}/{name}", "start.json"))))
            except MissingFile:
                pass  # a run whose first file never reached the store
    replace(store.kind, store.cache, rows)
    return len(rows)


def _json_name(key: str) -> str | None:
    """The source a key of <id>/sources/ names the record of; None for any other."""
    name = key.rsplit("/", 1)[1].removesuffix(PACKED)
    return name.removesuffix(".json") if name.endswith(".json") else None


INSERT = "INSERT OR REPLACE INTO held (session, kind, name, time, about) VALUES (?, ?, ?, ?, ?)"


def _change(kind, folder, edit, blank=False):
    """edit(connection) run on a local copy of the index (blank: an empty
    one), then the copy replacing the stored file, both tried again until no
    other command replaced it meanwhile."""
    while True:
        local, version = _copy(kind, folder, blank)
        try:
            connection = _connect(local.path())
            try:
                with connection:
                    edit(connection)
            finally:
                connection.close()
            with open(local.path(), "rb") as source, open(local.path(PACKED), "wb") as out:
                writer = blocks.Writer(out)
                while block := source.read(blocks.FRAME):
                    writer.write(block)
                writer.close()
            try:
                kind.replace(KEY, local.path(PACKED), version)
                return
            except Changed:
                pass
        finally:
            local.remove()
        time.sleep(random.uniform(0, PAUSE))


def _copy(kind, folder, blank=False):
    """(a LocalFile holding the stored index's bytes in its path(), empty
    when the store has none or for blank, and the stored file's version),
    after the copies a crash left in folder are removed."""
    for left in unowned(folder, COPY, ENDINGS):
        left.remove()
    local = LocalFile(Path(folder) / (COPY + secrets.token_hex(8)), ENDINGS)
    try:
        version = _download(kind, local.path(PACKED))
        if version is None or blank:
            local.path().write_bytes(b"")
        else:
            try:
                with zstd.open(local.path(PACKED)) as packed, open(local.path(), "wb") as out:
                    shutil.copyfileobj(packed, out, blocks.FRAME)
            except zstd.ZstdError as error:
                raise CorruptFile(f"the store index {KEY} is not zstd: {error}") from None
            local.path(PACKED).unlink()
    except BaseException:
        local.remove()
        raise
    return local, version


def _download(kind, target: Path):
    """The stored file into the local file target, PART bytes at a time,
    each part checked to be of the same version; read again from its start
    when it changed meanwhile. Its version, or None when there is none."""
    while True:
        version, at = None, 0
        with open(target, "wb") as out:
            try:
                while True:
                    data, found = kind.read(KEY, at, PART)
                    if not data and found is None:  # S3 answers a read from the end with no version
                        found = kind.read(KEY, 0, 0)[1]
                    if at == 0:
                        version = found
                    elif found != version:
                        break  # replaced meanwhile
                    out.write(data)
                    at += len(data)
                    if len(data) < PART:
                        return version
            except Absent:
                if at == 0:
                    return None
        time.sleep(random.uniform(0, PAUSE))


def _connect(path, threads=False, factory=sqlite3.Connection):
    connection = sqlite3.connect(path, check_same_thread=not threads, factory=factory)
    try:
        # a copy only this process uses, whose stored file is replaced whole: no journal on the disk, no sync
        connection.executescript("PRAGMA journal_mode=MEMORY; PRAGMA synchronous=OFF;")
        connection.executescript(SCHEMA)
    except sqlite3.DatabaseError as error:
        connection.close()
        raise CorruptFile(f"the store index {KEY} is not a SQLite database: {error}") from None
    return connection
