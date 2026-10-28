"""The store database: one SQLite file about the store and nothing else,
the full history of its sessions (wpt_store.sessions) and of their lines.

    sessions  folder (the session's run record, records/<folder>/), command,
              started and ended (UTC, ISO 8601; ended is NULL until the
              session ended), wpt_commit (the repository commit it ran at),
              dirty (1 when that working tree had changes)
    lines     session (its folder), number (its place in the session),
              source (id), got (what the download got, as a JSON object,
              such as the commit: for the record only; NULL when it got
              nothing to record), address (the one actually used), time
              (UTC), outcome (downloaded, reused or failed), size (bytes),
              code (the code that did it: wpt_utils.programs.code_name)
    marks     source (id, or ALL: every source), marked (UTC): a person's
              ask for a new download of that source, which holds until a
              line of it says downloaded at a later time (marked)
    git_objects
              object_id (a git blob's id), sha256 (its content's), bundle,
              start and length (where the store holds that content, as a
              wpt_store.Placed says): shared by every source, so a git
              source whose commit moved downloads only the files it does
              not hold, and takes the sha256 of the others from here

    removals  bundle (<session folder>/<number>), marked (UTC): a bundle a
              session's end replaced by a rewritten one, whose old copy
              stays in the store until no session that started before
              `marked` still runs; a later session then removes it (put
              gives each one such; forget drops its row once it is
              removed)

A database whose lines still have the columns revision and checksum (an
earlier shape) is read with them moved into got, as {"revision": ...,
"checksum": ...} of those not NULL.

Each fact is written once: the run record keeps a session's start and end,
not its lines. The file lies at KEY, compressed (wpt_store.blocks), and is
replaced whole, only if unchanged since it was read: on S3 a conditional
upload, in a folder under a lock (wpt_store.kinds, replace). When another
session replaced it meanwhile, it is read again, this session's rows are
put in again, and the replace is tried again: rows are keyed by the
session's folder and the line's number, so putting them in twice changes
nothing. SQLite's own journal is no help here: it does not work across
machines.
"""

import json
import random
import sqlite3
import time
from compression import zstd

from .blocks import compressed
from .errors import CorruptObject
from .kinds import Absent, Changed

KEY = "store.sqlite.zst"
DOWNLOADED, REUSED, FAILED = "downloaded", "reused", "failed"  # what a line says of its source
OUTCOMES = (DOWNLOADED, REUSED, FAILED)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sessions (
    folder TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    started TEXT NOT NULL,
    ended TEXT,
    wpt_commit TEXT NOT NULL,
    dirty INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lines (
    session TEXT NOT NULL REFERENCES sessions (folder),
    number INTEGER NOT NULL,
    source TEXT NOT NULL,
    got TEXT,
    address TEXT,
    time TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({", ".join(f"'{outcome}'" for outcome in OUTCOMES)})),
    size INTEGER,
    code TEXT NOT NULL,
    PRIMARY KEY (session, number)
);
CREATE TABLE IF NOT EXISTS git_objects (
    object_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    bundle TEXT NOT NULL,
    start INTEGER NOT NULL,
    length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS removals (
    bundle TEXT PRIMARY KEY,
    marked TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS marks (
    source TEXT PRIMARY KEY,
    marked TEXT NOT NULL
);
"""
ALL = "*"  # the source a mark of every source names (no source id holds it)
SESSION = ("folder", "command", "started", "ended", "wpt_commit", "dirty")
LINE = ("session", "number", "source", "got", "address", "time", "outcome", "size", "code")
GIT_OBJECT = ("object_id", "sha256", "bundle", "start", "length")
REMOVAL = ("bundle", "marked")
MARK = ("source", "marked")
# whether a source is marked: a mark of it, or of every source, later than its last download
MARKED = ("SELECT 1 FROM marks WHERE source IN (?, ?) AND marked > COALESCE((SELECT MAX(time) FROM lines"
          f" WHERE source = ? AND outcome = '{DOWNLOADED}'), '') LIMIT 1")
# the marked bundles no running session can read any more: every session that started before the mark has ended
REMOVABLE = ("SELECT bundle FROM removals AS removal WHERE NOT EXISTS "
             "(SELECT 1 FROM sessions WHERE ended IS NULL AND started < removal.marked) ORDER BY bundle")
PAUSE = 0.2  # seconds: the longest wait, drawn at random, before reading again after another session's replace


def read(kind, threads=False) -> sqlite3.Connection:
    """The database as it is in the store, as a connection to a copy in
    memory (empty, with its tables, when the store has none yet); threads:
    usable from any thread (one at a time). CorruptObject when the stored
    file is not one."""
    return _connect(_fetch(kind)[0], threads)


def put(kind, session: dict, lines: list[dict], git_objects=(), removals=()) -> list[str]:
    """Put one session's row (SESSION's fields), lines (LINE's fields), git
    objects (GIT_OBJECT's fields) and removals (REMOVAL's fields: the old
    copies of bundles the session replaced, marked now) into the database in
    the store, replacing what it holds for them. Returns the bundles the
    database now marks that may be removed: those with no session still
    running (not ended) that started before the bundle was marked. CorruptObject when the stored file is
    not a database (nothing written). A failure of the store leaves the
    database as it was, or as replaced with these rows."""
    while True:
        data, version = _fetch(kind)
        connection = _connect(data)
        with connection:
            connection.execute(_insert("sessions", SESSION), [session[name] for name in SESSION])
            connection.executemany(_insert("lines", LINE), [[line[name] for name in LINE] for line in lines])
            connection.executemany(_insert("git_objects", GIT_OBJECT),
                                   [[row[name] for name in GIT_OBJECT] for row in git_objects])
            connection.executemany(_insert("removals", REMOVAL), [[row[name] for name in REMOVAL] for row in removals])
            removable = [bundle for bundle, in connection.execute(REMOVABLE)]
        updated = compressed(connection.serialize())
        connection.close()
        try:
            kind.replace(KEY, updated, version)
            return removable
        except Changed:
            time.sleep(random.uniform(0, PAUSE))


def mark(kind, sources, at: str):
    """Mark sources (ids, or [ALL] for every source) for a new download,
    at the time at (UTC, as lines write times): a later mark of a source
    replaces its earlier one. CorruptObject as put's."""
    while True:
        data, version = _fetch(kind)
        connection = _connect(data)
        with connection:
            connection.executemany(_insert("marks", MARK), [[source, at] for source in sources])
        updated = compressed(connection.serialize())
        connection.close()
        try:
            kind.replace(KEY, updated, version)
            return
        except Changed:
            time.sleep(random.uniform(0, PAUSE))


def marked(connection, source: str) -> bool:
    """Whether a connection's database marks source for a new download
    (MARKED)."""
    return connection.execute(MARKED, (source, ALL, source)).fetchone() is not None


def _insert(table: str, fields) -> str:
    """The statement putting one row of fields into table, each column named
    (a table an earlier shape left may hold them in another order)."""
    return f"INSERT OR REPLACE INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})"


def forget(kind, bundles):
    """Drop the removals of these bundles, once they are removed from the
    store."""
    while bundles:
        data, version = _fetch(kind)
        connection = _connect(data)
        with connection:
            connection.executemany("DELETE FROM removals WHERE bundle = ?", [[bundle] for bundle in bundles])
        updated = compressed(connection.serialize())
        connection.close()
        try:
            kind.replace(KEY, updated, version)
            return
        except Changed:
            time.sleep(random.uniform(0, PAUSE))


def _fetch(kind):
    """(the database's bytes, or None when there is none, and its version)."""
    try:
        data, version = kind.read(KEY)
    except Absent:
        return None, None
    try:
        return zstd.decompress(data), version
    except zstd.ZstdError as error:
        raise CorruptObject(f"the store database {KEY} is not zstd: {error}") from None


def _connect(data, threads=False):
    connection = sqlite3.connect(":memory:", check_same_thread=not threads)
    try:
        if data:
            connection.deserialize(data)
        _move_pinned_columns(connection)
        connection.executescript(SCHEMA)
    except sqlite3.DatabaseError as error:
        connection.close()
        raise CorruptObject(f"the store database {KEY} is not a SQLite database: {error}") from None
    return connection


def _move_pinned_columns(connection):
    """Lines of an earlier shape, with the columns revision and checksum:
    those not NULL moved into got (see the module's docstring), the columns
    gone."""
    columns = [row[1] for row in connection.execute("PRAGMA table_info(lines)")]
    if "revision" not in columns:
        return
    connection.create_function("pinned", 2, lambda revision, checksum: json.dumps(
        {key: value for key, value in (("revision", revision), ("checksum", checksum)) if value is not None},
        sort_keys=True) if revision is not None or checksum is not None else None, deterministic=True)
    with connection:
        connection.execute("ALTER TABLE lines ADD COLUMN got TEXT")
        connection.execute("UPDATE lines SET got = pinned(revision, checksum)")
        connection.execute("ALTER TABLE lines DROP COLUMN revision")
        connection.execute("ALTER TABLE lines DROP COLUMN checksum")
