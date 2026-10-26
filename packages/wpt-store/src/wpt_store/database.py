"""The store database: one SQLite file about the store and nothing else,
the full history of its sessions (wpt_store.sessions) and of their lines.

    sessions  folder (the session's run record, records/<folder>/), command,
              started and ended (UTC, ISO 8601; ended is NULL until the
              session ended), wpt_commit (the repository commit it ran at),
              dirty (1 when that working tree had changes)
    lines     session (its folder), number (its place in the session),
              source (id), revision, checksum (the one found, for a
              failure), address (the one actually used), time (UTC), outcome
              (downloaded, reused or failed), size (bytes), code (the code
              that did it: wpt_utils.programs.code_name)
    git_objects
              object_id (a git blob's id), sha256 (its content's), bundle,
              start and length (where the store holds that content, as a
              wpt_store.Placed says): shared by every source, so a git
              source whose revision moved downloads only the files it does
              not hold, and takes the sha256 of the others from here

    removals  bundle (<session folder>/<number>), marked (UTC): a bundle a
              session's end replaced by a rewritten one, whose old copy
              stays in the store until no session that started before
              `marked` still runs; a later session then removes it (put
              gives each one such; forget drops its row once it is
              removed)

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
    revision TEXT,
    checksum TEXT,
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
"""
SESSION = ("folder", "command", "started", "ended", "wpt_commit", "dirty")
LINE = ("session", "number", "source", "revision", "checksum", "address", "time", "outcome", "size", "code")
GIT_OBJECT = ("object_id", "sha256", "bundle", "start", "length")
REMOVAL = ("bundle", "marked")
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
            connection.execute(f"INSERT OR REPLACE INTO sessions VALUES ({', '.join('?' * len(SESSION))})",
                               [session[name] for name in SESSION])
            connection.executemany(f"INSERT OR REPLACE INTO lines VALUES ({', '.join('?' * len(LINE))})",
                                   [[line[name] for name in LINE] for line in lines])
            connection.executemany(f"INSERT OR REPLACE INTO git_objects VALUES ({', '.join('?' * len(GIT_OBJECT))})",
                                   [[row[name] for name in GIT_OBJECT] for row in git_objects])
            connection.executemany(f"INSERT OR REPLACE INTO removals VALUES ({', '.join('?' * len(REMOVAL))})",
                                   [[row[name] for name in REMOVAL] for row in removals])
            removable = [bundle for bundle, in connection.execute(REMOVABLE)]
        updated = compressed(connection.serialize())
        connection.close()
        try:
            kind.replace(KEY, updated, version)
            return removable
        except Changed:
            time.sleep(random.uniform(0, PAUSE))


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
        connection.executescript(SCHEMA)
    except sqlite3.DatabaseError as error:
        connection.close()
        raise CorruptObject(f"the store database {KEY} is not a SQLite database: {error}") from None
    return connection
