"""The store database: one SQLite file about the store and nothing else,
the full history of its sessions (wpt_store.sessions) and of their lines.

    sessions  folder (the session's run record, records/<folder>/), command,
              started and ended (UTC, ISO 8601; ended is NULL until the
              session ended), wpt_commit (the repository commit it ran at),
              dirty (1 when that working tree had changes)
    lines     session (its folder), number (its place in the session),
              source (id), revision, checksum (the one found, for a
              failure), address (the one actually used), time (UTC), outcome
              (downloaded, reused or failed), size (bytes), wpt_version

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
OUTCOMES = ("downloaded", "reused", "failed")
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
    wpt_version TEXT NOT NULL,
    PRIMARY KEY (session, number)
);
"""
SESSION = ("folder", "command", "started", "ended", "wpt_commit", "dirty")
LINE = ("session", "number", "source", "revision", "checksum", "address", "time", "outcome", "size", "wpt_version")
PAUSE = 0.2  # seconds: the longest wait, drawn at random, before reading again after another session's replace


def read(kind) -> sqlite3.Connection:
    """The database as it is in the store, as a connection to a copy in
    memory (empty, with its tables, when the store has none yet).
    CorruptObject when the stored file is not one."""
    return _connect(_fetch(kind)[0])


def put(kind, session: dict, lines: list[dict]):
    """Put one session's row (SESSION's fields) and lines (LINE's fields)
    into the database in the store, replacing what it holds for them.
    CorruptObject when the stored file is not a database (nothing written).
    A failure of the store leaves the database as it was, or as replaced
    with these rows."""
    while True:
        data, version = _fetch(kind)
        connection = _connect(data)
        with connection:
            connection.execute(f"INSERT OR REPLACE INTO sessions VALUES ({', '.join('?' * len(SESSION))})",
                               [session[name] for name in SESSION])
            connection.executemany(f"INSERT OR REPLACE INTO lines VALUES ({', '.join('?' * len(LINE))})",
                                   [[line[name] for name in LINE] for line in lines])
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


def _connect(data):
    connection = sqlite3.connect(":memory:")
    try:
        if data:
            connection.deserialize(data)
        connection.executescript(SCHEMA)
    except sqlite3.DatabaseError as error:
        connection.close()
        raise CorruptObject(f"the store database {KEY} is not a SQLite database: {error}") from None
    return connection
