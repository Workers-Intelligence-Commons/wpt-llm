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
not its lines. The file lies at KEY, compressed (wpt_store.blocks).

Every read and every change of it goes through a copy on this machine's
disk, in a local folder the caller gives (the store's cache folder): the
stored file is read part by part into a temporary file there and
decompressed into another one, which SQLite opens. No step holds the
database whole in memory, so only the disk bounds its size. A change
(put, mark, forget) is made in that copy, which is then compressed into a
file beside it and replaces the stored one only if that is unchanged since
it was read: on S3 a conditional upload, in a folder under a lock
(wpt_store.kinds, replace). When another session replaced it meanwhile,
it is read again, the change is made again, and the replace is tried
again. SQLite's own journal is no help here: it does not work across
machines.

A session's rows (its own row, its lines, its git objects, its removals)
go first into its own local file, a Rows: a SQLite file in WAL mode with
the same tables, in the same local folder, named after the session's
folder. put copies the rows of such files into the database. Rows are
keyed by the session's folder and the line's number (a git object by its
id, a removal by its bundle), so copying them twice changes nothing. A
session that crashed before its rows went in leaves its file, and the
next session on this machine puts them in (unowned_rows); a file is
removed only once its rows are in the database.

Each of these local files is owned through a lock file beside it
(wpt_store.local_files): a temporary copy a crash left is removed before
the next copy is made. A Rows file a crash left whose own file is gone
was being removed once its rows were in, and is removed; one without the
tables holds no rows, and is put in as empty.
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
from .errors import CorruptObject
from .kinds import Absent, Changed
from .local_files import LocalFile, unowned

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
SESSION_ROWS = {"lines": LINE, "git_objects": GIT_OBJECT, "removals": REMOVAL}  # a Rows file's tables besides sessions
# whether a source is marked: a mark of it, or of every source, later than its last download
MARKED = ("SELECT 1 FROM marks WHERE source IN (?, ?) AND marked > COALESCE((SELECT MAX(time) FROM lines"
          f" WHERE source = ? AND outcome = '{DOWNLOADED}'), '') LIMIT 1")
# the marked bundles no running session can read any more: every session that started before the mark has ended
REMOVABLE = ("SELECT bundle FROM removals AS removal WHERE NOT EXISTS "
             "(SELECT 1 FROM sessions WHERE ended IS NULL AND started < removal.marked) ORDER BY bundle")
PAUSE = 0.2  # seconds: the longest wait, drawn at random, before reading again after another session's replace
PART = 64 << 20  # bytes of the stored file read at once into its local copy
COPY = ".store-database-"  # the start of the name of a temporary copy of the database on this machine's disk
ROWS = ".session-rows-"  # the start of the name of a session's Rows file, before its folder
ROWS_END = ".sqlite"  # the end of that name, so no session's name is another's with SQLite's endings added
PACKED = ".zst"  # the end of the name of a local file holding the database compressed
ENDINGS = ("", "-journal", "-wal", "-shm", PACKED)  # what SQLite and a change add to a local file's name


def read(kind, folder, threads=False) -> sqlite3.Connection:
    """The database as it is in the store, as a connection to a copy on
    this machine's disk in folder (empty, with its tables, when the store
    has none yet), a Copy: removed when the connection closes, by close()
    or at the end of a `with` block over it. threads: usable from any
    thread (one at a time). CorruptObject when the stored file is not
    one."""
    local = _copy(kind, folder)[0]
    try:
        connection = _connect(local.path(), threads, factory=Copy)
    except BaseException:
        local.remove()
        raise
    connection.local = local
    return connection


class Copy(sqlite3.Connection):
    """A connection to a temporary copy of the database (read), whose files
    go when it closes. The end of a `with` block over it closes it too,
    where a plain sqlite3 connection's only ends a transaction and stays
    open."""

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


def put(kind, folder, rows) -> tuple[list[str], int]:
    """Put the rows of local session files (rows: Rows) into the database in
    the store, replacing what it holds for them, through a copy in folder;
    once they are in, each of rows is applied() (Rows). Returns the bundles
    the database now marks that may be removed (those with no session still
    running, not ended, that started before the bundle was marked) and the
    bytes the database now takes in the store.
    CorruptObject when the stored file is not a database (nothing written).
    A failure of the store leaves the database as it was, or as replaced
    with these rows, and rows as they were."""

    def put_in(connection):
        for one in rows:
            connection.execute("ATTACH DATABASE ? AS rows", (str(one.local.path()),))
            try:
                for table, fields in {"sessions": SESSION, **SESSION_ROWS}.items():
                    columns = ", ".join(fields)
                    connection.execute(f"INSERT OR REPLACE INTO main.{table} ({columns})"
                                       f" SELECT {columns} FROM rows.{table}")
                connection.commit()
            finally:
                connection.execute("DETACH DATABASE rows")
        return [bundle for bundle, in connection.execute(REMOVABLE)]

    removable, size = _change(kind, folder, put_in)
    for one in rows:
        one.applied()
    return removable, size


def mark(kind, folder, sources, at: str):
    """Mark sources (ids, or [ALL] for every source) for a new download,
    at the time at (UTC, as lines write times): a later mark of a source
    replaces its earlier one. Through a copy in folder; CorruptObject as
    put's."""
    def mark_in(connection):
        connection.executemany(_insert("marks", MARK), [[source, at] for source in sources])

    _change(kind, folder, mark_in)


def marked(connection, source: str) -> bool:
    """Whether a connection's database marks source for a new download
    (MARKED)."""
    return connection.execute(MARKED, (source, ALL, source)).fetchone() is not None


def forget(kind, folder, bundles):
    """Drop the removals of these bundles, once they are removed from the
    store, through a copy in folder."""
    def forget_in(connection):
        connection.executemany("DELETE FROM removals WHERE bundle = ?", [[bundle] for bundle in bundles])

    if bundles:
        _change(kind, folder, forget_in)


class Rows:
    """One session's rows on this machine's disk before they go into the
    database: a SQLite file in WAL mode, named after the session's folder,
    owned by this process (local, a LocalFile): a new session's (begin) or
    one a crashed session left (unowned_rows). add() puts rows in; put()
    copies them into the database and marks them applied(), which drops all
    but the session's row; close() releases the file, removing it only when
    every row it holds is in the database; discard() removes it whatever it
    holds."""

    def __init__(self, local: LocalFile, pending: bool):
        self.local = local
        self.pending = pending  # whether it holds rows the database may not hold yet
        try:
            self.connection = sqlite3.connect(local.path(), check_same_thread=False)
            try:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.executescript(SCHEMA)
            except BaseException:
                self.connection.close()
                raise
        except BaseException:
            self.connection = None
            if pending:
                local.release()
            else:
                local.remove()
            raise

    @classmethod
    def begin(cls, folder, session: str) -> "Rows":
        """A new session's file in folder, empty."""
        return cls(LocalFile(Path(folder) / f"{ROWS}{session}{ROWS_END}", ENDINGS), pending=False)

    def add(self, session: dict, lines=(), git_objects=(), removals=()):
        """Put a session's row (SESSION's fields), lines (LINE's fields), git
        objects (GIT_OBJECT's fields) and removals (REMOVAL's fields), on the
        disk when it returns."""
        with self.connection:
            self.connection.execute(_insert("sessions", SESSION), [session[name] for name in SESSION])
            for table, rows in (("lines", lines), ("git_objects", git_objects), ("removals", removals)):
                fields = SESSION_ROWS[table]
                self.connection.executemany(_insert(table, fields), [[row[name] for name in fields] for row in rows])
        self.pending = True

    def applied(self):
        """Every row is in the database (put): all but the session's row
        dropped."""
        with self.connection:
            for table in SESSION_ROWS:
                self.connection.execute(f"DELETE FROM {table}")
        self.pending = False

    def close(self):
        """The file closed and the lock released; the file removed when the
        database holds every row of it, kept otherwise for a later session
        to put in (unowned_rows)."""
        if self.connection is None:
            return
        self.connection.close()
        self.connection = None
        if self.pending:
            self.local.release()
        else:
            self.local.remove()

    def discard(self):
        """The file closed and removed, whatever it holds."""
        self.pending = False
        self.close()


def unowned_rows(folder) -> list[Rows]:
    """The Rows files in folder no live session owns (those a crashed
    session left), pending: put them in, then close() them. One whose own
    file is gone (a removal that did not finish) is removed, not given.
    sqlite3.DatabaseError for one that is not a SQLite file, none kept
    taken."""
    found = []
    try:
        for local in unowned(folder, ROWS, ENDINGS):
            if local.path().exists():
                found.append(Rows(local, pending=True))
            else:
                local.remove()
    except BaseException:
        for rows in found:
            rows.close()
        raise
    return found


def _change(kind, folder, edit):
    """edit(connection) run on a local copy of the database, then the copy
    replacing the stored file, both tried again until no other session
    replaced it meanwhile: what edit returned, and the bytes of the stored
    file."""
    while True:
        local, version = _copy(kind, folder)
        try:
            connection = _connect(local.path())
            try:
                with connection:
                    result = edit(connection)
            finally:
                connection.close()
            with open(local.path(), "rb") as source, open(local.path(PACKED), "wb") as out:
                writer = blocks.Writer(out)
                while block := source.read(blocks.FRAME):
                    writer.write(block)
                writer.close()
            try:
                kind.replace(KEY, local.path(PACKED), version)
                return result, local.path(PACKED).stat().st_size
            except Changed:
                pass
        finally:
            local.remove()
        time.sleep(random.uniform(0, PAUSE))


def _insert(table: str, fields) -> str:
    """The statement putting one row of fields into table, each column named
    (a table an earlier shape left may hold them in another order)."""
    return f"INSERT OR REPLACE INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})"


def _copy(kind, folder):
    """(a LocalFile holding the stored database's bytes in its path(), empty
    when the store has none, and the stored file's version), after the
    copies a crash left in folder are removed."""
    for left in unowned(folder, COPY, ENDINGS):
        left.remove()
    local = LocalFile(Path(folder) / (COPY + secrets.token_hex(8)), ENDINGS)
    try:
        version = _download(kind, local.path(PACKED))
        if version is None:
            local.path().write_bytes(b"")
        else:
            try:
                with zstd.open(local.path(PACKED)) as packed, open(local.path(), "wb") as out:
                    shutil.copyfileobj(packed, out, blocks.FRAME)
            except zstd.ZstdError as error:
                raise CorruptObject(f"the store database {KEY} is not zstd: {error}") from None
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
