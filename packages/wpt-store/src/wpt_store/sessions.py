"""Sessions: one run of one downloading command, start to end, at one
repository commit, and what it stored.

A session is named by its run record's folder (records/<folder>/, see
wpt_telemetry). The files it downloads go into its bundles,
bundles/<folder>/<number> (six digits, from 000001), each one file of the
store holding many downloaded files one after the other, compressed as
the store compresses every file (wpt_store.Store), in frames of about
1 MB, so one downloaded file is read back alone (Store.read_file). With
keep_local, a store whose files are not on this machine (S3) keeps a copy
of each sealed bundle in its cache folder; without it, the only copy left
is the store's. A bundle is cut only between files, once it holds BUNDLE
bytes or more; a file bigger than BUNDLE gets a bundle of its own. Where a
file went is a Placed: its bundle, start and length in the bundle's
original bytes, and its sha256, which every read checks. Session.read
gives a file back before its bundle is sealed too, from the bundle's
partial copy on this machine. A file several sources hold is stored again
for each of them: no table of every file the store holds, but for git
files, which the store database's git_objects table maps from their git
object id to where the store holds them (known_objects), so a git source
reuses what another one stored. That table gives out only places that
stay: in a bundle of a session that has ended (its bundles are sealed for
good, see the end below) or of this session.

A bundle fills on this machine's disk first, in the store's cache folder,
as PARTIAL<random>, owned by the session through a lock file beside it
(wpt_store.local_files) while it fills. Every new session removes the
unfinished bundles no live session owns.

A source's file list (file_list_bytes, parse_file_list) says where each of
its files went, one line per file, sorted by the UTF-8 bytes of the path:

    <mode> <sha256> <bundle> <start> <length> <path>

mode is the caller's, kept as it is (wpt_dataset's manifest modes). A file
list is handed to name(), which stores it once its files are, and tells
the caller where it lies.

The session's lines (one per download, use of a source already in the
store, failure) go into the store database
(wpt_store.database) with the session's own row each time a bundle is
sealed, and at the end, and so do the git objects of the file lists
named meanwhile (their git_objects table). They go first into the
session's own file on this machine's disk (database.Rows, in the store's
cache folder), then from it into the store database (database.put). An
action handed to then() or name() (such as setting the refs that name a
downloaded source's files) runs at the same moment, just before those
rows go in: once every file added before it is stored. A bundle is on the
disk (local store) or held by S3 before the database names it. A full
bundle is sealed while the next one fills; should it fill too before the
first is sealed, adding waits. A crash loses only the bundles not yet
sealed (the one being sealed and the one being filled) and the rows added
since the last one was sealed. Rows that reached the session's file but
not the database stay in that file, and the next session that starts on
this machine puts them in, then removes the file.

The session's row goes into the database when it starts, so that a session
that ends later sees it running. A session that cannot put it in removes
its file and raises. The end (end()) seals the bundles for good: a bundle holding bytes that
no named file list names (a try that failed, a download that failed and
named nothing) is rewritten without them. The one
being filled is rewritten before it is stored; one sealed earlier becomes
a new bundle of the session, every file list naming it is stored again
with its files' new places and its action runs again, and the old bundle
is marked for removal in the database together with the new places (the
`removals` table). A bundle no file list names at all is marked too.
Another session may still hold the old places, so the old copy stays until
no session that started before the mark is still running: the end of a
later session (or of this one, when no other runs) removes it. A session
that crashed never ends, and so keeps the copies marked after it started.
A list named with no action (name(files, None, ...)) only keeps its files:
the bytes the git map names that no file list of a tree names, such as the
blobs a git source's history needs, are kept by the rewrite and move with
it, and their rows follow them.

Failures: what the store raised while sealing a bundle or putting the
rows in the database. The bundles not yet sealed and the rows that had
not reached the session's file are then lost (those in it go in with the
next session), and the session cannot go on: every later add, line
and end raises that same failure and writes nothing; the actions handed
to then() or name() with them never run. A failing action stops the
session the same way. ValueError for an outcome the database does not
know, or a file added after the end.
"""

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import blocks, database
from .errors import CorruptObject
from .local_files import LocalFile, unowned
from .names import check_record_name

BUNDLE = 4 << 30  # bytes: a bundle is sealed once it holds this many or more
CHUNK = 1 << 20  # bytes of a local file read at once
PARTIAL = ".partial-bundle-"  # the start of the name of a bundle being filled on this machine's disk
PART = 64 << 20  # bytes of one file copied at once when a bundle is rewritten
QUERY = 500  # git object ids asked for in one query of the store database


@dataclass(frozen=True)
class Placed:
    """Where one downloaded file lies: bytes [start, start + length) of
    its bundle's original bytes; sha256 is the file's own."""

    bundle: str  # <session folder>/<number>
    start: int
    length: int
    sha256: str


def file_list_bytes(files: dict) -> bytes:
    """A file list from {path: (mode, Placed)}."""
    lines = []
    for path in sorted(files, key=lambda path: path.encode()):
        mode, placed = files[path]
        lines.append(f"{mode} {placed.sha256} {placed.bundle} {placed.start} {placed.length} {path}\n")
    return "".join(lines).encode()


def parse_file_list(data: bytes) -> dict:
    """{path: (mode, Placed)} from a file list. ValueError for a line not
    in its format."""
    files = {}
    for line in data.decode().splitlines():
        try:
            mode, sha256, bundle, start, length, path = line.split(" ", 5)
            files[path] = (mode, Placed(bundle, int(start), int(length), sha256))
        except ValueError:
            raise ValueError(f"not a file list line: {line!r}") from None
    return files


def now() -> str:
    """This moment in UTC, as the store database writes times."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def remove_unowned_partials(cache: Path) -> list[str]:
    """Remove every unfinished bundle in cache (PARTIAL...) that no live
    session owns (local_files.unowned). Their names."""
    removed = []
    for left in unowned(cache, PARTIAL):
        if left.path().exists():
            removed.append(left.path().name)
        left.remove()
    return removed


class _Partial:
    """A bundle being filled on this machine's disk: its file, its
    blocks.Writer, the original bytes written so far, its number, and the
    lock that says a live session owns it."""

    def __init__(self, store, number):
        self.local = LocalFile(store.cache / (PARTIAL + secrets.token_hex(8)))  # a new name: taken at once
        self.path = self.local.path()
        try:
            self.out = open(self.path, "xb")
        except BaseException:
            self.local.remove()
            raise
        self.writer = blocks.Writer(self.out, *store.compressing())
        self.size, self.number = 0, number

    def write(self, block):
        self.writer.write(block)
        self.size += len(block)

    def read(self, start, length) -> bytes:
        """Original bytes of the partial copy; the frame being filled ends there early."""
        self.writer.flush()
        self.out.flush()
        with open(self.path, "rb") as partial:
            return blocks.read(partial, blocks.SeekTable(self.writer.sizes), start, length)

    def seal(self):
        """(original size, stored size), the file whole and closed."""
        sizes = self.writer.close()
        self.out.close()
        return sizes

    def remove(self):
        """The file gone, if still there, and the lock released."""
        self.out.close()
        self.local.remove()


@dataclass(eq=False)
class _List:
    """A file list handed to name(): its files, its action, its git
    objects, the bundles of this session it touches whose bytes are not
    all named yet, and whether its action ran."""

    files: dict
    action: object
    git_objects: dict
    touching: set = field(default_factory=set)
    ran: bool = False


class Session:
    """A session of one command: add() files into its bundles, read() them
    back, name() the file lists that say where they lie, line() what
    happened to each source, end() when the command ends (or use it as a
    context manager). Several threads may add and record at once; a full
    bundle is sealed while the others go on filling the next one."""

    def __init__(self, store, folder, command, commit, dirty, bundle=BUNDLE, keep_local=True):
        check_record_name(folder)
        self.store, self.folder, self.bundle, self.keep_local = store, folder, bundle, keep_local
        self.row = {"folder": folder, "command": command, "started": now(), "ended": None, "wpt_commit": commit,
                    "dirty": int(bool(dirty))}
        remove_unowned_partials(store.cache)
        # the session's row goes in at once, so that a later session's end sees it running (what decides when it
        # removes the old copy of a rewritten bundle), with the rows sessions that crashed on this machine left;
        # the bundles marked that no session still running could read go now
        self._row_file, left = database.Rows.begin(store.cache, folder), []
        try:
            left = database.unowned_rows(store.cache)
            self._row_file.add(dict(self.row))
            removable = database.put(store.kind, store.cache, [self._row_file, *left])
        except BaseException:
            self._row_file.discard()  # a session that could not start ran nothing: its row alone is not kept
            raise
        finally:
            for rows in left:
                rows.close()  # removed once put in, kept otherwise
        self._remove(removable)
        # _lock guards everything below but _failure and the git objects' lookup. _sealing is held while one
        # bundle is sealed; it is taken under _lock, so bundles are sealed one at a time in the order they
        # were handed over
        self._lock, self._sealing = threading.Lock(), threading.Lock()
        self._number = 0  # the last bundle begun
        self._filling = None  # the bundle being filled: a _Partial
        self._lines, self._counted = [], 0  # lines not yet handed over with a bundle, and how many were added
        self._actions = []  # then()'s and name()'s actions not yet handed over with a bundle
        self._rows = []  # remember()'s git object rows not yet handed over with a bundle
        self._ended = False
        self._failure = None  # what a seal or a put raised: the session cannot go on
        # what the named file lists name, for the end's rewrite: each bundle's original bytes once handed over,
        # and for each bundle not known to be named whole, {(start, length): sha256} of its files named so far
        self._sizes, self._open = {}, {}
        self._lists, self._touching, self._listed = {}, {}, 0  # _List by number; bundle -> numbers of lists in it
        self._fresh = {}  # git object id -> Placed, of the lists this session named
        # the database read once, the sessions that had ended then, and which bundles the store holds
        self._known, self._ended_sessions, self._known_lock, self._bundles = None, set(), threading.Lock(), {}

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        try:
            self.end()
        except Exception as failure:
            if error is None:
                raise
            if failure is not error:
                error.add_note(f"the session's end could not be recorded either: {type(failure).__name__}: {failure}")

    def add(self, source) -> Placed:
        """Store one downloaded file (bytes, or a local file's path) in this
        session's bundles. Readable once its bundle is sealed: at once when
        it fills the bundle, at the end otherwise."""
        size = len(source) if isinstance(source, (bytes, bytearray, memoryview)) else os.path.getsize(source)
        placed = None
        while placed is None:
            with self._lock:
                self._check()
                if self._ended:
                    raise ValueError(f"session {self.folder} has ended")
                if self._filling is not None and self._filling.size + size > self.bundle:
                    full = self._hand_over()  # the file does not fit: it goes into the next bundle
                else:
                    placed = self._write(source)
                    full = self._hand_over() if placed.start + placed.length >= self.bundle else None
            if full is not None:
                self._seal(full)
        return placed

    def _write(self, source) -> Placed:
        """Under _lock: the file at the end of the bundle being filled,
        begun when there is none."""
        if self._filling is None:
            self._number += 1
            self._filling = _Partial(self.store, self._number)
            self._open[self._number] = {}
        filling, start = self._filling, self._filling.size
        digest = hashlib.sha256()
        for block in _blocks(source):
            digest.update(block)
            filling.write(block)  # what was read counts, should the file have changed since its size was taken
        return Placed(self._name(filling.number), start, filling.size - start, digest.hexdigest())

    def _name(self, number) -> str:
        return f"{self.folder}/{number:06d}"

    def _own(self, placed) -> int | None:
        """The number of a bundle of this session, or None for another session's."""
        folder, _, number = placed.bundle.rpartition("/")
        return int(number) if folder == self.folder else None

    def read(self, placed: Placed) -> bytes:
        """One file of the store, checked against its sha256: one this
        session added whether its bundle is sealed or not (one in the
        bundle being filled is read from that bundle's partial copy on this
        machine; one in a bundle being sealed once it is stored), one of
        another session's bundle from the store. CorruptObject when the
        bytes do not match their sha256."""
        number = self._own(placed)
        data = None
        with self._lock:
            self._check()
            if number is not None and self._filling is not None and number == self._filling.number:
                data = self._filling.read(placed.start, placed.length)
        if data is None:
            if number is not None:
                with self._sealing:  # the bundle is sealed or being sealed: wait for its seal to end
                    pass
                with self._lock:
                    self._check()
            return self.store.read_file(placed)
        if len(data) != placed.length or hashlib.sha256(data).hexdigest() != placed.sha256:
            raise CorruptObject(f"bundle {placed.bundle}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")
        return data

    def line(self, source, outcome, code, *, got=None, address=None, size=None, time=None):
        """Record what happened to one source: outcome is downloaded, reused
        or failed (wpt_store.database.OUTCOMES); got, what a download got
        (a dict JSON can hold, such as the commit), for the record only;
        time is now unless given."""
        if outcome not in database.OUTCOMES:
            raise ValueError(f"outcome must be one of {', '.join(database.OUTCOMES)}: {outcome!r}")
        with self._lock:
            self._check()
            self._counted += 1
            self._lines.append({"session": self.folder, "number": self._counted, "source": source,
                                "got": json.dumps(got, sort_keys=True) if got is not None else None,
                                "address": address,
                                "time": time or now(), "outcome": outcome, "size": size,
                                "code": code})

    def then(self, action):
        """Run action() once every file added so far is stored: when the
        bundle being filled (or, when none is, the next one or the end) is
        sealed, before the rows recorded so far go into the database."""
        with self._lock:
            self._check()
            if self._ended:
                raise ValueError(f"session {self.folder} has ended")
            self._actions.append(action)

    def name(self, files: dict, action, git_objects=None):
        """Name a file list ({path: (mode, Placed)}) once every file added
        so far is stored, when then() would run an action: its bytes
        (file_list_bytes) are stored as an object, then action(that
        object's sha256) runs. With no action (None), no list is stored: the
        files are only kept, with their git objects (the bytes the git map
        names that no tree names; the keys of files are then the caller's
        own). git_objects, {git object id: path} of files
        of the list that are git blobs: those this session placed go into
        the store database's git_objects table with where they lie, and
        known_objects() gives them from now on. Should the end rewrite a
        bundle sealed earlier that the list's files lie in, the list is
        stored again with their new places and action runs again with it.
        KeyError for a git object whose path the list does not hold."""
        with self._lock:
            self._check()
            if self._ended:
                raise ValueError(f"session {self.folder} has ended")
            named = _List(dict(files), action, dict(git_objects or {}))
            for object_id, path in named.git_objects.items():
                if path not in named.files:
                    raise KeyError(f"git object {object_id}: {path!r} is not a file of the list")
            self._listed += 1
            for _, placed in named.files.values():
                number = self._own(placed)
                if number in self._open:
                    self._open[number][(placed.start, placed.length)] = placed.sha256
                    named.touching.add(number)
                    self._touching.setdefault(number, set()).add(self._listed)
            if named.touching:
                self._lists[self._listed] = named
            for object_id, path in named.git_objects.items():
                self._fresh[object_id] = named.files[path][1]
            for number in list(named.touching):
                self._whole_yet(number)
            self._actions.append(lambda: self._store_list(named))

    def remember(self, git_objects: dict):
        """Put {git object id: Placed} of files the store holds already (a
        tree an earlier session stored) into the store database's
        git_objects table, with the rows of the next bundle sealed, and
        give them from known_objects() from now on: those in a bundle of a
        session that has ended. The others are left out: a session still
        running may yet rewrite its bundle and remove it, and this
        session's own files go in with the lists that name them."""
        lasting = {object_id: placed for object_id, placed in git_objects.items() if self._lasting(placed.bundle)}
        with self._lock:
            self._check()
            self._fresh |= lasting
            self._rows += [{"object_id": object_id, "sha256": placed.sha256, "bundle": placed.bundle,
                            "start": placed.start, "length": placed.length}
                           for object_id, placed in sorted(lasting.items())]

    def marked(self, source: str) -> bool:
        """Whether the store database, as read once per session, marks
        source for a new download (database.marked)."""
        with self._known_lock:
            self._read_known()
            return database.marked(self._known, source)

    def known_objects(self, object_ids) -> dict[str, Placed]:
        """{git object id: where the store holds its content} for each of
        object_ids the store database's git_objects table (read once per
        session) or this session's named lists know, whose bundle is this
        session's or one of a session that had ended when the table was
        read (no other session can then remove it), and is held by the
        store."""
        object_ids = list(dict.fromkeys(object_ids))
        found = {}
        with self._known_lock:
            self._read_known()
            for at in range(0, len(object_ids), QUERY):
                part = object_ids[at:at + QUERY]
                for object_id, sha256, bundle, start, length in self._known.execute(
                        f"SELECT object_id, sha256, bundle, start, length FROM git_objects WHERE object_id IN "
                        f"({', '.join('?' * len(part))})", part):
                    if bundle.rpartition("/")[0] in self._ended_sessions:
                        found[object_id] = Placed(bundle, start, length, sha256)
        with self._lock:
            found |= {object_id: self._fresh[object_id] for object_id in object_ids if object_id in self._fresh}
        return {object_id: placed for object_id, placed in found.items() if self._held(placed.bundle)}

    def _read_known(self):
        """Under _known_lock: the store database read, once per session,
        with the sessions that had ended then."""
        if self._known is None:
            self._known = self.store.database(threads=True)
            self._ended_sessions = {folder for folder, in self._known.execute(
                "SELECT folder FROM sessions WHERE ended IS NOT NULL")}

    def _lasting(self, bundle) -> bool:
        """Whether a bundle is another session's that had ended when the
        store database was read: sealed for good, never removed."""
        folder = bundle.rpartition("/")[0]
        with self._known_lock:
            self._read_known()
            return folder != self.folder and folder in self._ended_sessions

    def _held(self, bundle) -> bool:
        """Whether the store holds a bundle (this session's always are)."""
        if bundle.rpartition("/")[0] == self.folder:
            return True
        with self._known_lock:
            if bundle not in self._bundles:
                self._bundles[bundle] = self.store.has_bundle(bundle)
            return self._bundles[bundle]

    def end(self):
        """Seal the bundles for good, the last one included, rewriting each
        that holds bytes no named file list names (see the module's
        docstring), and put the session's end and its last rows in the
        database. Nothing more once ended."""
        with self._lock:
            if self._ended:
                self._check()
                return
            self._ended = True
        try:
            with self._sealing:  # a bundle another thread handed over is sealed first
                pass
            with self._lock:
                self._check()
            marks, rows = [], []
            try:
                with self._lock:
                    self._check()
                    earlier = [number for number in sorted(self._open) if number in self._sizes]
                for number in earlier:
                    rows += self._rewrite_sealed(number)
                    marks.append(self._name(number))
                with self._lock:
                    self._check()
                    self._rewrite_filling()
                    self.row["ended"] = now()
                    last = self._hand_over(rows, [{"bundle": bundle, "marked": now()} for bundle in marks])
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                raise
            removable = self._seal(last)
        finally:
            # the rows file goes once the database holds all of it, and stays for a later session otherwise
            self._row_file.close()
            with self._known_lock:
                if self._known is not None:
                    self._known.close()
        self._remove(removable)

    def _remove(self, removable):
        """Remove the bundles the database marks that no session which
        started before the mark still runs (put and Session.end give them),
        and their marks."""
        for bundle in removable:
            self.store.drop_bundle(bundle)
        database.forget(self.store.kind, self.store.cache, removable)

    def _rewrite_sealed(self, number) -> list[dict]:
        """A bundle sealed earlier rewritten as a new bundle of the session
        holding only its named files (none when it holds none), every list
        naming it stored again with their new places and its action run
        again: the git object rows to put."""
        named = sorted(self._open.pop(number).items())
        moved = {}
        if named:
            self._number += 1
            rewritten = _Partial(self.store, self._number)
            try:
                for (start, length), sha256 in named:
                    moved[(start, length)] = Placed(self._name(rewritten.number), rewritten.size, length, sha256)
                    for data in self.store.file_parts(Placed(self._name(number), start, length, sha256), PART):
                        rewritten.write(data)
                original, stored = rewritten.seal()
                self.store.keep_compressed(f"bundles/{self._name(rewritten.number)}", rewritten.path, original,
                                           stored, self.keep_local)
            finally:
                rewritten.remove()
        rows = []
        for named_list in self._moved(number, moved):
            if named_list.ran:
                rows += self._store_list(named_list)
        return rows

    def _rewrite_filling(self):
        """Under _lock: the bundle being filled rewritten on this machine's
        disk without the bytes no named list names, before it is stored;
        dropped when it holds no named file."""
        filling = self._filling
        if filling is None:
            return
        named = sorted(self._open.pop(filling.number, {}).items())
        if sum(length for (_, length), _ in named) == filling.size:
            return
        moved = {}
        if named:
            rewritten = _Partial(self.store, filling.number)
            try:
                for (start, length), sha256 in named:
                    moved[(start, length)] = Placed(self._name(filling.number), rewritten.size, length, sha256)
                    for at in range(0, length, PART):
                        rewritten.write(filling.read(start + at, min(PART, length - at)))
            except BaseException:
                rewritten.remove()
                raise
            self._filling = rewritten
        else:
            self._filling = None
        filling.remove()
        self._moved(filling.number, moved)

    def _moved(self, number, moved) -> list:
        """The lists naming files of a rewritten bundle, each file given its
        new place (moved: {(start, length): Placed})."""
        lists = [self._lists[listed] for listed in sorted(self._touching.pop(number, ()))]
        for named_list in lists:
            named_list.files = {path: (mode, moved.get((p.start, p.length), p) if self._own(p) == number else p)
                                for path, (mode, p) in named_list.files.items()}
            named_list.touching.discard(number)
        return lists

    def _store_list(self, named_list) -> list[dict]:
        """Outside _lock, at a seal: a named list stored and its action run;
        the git object rows of its files this session placed."""
        if named_list.action is not None:
            named_list.action(self.store.put_bytes(file_list_bytes(named_list.files)))
        named_list.ran = True
        placed = {object_id: named_list.files[path][1] for object_id, path in named_list.git_objects.items()}
        return [{"object_id": object_id, "sha256": where.sha256, "bundle": where.bundle, "start": where.start,
                 "length": where.length}
                for object_id, where in sorted(placed.items()) if self._own(where) is not None]

    def _whole_yet(self, number):
        """Under _lock: once a bundle handed over has every byte named, it
        needs no rewrite; the lists kept for it alone are let go."""
        if number not in self._sizes or number not in self._open:
            return
        if sum(length for _, length in self._open[number]) != self._sizes[number]:
            return
        del self._open[number]
        for listed in self._touching.pop(number, ()):
            named_list = self._lists[listed]
            named_list.touching.discard(number)
            if not named_list.touching:
                del self._lists[listed]

    def _check(self):
        """Under _lock: raise the failure that stopped the session, after
        dropping what would only have gone with it (the bundle being filled
        and the rows not handed over)."""
        if self._failure is None:
            return
        if self._filling is not None:
            self._filling.remove()
            self._filling = None
        self._lines, self._actions, self._rows = [], [], []
        raise self._failure

    def _hand_over(self, rows=(), marks=()):
        """Under _lock: the bundle being filled (or None), the session's row,
        the lines and the actions so far, and rows (git objects) and marks
        (bundles to remove), to seal and put together; the next add begins a
        new bundle. Takes _sealing, which _seal releases."""
        filling = self._filling
        if filling is not None:
            self._sizes[filling.number] = filling.size
            self._whole_yet(filling.number)
        handed = (filling, dict(self.row), self._lines, self._actions, self._rows + list(rows), list(marks))
        self._filling, self._lines, self._actions, self._rows = None, [], [], []
        self._sealing.acquire()
        return handed

    def _seal(self, handed):
        """Outside _lock, holding _sealing: store the bundle, run the actions
        handed over with it, then put its rows and marks in the database. A
        failure stops the session. The bundles the database then marks that
        no running session can read any more: those it gives back."""
        filling, row, lines, actions, rows, marks = handed
        try:
            if self._failure is not None:  # an earlier bundle was lost: this one goes with it
                raise self._failure
            if filling is not None:
                original, stored = filling.seal()
                self.store.keep_compressed(f"bundles/{self._name(filling.number)}", filling.path, original, stored,
                                           self.keep_local)
            for action in actions:
                rows += action() or []
            self._row_file.add(row, lines, rows, marks)
            return database.put(self.store.kind, self.store.cache, [self._row_file])
        except BaseException as error:
            if self._failure is None:
                self._failure = error
            raise
        finally:
            if filling is not None:
                filling.remove()
            self._sealing.release()


def _blocks(source):
    if isinstance(source, (bytes, bytearray, memoryview)):  # in parts, so a large file is never copied whole
        view = memoryview(source)
        for at in range(0, len(view), CHUNK):
            yield view[at:at + CHUNK]
        return
    with open(source, "rb") as file:
        while block := file.read(CHUNK):
            yield block
