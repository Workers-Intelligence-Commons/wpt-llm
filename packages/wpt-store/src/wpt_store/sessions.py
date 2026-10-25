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
is the store's. A
bundle is cut only between files, once it holds BUNDLE bytes or more; a
file bigger than BUNDLE gets a bundle of its own. Where a file went is a
Placed: its bundle, start and length in the bundle's original bytes, and
its sha256, which every read checks. Session.read gives a file back
before its bundle is sealed too, from the bundle's partial copy on this
machine. A file several sources hold is
stored again for each of them: no table of every file the store holds.

A source's file list (file_list_bytes, parse_file_list) says where each of
its files went, one line per file, sorted by the UTF-8 bytes of the path:

    <mode> <sha256> <bundle> <start> <length> <path>

mode is the caller's, kept as it is (wpt_dataset's manifest modes).

The session's lines (one per download, check against upstream, use of a
source already in the store, failure) go into the store database
(wpt_store.database) with the session's own row each time a bundle is
sealed, and at the end. An action handed to then() (such as setting the
refs that name a downloaded source's files) runs at the same moment, just
before those lines go in: once every file added before it is stored. A full bundle is sealed while the next one fills;
should it fill too before the first is sealed, adding waits. A crash
loses only the bundles not yet sealed (the one being sealed and the one
being filled) and the lines added since the last one was sealed.

Failures: what the store raised while sealing a bundle or putting the
rows in the database. The bundles not yet sealed and the lines not yet
put are then lost, and the session cannot go on: every later add, line
and end raises that same failure and writes nothing; the actions handed
to then() with them never run. A failing action stops the session the same
way. ValueError for an outcome the database does not know, or a file
added after the end.
"""

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import blocks, database
from .errors import CorruptObject
from .names import check_record_name

BUNDLE = 4 << 30  # bytes: a bundle is sealed once it holds this many or more
CHUNK = 1 << 20  # bytes of a local file read at once


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


class Session:
    """A session of one command: add() files into its bundles, read() them
    back, line() what happened to each source, end() when the command ends (or use it as a
    context manager). Several threads may add and record at once; a full
    bundle is sealed while the others go on filling the next one."""

    def __init__(self, store, folder, command, commit, dirty, bundle=BUNDLE, keep_local=True):
        check_record_name(folder)
        self.store, self.folder, self.bundle, self.keep_local = store, folder, bundle, keep_local
        self.row = {"folder": folder, "command": command, "started": now(), "ended": None, "wpt_commit": commit,
                    "dirty": int(bool(dirty))}
        # _lock guards everything below but _failure. _sealing is held while one bundle is sealed; it is
        # taken under _lock, so bundles are sealed one at a time in the order they were handed over
        self._lock, self._sealing = threading.Lock(), threading.Lock()
        self._number = 0  # the last bundle begun
        self._filling = None  # the bundle being filled: (temporary file, its Writer, original bytes so far)
        self._lines, self._counted = [], 0  # lines not yet handed over with a bundle, and how many were added
        self._actions = []  # then()'s actions not yet handed over with a bundle
        self._ended = False
        self._failure = None  # what a seal or a put raised: the session cannot go on

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
                if self._filling is not None and self._filling[2] + size > self.bundle:
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
            self.store.cache.mkdir(parents=True, exist_ok=True)
            out = tempfile.NamedTemporaryFile(dir=self.store.cache, prefix=".partial-bundle-", delete=False)
            self._filling = (out, blocks.Writer(out, *self.store.compressing()), 0)
        out, writer, start = self._filling
        digest, size = hashlib.sha256(), 0
        for block in _blocks(source):
            digest.update(block)
            writer.write(block)
            size += len(block)  # what was read, should the file have changed since its size was taken
        self._filling = (out, writer, start + size)
        return Placed(f"{self.folder}/{self._number:06d}", start, size, digest.hexdigest())

    def read(self, placed: Placed) -> bytes:
        """One file this session added, checked against its sha256, whether
        its bundle is sealed or not: one in the bundle being filled is read
        from that bundle's partial copy on this machine, whose frame being
        filled ends there early; one in a bundle being sealed once it is
        stored. ValueError for a file of another session; CorruptObject
        when the bytes do not match their sha256."""
        folder, _, number = placed.bundle.rpartition("/")
        if folder != self.folder:
            raise ValueError(f"{placed.bundle} is not a bundle of session {self.folder}")
        with self._lock:
            self._check()
            data = None
            if self._filling is not None and int(number) == self._number:
                out, writer, _ = self._filling
                writer.flush()
                out.flush()
                with open(out.name, "rb") as partial:
                    data = blocks.read(partial, blocks.SeekTable(writer.sizes), placed.start, placed.length)
        if data is None:
            with self._sealing:  # the bundle is sealed or being sealed: wait for its seal to end
                pass
            with self._lock:
                self._check()
            return self.store.read_file(placed)
        if len(data) != placed.length or hashlib.sha256(data).hexdigest() != placed.sha256:
            raise CorruptObject(f"bundle {placed.bundle}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")
        return data

    def line(self, source, outcome, wpt_version, *, revision=None, checksum=None, address=None, size=None,
             time=None):
        """Record what happened to one source: outcome is downloaded, reused
        or failed (wpt_store.database.OUTCOMES); time is now unless given."""
        if outcome not in database.OUTCOMES:
            raise ValueError(f"outcome must be one of {', '.join(database.OUTCOMES)}: {outcome!r}")
        with self._lock:
            self._check()
            self._counted += 1
            self._lines.append({"session": self.folder, "number": self._counted, "source": source,
                                "revision": revision, "checksum": checksum, "address": address,
                                "time": time or now(), "outcome": outcome, "size": size,
                                "wpt_version": wpt_version})

    def then(self, action):
        """Run action() once every file added so far is stored: when the
        bundle being filled (or, when none is, the next one or the end) is
        sealed, before the lines recorded so far go into the database."""
        with self._lock:
            self._check()
            if self._ended:
                raise ValueError(f"session {self.folder} has ended")
            self._actions.append(action)

    def end(self):
        """Seal the last bundle and put the session's end and its last lines
        in the database. Nothing more once ended."""
        with self._lock:
            self._check()
            if self._ended:
                return
            self._ended = True
            self.row["ended"] = now()
            last = self._hand_over()
        self._seal(last)

    def _check(self):
        """Under _lock: raise the failure that stopped the session, after
        dropping what would only have gone with it (the bundle being filled
        and the lines not handed over)."""
        if self._failure is None:
            return
        if self._filling is not None:
            out = self._filling[0]
            out.close()
            Path(out.name).unlink(missing_ok=True)
            self._filling = None
        self._lines, self._actions = [], []
        raise self._failure

    def _hand_over(self):
        """Under _lock: the bundle being filled (or None), its number, the
        session's row and the lines so far, to seal and put together; the
        next add begins a new bundle. Takes _sealing, which _seal releases."""
        handed = (self._filling, self._number, dict(self.row), self._lines, self._actions)
        self._filling, self._lines, self._actions = None, [], []
        self._sealing.acquire()
        return handed

    def _seal(self, handed):
        """Outside _lock, holding _sealing: store the bundle, run the actions
        handed over with it, then put its lines in the database. A failure
        stops the session."""
        filling, number, row, lines, actions = handed
        try:
            if self._failure is not None:  # an earlier bundle was lost: this one goes with it
                raise self._failure
            if filling is not None:
                out, writer, _ = filling
                original, stored = writer.close()
                out.close()
                self.store.keep_compressed(f"bundles/{self.folder}/{number:06d}", Path(out.name), original, stored,
                                           self.keep_local)
            for action in actions:
                action()
            database.put(self.store.kind, row, lines)
        except BaseException as error:
            if self._failure is None:
                self._failure = error
            raise
        finally:
            if filling is not None:
                filling[0].close()
                Path(filling[0].name).unlink(missing_ok=True)
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
