"""Storage for everything WPT downloads or builds.

WPT hosts no data: each machine points the WPT_STORE setting at its own store,

    WPT_STORE=file:///home/me/wpt-store      a local folder
    WPT_STORE=s3://bucket/prefix             an S3-compatible bucket

The prefix picks the store's kind (wpt_store.kinds: Folder, Bucket), which
supplies five operations on files; everything here is built on them, alike
for both kinds. A store holds real named things, the same on a folder and
on a bucket:

    index                              each session ID and what it holds (wpt_store.index)
    <id>/sources/<source>.tar.zst      a source's files (wpt_store.sessions)
    <id>/sources/<source>.json         when, address, size, or the error, and what reading needs
    <id>/datasets/<name>/              dataset.json, its shards, vocabulary, document list, credits,
                                       conversations and tasks (wpt_store.sessions)
    <id>/models/<name>/                the model's real files (wpt_store.models)
    <id>/runs/<run>/                   each command's log
    cache/<stage>/<fingerprint>        results that can be rebuilt, shared by every ID
    cut-key                            the secret cuts are matched under (wpt_dataset.erasure)

An ID is a session's (wpt_store.sessions): Store.at(session, start) is the
store as one command works in it, and a store given no ID refuses whatever
needs one (SessionNeeded).

Hashes do two jobs. Every read checks the bytes against their sha256, which
what names a file holds: a File (key, sha256) as a record gives it, a
wpt_store.Placed for one file of a source. A record itself, written by
put_json, holds the sha256 of its own fields, `check`, and a result holds
its own in its first line. And finished work is found from its inputs: its
fingerprint names its result.

    put_result(stage, fingerprint, data, made_by), get_result(stage, fingerprint), has_result
        a result, cache/<stage>/<fingerprint>: its first line says its
        sha256 and what made it (wpt_utils.programs.code_name), so finding
        it says that at no further read
    put_bytes(data), put_file(path)
        a result too large to hold in another one, or whose content is all
        that names it (a cut's or a patch's output, a token file), under
        the fingerprint its sha256 is: cache/files/<sha256>. They give that
        sha256, which has(), size(), path(), get_bytes(), read_range(),
        release() and delete() take, as they take a File
    put_named(key, source), put_json(key, record), get_json(key), delete_json(key)
        a file of a session under the key its caller names

Every file a store writes is compressed while it is written, with zstd in
frames of about 1 MB that a range read takes a few at a time (wpt_store.blocks),
and kept so only when that saves space, under its key plus .zst then, under
its key as it is otherwise (Store.keep_smaller). Which of the two names a
file is stored under is found once and kept in memory. A source's tar is
always compressed: its name says so. Large files are compressed several
frames at once, as wpt_utils.parallel allows, and sent to S3 in parts that
every upload of the process sends together, at most wpt_store.kinds.UPLOADS
at once in total. Record files are compressed, except the one a local store
adds lines to while its record runs.

A run's record is one folder of files written by name, a command's own
account of one run (see wpt_telemetry), <id>/runs/<run>/, named here
"<id>/<run>", built on the same five operations as everything else, alike
for both kinds. A file is written whole, or added to line by line, never
both. A file written whole is in the store, on the disk for a local store,
when the call returns. An S3 store holds added lines back until
flush_records() sends them together, as one part of the file, since each
request costs time (and money on S3). A local store appends each line at
once to one plain file, <id>/runs/<run>/<name>.open, so a crash loses none
that was added (flush_records() puts them on the disk), and
end_records(folder), when the record ends, compresses each such file into
one part. Lines added by several processes read back in the order they were
added.

For S3, the credentials and a non-AWS endpoint come from the store's own
settings, in the environment or in .env: WPT_STORE_ACCESS_KEY_ID and
WPT_STORE_SECRET_ACCESS_KEY (both needed), WPT_STORE_SESSION_TOKEN,
WPT_STORE_ENDPOINT_URL and WPT_STORE_REGION (default us-east-1). The
store reads no AWS_* setting (no AWS_ENDPOINT_URL, no AWS_PROFILE) and no
AWS configuration or credentials file, so the keys a cloud provider's
other tools use never reach it. A store keeps local copies in its cache
folder: an S3 store every file read as a file, after its check, and the
tars its downloads made unless they stream; a local store the compressed
ones read as a file, which it cannot give in place. release() removes a
copy; copies_in(folder) is the same store making in folder only the copies
its work asks for as files, for work that keeps none. A local store keeps
that folder and its scratch folder under .local/ at its root, which is no
part of the store. A store sent to another process (pickled) opens the same
store there, in the same session.

Failures: StoreSettingError (no or bad WPT_STORE), SessionNeeded (an ID
needed and not given), MissingFile (no such file or record file),
CorruptFile (stored bytes do not match their sha256, a compressed file or
the index is not in its format, or a record part is not in the store's
format; the copy is not used), MixedRecord (a whole write to a file added
to line by line, or the reverse; nothing is written). Writing raises what
the folder or the bucket raised (OSError, botocore's errors): nothing is
known to be stored.
"""

import collections
import contextlib
import copy
import hashlib
import itertools
import json
import os
import re
import secrets
import tempfile
import threading
import time
from compression import zstd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

from wpt_utils.files import lock_file, sha256_file, unlock_file
from wpt_utils.parallel import Sizing, in_order

from . import blocks, index, sessions
from .ahead import REQUESTS, Ahead
from .errors import CorruptFile, MissingFile, MixedRecord, SessionNeeded, StoreError, StoreSettingError
from .kinds import UPLOADS, Absent, Bucket, Folder, atomic_copy, sync_file, sync_folder
from .names import NAME, NAME_CHARACTERS, File, as_record_name, check_name, check_record_name, check_session
from .models import copy_out, has_model, keep_model, model_file, put_model
from .sessions import (Held, Packing, dataset_key, dataset_vocabulary, failed_addresses, find_dataset, given, held, held_all, held_here,
                       keep_dataset, keep_failure, latest_dataset, needed, new_id, remove_unowned_partials, source_row,
                       started)
from .settings import now, read_env_file
from .tars import Placed

SHA256 = re.compile(r"[0-9a-f]{64}")
CHUNK = 1 << 20
PACKED = ".zst"  # the end of a key whose file is compressed
OPEN = ".open"  # the end of the key of the plain file a local store adds a record file's lines to while it runs
TAIL = 64 << 10  # bytes read at a compressed file's end for its seek table: the whole table up to 8,000 frames
DOWNLOAD = 64 << 20  # bytes of one file fetched in one request, when a whole file is copied from the store
RUN_BYTES = 16 << 20  # original bytes of frames one request of fetch_files takes, past which the next run starts
FRAMES_KEPT = 64  # decompressed frames kept in memory for range reads (about 64 MB)
READ_AHEAD = 16  # frames one range read of a kind that reads in place (a folder) fetches at most when it needs some
# that are not kept: about 16 MB per request, fewer than FRAMES_KEPT so that all of them stay kept
TABLES_KEPT = 1024  # seek tables kept in memory
KEYS_KEPT = 1 << 16  # keys files were found under (compressed or not) kept in memory
HELD_KEPT = 16  # source records kept parsed in memory, with their versions (wpt_store.sessions): one of 200,000
# files takes about 100 MB
CACHE, FILES = "cache", "files"  # the folder of what can be rebuilt, and its stage of results named by their content
RUNS = index.RUNS  # the folder of a session's run records
CHECK = "check"  # the field of a record (put_json) holding the sha256 of its other fields


WHOLE, APPENDED = "written whole", "added to line by line"  # the two kinds of record file


def _check_sha(sha):
    if not SHA256.fullmatch(sha):
        raise ValueError(f"not a sha256: {sha!r}")


def _run_prefix(folder, name=None) -> str:
    """The key of a run's folder ("<id>/<run>"), ending in '/'; ValueError
    for any other shape, or a file name the store does not allow."""
    session, slash, run = folder.partition("/")
    if not slash:
        raise ValueError(f"a run is named <session>/<run>: {folder!r}")
    check_session(session)
    check_record_name(run)
    if name is not None:
        check_record_name(name)
    return f"{session}/{RUNS}/{run}/"


def _result_key(stage, fingerprint) -> str:
    check_name(stage)
    check_name(fingerprint)
    return f"{CACHE}/{stage}/{fingerprint}"


def _record_bytes(record: dict) -> bytes:
    """A record as put_json writes it, without its CHECK: its sha256 is what
    CHECK holds."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8", "surrogatepass")


class _Kept:
    """The last `size` values by key, for several threads."""

    def __init__(self, size):
        self.size, self._values, self._lock = size, collections.OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._values:
                self._values.move_to_end(key)
            return self._values.get(key)

    def put(self, key, value):
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.size:
                self._values.popitem(last=False)

    def drop(self, test):
        with self._lock:
            for key in [key for key in self._values if test(key)]:
                del self._values[key]


class _Ticks:
    """The moments and numbers of a store's additions to its record files,
    for several threads and for every view of the store: no moment and no
    number twice."""

    def __init__(self):
        self._lock, self._last, self._added = threading.Lock(), 0, 0

    def next(self) -> tuple[int, int]:
        with self._lock:
            self._last = max(time.time_ns(), self._last + 1)
            self._added += 1
            return self._last, self._added


class Store:
    """Operations every store offers, built on its kind's five (see
    wpt_store.kinds), run records included. Subclasses give the kind,
    `cache` (a local folder for copies of files) and `scratch` (a local
    folder for files derived from others, such as token files for
    training, which may be deleted at any time)."""

    scratch: Path
    local = False  # whether the kind's files are on this machine's disk already, so no copy of them is kept

    def __init__(self, kind, cache):
        self.kind, self.cache = kind, Path(cache)
        self.session = None  # the ID of a view of at(): the session it works in
        self.start = None  # the ID that session started from (at), which it reads what it lacks from
        self.copies_only = False  # a view of copies_in: only path() and file_path() make a local copy
        self._copies = Folder(self.cache)  # the local copies of tars, read through the same five operations
        self._record_kinds = {}  # (folder, name) -> WHOLE or APPENDED, as this store learned it
        self._tables, self._frames, self._keys = _Kept(TABLES_KEPT), _Kept(FRAMES_KEPT), _Kept(KEYS_KEPT)
        self.held_records = _Kept(HELD_KEPT)  # (session, source) -> (its version, its parsed record) (wpt_store.sessions)
        self._ahead = Ahead()  # reads of a kind that waits (S3) with several requests in flight
        self._compressing, self._compressing_lock = None, threading.Lock()
        # record lines added and not yet sent: (folder, name) -> [(time, number, data)]; see _raw_append_record
        self._writer, self._pending, self._ticks = secrets.token_hex(4), {}, _Ticks()
        self._pending_lock = threading.Lock()
        self._record_started = set()  # (folder, name) of the plain record files this store has added to

    # Sessions

    def at(self, session: str, start: str | None = None) -> "Store":
        """The same store as one command works in it: under the ID session,
        reading from the ID start what the session does not hold."""
        check_session(session)
        if start is not None:
            check_session(start)
        self.compressing()  # made before the copy, so every view shares this store's one
        view = copy.copy(self)
        view.session, view.start = session, start
        return view

    def needed(self) -> str:
        """The session of this view of at(), SessionNeeded without one."""
        if self.session is None:
            raise SessionNeeded(sessions.NO_SESSION)
        return self.session

    def reading(self) -> list[str]:
        """The sessions a read looks in, in order: this view's, then the one
        it started from."""
        return [self.needed(), *([self.start] if self.start else [])]

    def sessions(self) -> list[str]:
        """Every session the store holds, sorted, from its folders."""
        found = (key.removesuffix("/") for key in self.kind.list("", deep=False) if key.endswith("/"))
        return sorted(name for name in found if name != CACHE)

    def index(self, threads=False):
        """The store's index (wpt_store.index), as a sqlite3 connection to a
        copy in the cache folder, removed when the connection closes
        (close(), or the end of a `with` block over it: there it closes,
        where a plain sqlite3 connection's `with` only ends a transaction);
        threads: usable from any thread, one at a time."""
        return index.read(self.kind, self.cache, threads)

    def note(self, rows):
        """Put rows (wpt_store.index.held_row) into the index."""
        index.put(self.kind, self.cache, rows)

    # Results

    def put_result(self, stage: str, fingerprint: str, data: bytes, made_by: str | None = None) -> str:
        """Keep a result under the fingerprint of what made it, replacing
        any earlier one. made_by, one word, says what made it
        (wpt_utils.programs.code_name). Its sha256."""
        sha = hashlib.sha256(data).hexdigest()
        self.keep_smaller(_result_key(stage, fingerprint), b"%s %s\n" % (sha.encode(), (made_by or "").encode()) + data)
        return sha

    def get_result(self, stage: str, fingerprint: str) -> tuple[bytes | None, str | None]:
        """(a result's bytes, checked against their sha256; what made it,
        as put_result got it), (None, None) when there is none."""
        key = _result_key(stage, fingerprint)
        whole = self._whole(key)
        if whole is None:
            return None, None
        head, _, data = whole.partition(b"\n")
        sha, _, made_by = head.decode(errors="replace").partition(" ")
        if hashlib.sha256(data).hexdigest() != sha:
            raise CorruptFile(f"{key} reads back as {hashlib.sha256(data).hexdigest()}, not {sha}")
        return data, made_by or None

    def has_result(self, stage: str, fingerprint: str) -> bool:
        """Whether a result is kept, without reading it."""
        return self._stored(_result_key(stage, fingerprint), fresh=True) is not None

    def _whole(self, key) -> bytes | None:
        """The original bytes of the file under key, compressed or not, None
        without one."""
        try:
            data = self.kind.read(key + PACKED)[0]
        except Absent:
            pass
        else:
            return self._decompressed(key + PACKED, data)
        try:
            return self.kind.read(key)[0]
        except Absent:
            return None

    # Files named by their sha256, or by what a record says of them

    def _file(self, what) -> File:
        """A File from what names one: a File, a record's reference to one
        ({"key", "sha256"}), or the sha256 of a file of cache/files/."""
        if isinstance(what, File):
            return what
        if isinstance(what, dict):
            return File.of(what)
        if not SHA256.fullmatch(what):
            raise ValueError(f"not a sha256: {what!r}")
        return File(f"{CACHE}/{FILES}/{what}", what)

    def _local(self, file: File) -> Path:
        return self.cache / file.sha256[:2] / file.sha256

    def file(self, sha: str) -> File:
        """The File of cache/files/ that the sha256 names."""
        return self._file(sha)

    def put_bytes(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if not self.has(sha):
            self._put(sha, data)
        return sha

    def put_file(self, path) -> str:
        sha = sha256_file(path)
        if not self.has(sha):
            self._put(sha, Path(path))
        return sha

    def _put(self, sha, source):
        try:
            self.keep_smaller(self._file(sha).key, source)
        except PermissionError:
            # on Windows, replacing a file another process has open is refused; the
            # name is the content's hash, so a file stored meanwhile holds the same bytes
            if not self.has(sha):
                raise

    def put_named(self, key: str, source) -> File:
        """Store source (bytes, or a local file, left as it is) under key,
        replacing any earlier file: the File that reads it back."""
        sha = sha256_file(source) if isinstance(source, (str, Path)) else hashlib.sha256(source).hexdigest()
        self.keep_smaller(key, Path(source) if isinstance(source, str) else source)
        return File(key, sha)

    def put_json(self, key: str, record: dict) -> dict:
        """Store a record, a JSON object, under key, replacing any earlier
        one, with the sha256 of its fields in CHECK, the ID it started from
        in `from` when the session has one, over any `from` a record copied
        from another session brought: the record as stored. A record of the
        session is on the disk, or held by S3, when this returns."""
        record = {name: value for name, value in record.items() if name != CHECK}
        if self.start is not None and self.session is not None and key.startswith(f"{self.session}/"):
            record["from"] = self.start
        record[CHECK] = hashlib.sha256(_record_bytes(record)).hexdigest()
        self.keep_smaller(key, _record_bytes(record), durable=True)
        return record

    def get_json(self, key: str) -> dict | None:
        """The record under key, checked against its CHECK, None without one."""
        data = self._whole(key)
        if data is None:
            return None
        try:
            record = json.loads(data)
            check = record.pop(CHECK)
        except (ValueError, KeyError, TypeError, AttributeError):
            raise CorruptFile(f"{key} is not a record") from None
        if hashlib.sha256(_record_bytes(record)).hexdigest() != check:
            raise CorruptFile(f"{key} does not match its check")
        record[CHECK] = check
        return record

    def version(self, key: str) -> str | None:
        """What the kind says of the file under key now, compressed or not,
        None without one: it changes whenever the file is replaced, by any
        process."""
        for stored in (key + PACKED, key):
            try:
                return self.kind.read(stored, 0, 0)[1]
            except Absent:
                pass
        return None

    def delete_json(self, key: str):
        """Remove the record under key; nothing when it is absent."""
        for stored in (key + PACKED, key):
            self.kind.delete(stored)
        self.forget(key)

    def has(self, what) -> bool:
        """Asks the store each time, so a file deleted by another process
        is not taken as stored."""
        return self._stored(self._file(what).key, fresh=True) is not None

    def size(self, what) -> int:
        """The file's length in bytes, without reading or copying it (and
        so without checking it against its sha256)."""
        file = self._file(what)
        key = self._found(file)
        with self._reading(file.key, key):
            if key.endswith(PACKED):
                return self._table(key).size
            on_disk = self.kind.on_disk(key)
            total = on_disk.stat().st_size if on_disk is not None else self.kind.list(key).get(key)
        if total is None:
            self.forget(file.key)
            raise MissingFile(f"{file.key} is not in the store")
        return total

    def path(self, what) -> Path:
        """A local file holding the file, checked against its sha256. A
        copy in the cache folder when it is not a local file as it is
        (every file of an S3 store, a local store's compressed ones),
        and always in a view of copies_in: the copy stays until release()."""
        file = self._file(what)
        key = self._found(file)
        local = self._local(file)
        path = None if key.endswith(PACKED) or self.copies_only else self.kind.on_disk(key)
        if path is None:
            if not local.is_file():
                with self._reading(file.key, key):
                    self._copy(key, local)
            path = local
        actual = sha256_file(path)
        if actual != file.sha256:
            if path == local:
                local.unlink(missing_ok=True)
            raise CorruptFile(f"{file.key} reads back as {actual}, not {file.sha256}")
        return path

    def get_bytes(self, what) -> bytes:
        """The file's bytes, checked against its sha256. A file that is
        not a local one is read through a copy in the cache folder (kept
        until release()), except in a view of copies_in, which reads it
        straight from the store and keeps no copy."""
        file = self._file(what)
        key = self._found(file)
        on_disk = self.kind.on_disk(key)
        if on_disk is None and self.copies_only and not self._local(file).is_file():
            with self._reading(file.key, key):
                data = self._range(key, 0, self.size(file))
            if hashlib.sha256(data).hexdigest() != file.sha256:
                raise CorruptFile(f"{file.key} reads back as {hashlib.sha256(data).hexdigest()}, not {file.sha256}")
            return data
        if on_disk is None or self._local(file).is_file():
            return self.path(file).read_bytes()
        try:
            data = on_disk.read_bytes()
        except FileNotFoundError:
            self.forget(file.key)
            raise MissingFile(f"{file.key} is not in the store") from None
        if key.endswith(PACKED):
            data = self._decompressed(key, data)
        actual = hashlib.sha256(data).hexdigest()
        if actual != file.sha256:
            raise CorruptFile(f"{file.key} reads back as {actual}, not {file.sha256}")
        return data

    def read_range(self, what, start: int, length: int) -> bytes:
        """length bytes of a file from offset start, without fetching the
        rest of it. A part cannot be checked against the file's sha256:
        the caller checks it against a hash of its own."""
        file = self._file(what)
        key = self._found(file)
        local = self._local(file)
        if local.is_file():
            with open(local, "rb") as held:
                held.seek(start)
                data = held.read(length)
        else:
            with self._reading(file.key, key):
                data = self._range(key, start, length)
        if len(data) != length:
            raise CorruptFile(f"{file.key}: {length} bytes at {start} read back as {len(data)}")
        return data

    def release(self, what):
        """Remove the local copy of a file (it stays in the store); nothing
        when there is none, as for a local store's file that is not
        compressed, whose copy is the file itself."""
        self._local(self._file(what)).unlink(missing_ok=True)

    def delete(self, what):
        """Remove a file, and its local copy; nothing when it is absent.
        Only for one that nothing else can point at, such as a training
        checkpoint that a newer one replaced."""
        file = self._file(what)
        self.release(file)
        for stored in (file.key + PACKED, file.key):
            self.kind.delete(stored)
        self.forget(file.key)

    def copies_in(self, folder) -> "Store":
        """The same store, whose local copies of files go into folder
        instead of its cache folder, for work that keeps none: removing
        folder afterwards removes every copy that work made. Only a file
        the work asks for as a file (path(), file_path()) is copied, a local
        store's own files too, so the work never opens a file of the store
        itself; everything else is read from the store (get_bytes,
        read_range, read_file, file_parts). A copy folder already holds is
        read instead. Copies already in the cache folder are not read. Not
        for run records."""
        self.compressing()  # made before the copy, so every view shares this store's one
        view = copy.copy(self)
        view.cache, view.copies_only = Path(folder), True
        return view

    def __reduce__(self):
        """A store sent to another process (a worker process of the same
        run) is the same store opened there, with the same cache folder,
        in the same session: a view of copies_in stays one."""
        return _reopened, (self._opening(), str(self.cache), self.copies_only, self.session, self.start)

    def _opening(self):
        """(a class, its arguments) that open this store again."""
        raise NotImplementedError

    def _found(self, file: File):
        key = self._stored(file.key)
        if key is None:
            raise MissingFile(f"{file.key} is not in the store")
        return key

    # One file of a source's tar

    def read_file(self, placed: Placed) -> bytes:
        """One file of a source, checked against its sha256: from the copy
        fetch_files made, when there is one. MissingFile when its tar is
        not in the store."""
        if placed.text is not None:
            return placed.text
        if (local := self._held(placed)) is not None:
            return local.read_bytes()
        key, kind = self._archive(placed)
        with self._reading(placed.archive, key):
            data = self._range(key, placed.start, placed.length, kind)
        if len(data) != placed.length or hashlib.sha256(data).hexdigest() != placed.sha256:
            raise CorruptFile(f"{placed.archive}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")
        return data

    def _held(self, placed: Placed) -> Path | None:
        """The local copy of one file of a source that a view of copies_in
        holds, None without one."""
        local = self.cache / placed.sha256[:2] / placed.sha256
        return local if self.copies_only and local.is_file() else None

    def fetch_files(self, places: list[Placed]):
        """In a view of copies_in, a local copy of each of these files of
        sources it does not hold, each checked against its sha256, for
        read_file to read: only the frames that hold them are fetched, and
        the frames next to each other go in one request, up to REQUESTS
        requests at once. A frame is fetched once, however many files it
        holds, or once more where two runs of frames meet (_runs). Each
        request's frames are let go once its files are copied, so memory
        holds REQUESTS runs at most, however many frames the files touch."""
        places = [placed for placed in places if self._held(placed) is None]
        for placed in places:
            if placed.text is not None:
                self._copy_fetched(placed, placed.text)
        places = [placed for placed in places if placed.text is None]
        for archive, placed in itertools.groupby(sorted(places, key=lambda p: p.archive), key=lambda p: p.archive):
            placed = list(placed)
            key, kind = self._archive(placed[0])
            with self._reading(archive, key):
                self._fetch_frames(key, kind, placed)

    def _copy_fetched(self, file: Placed, data: bytes):
        """Keep one fetched file's bytes as its local copy, once they match
        its sha256."""
        if len(data) != file.length or hashlib.sha256(data).hexdigest() != file.sha256:
            raise CorruptFile(f"{file.archive}: the {file.length} bytes at {file.start} do not match "
                                f"their sha256 {file.sha256}")
        local = self.cache / file.sha256[:2] / file.sha256
        local.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=local.parent, prefix=".partial-")
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        _replace_file(tmp, local)

    def _fetch_frames(self, key, kind, places):
        """Copy these files out of a source's tar (fetch_files), one run of
        frames at a time on each of REQUESTS threads."""
        table = self._table(key, kind)
        frameless = [file for file in places if not table.frames(file.start, file.length)]
        for file in frameless:
            self._copy_fetched(file, b"")

        def copied(run):
            frames, files = run
            at = table.stored[frames.start]
            data = kind.read(key, at, table.stored[frames.stop] - at)[0]
            held = b"".join(self._frame(key, table, frame, data[table.stored[frame] - at:table.stored[frame + 1] - at])
                            for frame in frames)
            del data
            for file in files:
                start = file.start - table.original[frames.start]
                self._copy_fetched(file, held[start:start + file.length])

        with ThreadPoolExecutor(REQUESTS) as pool:
            list(pool.map(copied, _runs(table, [file for file in places if file not in frameless])))

    def file_path(self, placed: Placed) -> Path:
        """A local file holding one file of a source, checked against
        its sha256: a copy in the cache folder, where path() keeps a file's,
        read part by part and kept until release(placed.sha256)."""
        local = self.cache / placed.sha256[:2] / placed.sha256
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=local.parent, prefix=".partial-")
            try:
                with os.fdopen(fd, "wb") as out:
                    for data in self.file_parts(placed):
                        out.write(data)
                _replace_file(tmp, local)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return local

    def file_parts(self, placed: Placed, size=DOWNLOAD):
        """One file of a source, in parts of at most size bytes, read one
        at a time, so a large file is never in memory whole: from the copy a
        view of copies_in holds, when there is one; CorruptFile, at the
        latest after the last part, when the bytes do not match its sha256
        (the parts given are then not that file)."""
        if placed.text is not None:
            yield placed.text
            return
        if (local := self._held(placed)) is not None:
            with open(local, "rb") as copy:
                while data := copy.read(size):
                    yield data
            return
        key, kind = self._archive(placed)
        digest, done = hashlib.sha256(), 0
        with self._reading(placed.archive, key):
            while done < placed.length:
                data = self._range(key, placed.start + done, min(size, placed.length - done), kind)
                if not data:
                    break  # the tar ends first
                digest.update(data)
                done += len(data)
                yield data
        if done != placed.length or digest.hexdigest() != placed.sha256:
            raise CorruptFile(f"{placed.archive}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")

    def _archive(self, placed):
        """(the key to read a source's tar under, the kind to read it from:
        its local copy's when there is one, of that very version)."""
        copied = _copy_key(placed.archive, placed.version)
        return (copied, self._copies) if self._copies.on_disk(copied) is not None else (placed.archive, self.kind)

    def keep_archive(self, key, path, version, replaced=None, local=False):
        """Move a source's tar, the local file path of sha256 version, into
        the store under key, on the disk before any record names it. local:
        a store whose files are not on this machine keeps a copy in its
        cache folder, removed again when the store refused the file. replaced:
        the version it replaces, whose copy goes."""
        copy = self.cache.joinpath(*_copy_key(key, version).split("/")) if local and not self.local else None
        if copy is not None:
            atomic_copy(path, copy)
        try:
            self.kind.write(key, path, move=True, durable=True)
        except BaseException:
            if copy is not None:
                copy.unlink(missing_ok=True)
            raise
        self.forget(key)
        if replaced is not None and replaced != version:
            self._copies.delete(_copy_key(key, replaced))

    # Compressed files

    def compressing(self):
        """(executor, sizing) for wpt_store.blocks.Writer, made once per
        store: as many frames at once as wpt_utils.parallel allows."""
        with self._compressing_lock:
            if self._compressing is None:
                self._compressing = blocks.parallel()
            return self._compressing

    def keep_smaller(self, key, source, durable=False):
        """Store source (bytes, or a local file, left as it is) under
        key + PACKED, compressed, when that is smaller, else under key as
        it is, replacing either. durable: as the kind's write."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            packed = blocks.compressed(source, *(self.compressing() if len(source) > blocks.FRAME else (None, None)))
            if len(packed) < len(source):
                self._write_one(key, PACKED, packed, durable=durable)
            else:
                self._write_one(key, "", bytes(source), durable=durable)
            return
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache, prefix=".partial-", delete=False) as out:
            try:
                writer = blocks.Writer(out, *self.compressing())
                with open(source, "rb") as file:
                    while block := file.read(CHUNK):
                        writer.write(block)
                original, stored = writer.close()
                out.close()
                if stored < original:
                    self._write_one(key, PACKED, Path(out.name), move=True, durable=durable)
                else:
                    self._write_one(key, "", Path(source), durable=durable)
            finally:
                Path(out.name).unlink(missing_ok=True)

    def _write_one(self, key, ending, source, **how):
        """Write key + ending, then remove the other of a file's two names,
        so a file replaced is never read under the one it left."""
        self.kind.write(key + ending, source, **how)
        self.kind.delete(key + ("" if ending else PACKED))
        self.forget(key)

    def _stored(self, key, fresh=False):
        """The key under which a file is stored, compressed or not, or None.
        A key found is kept in memory and given again, unless fresh."""
        found = None if fresh else self._keys.get(key)
        if found is not None:
            return found
        for stored in (key + PACKED, key):
            try:
                self.kind.read(stored, 0, 0)
                self._keys.put(key, stored)
                return stored
            except Absent:
                pass
        return None

    @contextlib.contextmanager
    def _reading(self, what, key):
        """A file found earlier and gone since (deleted by another process)
        is a MissingFile, and forgotten. what: what the failure names."""
        try:
            yield
        except Absent:
            self.forget(key.removesuffix(PACKED))
            raise MissingFile(f"{what} is not in the store") from None

    def _table(self, key, kind=None) -> blocks.SeekTable:
        kind = kind or self.kind
        table = self._tables.get(key)
        if table is None:
            end = kind.read(key, -TAIL)[0]
            try:
                table = blocks.SeekTable.parse(end)
                if isinstance(table, int):
                    table = blocks.SeekTable.parse(kind.read(key, -table)[0])
            except blocks.BadTable as error:
                raise CorruptFile(f"{key}: {error}") from None
            if isinstance(table, int):
                raise CorruptFile(f"{key}: shorter than its seek table")
            self._tables.put(key, table)
        return table

    def _range(self, key, start, length, kind=None):
        """length original bytes of a stored file from start (fewer when it
        ends first): read in place when it is not compressed, else from
        the frames that hold them, each decompressed once and kept a while;
        the frames not kept are read in one request, with up to READ_AHEAD
        frames from the first of them on.
        kind: where to read it (default: the store's own kind)."""
        kind = kind or self.kind
        if not key.endswith(PACKED):
            return kind.read(key, start, length)[0] if length else b""
        table = self._table(key, kind)
        frames = table.frames(start, length)
        if not frames:
            return b""
        found = {i: self._frames.get((key, i)) for i in frames}
        wanted = [i for i in frames if found[i] is None]
        if not wanted and kind.ahead:
            self._ahead.reached(key, table, frames[-1], lambda at, count: kind.read(key, at, count)[0])
        if wanted:
            first = wanted[0]
            if kind.ahead:  # the blocks after those needed are requested ahead, not with this read
                first, last = self._ahead.cover(table, first, wanted[-1])
                data = self._ahead.fetch(key, table, first, last, lambda at, count: kind.read(key, at, count)[0])
            else:
                last = max(wanted[-1], min(first + READ_AHEAD, len(table.sizes)) - 1)
                data = kind.read(key, table.stored[first], table.stored[last + 1] - table.stored[first])[0]
            for i in range(first, last + 1):
                if found.get(i) is None:
                    frame = self._frames.get((key, i))  # another thread may have kept it meanwhile
                    if frame is None:
                        at = table.stored[i] - table.stored[first]
                        frame = self._frame(key, table, i, data[at:at + table.sizes[i][0]])
                        self._frames.put((key, i), frame)
                    if i in found:
                        found[i] = frame
        joined = b"".join(found[i] for i in frames)
        at = start - table.original[frames[0]]
        return joined[at:at + length]

    def _frame(self, key, table, i, data):
        try:
            frame = zstd.decompress(data)
        except zstd.ZstdError as error:
            raise CorruptFile(f"{key}: frame {i} is not zstd: {error}") from None
        if len(frame) != table.sizes[i][1]:
            raise CorruptFile(f"{key}: frame {i} holds {len(frame)} bytes, not {table.sizes[i][1]}")
        return frame

    def _decompressed(self, key, data):
        try:
            return zstd.decompress(data)
        except zstd.ZstdError as error:
            raise CorruptFile(f"{key}: not zstd: {error}") from None

    def _copy(self, key, target: Path):
        """A whole stored file's original bytes into the local file target,
        several parts fetched at once as wpt_utils.parallel allows."""
        if key.endswith(PACKED):
            table = self._table(key)
            parts, group, size = [], [], 0
            for i, (stored, _) in enumerate(table.sizes):  # frames grouped into requests of about DOWNLOAD bytes
                group.append(i)
                size += stored
                if size >= DOWNLOAD:
                    parts.append(group)
                    group, size = [], 0
            if group:
                parts.append(group)

            def fetch(group):
                first, last = group[0], group[-1]
                data = self.kind.read(key, table.stored[first], table.stored[last + 1] - table.stored[first])[0]
                return b"".join(self._frame(key, table, i, data[table.stored[i] - table.stored[first]:
                                                                 table.stored[i + 1] - table.stored[first]])
                                for i in group)
        else:
            total = self.kind.list(key).get(key)
            if total is None:
                raise MissingFile(f"{key} is not in the store")
            parts = [(at, min(DOWNLOAD, total - at)) for at in range(0, total, DOWNLOAD)]

            def fetch(part):
                return self.kind.read(key, *part)[0]

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
        try:
            with os.fdopen(fd, "wb") as out:
                if len(parts) > 1:
                    sizing = Sizing("downloads")
                    with ThreadPoolExecutor(sizing.limit) as executor:
                        for data in in_order(fetch, parts, executor, sizing):
                            out.write(data)
                else:
                    for part in parts:
                        out.write(fetch(part))
            _replace_file(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def forget(self, key):
        """Drop what is kept in memory about a file that changed or went."""
        self._keys.drop(lambda kept: kept == key)
        for stored in (key, key + PACKED):
            self._tables.drop(lambda kept: kept == stored)
            self._frames.drop(lambda kept: kept[0] == stored)
            self._ahead.forget(stored)

    # Record folders

    def write_record(self, folder: str, name: str, data: bytes):
        """Write a record file whole, replacing any earlier one; a crash
        meanwhile leaves the earlier one or nothing, never a part. In the
        store when it returns. MixedRecord for a file added to line by line."""
        _run_prefix(folder, name)
        self._claim(folder, name, WHOLE)
        self._raw_write_record(folder, name, data)

    def append_record(self, folder: str, name: str, data: bytes):
        """Add lines at the end of a record file, creating it, held back
        until flush_records(). Several processes may add to one file;
        reading returns every addition in the order they were made, and a
        line another writer left cut (it ended without its newline) is
        ended first, so the next writer's lines start on their own.
        MixedRecord for a file written whole."""
        _run_prefix(folder, name)
        self._claim(folder, name, APPENDED)
        self._raw_append_record(folder, name, data)

    def flush_records(self):
        """Send the lines append_record holds back, one part per file, on
        the disk for a local store when it returns. Raises what the folder
        or the bucket raised: the lines not sent are kept, to send again."""
        self._raw_flush_records()

    def end_records(self, folder: str):
        """The record folder's run ended: compress each plain file a local
        store added lines to into one part, and remove the plain file. An S3
        store has only parts, so nothing. A file read back is the same
        before and after; lines added to a file again later go to a new plain
        file, read after the parts. A file that grew while it was compressed
        is left as it was."""
        prefix = _run_prefix(folder)
        for key in self.kind.list(prefix, deep=False):
            if not key.endswith(OPEN):
                continue
            try:
                data, _ = self.kind.read(key)
            except Absent:
                continue  # ended by another process
            name = key.rsplit("/", 1)[1].removesuffix(OPEN)
            moment, number = self._ticks.next()
            part = f"{prefix}{name}/{moment:020d}-{self._writer}-{number:010d}{PACKED}"
            if data:
                self.kind.write(part, blocks.compressed(b"%d %d %d\n" % (moment, number, len(data)) + data),
                                durable=True)
            if self.kind.list(key).get(key) == len(data):
                self.kind.delete(key)
            elif data:
                self.kind.delete(part)  # lines came meanwhile: the plain file stays, whole
            self._record_started.discard((folder, name))

    def _claim(self, folder, name, kind):
        """Refuse to mix the two kinds of record file. The kind is read from
        the store the first time this store touches a file, so it holds for
        files that another process made."""
        known = self._record_kinds.get((folder, name))
        if known is None:
            known = self._raw_record_kind(folder, name) or kind
        if known != kind:
            raise MixedRecord(f"record file {folder}/{name} is {known}; it cannot also be {kind}")
        self._record_kinds[(folder, name)] = known

    def read_record(self, folder: str, name: str) -> bytes:
        _run_prefix(folder, name)
        data = self._raw_read_record(folder, name)
        if data is None:
            raise MissingFile(f"no record file {folder}/{name} in the store")
        return data

    def record_reader(self, folder: str, name: str) -> "RecordReader":
        """A reader of the lines added to a record file, for a run that is
        still going: each read() gives only what is new since the last."""
        _run_prefix(folder, name)
        return RecordReader(self, folder, name)

    def record_files(self, folder: str) -> list[str]:
        """The names of a run's files, sorted; [] for no such run."""
        return sorted(self._record_files_under(_run_prefix(folder)))

    def record_folders(self) -> list[str]:
        """Every run, <id>/<run>, of this view's session and the one it
        started from, or of every session for a store with no session,
        oldest first (a run's name starts with its UTC time)."""
        sessions = [self.session, *([self.start] if self.start else [])] if self.session else self.sessions()
        return sorted((f"{session}/{name}" for session in sessions
                       for name, is_folder in self._record_files_under(f"{session}/{RUNS}/").items() if is_folder),
                      key=lambda folder: folder.partition("/")[2])

    # A record file written whole is <id>/runs/<run>/<name>.zst. A file added
    # to line by line is a folder of parts, and, in a local store while its
    # record runs, also one plain file <id>/runs/<run>/<name>.open that its
    # lines are appended to, read after the parts (end_records makes it a
    # part). The parts are <id>/runs/<run>/<name>/<time>-<writer>-<number>.zst:
    # each part holds the additions one flush_records() sent, each one framed
    # as "<time> <number> <length>\n" and its bytes. <time> is when it was
    # added (nanoseconds since 1970, never twice the same for one writer),
    # <writer> is this store object's random id, and <number> counts its
    # additions; a part is named after its first addition. Reading sorts
    # every addition of every part by (time, writer, number): the order they
    # were made, across processes, whenever each was sent. Every one of these
    # files is compressed (records always shrink), and written durably: on
    # the disk when the write returns.

    def _record_files_under(self, prefix) -> dict:
        """{name: whether it is a folder of parts} right under a key
        prefix ending in '/', each name without .zst."""
        names = {}
        for key in self.kind.list(prefix, deep=False):
            name = key[len(prefix):].removesuffix("/")
            names[name.removesuffix(PACKED) if name.endswith(PACKED) else name.removesuffix(OPEN)] = key.endswith("/")
        return names

    def _raw_record_kind(self, folder, name):
        prefix = _run_prefix(folder)
        if self.kind.list(f"{prefix}{name}/") or self.kind.list(f"{prefix}{name}{OPEN}"):
            return APPENDED
        return WHOLE if self._record_file(folder, name) is not None else None

    def _raw_write_record(self, folder, name, data):
        self.kind.write(f"{_run_prefix(folder)}{name}{PACKED}", blocks.compressed(data), durable=True)

    def _raw_append_record(self, folder, name, data):
        moment, number = self._ticks.next()
        with self._pending_lock:
            self._pending.setdefault((folder, name), []).append((moment, number, data))

    def _raw_flush_records(self):
        # A group leaves _pending only once the store took its part, so a send
        # that failed or was interrupted (Ctrl+C) loses no line. When the
        # interruption came after the store took it, the next flush sends the
        # group again under the same key, since its first addition is the
        # same: the store replaces that part with the group and the lines added
        # since, and no line is ever stored twice.
        with self._pending_lock:
            for (folder, name), additions in list(self._pending.items()):
                first_time, first_number, _ = additions[0]
                body = b"".join(b"%d %d %d\n" % (moment, number, len(data)) + data for moment, number, data in additions)
                self.kind.write(f"{_run_prefix(folder)}{name}/{first_time:020d}-{self._writer}-{first_number:010d}{PACKED}",
                                blocks.compressed(body), durable=True)
                del self._pending[(folder, name)]

    def _record_file(self, folder, name):
        key = f"{_run_prefix(folder)}{name}{PACKED}"
        try:
            return self._decompressed(key, self.kind.read(key)[0])
        except Absent:
            return None

    def _raw_read_record(self, folder, name):
        prefix = _run_prefix(folder)
        parts = self.kind.list(f"{prefix}{name}/")
        try:
            plain = self.kind.read(f"{prefix}{name}{OPEN}")[0]
        except Absent:
            plain = None
        if not parts and plain is None:
            return self._record_file(folder, name)
        joined = self._joined_parts(parts)
        if plain:
            if joined and not joined[-1].endswith(b"\n"):
                joined.append(b"\n")
            joined.append(plain)
        return b"".join(joined)

    def _joined_parts(self, parts):
        """The additions of these parts of a record file, in the order they
        were made, as a list of bytes."""
        additions = []
        for key in parts:
            writer = key.rsplit("/", 1)[1].split("-")[1]
            try:
                body = self.kind.read(key)[0]
            except Absent:
                continue  # removed since it was listed
            body = self._decompressed(key, body)
            at = 0
            try:
                while at < len(body):
                    end = body.index(b"\n", at)
                    moment, number, length = (int(field) for field in body[at:end].split(b" "))
                    additions.append((moment, writer, number, body[end + 1:end + 1 + length]))
                    at = end + 1 + length
            except ValueError:
                raise CorruptFile(f"record part {key} is not in the store's format") from None
        joined, last_writer = [], None
        for _, writer, _, data in sorted(additions):
            # a line another writer left cut (it ended without its newline) is ended first, so this writer's
            # own lines stay whole and readable
            if joined and writer != last_writer and not joined[-1].endswith(b"\n"):
                joined.append(b"\n")
            joined.append(data)
            last_writer = writer if data else last_writer
        return joined


class RecordReader:
    """The new lines of one record file (Store.record_reader): each read()
    fetches only the parts it has not read yet and, for a local store's run
    that is still going, the end of its plain file. When the run ends, that
    file becomes a part that repeats what was read of it, which is left out.
    A read that raises moves nothing: the next one fetches the same lines."""

    def __init__(self, store, folder, name):
        self.store, self.seen, self.plain = store, set(), 0
        prefix = _run_prefix(folder)
        self.prefix, self.open_key = f"{prefix}{name}/", f"{prefix}{name}{OPEN}"

    def read(self) -> bytes:
        kind = self.store.kind
        new = [key for key in kind.list(self.prefix) if key not in self.seen]
        data = b"".join(self.store._joined_parts(new))
        try:
            more = kind.read(self.open_key, self.plain)[0]
        except Absent:
            more = b""
            if new:
                data, self.plain = data[self.plain:], 0
        self.seen.update(new)
        self.plain += len(more)
        return data + more


class LocalStore(Store):
    """A store in one local folder (wpt_store.kinds.Folder), laid out as
    the module's docstring says; .local/cache/ and .local/scratch/ beside
    it are this machine's, and no part of the store."""

    local = True

    def __init__(self, root):
        self.root = Path(root)
        super().__init__(Folder(self.root), self.root / ".local" / "cache")
        self.scratch = self.root / ".local" / "scratch"
        self._unsynced = set()  # plain record files added to since the last flush_records

    def __repr__(self):
        return f"LocalStore({str(self.root)!r})"

    def _raw_append_record(self, folder, name, data):
        """Add the lines at once to the plain file of the record file: one
        write, so lines of several processes never mix, and none is lost
        when this process dies. A line an earlier writer left cut (the file
        does not end with a newline) is ended first, once per writer."""
        path = self.root.joinpath(*_run_prefix(folder).split("/")) / f"{name}{OPEN}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._pending_lock:
            first = (folder, name) not in self._record_started
            self._record_started.add((folder, name))
            self._unsynced.add(path)
        if first:
            lock = lock_file(path.with_name(f".{path.name}.lock"))  # two first writers must not both end a cut line
            try:
                data = (b"\n" if _cut(path) else b"") + data
                _append(path, data)
            finally:
                unlock_file(lock)
        else:
            _append(path, data)

    def _raw_flush_records(self):
        with self._pending_lock:
            paths = list(self._unsynced)
        for path in paths:
            try:
                sync_file(path)
            except FileNotFoundError:
                pass  # compressed into a part meanwhile, which is on the disk
            sync_folder(path.parent)
        with self._pending_lock:
            self._unsynced.difference_update(paths)

    def _opening(self):
        return LocalStore, (str(self.root),)


S3_SETTINGS = {"access_key_id": "WPT_STORE_ACCESS_KEY_ID", "secret_access_key": "WPT_STORE_SECRET_ACCESS_KEY",
               "session_token": "WPT_STORE_SESSION_TOKEN", "endpoint_url": "WPT_STORE_ENDPOINT_URL",
               "region": "WPT_STORE_REGION"}
S3_REQUIRED = ("access_key_id", "secret_access_key")
S3_DEFAULT_REGION = "us-east-1"


def s3_settings(environ=None, env_file=None) -> dict:
    """{name: value} of the S3 store's own settings (S3_SETTINGS) set in the
    environment, else in env_file (None: no file). StoreSettingError when a
    key of S3_REQUIRED is not set. No AWS_* setting is read."""
    environ = os.environ if environ is None else environ
    from_file = read_env_file(env_file) if env_file is not None else {}
    found = {name: value for name, variable in S3_SETTINGS.items()
             if (value := environ.get(variable) or from_file.get(variable))}
    missing = [S3_SETTINGS[name] for name in S3_REQUIRED if name not in found]
    if missing:
        raise StoreSettingError(f"an s3 store needs {' and '.join(missing)} (in the environment or in .env); it reads "
                                f"no AWS_* setting")
    return found


class S3Store(Store):
    """The same layout as a LocalStore, as keys under s3://bucket/prefix
    (wpt_store.kinds.Bucket); its cache and scratch folders are local. Its
    credentials, endpoint and region are `settings` (s3_settings; read from
    the environment when not given)."""

    def __init__(self, bucket, prefix, cache_dir=None, client=None, settings=None):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.settings = (s3_settings() if settings is None else settings) if client is None else settings
        if client is None:
            client = boto3.Session(botocore_session=_own_session()).client(
                "s3", aws_access_key_id=self.settings["access_key_id"],
                aws_secret_access_key=self.settings["secret_access_key"],
                aws_session_token=self.settings.get("session_token"),
                endpoint_url=self.settings.get("endpoint_url"),
                region_name=self.settings.get("region", S3_DEFAULT_REGION),
                # as many open connections as the upload parts at once, as a whole file's copy fetches, and as the reads
                # in flight at once (wpt_store.ahead)
                config=Config(ignore_configured_endpoint_urls=True,
                              max_pool_connections=UPLOADS + Sizing("downloads").limit + REQUESTS))
        self.client = client
        cache = Path(cache_dir) if cache_dir else _default_cache() / bucket / (self.prefix or "_")
        super().__init__(Bucket(self.client, bucket, self.prefix), cache)
        self.scratch = self.cache / "scratch"
        self._own_cache = str(cache)

    def __repr__(self):
        return f"S3Store(s3://{self.bucket}/{self.prefix})"

    def _opening(self):
        return S3Store, (self.bucket, self.prefix, self._own_cache, None, self.settings)  # its client, from the same settings


def _own_session():
    """A new botocore session that reads no AWS_* environment variable and
    no AWS configuration or credentials file (a client is built from its
    arguments alone). One per client: boto3 adds its methods to a
    session's client class once."""
    import botocore.session
    from botocore.configprovider import BOTOCORE_DEFAUT_SESSION_VARIABLES

    files = ("config_file", "credentials_file")
    return botocore.session.Session(session_vars={
        name: (config, None, os.devnull if name in files else default, convert)
        for name, (config, _variable, default, convert) in BOTOCORE_DEFAUT_SESSION_VARIABLES.items()})


def _cut(path: Path) -> bool:
    """Whether a file ends with a line that has no newline (an empty or
    missing file does not)."""
    try:
        with open(path, "rb") as file:
            if not file.seek(0, os.SEEK_END):
                return False
            file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _append(path: Path, data: bytes):
    """data at the end of a file, creating it, in as few writes as the system
    allows (one, on any file system that writes it whole)."""
    with open(path, "ab", buffering=0) as file:
        view = memoryview(data)
        while view:
            view = view[file.write(view):]


def _reopened(opening, cache, copies_only, session, start) -> Store:
    kind, arguments = opening
    store = kind(*arguments)
    store.cache, store.copies_only, store.session, store.start = Path(cache), copies_only, session, start
    return store


def local_folder(url: str) -> Path | None:
    """The folder a file:// WPT_STORE value names; None for any other
    value. StoreSettingError for a file:// value that is not local."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in ("", "localhost"):
        raise StoreSettingError(f"file URLs must name a local path: {url!r}")
    path = unquote(parsed.path)
    if os.name == "nt" and re.match(r"/[A-Za-z]:", path):
        path = path[1:]  # file:///C:/x -> C:/x
    return Path(path)


def open_store(url: str, environ=None, env_file=None) -> Store:
    """A store from a WPT_STORE value; an s3 store reads its own settings
    (s3_settings) from environ (default: the environment) and env_file."""
    folder = local_folder(url)
    if folder is not None:
        return LocalStore(folder)
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise StoreSettingError(f"s3 URLs need a bucket: {url!r}")
        return S3Store(parsed.netloc, parsed.path, settings=s3_settings(environ, env_file))
    raise StoreSettingError(f"WPT_STORE must start with file:// or s3://, got {url!r}")


def store_from_settings(env_file=".env", environ=None) -> Store:
    """The store named by WPT_STORE in the environment, else in env_file (an s3 store's keys likewise)."""
    environ = os.environ if environ is None else environ
    url = environ.get("WPT_STORE") or read_env_file(env_file).get("WPT_STORE")
    if not url:
        raise StoreSettingError("WPT_STORE is not set: put WPT_STORE=file:///path or s3://bucket/prefix in .env")
    return open_store(url, environ, env_file)


def _default_cache():
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return Path(base or Path.home() / ".cache") / "wpt-store"


def _copy_key(key, version) -> str:
    """The key, in the cache folder, of the local copy of the version of a
    source's tar: it ends as a compressed file's key does."""
    return f"{key}.{version}{PACKED}"


def _runs(table, places) -> list[tuple[range, list[Placed]]]:
    """(frames next to each other, the files they hold) that together hold
    these files of a source's tar, each at least one frame long. A file
    is never split between two runs. A run past RUN_BYTES of original bytes
    takes no more files, so when files lie one against the other, the frame
    two of them share belongs to both runs."""
    runs = []
    for file in sorted(places, key=lambda place: place.start):
        held = table.frames(file.start, file.length)
        frames, files = runs[-1] if runs else (range(0), [])
        size = table.original[frames.stop] - table.original[frames.start]
        if frames and held.start <= frames.stop and size < RUN_BYTES:
            files.append(file)
            runs[-1] = range(frames.start, max(frames.stop, held.stop)), files
        else:
            runs.append((held, [file]))
    return runs


def _replace_file(tmp, target: Path):
    """os.replace for a file named by its hash. On Windows, replacing a file
    another process has open is refused; the name is the content's hash, so a
    target that exists already holds the same bytes."""
    try:
        os.replace(tmp, target)
    except PermissionError:
        if not target.is_file():
            raise
        Path(tmp).unlink(missing_ok=True)
