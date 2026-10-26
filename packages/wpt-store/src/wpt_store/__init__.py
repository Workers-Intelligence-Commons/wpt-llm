"""Content-addressed storage for everything WPT downloads or builds.

WPT hosts no data: each machine points the WPT_STORE setting at its own store,

    WPT_STORE=file:///home/me/wpt-store      a local folder
    WPT_STORE=s3://bucket/prefix             an S3-compatible bucket

The prefix picks the store's kind (wpt_store.kinds: Folder, Bucket), which
supplies five operations on files; everything here is built on them, alike
for both kinds.

An object is stored under the sha256 of its bytes and checked against it on
every read. A ref is a small named pointer to an object (for example a build
fingerprint to its result), so "already done" is a lookup.

Every file a store writes is compressed while it is written, with zstd in
frames of about 1 MB that a range read takes one by one (wpt_store.blocks),
and kept so only when that saves space: objects/<first 2 hex>/<sha256>.zst
then, objects/<first 2 hex>/<sha256> as it is otherwise (Store.keep_smaller
for objects, Store.keep_compressed for bundles). Which of the two names a
file is stored under is found once and kept in memory. Large files are compressed several frames
at once, and sent to S3 several parts at once, as wpt_utils.parallel allows.
Refs, too small to shrink, are written as they are; record files are
compressed, except the one a local store adds lines to while its record
runs.

A session (wpt_store.sessions) stores the files one downloading command
fetches in its own bundles, and its lines in the store database
(wpt_store.database), which Store.database() reads. An S3 store keeps a
local copy of each bundle its sessions stored, in its cache folder,
cache/bundles/<session>/<number>[.zst], and reads that copy first, unless
the session streams (Store.session): the copy is then gone once the
bucket holds the bundle.

A record folder holds files written by name, not by content: a command's
own account of one run (see wpt_telemetry), records/<folder>/, built on the
same five operations as everything else, alike for both kinds. A file is
written whole, or added to line by line, never both. A file written whole
is in the store, on the disk for a local store, when the call returns.
An S3 store holds added lines back until flush_records() sends them
together, as one part of the file, since each request costs time (and
money on S3). A local store appends each line at once to one plain file,
records/<folder>/<name>.open, so a crash loses none that was added (flush_records()
puts them on the disk), and end_records(folder), when the record ends,
compresses each such file into one part. Lines added by several processes
read back in the order they were added.

For S3, the credentials and a non-AWS endpoint come from the store's own
settings, in the environment or in .env: WPT_STORE_ACCESS_KEY_ID and
WPT_STORE_SECRET_ACCESS_KEY (both needed), WPT_STORE_SESSION_TOKEN,
WPT_STORE_ENDPOINT_URL and WPT_STORE_REGION (default us-east-1). The
store reads no AWS_* setting (no AWS_ENDPOINT_URL, no AWS_PROFILE) and no
AWS configuration or credentials file, so the keys a cloud provider's
other tools use never reach it. A store keeps local copies of objects in its cache folder: an S3 store every object
read as a file, after its check; a local store the compressed ones read as
a file, which it cannot give in place. release() removes a copy;
copies_in(folder) is the same store making in folder only the copies its
work asks for as files, for work that keeps none. A store sent to another
process (pickled) opens the same store there.

Failures: StoreSettingError (no or bad WPT_STORE), MissingObject (no such
object, ref target, bundle or record file), CorruptObject (stored bytes do
not match their sha256, a compressed file or the store database is not in
its format, or a record part is not in the store's format; the copy is not
used), MixedRecord (a whole write to a file added to line by line, or the
reverse; nothing is written). Writing raises what the folder or the bucket
raised (OSError, botocore's errors): nothing is known to be stored.
"""

import collections
import contextlib
import copy
import hashlib
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

from . import blocks, database
from .errors import CorruptObject, MissingObject, MixedRecord, StoreError, StoreSettingError
from .kinds import Absent, Bucket, Folder, atomic_copy, sync_file, sync_folder
from .names import NAME, NAME_CHARACTERS, as_record_name, check_record_name, check_ref
from .sessions import Placed, Session, file_list_bytes, parse_file_list, remove_unowned_partials

SHA256 = re.compile(r"[0-9a-f]{64}")
CHUNK = 1 << 20
PACKED = ".zst"  # the end of a key whose file is compressed
OPEN = ".open"  # the end of the key of the plain file a local store adds a record file's lines to while it runs
TAIL = 64 << 10  # bytes read at a compressed file's end for its seek table: the whole table up to 8,000 frames
DOWNLOAD = 64 << 20  # bytes of one file fetched in one request, when a whole file is copied from the store
FRAMES_KEPT = 64  # decompressed frames kept in memory for range reads (about 64 MB)
TABLES_KEPT = 1024  # seek tables kept in memory
KEYS_KEPT = 1 << 16  # keys files were found under (compressed or not) kept in memory


WHOLE, APPENDED = "written whole", "added to line by line"  # the two kinds of record file


def _check_sha(sha):
    if not SHA256.fullmatch(sha):
        raise ValueError(f"not a sha256: {sha!r}")


def _check_record(folder, name=None):
    for part in (folder, name) if name is not None else (folder,):
        check_record_name(part)


def _object_key(sha):
    return f"objects/{sha[:2]}/{sha}"


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


class Store:
    """Operations every store offers, built on its kind's five (see
    wpt_store.kinds), record folders included. Subclasses give the kind,
    `cache` (a local folder for copies of objects) and `scratch` (a local
    folder for files derived from objects, such as token files for
    training, which may be deleted at any time)."""

    scratch: Path
    local = False  # whether the kind's files are on this machine's disk already, so no copy of them is kept

    def __init__(self, kind, cache):
        self.kind, self.cache = kind, Path(cache)
        self.copies_only = False  # a view of copies_in: only path() and file_path() make a local copy
        self._copies = Folder(self.cache)  # the local copies of bundles, read through the same five operations
        self._record_kinds = {}  # (folder, name) -> WHOLE or APPENDED, as this store learned it
        self._tables, self._frames, self._keys = _Kept(TABLES_KEPT), _Kept(FRAMES_KEPT), _Kept(KEYS_KEPT)
        self._compressing, self._compressing_lock = None, threading.Lock()
        # record lines added and not yet sent: (folder, name) -> [(time, number, data)]; see _raw_append_record
        self._writer, self._pending, self._added, self._last = secrets.token_hex(4), {}, 0, 0
        self._pending_lock = threading.Lock()
        self._record_started = set()  # (folder, name) of the plain record files this store has added to

    # Objects

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
            self.keep_smaller(_object_key(sha), source)
        except PermissionError:
            # on Windows, replacing a file another process has open is refused; the
            # name is the content's hash, so an object stored meanwhile holds the same bytes
            if not self.has(sha):
                raise

    def has(self, sha: str) -> bool:
        """Asks the store each time, so an object deleted by another process
        is not taken as stored."""
        _check_sha(sha)
        return self._stored(_object_key(sha), fresh=True) is not None

    def size(self, sha: str) -> int:
        """The object's length in bytes, without reading or copying it (and
        so without checking it against its sha256)."""
        _check_sha(sha)
        key = self._found(sha)
        with self._reading(f"object {sha}", key):
            if key.endswith(PACKED):
                return self._table(key).size
            on_disk = self.kind.on_disk(key)
            total = on_disk.stat().st_size if on_disk is not None else self.kind.list(key).get(key)
        if total is None:
            self._forget(_object_key(sha))
            raise MissingObject(f"object {sha} is not in the store")
        return total

    def path(self, sha: str) -> Path:
        """A local file holding the object, checked against its sha256. A
        copy in the cache folder when the object is not a local file as it
        is (every object of an S3 store, a local store's compressed ones),
        and always in a view of copies_in: the copy stays until release()."""
        _check_sha(sha)
        key = self._found(sha)
        local = self.cache / sha[:2] / sha
        path = None if key.endswith(PACKED) or self.copies_only else self.kind.on_disk(key)
        if path is None:
            if not local.is_file():
                with self._reading(f"object {sha}", key):
                    self._copy(key, local)
            path = local
        actual = sha256_file(path)
        if actual != sha:
            if path == local:
                local.unlink(missing_ok=True)
            raise CorruptObject(f"object {sha} reads back as {actual}")
        return path

    def get_bytes(self, sha: str) -> bytes:
        """The object's bytes, checked against its sha256. An object that is
        not a local file is read through a copy in the cache folder (kept
        until release()), except in a view of copies_in, which reads it
        straight from the store and keeps no copy."""
        _check_sha(sha)
        key = self._found(sha)
        on_disk = self.kind.on_disk(key)
        if on_disk is None and self.copies_only and not (self.cache / sha[:2] / sha).is_file():
            with self._reading(f"object {sha}", key):
                data = self._range(key, 0, self.size(sha))
            if hashlib.sha256(data).hexdigest() != sha:
                raise CorruptObject(f"object {sha} reads back as {hashlib.sha256(data).hexdigest()}")
            return data
        if on_disk is None or (self.cache / sha[:2] / sha).is_file():
            return self.path(sha).read_bytes()
        try:
            data = on_disk.read_bytes()
        except FileNotFoundError:
            self._forget(key.removesuffix(PACKED))
            raise MissingObject(f"object {sha} is not in the store") from None
        if key.endswith(PACKED):
            data = self._decompressed(key, data)
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha:
            raise CorruptObject(f"object {sha} reads back as {actual}")
        return data

    def read_range(self, sha: str, start: int, length: int) -> bytes:
        """length bytes of an object from offset start, without fetching the
        rest of it. A part cannot be checked against the object's sha256:
        the caller checks it against a hash of its own."""
        _check_sha(sha)
        key = self._found(sha)
        local = self.cache / sha[:2] / sha
        if local.is_file():
            with open(local, "rb") as file:
                file.seek(start)
                data = file.read(length)
        else:
            with self._reading(f"object {sha}", key):
                data = self._range(key, start, length)
        if len(data) != length:
            raise CorruptObject(f"object {sha}: {length} bytes at {start} read back as {len(data)}")
        return data

    def release(self, sha: str):
        """Remove the local copy of an object (it stays in the store);
        nothing when there is none, as for a local store's object that is
        not compressed, whose copy is the object itself."""
        _check_sha(sha)
        (self.cache / sha[:2] / sha).unlink(missing_ok=True)

    def delete(self, sha: str):
        """Remove an object, and its local copy; nothing when it is absent.
        Objects are shared by content: only for one that nothing else can
        point at, such as a training checkpoint that a newer one replaced."""
        _check_sha(sha)
        self.release(sha)
        key = _object_key(sha)
        for stored in (key + PACKED, key):
            self.kind.delete(stored)
        self._forget(key)

    def copies_in(self, folder) -> "Store":
        """The same store, whose local copies of objects and of a session's
        files go into folder instead of its cache folder, for work that
        keeps none: removing folder afterwards removes every copy that work
        made. Only a file the work asks for as a file (path(), file_path())
        is copied, a local store's own files too, so the work never opens
        a file of the store itself; everything else is read from the store
        (get_bytes, read_range, read_file, file_parts). Copies already in
        the cache folder are not read. For objects and refs only, not
        record folders."""
        self.compressing()  # made before the copy, so every view shares this store's one
        view = copy.copy(self)
        view.cache, view.copies_only = Path(folder), True
        return view

    def __reduce__(self):
        """A store sent to another process (a worker process of the same
        run) is the same store opened there, with the same cache folder:
        a view of copies_in stays one."""
        return _reopened, (self._opening(), str(self.cache), self.copies_only)

    def _opening(self):
        """(a class, its arguments) that open this store again."""
        raise NotImplementedError

    def _found(self, sha):
        key = self._stored(_object_key(sha))
        if key is None:
            raise MissingObject(f"object {sha} is not in the store")
        return key

    # Refs

    def set_ref(self, kind: str, name: str, sha: str):
        check_ref(kind, name)
        _check_sha(sha)
        if not self.has(sha):
            raise MissingObject(f"ref {kind}/{name} would point at missing object {sha}")
        self.kind.write(f"refs/{kind}/{name}", sha.encode())

    def get_ref(self, kind: str, name: str) -> str | None:
        """The object a ref points at, or None when the ref does not exist."""
        check_ref(kind, name)
        try:
            sha = self.kind.read(f"refs/{kind}/{name}")[0].decode().strip()
        except Absent:
            return None
        if not SHA256.fullmatch(sha):
            raise CorruptObject(f"ref {kind}/{name} holds {sha!r}, not a sha256")
        return sha

    def delete_ref(self, kind: str, name: str):
        """Remove a ref (not its object); nothing when it is absent."""
        check_ref(kind, name)
        self.kind.delete(f"refs/{kind}/{name}")

    # Sessions, bundles and the store database

    def session(self, folder: str, command: str, commit: str, dirty: bool, stream=False) -> Session:
        """A new session (wpt_store.sessions), named by its run record's
        folder. stream: no local copy of its bundles stays (see the module's
        docstring)."""
        return Session(self, folder, command, commit, dirty, keep_local=not stream)

    def read_file(self, placed: Placed) -> bytes:
        """One file a session stored, checked against its sha256.
        MissingObject when its bundle is not in the store (or not sealed)."""
        key, kind = self._bundle(placed)
        with self._reading(f"bundle {placed.bundle}", key):
            data = self._range(key, placed.start, placed.length, kind)
        if len(data) != placed.length or hashlib.sha256(data).hexdigest() != placed.sha256:
            raise CorruptObject(f"bundle {placed.bundle}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")
        return data

    def file_path(self, placed: Placed) -> Path:
        """A local file holding one file a session stored, checked against
        its sha256: a copy in the cache folder, where path() keeps an
        object's, read part by part and kept until release(placed.sha256)."""
        local = self.cache / placed.sha256[:2] / placed.sha256
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=local.parent, prefix=".partial-")
            try:
                with os.fdopen(fd, "wb") as out:
                    for data in self.file_parts(placed):
                        out.write(data)
                _replace_object(tmp, local)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return local

    def file_parts(self, placed: Placed, size=DOWNLOAD):
        """One file a session stored, in parts of at most size bytes, read
        one at a time, so a large file is never in memory whole;
        CorruptObject, at the latest after the last part, when the bytes do
        not match its sha256 (the parts given are then not that file)."""
        key, kind = self._bundle(placed)
        digest, done = hashlib.sha256(), 0
        with self._reading(f"bundle {placed.bundle}", key):
            while done < placed.length:
                data = self._range(key, placed.start + done, min(size, placed.length - done), kind)
                if not data:
                    break  # the bundle ends first
                digest.update(data)
                done += len(data)
                yield data
        if done != placed.length or digest.hexdigest() != placed.sha256:
            raise CorruptObject(f"bundle {placed.bundle}: the {placed.length} bytes at {placed.start} "
                                f"do not match their sha256 {placed.sha256}")

    def _bundle(self, placed):
        """(the key a bundle is stored under, the kind to read it from: its
        local copy's when there is one)."""
        key = self._stored(f"bundles/{placed.bundle}")
        if key is None:
            raise MissingObject(f"bundle {placed.bundle} is not in the store")
        return key, self._copies if self._copies.on_disk(key) is not None else self.kind

    def database(self, threads=False):
        """The store database (wpt_store.database), as a sqlite3 connection
        to a copy in memory; threads: usable from any thread, one at a time."""
        return database.read(self.kind, threads)

    def has_bundle(self, bundle: str) -> bool:
        """Whether the store holds a sealed bundle (<session folder>/<number>)."""
        return self._stored(f"bundles/{bundle}", fresh=True) is not None

    def drop_bundle(self, bundle: str):
        """Remove a bundle (<session folder>/<number>) and its local copy;
        nothing when absent. Only for one that no file list names any more
        (wpt_store.sessions, end)."""
        key = f"bundles/{bundle}"
        for stored in (key + PACKED, key):
            self.kind.delete(stored)
            self._copies.delete(stored)
        self._forget(key)

    # Compressed files

    def compressing(self):
        """(executor, sizing) for wpt_store.blocks.Writer, made once per
        store: as many frames at once as wpt_utils.parallel allows."""
        with self._compressing_lock:
            if self._compressing is None:
                self._compressing = blocks.parallel()
            return self._compressing

    def keep_smaller(self, key, source):
        """Store source (bytes, or a local file, left as it is) under
        key + PACKED, compressed, when that is smaller, else under key as
        it is."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            packed = blocks.compressed(source, *(self.compressing() if len(source) > blocks.FRAME else (None, None)))
            if len(packed) < len(source):
                self.kind.write(key + PACKED, packed)
            else:
                self.kind.write(key, bytes(source))
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
                    self.kind.write(key + PACKED, Path(out.name), move=True)
                else:
                    self.kind.write(key, Path(source))
            finally:
                Path(out.name).unlink(missing_ok=True)

    def keep_compressed(self, key, path, original, stored, local=False):
        """Store a local file wpt_store.blocks.Writer compressed (original
        and stored: the sizes its close() gave) as keep_smaller would its
        original bytes. The file is gone afterwards, stored or not. local:
        a store whose files are not on this machine keeps a copy in its
        cache folder, under the same key, once the write succeeded."""
        try:
            if stored < original:
                self._write_kept(key + PACKED, path, local)
                return
            with tempfile.NamedTemporaryFile(dir=self.cache, prefix=".partial-", delete=False) as out:
                try:
                    with zstd.open(path) as packed:
                        while block := packed.read(CHUNK):
                            out.write(block)
                    out.close()
                    self._write_kept(key, Path(out.name), local)
                finally:
                    Path(out.name).unlink(missing_ok=True)
        finally:
            Path(path).unlink(missing_ok=True)

    def _write_kept(self, key, path, local):
        """Move a local file into the store under key; with local (and a
        store not on this machine), a copy of it into the cache folder too,
        removed again when the store refused the file."""
        copy = self.cache.joinpath(*key.split("/")) if local and not self.local else None
        if copy is not None:
            atomic_copy(path, copy)
        try:
            self.kind.write(key, path, move=True, durable=True)  # on the disk before the database names it
        except BaseException:
            if copy is not None:
                copy.unlink(missing_ok=True)
            raise

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
        is a MissingObject, and forgotten."""
        try:
            yield
        except Absent:
            self._forget(key.removesuffix(PACKED))
            raise MissingObject(f"{what} is not in the store") from None

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
                raise CorruptObject(f"{key}: {error}") from None
            if isinstance(table, int):
                raise CorruptObject(f"{key}: shorter than its seek table")
            self._tables.put(key, table)
        return table

    def _range(self, key, start, length, kind=None):
        """length original bytes of a stored file from start (fewer when it
        ends first): read in place when it is not compressed, else from
        the frames that hold them, each decompressed once and kept a while.
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
        if wanted:
            first, last = wanted[0], wanted[-1]
            data = kind.read(key, table.stored[first], table.stored[last + 1] - table.stored[first])[0]
            for i in range(first, last + 1):
                if found.get(i) is None:
                    at = table.stored[i] - table.stored[first]
                    found[i] = self._frame(key, table, i, data[at:at + table.sizes[i][0]])
                    self._frames.put((key, i), found[i])
        joined = b"".join(found[i] for i in frames)
        at = start - table.original[frames[0]]
        return joined[at:at + length]

    def _frame(self, key, table, i, data):
        try:
            frame = zstd.decompress(data)
        except zstd.ZstdError as error:
            raise CorruptObject(f"{key}: frame {i} is not zstd: {error}") from None
        if len(frame) != table.sizes[i][1]:
            raise CorruptObject(f"{key}: frame {i} holds {len(frame)} bytes, not {table.sizes[i][1]}")
        return frame

    def _decompressed(self, key, data):
        try:
            return zstd.decompress(data)
        except zstd.ZstdError as error:
            raise CorruptObject(f"{key}: not zstd: {error}") from None

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
                raise MissingObject(f"{key} is not in the store")
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
            _replace_object(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _forget(self, key):
        """Drop what is kept in memory about a file that changed or went."""
        self._keys.drop(lambda kept: kept == key)
        for stored in (key, key + PACKED):
            self._tables.drop(lambda kept: kept == stored)
            self._frames.drop(lambda kept: kept[0] == stored)

    # Record folders

    def write_record(self, folder: str, name: str, data: bytes):
        """Write a record file whole, replacing any earlier one; a crash
        meanwhile leaves the earlier one or nothing, never a part. In the
        store when it returns. MixedRecord for a file added to line by line."""
        _check_record(folder, name)
        self._claim(folder, name, WHOLE)
        self._raw_write_record(folder, name, data)

    def append_record(self, folder: str, name: str, data: bytes):
        """Add lines at the end of a record file, creating it, held back
        until flush_records(). Several processes may add to one file;
        reading returns every addition in the order they were made, and a
        line another writer left cut (it ended without its newline) is
        ended first, so the next writer's lines start on their own.
        MixedRecord for a file written whole."""
        _check_record(folder, name)
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
        _check_record(folder)
        for key in self.kind.list(f"records/{folder}/", deep=False):
            if not key.endswith(OPEN):
                continue
            try:
                data, _ = self.kind.read(key)
            except Absent:
                continue  # ended by another process
            name = key.rsplit("/", 1)[1].removesuffix(OPEN)
            with self._pending_lock:
                self._last = max(time.time_ns(), self._last + 1)
                self._added += 1
                moment, number = self._last, self._added
            part = f"records/{folder}/{name}/{moment:020d}-{self._writer}-{number:010d}{PACKED}"
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
        _check_record(folder, name)
        data = self._raw_read_record(folder, name)
        if data is None:
            raise MissingObject(f"no record file {folder}/{name} in the store")
        return data

    def record_files(self, folder: str) -> list[str]:
        """The names of a record folder's files, sorted; [] for no folder."""
        _check_record(folder)
        return sorted(self._raw_record_files(folder))

    def record_folders(self) -> list[str]:
        """Every record folder, sorted by name."""
        return sorted(self._raw_record_folders())

    # A record file written whole is records/<folder>/<name>.zst. A file
    # added to line by line is a folder of parts, and, in a local store while
    # its record runs, also one plain file records/<folder>/<name>.open that
    # its lines are appended to, read after the parts (end_records makes it
    # a part). The parts are:
    # records/<folder>/<name>/<time>-<writer>-<number>.zst: each part holds
    # the additions one flush_records() sent, each one framed as
    # "<time> <number> <length>\n" and its bytes. <time> is when it was added
    # (nanoseconds since 1970, never twice the same for one writer), <writer>
    # is this store object's random id, and <number> counts its additions; a
    # part is named after its first addition. Reading sorts every addition
    # of every part by (time, writer, number): the order they were made,
    # across processes, whenever each was sent. Every one of these files is
    # compressed (records always shrink), and written durably: on the disk
    # when the write returns. A record file stored uncompressed, without
    # .zst, is read as it is (found by trying, as objects are): the layout
    # records had before, which nothing rewrites. A local store kept a file
    # added to line by line as one file then: it reads as a file written
    # whole.

    def _record_files_under(self, prefix) -> dict:
        """{name: whether it is a folder of parts} right under a key
        prefix ending in '/', each name without .zst."""
        names = {}
        for key in self.kind.list(prefix, deep=False):
            name = key[len(prefix):].removesuffix("/")
            names[name.removesuffix(PACKED) if name.endswith(PACKED) else name.removesuffix(OPEN)] = key.endswith("/")
        return names

    def _raw_record_kind(self, folder, name):
        if self.kind.list(f"records/{folder}/{name}/") or self.kind.list(f"records/{folder}/{name}{OPEN}"):
            return APPENDED
        return WHOLE if self._record_object(folder, name) is not None else None

    def _raw_write_record(self, folder, name, data):
        self.kind.write(f"records/{folder}/{name}{PACKED}", blocks.compressed(data), durable=True)

    def _raw_append_record(self, folder, name, data):
        with self._pending_lock:
            self._last = max(time.time_ns(), self._last + 1)
            self._added += 1
            self._pending.setdefault((folder, name), []).append((self._last, self._added, data))

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
                self.kind.write(f"records/{folder}/{name}/{first_time:020d}-{self._writer}-{first_number:010d}{PACKED}",
                                blocks.compressed(body), durable=True)
                del self._pending[(folder, name)]

    def _record_object(self, folder, name):
        key = f"records/{folder}/{name}"
        try:
            return self._decompressed(key + PACKED, self.kind.read(key + PACKED)[0])
        except Absent:
            pass
        try:
            return self.kind.read(key)[0]  # the earlier layout, uncompressed
        except Absent:
            return None

    def _raw_read_record(self, folder, name):
        parts = self.kind.list(f"records/{folder}/{name}/")
        try:
            plain = self.kind.read(f"records/{folder}/{name}{OPEN}")[0]
        except Absent:
            plain = None
        if not parts and plain is None:
            return self._record_object(folder, name)
        additions = []
        for key in parts:
            writer = key.rsplit("/", 1)[1].split("-")[1]
            try:
                body = self.kind.read(key)[0]
            except Absent:
                continue  # removed since it was listed
            if key.endswith(PACKED):  # else a part of the earlier layout, uncompressed
                body = self._decompressed(key, body)
            at = 0
            try:
                while at < len(body):
                    end = body.index(b"\n", at)
                    moment, number, length = (int(field) for field in body[at:end].split(b" "))
                    additions.append((moment, writer, number, body[end + 1:end + 1 + length]))
                    at = end + 1 + length
            except ValueError:
                raise CorruptObject(f"record part {key} is not in the store's format") from None
        joined, last_writer = [], None
        for _, writer, _, data in sorted(additions):
            # a line another writer left cut (it ended without its newline) is ended first, so this writer's
            # own lines stay whole and readable
            if joined and writer != last_writer and not joined[-1].endswith(b"\n"):
                joined.append(b"\n")
            joined.append(data)
            last_writer = writer if data else last_writer
        if plain:
            if joined and not joined[-1].endswith(b"\n"):
                joined.append(b"\n")
            joined.append(plain)
        return b"".join(joined)

    def _raw_record_files(self, folder):
        return set(self._record_files_under(f"records/{folder}/"))

    def _raw_record_folders(self):
        return {name for name, is_folder in self._record_files_under("records/").items() if is_folder}


class LocalStore(Store):
    """A store in one local folder (wpt_store.kinds.Folder):
    objects/<first 2 hex>/<sha256>[.zst], refs/<kind>/<name>,
    bundles/<session>/<number>[.zst], store.sqlite.zst, records/<folder>/,
    and the local folders cache/ and scratch/."""

    local = True

    def __init__(self, root):
        self.root = Path(root)
        super().__init__(Folder(self.root), self.root / "cache")
        self.scratch = self.root / "scratch"
        self._unsynced = set()  # plain record files added to since the last flush_records

    def __repr__(self):
        return f"LocalStore({str(self.root)!r})"

    def _raw_append_record(self, folder, name, data):
        """Add the lines at once to the plain file of the record file: one
        write, so lines of several processes never mix, and none is lost
        when this process dies. A line an earlier writer left cut (the file
        does not end with a newline) is ended first, once per writer."""
        path = self.root / "records" / folder / f"{name}{OPEN}"
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
    """The same layout as LocalStore, as keys under s3://bucket/prefix
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
                config=Config(ignore_configured_endpoint_urls=True))
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


def _reopened(opening, cache, copies_only) -> Store:
    kind, arguments = opening
    store = kind(*arguments)
    store.cache, store.copies_only = Path(cache), copies_only
    return store


def open_store(url: str, environ=None, env_file=None) -> Store:
    """A store from a WPT_STORE value; an s3 store reads its own settings
    (s3_settings) from environ (default: the environment) and env_file."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise StoreSettingError(f"file URLs must name a local path: {url!r}")
        path = unquote(parsed.path)
        if os.name == "nt" and re.match(r"/[A-Za-z]:", path):
            path = path[1:]  # file:///C:/x -> C:/x
        return LocalStore(path)
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


def read_env_file(path) -> dict:
    """KEY=VALUE lines; blank lines and # comments are skipped."""
    pairs = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return pairs
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key.strip()] = value.strip().strip("'\"")
    return pairs


def _default_cache():
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return Path(base or Path.home() / ".cache") / "wpt-store"


def _replace_object(tmp, target: Path):
    """os.replace for a file named by its hash. On Windows, replacing a file
    another process has open is refused; the name is the content's hash, so a
    target that exists already holds the same bytes."""
    try:
        os.replace(tmp, target)
    except PermissionError:
        if not target.is_file():
            raise
        Path(tmp).unlink(missing_ok=True)
