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
Refs, too small to shrink, and record folders are written as they are.

A session (wpt_store.sessions) stores the files one downloading command
fetches in its own bundles, and its lines in the store database
(wpt_store.database), which Store.database() reads. An S3 store keeps a
local copy of each bundle its sessions stored, in its cache folder,
cache/bundles/<session>/<number>[.zst], and reads that copy first, unless
the session streams (Store.session): the copy is then gone once the
bucket holds the bundle.

A record folder holds files written by name, not by content: a command's
own account of one run (see wpt_telemetry), records/<folder>/<name>. A file
is written whole, or added to line by line, never both. A file written whole
is in the store when the call returns. A line added to a local store is on
the disk when the call returns; an S3 store holds added lines back until
flush_records() sends them together, since each S3 request costs time and
money. Lines added by several processes read back in the order they were
added.

For S3, credentials and a non-AWS endpoint come from boto3's own settings
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL...). A store
keeps local copies of objects in its cache folder: an S3 store every object
read as a file, after its check; a local store the compressed ones read as
a file, which it cannot give in place. release() removes a copy;
copies_in(folder) is the same store making its new copies in folder, for
work that keeps none.

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

from wpt_utils.files import sha256_file
from wpt_utils.parallel import Sizing, in_order

from . import blocks, database
from .errors import CorruptObject, MissingObject, MixedRecord, StoreError, StoreSettingError
from .kinds import Absent, Bucket, Folder, atomic_copy, atomic_write, sync_folder
from .names import NAME, NAME_CHARACTERS, as_record_name, check_record_name, check_ref
from .sessions import Placed, Session, file_list_bytes, parse_file_list

SHA256 = re.compile(r"[0-9a-f]{64}")
CHUNK = 1 << 20
PACKED = ".zst"  # the end of a key whose file is compressed
TAIL = 64 << 10  # bytes read at a compressed file's end for its seek table: the whole table up to 8,000 frames
DOWNLOAD = 64 << 20  # bytes of one file fetched in one request, when a whole file is copied from the store
DOWNLOAD_MEMORY = 2.5 * DOWNLOAD  # bytes one such request takes: its bytes, decompressed, and written
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
    wpt_store.kinds). Subclasses give the kind, `cache` (a local folder for
    copies of objects) and `scratch` (a local folder for files derived from
    objects, such as token files for training, which may be deleted at
    any time), and the _raw_* methods of record folders."""

    scratch: Path
    local = False  # whether the kind's files are on this machine's disk already, so no copy of them is kept

    def __init__(self, kind, cache):
        self.kind, self.cache = kind, Path(cache)
        self._copies = Folder(self.cache)  # the local copies of bundles, read through the same five operations
        self._record_kinds = {}  # (folder, name) -> WHOLE or APPENDED, as this store learned it
        self._tables, self._frames, self._keys = _Kept(TABLES_KEPT), _Kept(FRAMES_KEPT), _Kept(KEYS_KEPT)
        self._compressing, self._compressing_lock = None, threading.Lock()

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
        is (every object of an S3 store, a local store's compressed ones):
        the copy stays until release()."""
        _check_sha(sha)
        key = self._found(sha)
        local = self.cache / sha[:2] / sha
        path = None if key.endswith(PACKED) else self.kind.on_disk(key)
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
        _check_sha(sha)
        key = self._found(sha)
        on_disk = self.kind.on_disk(key)
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
        """The same store, whose new local copies of objects and of a
        session's files (path(), file_path()) go into folder instead of its
        cache folder, for work that keeps none: removing folder afterwards
        removes every copy that work made. Copies already in the cache
        folder are not read. For objects and refs only, not record
        folders."""
        self.compressing()  # made before the copy, so every view shares this store's one
        view = copy.copy(self)
        view.cache = Path(folder)
        return view

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
            key, kind = self._bundle(placed)
            local.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=local.parent, prefix=".partial-")
            try:
                digest = hashlib.sha256()
                with os.fdopen(fd, "wb") as out, self._reading(f"bundle {placed.bundle}", key):
                    for at in range(0, placed.length, DOWNLOAD):
                        data = self._range(key, placed.start + at, min(DOWNLOAD, placed.length - at), kind)
                        digest.update(data)
                        out.write(data)
                if digest.hexdigest() != placed.sha256:
                    raise CorruptObject(f"bundle {placed.bundle}: the {placed.length} bytes at {placed.start} "
                                        f"do not match their sha256 {placed.sha256}")
                _replace_object(tmp, local)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return local

    def _bundle(self, placed):
        """(the key a bundle is stored under, the kind to read it from: its
        local copy's when there is one)."""
        key = self._stored(f"bundles/{placed.bundle}")
        if key is None:
            raise MissingObject(f"bundle {placed.bundle} is not in the store")
        return key, self._copies if self._copies.on_disk(key) is not None else self.kind

    def database(self):
        """The store database (wpt_store.database), as a sqlite3 connection
        to a copy in memory."""
        return database.read(self.kind)

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
            self.kind.write(key, path, move=True)
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
                    sizing = Sizing(DOWNLOAD_MEMORY, "downloads")
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
        """Add lines at the end of a record file, creating it. Several
        processes may add to one file; reading returns every addition in the
        order they were made. The first addition of this store to a file
        whose last line is cut (a crash while it was written) ends that line
        first, so the new lines start on their own. A local store has the
        lines on the disk when it returns; an S3 store holds them until
        flush_records(). MixedRecord for a file written whole."""
        _check_record(folder, name)
        self._claim(folder, name, APPENDED)
        self._raw_append_record(folder, name, data)

    def flush_records(self):
        """Send the lines append_record holds back (S3); a local store holds
        none. Raises what the bucket raised: the lines not sent are kept, to
        send again."""
        self._raw_flush_records()

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
        self._appending = set()  # (folder, name) this store added to: their marker exists and their last line ends

    def __repr__(self):
        return f"LocalStore({str(self.root)!r})"

    def _record(self, folder, name=None):
        return self.root / "records" / folder / name if name else self.root / "records" / folder

    def _raw_write_record(self, folder, name, data):
        atomic_write(self._record(folder, name), data, durable=True)

    # A file added to line by line has a marker beside it, .appended-<name>,
    # so a later process knows its kind (see Store._claim).

    def _marker(self, folder, name):
        return self._record(folder) / f".appended-{name}"

    def _raw_record_kind(self, folder, name):
        if self._marker(folder, name).exists():
            return APPENDED
        return WHOLE if self._record(folder, name).exists() else None

    def _raw_append_record(self, folder, name, data):
        path = self._record(folder, name)
        first = (folder, name) not in self._appending
        created = first and not self._marker(folder, name).exists()
        if created:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._marker(folder, name).touch()
        with open(path, "a+b") as out:
            if first and out.seek(0, os.SEEK_END) > 0:
                out.seek(-1, os.SEEK_END)
                if out.read(1) != b"\n":  # the last line was cut by a crash: end it
                    data = b"\n" + data
            out.write(data)
            out.flush()
            os.fsync(out.fileno())  # on the disk before the caller goes on
        if created:
            sync_folder(path.parent)
        self._appending.add((folder, name))

    def _raw_flush_records(self):
        pass  # every line is on the disk already

    def _raw_read_record(self, folder, name):
        try:
            return self._record(folder, name).read_bytes()
        except FileNotFoundError:
            return None

    def _raw_record_files(self, folder):
        found = self._record(folder)
        return [p.name for p in found.iterdir() if p.is_file() and not p.name.startswith(".")] if found.is_dir() else []

    def _raw_record_folders(self):
        found = self.root / "records"
        return [p.name for p in found.iterdir() if p.is_dir()] if found.is_dir() else []


class S3Store(Store):
    """The same layout as LocalStore, as keys under s3://bucket/prefix
    (wpt_store.kinds.Bucket); its cache and scratch folders are local."""

    def __init__(self, bucket, prefix, cache_dir=None, client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3")
        cache = Path(cache_dir) if cache_dir else _default_cache() / bucket / (self.prefix or "_")
        super().__init__(Bucket(self.client, bucket, self.prefix), cache)
        self.scratch = self.cache / "scratch"
        # lines added and not yet sent: (folder, name) -> [(time, number, data)]; see _raw_append_record
        self._writer, self._pending, self._added, self._last = secrets.token_hex(4), {}, 0, 0
        self._pending_lock = threading.Lock()

    def __repr__(self):
        return f"S3Store(s3://{self.bucket}/{self.prefix})"

    # An S3 object cannot grow. A file added to line by line is a folder of
    # parts, records/<folder>/<name>/<time>-<writer>-<number>: each part holds
    # the additions one flush_records() sent, each one framed as
    # "<time> <number> <length>\n" and its bytes. <time> is when it was added
    # (nanoseconds since 1970, never twice the same for one writer), <writer>
    # is this store's random id, and <number> counts its additions; a part is
    # named after its first addition. Reading sorts every addition of every
    # part by (time, writer, number): the order they were made, across
    # processes, whenever each was sent.

    def _raw_record_kind(self, folder, name):
        if self.kind.list(f"records/{folder}/{name}/"):
            return APPENDED
        return WHOLE if self._record_object(folder, name) is not None else None

    def _raw_write_record(self, folder, name, data):
        self.kind.write(f"records/{folder}/{name}", data)

    def _raw_append_record(self, folder, name, data):
        with self._pending_lock:
            self._last = max(time.time_ns(), self._last + 1)
            self._added += 1
            self._pending.setdefault((folder, name), []).append((self._last, self._added, data))

    def _raw_flush_records(self):
        # A group leaves _pending only once S3 answered its upload, so a send
        # that failed or was interrupted (Ctrl+C) loses no line. When the
        # interruption came after S3 stored it, the next flush sends the
        # group again under the same key, since its first addition is the
        # same: S3 replaces that object with the group and the lines added
        # since, and no line is ever stored twice.
        with self._pending_lock:
            for (folder, name), additions in list(self._pending.items()):
                first_time, first_number, _ = additions[0]
                body = b"".join(b"%d %d %d\n" % (moment, number, len(data)) + data for moment, number, data in additions)
                self.kind.write(f"records/{folder}/{name}/{first_time:020d}-{self._writer}-{first_number:010d}", body)
                del self._pending[(folder, name)]

    def _record_object(self, folder, name):
        try:
            return self.kind.read(f"records/{folder}/{name}")[0]
        except Absent:
            return None

    def _raw_read_record(self, folder, name):
        parts = self.kind.list(f"records/{folder}/{name}/")
        if not parts:
            return self._record_object(folder, name)
        additions = []
        for key in parts:
            writer = key.rsplit("/", 1)[1].split("-")[1]
            body = self.kind.read(key)[0]
            at = 0
            try:
                while at < len(body):
                    end = body.index(b"\n", at)
                    moment, number, length = (int(field) for field in body[at:end].split(b" "))
                    additions.append((moment, writer, number, body[end + 1:end + 1 + length]))
                    at = end + 1 + length
            except ValueError:
                raise CorruptObject(f"record part {key} is not in the store's format") from None
        return b"".join(data for *_, data in sorted(additions))

    def _raw_record_files(self, folder):
        files, appended = self._folder(f"records/{folder}/")
        return set(files) | set(appended)

    def _raw_record_folders(self):
        _, folders = self._folder("records/")
        return folders

    def _folder(self, prefix):
        """(names of the files, names of the folders) right under a store
        key prefix ending in '/': S3's listing by delimiter, which the
        kind's list does not offer, so a folder's own files are not all
        listed to find its sub-folders."""
        prefix = self.kind.key(prefix)
        files, folders = [], []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix,
                                                                           Delimiter="/"):
            files += [item["Key"][len(prefix):] for item in page.get("Contents", [])]
            folders += [item["Prefix"][len(prefix):-1] for item in page.get("CommonPrefixes", [])]
        return files, folders


def open_store(url: str) -> Store:
    """A store from a WPT_STORE value."""
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
        return S3Store(parsed.netloc, parsed.path)
    raise StoreSettingError(f"WPT_STORE must start with file:// or s3://, got {url!r}")


def store_from_settings(env_file=".env", environ=None) -> Store:
    """The store named by WPT_STORE in the environment, else in env_file."""
    environ = os.environ if environ is None else environ
    url = environ.get("WPT_STORE") or read_env_file(env_file).get("WPT_STORE")
    if not url:
        raise StoreSettingError("WPT_STORE is not set: put WPT_STORE=file:///path or s3://bucket/prefix in .env")
    return open_store(url)


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
