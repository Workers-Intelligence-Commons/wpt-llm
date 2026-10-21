"""Content-addressed storage for everything WPT downloads or builds.

WPT hosts no data: each machine points the WPT_STORE setting at its own store,

    WPT_STORE=file:///home/me/wpt-store      a local folder
    WPT_STORE=s3://bucket/prefix             an S3-compatible bucket

An object is stored under the sha256 of its bytes and checked against it on
every read. A ref is a small named pointer to an object (for example a build
fingerprint to its result), so "already done" is a lookup.

A record folder holds files written by name, not by content: a command's
own account of one run (see wpt_telemetry), records/<folder>/<name>. A file
is written whole, or added to line by line, never both. A file written whole
is in the store when the call returns. A line added to a local store is on
the disk when the call returns; an S3 store holds added lines back until
flush_records() sends them together, since each S3 request costs time and
money. Lines added by several processes read back in the order they were
added.

For S3, credentials and a non-AWS endpoint come from boto3's own settings
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL...). Objects read
from S3 are cached locally after their check.

Failures: StoreSettingError (no or bad WPT_STORE), MissingObject (no such
object, ref target or record file), CorruptObject (stored bytes do not match
their sha256, or a record part is not in the store's format; the copy is not
used), MixedRecord (a whole write to a file added to line by line, or the
reverse; nothing is written). Writing raises what the folder or the bucket
raised (OSError, botocore's errors): nothing is known to be stored.
"""

import hashlib
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from wpt_utils.files import sha256_file

SHA256 = re.compile(r"[0-9a-f]{64}")
NAME_CHARACTERS = "A-Za-z0-9._-"  # what a ref or record name may hold, as a regular expression class
NAME = re.compile(f"[{NAME_CHARACTERS}]+")
CHUNK = 1 << 20


class StoreError(Exception):
    pass


class StoreSettingError(StoreError):
    pass


class MissingObject(StoreError):
    pass


class CorruptObject(StoreError):
    pass


class MixedRecord(StoreError):
    """A record file is written whole or added to line by line, never both."""


WHOLE, APPENDED = "written whole", "added to line by line"  # the two kinds of record file


def _check_sha(sha):
    if not SHA256.fullmatch(sha):
        raise ValueError(f"not a sha256: {sha!r}")


def _check_ref(kind, name):
    for part in (kind, name):
        if not NAME.fullmatch(part):
            raise ValueError(f"ref parts use letters, digits, '.', '_' and '-' only: {part!r}")


def check_record_name(part):
    """ValueError unless part can name a record folder or file: letters,
    digits, '.', '_' and '-' only, not starting with '.'."""
    if not NAME.fullmatch(part) or part.startswith("."):
        raise ValueError(f"record folders and files use letters, digits, '.', '_' and '-' only, "
                         f"and do not start with '.': {part!r}")


def as_record_name(text) -> str:
    """text made into a part of a record folder or file name that
    check_record_name accepts: '-' for each character it does not allow,
    without the leading '.'s, and '-' for nothing left."""
    return re.sub(f"[^{NAME_CHARACTERS}]", "-", str(text)).lstrip(".") or "-"


def _check_record(folder, name=None):
    for part in (folder, name) if name is not None else (folder,):
        check_record_name(part)


class Store:
    """Operations every store offers. Subclasses provide the _raw_* methods
    and `scratch`: a local folder for files derived from objects (such as
    token files for training), which may be deleted at any time."""

    scratch: Path

    def __init__(self):
        self._record_kinds = {}  # (folder, name) -> WHOLE or APPENDED, as this store learned it

    def put_bytes(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if not self.has(sha):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "object"
                path.write_bytes(data)
                self._raw_put(sha, path)
        return sha

    def put_file(self, path) -> str:
        sha = sha256_file(path)
        if not self.has(sha):
            self._raw_put(sha, Path(path))
        return sha

    def has(self, sha: str) -> bool:
        _check_sha(sha)
        return self._raw_has(sha)

    def path(self, sha: str) -> Path:
        """A local file holding the object, checked against its sha256."""
        _check_sha(sha)
        if not self._raw_has(sha):
            raise MissingObject(f"object {sha} is not in the store")
        path = self._raw_local(sha)
        actual = sha256_file(path)
        if actual != sha:
            self._raw_discard_local(sha)
            raise CorruptObject(f"object {sha} reads back as {actual}")
        return path

    def get_bytes(self, sha: str) -> bytes:
        return self.path(sha).read_bytes()

    def read_range(self, sha: str, start: int, length: int) -> bytes:
        """length bytes of an object from offset start, without fetching the
        rest of it. A part cannot be checked against the object's sha256:
        the caller checks it against a hash of its own."""
        _check_sha(sha)
        if not self._raw_has(sha):
            raise MissingObject(f"object {sha} is not in the store")
        data = self._raw_read_range(sha, start, length)
        if len(data) != length:
            raise CorruptObject(f"object {sha}: {length} bytes at {start} read back as {len(data)}")
        return data

    def release(self, sha: str):
        """Remove the local copy of a remote object (it stays in the store);
        nothing for a local store, whose copy is the object itself."""
        _check_sha(sha)
        self._raw_discard_local(sha)

    def set_ref(self, kind: str, name: str, sha: str):
        _check_ref(kind, name)
        _check_sha(sha)
        if not self._raw_has(sha):
            raise MissingObject(f"ref {kind}/{name} would point at missing object {sha}")
        self._raw_set_ref(kind, name, sha)

    def get_ref(self, kind: str, name: str) -> str | None:
        """The object a ref points at, or None when the ref does not exist."""
        _check_ref(kind, name)
        sha = self._raw_get_ref(kind, name)
        if sha is not None and not SHA256.fullmatch(sha):
            raise CorruptObject(f"ref {kind}/{name} holds {sha!r}, not a sha256")
        return sha

    def delete(self, sha: str):
        """Remove an object, and its local copy; nothing when it is absent.
        Objects are shared by content: only for one that nothing else can
        point at, such as a training checkpoint that a newer one replaced."""
        _check_sha(sha)
        self._raw_discard_local(sha)
        self._raw_delete(sha)

    def delete_ref(self, kind: str, name: str):
        """Remove a ref (not its object); nothing when it is absent."""
        _check_ref(kind, name)
        self._raw_delete_ref(kind, name)

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
    """objects/<first 2 hex>/<sha256> and refs/<kind>/<name> under one folder."""

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.scratch = self.root / "scratch"
        self._appending = set()  # (folder, name) this store added to: their marker exists and their last line ends

    def __repr__(self):
        return f"LocalStore({str(self.root)!r})"

    def _object(self, sha):
        return self.root / "objects" / sha[:2] / sha

    def _raw_has(self, sha):
        return self._object(sha).is_file()

    def _raw_put(self, sha, path):
        _atomic_copy(path, self._object(sha))

    def _raw_local(self, sha):
        return self._object(sha)

    def _raw_discard_local(self, sha):
        pass  # the stored copy is the only one; keep it for inspection

    def _raw_read_range(self, sha, start, length):
        with open(self._object(sha), "rb") as file:
            file.seek(start)
            return file.read(length)

    def _raw_set_ref(self, kind, name, sha):
        _atomic_write(self.root / "refs" / kind / name, sha.encode())

    def _raw_get_ref(self, kind, name):
        try:
            return (self.root / "refs" / kind / name).read_text().strip()
        except FileNotFoundError:
            return None

    def _raw_delete(self, sha):
        self._object(sha).unlink(missing_ok=True)

    def _raw_delete_ref(self, kind, name):
        (self.root / "refs" / kind / name).unlink(missing_ok=True)

    def _record(self, folder, name=None):
        return self.root / "records" / folder / name if name else self.root / "records" / folder

    def _raw_write_record(self, folder, name, data):
        _atomic_write(self._record(folder, name), data, durable=True)

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
            _sync_folder(path.parent)
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
    """The same layout as LocalStore, as keys under s3://bucket/prefix."""

    def __init__(self, bucket, prefix, cache_dir=None, client=None):
        import boto3

        super().__init__()
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3")
        self.cache = Path(cache_dir) if cache_dir else _default_cache() / bucket / (self.prefix or "_")
        self.scratch = self.cache / "scratch"
        # lines added and not yet sent: (folder, name) -> [(time, number, data)]; see _raw_append_record
        self._writer, self._pending, self._added, self._last = secrets.token_hex(4), {}, 0, 0
        self._pending_lock = threading.Lock()

    def __repr__(self):
        return f"S3Store(s3://{self.bucket}/{self.prefix})"

    def _key(self, *parts):
        return "/".join(p for p in (self.prefix, *parts) if p)

    def _object_key(self, sha):
        return self._key("objects", sha[:2], sha)

    def _raw_has(self, sha):
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._object_key(sha))
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _raw_put(self, sha, path):
        self.client.upload_file(str(path), self.bucket, self._object_key(sha))

    def _raw_local(self, sha):
        local = self.cache / sha[:2] / sha
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=local.parent)
            os.close(fd)
            try:
                self.client.download_file(self.bucket, self._object_key(sha), tmp)
                _replace_object(tmp, local)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return local

    def _raw_discard_local(self, sha):
        (self.cache / sha[:2] / sha).unlink(missing_ok=True)

    def _raw_read_range(self, sha, start, length):
        local = self.cache / sha[:2] / sha
        if local.is_file():
            with open(local, "rb") as file:
                file.seek(start)
                return file.read(length)
        if length == 0:
            return b""
        answer = self.client.get_object(Bucket=self.bucket, Key=self._object_key(sha),
                                        Range=f"bytes={start}-{start + length - 1}")
        return answer["Body"].read()

    def _raw_set_ref(self, kind, name, sha):
        self.client.put_object(Bucket=self.bucket, Key=self._key("refs", kind, name), Body=sha.encode())

    def _raw_get_ref(self, kind, name):
        from botocore.exceptions import ClientError

        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key("refs", kind, name))["Body"]
            return body.read().decode().strip()
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def _raw_delete(self, sha):
        self.client.delete_object(Bucket=self.bucket, Key=self._object_key(sha))  # S3 answers the same when absent

    def _raw_delete_ref(self, kind, name):
        self.client.delete_object(Bucket=self.bucket, Key=self._key("refs", kind, name))

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
        parts, _ = self._list(self._key("records", folder, name) + "/", max_keys=1)
        if parts:
            return APPENDED
        return WHOLE if self._record_object(folder, name) is not None else None

    def _raw_write_record(self, folder, name, data):
        self.client.put_object(Bucket=self.bucket, Key=self._key("records", folder, name), Body=data)

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
                key = self._key("records", folder, name, f"{first_time:020d}-{self._writer}-{first_number:010d}")
                self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
                del self._pending[(folder, name)]

    def _list(self, prefix, delimiter=None, max_keys=None):
        """(keys, sub-prefixes) under prefix, every page (only the first
        max_keys keys, when given)."""
        keys, prefixes = [], []
        options = {"Bucket": self.bucket, "Prefix": prefix, **({"Delimiter": delimiter} if delimiter else {})}
        if max_keys:
            page = self.client.list_objects_v2(**options, MaxKeys=max_keys)
            return [item["Key"] for item in page.get("Contents", [])], []
        for page in self.client.get_paginator("list_objects_v2").paginate(**options):
            keys += [item["Key"] for item in page.get("Contents", [])]
            prefixes += [item["Prefix"] for item in page.get("CommonPrefixes", [])]
        return keys, prefixes

    def _record_object(self, folder, name):
        from botocore.exceptions import ClientError

        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key("records", folder, name))["Body"].read()
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def _raw_read_record(self, folder, name):
        parts, _ = self._list(self._key("records", folder, name) + "/")
        if not parts:
            return self._record_object(folder, name)
        additions = []
        for key in parts:
            writer = key.rsplit("/", 1)[1].split("-")[1]
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
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
        prefix = self._key("records", folder) + "/"
        keys, appended = self._list(prefix, "/")
        return {key[len(prefix):] for key in keys} | {p[len(prefix):-1] for p in appended}

    def _raw_record_folders(self):
        prefix = self._key("records") + "/"
        _, folders = self._list(prefix, "/")
        return [p[len(prefix):-1] for p in folders]


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


def _atomic_copy(source, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
            while block := src.read(CHUNK):
                out.write(block)
        _replace_object(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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


def _atomic_write(target: Path, data: bytes, durable=False):
    """durable: on the disk, name included, when it returns, not only in the
    system's memory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            if durable:
                out.flush()
                os.fsync(out.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    if durable:
        _sync_folder(target.parent)


def _sync_folder(folder: Path):
    """Put a folder's list of names on the disk, so a new file's name survives
    a power cut. Windows cannot open a folder this way: there, only the
    file itself is synced."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
