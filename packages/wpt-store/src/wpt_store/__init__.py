"""Content-addressed storage for everything WPT downloads or builds.

WPT hosts no data: each machine points the WPT_STORE setting at its own store,

    WPT_STORE=file:///home/me/wpt-store      a local folder
    WPT_STORE=s3://bucket/prefix             an S3-compatible bucket

An object is stored under the sha256 of its bytes and checked against it on
every read. A ref is a small named pointer to an object (for example a build
fingerprint to its result), so "already done" is a lookup.

For S3, credentials and a non-AWS endpoint come from boto3's own settings
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL...). Objects read
from S3 are cached locally after their check.

Failures: StoreSettingError (no or bad WPT_STORE), MissingObject (no such
object or ref target), CorruptObject (stored bytes do not match their sha256;
the copy is not used).
"""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9._-]+")
CHUNK = 1 << 20


class StoreError(Exception):
    pass


class StoreSettingError(StoreError):
    pass


class MissingObject(StoreError):
    pass


class CorruptObject(StoreError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _check_sha(sha):
    if not SHA256.fullmatch(sha):
        raise ValueError(f"not a sha256: {sha!r}")


def _check_ref(kind, name):
    for part in (kind, name):
        if not NAME.fullmatch(part):
            raise ValueError(f"ref parts use letters, digits, '.', '_' and '-' only: {part!r}")


class Store:
    """Operations every store offers. Subclasses provide the _raw_* methods
    and `scratch`: a local folder for files derived from objects (such as
    token files for training), which may be deleted at any time."""

    scratch: Path

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


class LocalStore(Store):
    """objects/<first 2 hex>/<sha256> and refs/<kind>/<name> under one folder."""

    def __init__(self, root):
        self.root = Path(root)
        self.scratch = self.root / "scratch"

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


class S3Store(Store):
    """The same layout as LocalStore, as keys under s3://bucket/prefix."""

    def __init__(self, bucket, prefix, cache_dir=None, client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3")
        self.cache = Path(cache_dir) if cache_dir else _default_cache() / bucket / (self.prefix or "_")
        self.scratch = self.cache / "scratch"

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
                os.replace(tmp, local)
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
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
