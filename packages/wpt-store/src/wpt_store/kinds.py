"""Store kinds: where a store keeps its files, a local folder (Folder) or
an S3-compatible bucket (Bucket). WPT_STORE's prefix picks one
(wpt_store.open_store). A file is named by a key, '/'-separated parts such
as objects/ab/<sha256>, and a kind supplies five operations on them,
nothing else; objects, refs, compression, bundles and the store database
are built above them, alike for every kind:

    write(key, source, move=False, durable=False)
                                      write a whole file, from bytes or a local
                                      file, replacing one already there; with
                                      move, the local file is gone afterwards;
                                      durable: on the disk, name included, when
                                      it returns (a folder syncs it; S3 holds
                                      every file it answered for)
    read(key, start=0, length=None)   (bytes, version): length bytes from start
                                      (start < 0: that many from the end; length
                                      None: up to the end; fewer when the file
                                      ends first) and the file's version
    replace(key, source, version)     the new version: source (a local file,
                                      left as it is) written whole only if
                                      the file is still at version (None:
                                      only if there is no file); Changed
                                      otherwise, and nothing is written
    list(prefix, deep=True)           {key: size} of every file whose key starts
                                      with prefix; not deep: only those with no
                                      '/' after prefix, and each folder holding
                                      the others once, as its key ending in '/'
                                      with size None (S3's listing by
                                      delimiter), so a folder's sub-folders are
                                      found without listing all their files
    delete(key)                       remove a file; nothing when absent

A version is a string the kind gives and compares, nothing more: the
file's ETag on S3, its file number, time and size in a folder. S3 checks it
itself (If-Match and If-None-Match); a folder checks it while holding a
lock the operating system keeps on a file beside the target, which no
other process can then take.

A kind may also say where a file lies on this machine's disk (on_disk), so
the store reads a local folder's files in place instead of copying them.

Failures: Absent (read of a key with no file), Changed (replace refused,
nothing written). Anything else is what the folder or the bucket raised
(OSError, botocore's errors): a write is then not known to be stored, and
a folder leaves no partial file under the key.
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from wpt_utils.files import lock_file, unlock_file
from wpt_utils.parallel import POLL, SETTLE, Sizing

CHUNK = 1 << 20  # bytes copied at once
PART = 64 << 20  # bytes per part of an S3 upload: a file larger than this goes up in parts, several at once
# parts sent to S3 at once by every upload of this process together: R2 gives each connection a few MiB/s, so the
# link fills only with many connections at once
UPLOADS = 32


class Absent(Exception):
    """No file under that key."""


class Changed(Exception):
    """replace refused: the file is no longer at the version given."""


class Folder:
    """A store's files under one local folder, each key a path below it."""

    def __init__(self, root):
        self.root = Path(root)

    def __repr__(self):
        return f"Folder({str(self.root)!r})"

    def _path(self, key):
        return self.root.joinpath(*key.split("/"))

    def on_disk(self, key) -> Path | None:
        path = self._path(key)
        return path if path.is_file() else None

    def write(self, key, source, move=False, durable=False):
        target = self._path(key)
        if isinstance(source, (bytes, bytearray, memoryview)):
            atomic_write(target, source, durable)
            return
        if move:
            target.parent.mkdir(parents=True, exist_ok=True)
            if durable:
                sync_file(source)
            try:
                os.replace(source, target)
                if durable:
                    sync_folder(target.parent)
                return
            except OSError:
                if not os.path.exists(source):
                    raise
                # another file system (or a target another process has open, on Windows): copy instead
        atomic_copy(source, target, durable)
        if move:
            Path(source).unlink()

    def read(self, key, start=0, length=None):
        try:
            file = open(self._path(key), "rb")
        except (FileNotFoundError, NotADirectoryError):
            raise Absent(key) from None
        with file:
            stat = os.fstat(file.fileno())
            file.seek(max(0, stat.st_size + start) if start < 0 else start)
            return (file.read() if length is None else file.read(length)), _version(stat)

    def replace(self, key, source, version):
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_file(target.with_name(f".{target.name}.lock"))
        try:
            try:
                current = _version(target.stat())
            except FileNotFoundError:
                current = None
            if current != version:
                raise Changed(key)
            atomic_copy(source, target, durable=True)
            return _version(target.stat())
        finally:
            unlock_file(lock)

    def list(self, prefix, deep=True):
        found = {}
        folder, _, _ = prefix.rpartition("/")
        base = self._path(folder) if folder else self.root
        if not base.is_dir():
            return found
        if not deep:
            for entry in os.scandir(base):
                key = f"{folder}/{entry.name}" if folder else entry.name
                if entry.name.startswith(".") or not key.startswith(prefix):
                    continue  # a lock, a partial file, a record file's marker: not the store's files
                try:
                    if entry.is_dir():
                        found[key + "/"] = None
                    elif entry.is_file():
                        found[key] = entry.stat().st_size
                except FileNotFoundError:
                    pass  # removed meanwhile
            return found
        for directory, _, names in os.walk(base):
            for name in names:
                if name.startswith("."):
                    continue  # a lock, a partial file, a record file's marker: not the store's files
                path = Path(directory) / name
                key = path.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    try:
                        found[key] = path.stat().st_size
                    except FileNotFoundError:
                        pass  # removed meanwhile
        return found

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)


def _version(stat):
    return f"{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}"  # every replace makes a new file, so a new number


class _Parts:
    """The parts every upload of this process sends, on threads of their
    own: at most UPLOADS at once in total, however many uploads run, each
    more only as wpt_utils.parallel's rule for network work allows (a
    Sizing with UPLOADS as its network limit: the memory the running parts
    are measured taking, not the cores)."""

    def __init__(self):
        self.sizing = Sizing("upload parts", network=UPLOADS)
        self.executor = ThreadPoolExecutor(UPLOADS, thread_name_prefix="upload-part")
        self.changed = threading.Condition()
        self.started = []  # the monotonic time each running part started

    def run(self, send, numbers) -> list:
        """send(number) for each of numbers, results in their order. The
        first failure is raised once every part already started has ended;
        the parts not started by then never start."""
        failed = threading.Event()  # set by this run's first failure: its parts still waiting never start
        futures = [self.executor.submit(self._one, send, number, failed) for number in numbers]
        try:
            return [future.result() for future in futures]
        except BaseException as error:
            self._stop(failed)
            for future in futures:
                future.cancel()
            wait(futures)
            if isinstance(error, _NotStarted):  # a part waiting before the failing one: raise the failure itself
                errors = (future.exception() for future in futures if not future.cancelled())
                raise next(found for found in errors if found is not None and not isinstance(found, _NotStarted))
            raise

    def _stop(self, failed):
        with self.changed:
            failed.set()
            self.changed.notify_all()  # wakes this run's waiting parts at once

    def _one(self, send, number, failed):
        with self.changed:
            while True:
                if failed.is_set():
                    raise _NotStarted()
                now = time.monotonic()
                unseen = sum(now - started < SETTLE for started in self.started)
                if self.sizing.may_start(len(self.started), unseen):
                    break
                self.changed.wait(POLL)  # a part's end wakes it sooner
            started = time.monotonic()
            self.started.append(started)
        try:
            return send(number)
        except BaseException:
            self._stop(failed)
            raise
        finally:
            with self.changed:
                self.started.remove(started)
                self.changed.notify_all()


class _NotStarted(Exception):
    """A part that never started because another part of its upload failed."""


_parts, _parts_lock = None, threading.Lock()


def _upload_parts() -> _Parts:
    """This process's _Parts, made at its first upload in parts."""
    global _parts
    with _parts_lock:
        if _parts is None:
            _parts = _Parts()
        return _parts


class Bucket:
    """A store's files as keys under s3://bucket/prefix. A file larger than
    part goes up in parts of that size, sent with those of every other
    upload of this process, UPLOADS at most at once in total (_Parts); a
    replace in parts puts its condition on the request that completes it,
    which S3 and R2 refuse as they refuse a conditional single upload."""

    def __init__(self, client, bucket, prefix, part=PART):
        self.client, self.bucket, self.prefix, self.part = client, bucket, prefix.strip("/"), part

    def __repr__(self):
        return f"Bucket(s3://{self.bucket}/{self.prefix})"

    def key(self, key):
        """The bucket's own key for a store key."""
        return f"{self.prefix}/{key}" if self.prefix else key

    def on_disk(self, key):
        return None

    def write(self, key, source, move=False, durable=False):
        # S3 answers a write once it holds the file: every write is durable
        if isinstance(source, (bytes, bytearray, memoryview)):
            self.client.put_object(Bucket=self.bucket, Key=self.key(key), Body=bytes(source))
            return
        size = os.path.getsize(source)
        if size <= self.part:
            with open(source, "rb") as file:
                self.client.put_object(Bucket=self.bucket, Key=self.key(key), Body=file.read())
        else:
            self._write_parts(key, source, size)
        if move:
            Path(source).unlink()

    def _write_parts(self, key, source, size, guard=None):
        """source sent in parts, then completed under guard (the conditional
        headers of a replace, none for a write): the new version's ETag."""
        upload = self.client.create_multipart_upload(Bucket=self.bucket, Key=self.key(key))["UploadId"]

        def send(number):
            with open(source, "rb") as file:
                file.seek((number - 1) * self.part)
                body = file.read(self.part)
            answer = self.client.upload_part(Bucket=self.bucket, Key=self.key(key), UploadId=upload,
                                             PartNumber=number, Body=body)
            return {"PartNumber": number, "ETag": answer["ETag"]}

        try:
            parts = _upload_parts().run(send, range(1, -(-size // self.part) + 1))
            return self.client.complete_multipart_upload(Bucket=self.bucket, Key=self.key(key), UploadId=upload,
                                                         MultipartUpload={"Parts": parts}, **(guard or {}))["ETag"]
        except BaseException:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=self.key(key), UploadId=upload)
            raise

    def read(self, key, start=0, length=None):
        from botocore.exceptions import ClientError

        try:
            if length == 0:
                return b"", self.client.head_object(Bucket=self.bucket, Key=self.key(key))["ETag"]
            if start < 0:
                span = {"Range": f"bytes={start}"}  # the last -start bytes
            elif length is None:
                span = {"Range": f"bytes={start}-"} if start else {}
            else:
                span = {"Range": f"bytes={start}-{start + length - 1}"}
            answer = self.client.get_object(Bucket=self.bucket, Key=self.key(key), **span)
            data = answer["Body"].read()
            return (data if length is None else data[:length]), answer["ETag"]
        except ClientError as error:
            code = error.response["Error"]["Code"]
            if code in ("404", "NoSuchKey", "NotFound"):
                raise Absent(key) from None
            if code in ("416", "InvalidRange"):  # start past the end
                return b"", None
            raise

    def replace(self, key, source, version):
        from botocore.exceptions import ClientError

        guard = {"IfNoneMatch": "*"} if version is None else {"IfMatch": version}
        size = os.path.getsize(source)
        try:
            if size > self.part:  # a single upload is refused above 5 GiB
                return self._write_parts(key, source, size, guard)
            with open(source, "rb") as body:  # sent as it is read from the disk
                return self.client.put_object(Bucket=self.bucket, Key=self.key(key), Body=body, **guard)["ETag"]
        except ClientError as error:
            # 409: another conditional write to the key was under way; it may have changed it
            if error.response["Error"]["Code"] in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise Changed(key) from None
            raise

    def list(self, prefix, deep=True):
        found, start = {}, len(self.key(""))
        by_folder = {} if deep else {"Delimiter": "/"}
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.key(prefix),
                                                                           **by_folder):
            for item in page.get("Contents", []):
                found[item["Key"][start:]] = item["Size"]
            for item in page.get("CommonPrefixes", []):
                found[item["Prefix"][start:]] = None
        return found

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=self.key(key))  # S3 answers the same when absent


def atomic_copy(source, target: Path, durable=False):
    """Copy a file to target whole: a failure leaves target as it was.
    durable: as atomic_write's."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
            while block := src.read(CHUNK):
                out.write(block)
            if durable:
                out.flush()
                os.fsync(out.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    if durable:
        sync_folder(target.parent)


def atomic_write(target: Path, data: bytes, durable=False):
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
        sync_folder(target.parent)


def sync_file(path):
    """Put a file's bytes on the disk, not only in the system's memory."""
    with open(path, "rb+") as file:
        os.fsync(file.fileno())


def sync_folder(folder: Path):
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
