"""What every process of a gpu worker's job starts with: `python -m
wpt_worker.job`, run by the worker's server from the manager's checkout.

The process connects to the server (link.py, through its channel to the
manager), sends what it prints to the manager, which prints it, and reads
the job's spec (JSON, the file WPT_WORKER_SPEC names); spec["task"] names
what it runs (TASKS). The worker has no tokenizer: wpt_tokenizer cannot be
imported here (NoTokenizer), so nothing on the worker turns text into
tokens or back by mistake; it deals only in the token ids the manager sends.

    LinkStore     the manager's store as the job reads it: objects the manager allows, fetched
                  part by part into the job's folder and checked against their sha256, and the
                  run's record, each write and line given to the manager's store before the call
                  returns, and sent on from there when flush_records() returns (an S3 store holds
                  lines back until then, as on the manager)
    download()    a file the manager offers, into a local file, checked against its sha256
    upload()      a local file to the manager, part by part: its upload id
"""

import hashlib
import importlib.abc
import io
import json
import os
import socket
import sys
from pathlib import Path

from .link import PART, Connection, LinkError, RemoteError, receive, send

TOKENIZER = "wpt_tokenizer"
TASKS = {"wpt-train": "wpt_worker.training", "wpt-eval": "wpt_worker.evaluation"}  # spec["task"]: the module whose run(spec, link, work) is the job


class NoTokenizer(importlib.abc.MetaPathFinder):
    """Refuses to import wpt_tokenizer."""

    def find_spec(self, name, path, target=None):
        if name == TOKENIZER or name.startswith(TOKENIZER + "."):
            raise ModuleNotFoundError(f"No module named {name!r}: the gpu worker has no tokenizer; it deals only in "
                                      "token ids, which the wpt manager sends", name=name)
        return None


class Relay(io.TextIOBase):
    """A text stream whose writes the manager prints (its standard output
    or error output); when the link is gone, they go to this process's own."""

    def __init__(self, link, stream, fallback):
        self.link, self.stream, self.fallback = link, stream, fallback

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, text):
        if text:
            try:
                self.link.tell("print", stream=self.stream, text=text)
            except LinkError:
                self.fallback.write(text)
        return len(text)

    def reconfigure(self, **_):
        pass


def connect_socket(environ, hello) -> socket.socket:
    """A socket to the worker's server, on this machine, greeted with hello
    (its "kind" and more) and the job's key, and welcomed. LinkError when
    the server refuses it."""
    host, port = environ["WPT_WORKER_JOB"].rsplit(":", 1)
    sock = socket.create_connection((host, int(port)))
    send(sock, {**hello, "key": environ["WPT_WORKER_JOB_KEY"]})
    header, _ = receive(sock)
    if header.get("kind") != "welcome":
        sock.close()
        raise LinkError(f"the gpu worker's server refused this process: {header.get('message', '')}")
    return sock


def connect(environ=None) -> Connection:
    """This process's link to the manager, through the worker's server."""
    environ = os.environ if environ is None else environ
    return Connection(connect_socket(environ, {"kind": "hello"}), name="job")


def remote(error: RemoteError):
    """The exception of this side that a RemoteError stands for: the
    store's own errors and RecordError as themselves (so a store call fails
    here as it would on the manager), the rest as the RemoteError."""
    from wpt_store import CorruptObject, MissingObject
    from wpt_telemetry import RecordError

    known = {"MissingObject": MissingObject, "CorruptObject": CorruptObject, "RecordError": RecordError}
    found = known.get(error.kind)
    return found(error.message) if found is not None else error


def call(link, kind, body=b"", **fields):
    """link.call, a RemoteError raised as remote() says."""
    try:
        return link.call(kind, body, **fields)
    except RemoteError as error:
        raise remote(error) from None


def download(link, offer, target: Path) -> Path:
    """The file the manager offers ({"file": its id, "size", "sha256"}) into
    target, part by part, checked against its sha256 before it takes
    target's name. CorruptObject when it differs."""
    from wpt_store import CorruptObject

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    try:
        with open(partial, "wb") as out:
            for start in range(0, offer["size"], PART):
                _, data = call(link, "file.range", file=offer["file"], start=start,
                               length=min(PART, offer["size"] - start))
                digest.update(data)
                out.write(data)
        if digest.hexdigest() != offer["sha256"]:
            raise CorruptObject(f"{offer['file']} arrived as {digest.hexdigest()}, not {offer['sha256']}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def upload(link, path) -> str:
    """Send a local file to the manager, part by part; the id the manager
    gave it, which a later request names."""
    header, _ = call(link, "upload.start")
    upload_id = header["upload"]
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while data := file.read(PART):
            digest.update(data)
            call(link, "upload.part", data, upload=upload_id)
    call(link, "upload.end", upload=upload_id, sha256=digest.hexdigest())
    return upload_id


class LinkStore:
    """The manager's store, as a job's process reads it (what
    wpt_store.shards, the environments and wpt_telemetry call): objects by
    their sha256, which only the manager holds and gives part by part; the
    record folder of the run, whose every write goes to the manager's store
    before the call returns, and whose lines that store holds back are sent
    when flush_records() returns. Copies of objects go into folder/objects
    (path(), until release()), files derived from them into folder/scratch.
    The manager answers only for the objects of the run's dataset and
    states, and only for its record folder."""

    def __init__(self, link, folder):
        self.link = link
        self.cache, self.scratch = Path(folder) / "objects", Path(folder) / "scratch"

    def __repr__(self):
        return "the wpt manager's store"

    def path(self, sha) -> Path:
        local = self.cache / sha[:2] / sha
        if not local.is_file():
            header, _ = call(self.link, "object.offer", sha=sha)
            download(self.link, header, local)
        return local

    def get_bytes(self, sha) -> bytes:
        return self.path(sha).read_bytes()

    def release(self, sha):
        (self.cache / sha[:2] / sha).unlink(missing_ok=True)

    def read_record(self, folder, name) -> bytes:
        return call(self.link, "record.read", folder=folder, name=name)[1]

    def write_record(self, folder, name, data):
        call(self.link, "record.write", data, folder=folder, name=name)

    def append_record(self, folder, name, data):
        call(self.link, "record.append", data, folder=folder, name=name)

    def flush_records(self):
        call(self.link, "record.flush")


def main():
    sys.meta_path.insert(0, NoTokenizer())
    link = connect()
    sys.stdout = Relay(link, "out", sys.__stdout__)
    sys.stderr = Relay(link, "err", sys.__stderr__)
    spec = json.loads(Path(os.environ["WPT_WORKER_SPEC"]).read_text(encoding="utf-8"))
    module = __import__(TASKS[spec["task"]], fromlist=["run"])
    code = module.run(spec, link, Path(os.environ["WPT_WORKER_WORK"]))
    sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
