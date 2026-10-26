"""The wpt manager's side of the link to its gpu workers (docs/machines.md).

    start_workers(environ)   the workers the settings name: each key of WPT_GPU_WORKERS, or,
                             with none, a local worker this process starts on this machine
                             (python -m wpt_worker --local), reached over the same link
    machines(keys)           where each of a run's workers stands (its number, how many there
                             are, how they exchange gradients), for its job's spec
    run_jobs(workers, ...)   every worker's job at once, each link served on a thread of its own
    worker.send_checkout(left_out)   the manager's own checkout, uncommitted changes included,
                             its data folder (left_out, which the service layer names) left out
                             (checkout())
    worker.run(spec, ...)    one worker's job: started from that checkout, its requests answered
                             by the handlers given (on this thread), until it ends; its exit code
    Files                    what the manager offers the job (file.range) and receives from it
                             (upload.start, upload.part, upload.end)

A local worker gets this process's environment without WPT_STORE and the
store's credentials (AWS_*): a gpu worker never holds them, wherever it
runs. What the job prints, the manager prints. Ctrl+C or SIGTERM to the
manager is passed on to the job, which finishes its update, saves, and
stops; a second one again, which stops it at once. On several workers, a
job that fails or a link that breaks stops the other workers' jobs, which
would wait for it.

How a run's workers exchange gradients (machines()): one worker trains
alone; workers in one place (each key gives the same place, or no key
gives one and every worker's address is private) meet through NCCL at
worker 0's private address (its key's private, else its address), on the
port after its own, and each worker's job gets its own private address,
the one its NCCL traffic keeps to; workers in different places exchange
gradients over their TLS links (wpt_worker.peers): each worker's job gets
worker 0's address and certificate pin, with a secret made for the run in
place of worker 0's token, which never leaves the manager. Worker 0's
address must not be this machine's own (127.0.0.1) while another worker's
is not: that worker would reach itself.
"""

import hashlib
import io
import itertools
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
from compression import zstd
from pathlib import Path

import wpt_worker
from wpt_utils.programs import MissingTool, program

from .link import PART, Connection, Key, LinkError, dial, is_loopback, is_private
from .training import LINK, NCCL, ONE_MACHINE

WORKERS = "WPT_GPU_WORKERS"
SECRETS = re.compile(r"WPT_STORE|AWS_.*")  # what a local worker's environment never holds


class NoCheckout(Exception):
    """The manager does not run from a git checkout of the repository, so it
    has no code to send its gpu worker."""


def keys(environ) -> list[Key]:
    """The gpu workers WPT_GPU_WORKERS lists (keys separated by spaces or
    commas); ValueError for one that is not a key, and for a number of
    workers to rent, which wpt-manager rents."""
    found = [text for text in re.split(r"[\s,]+", environ.get(WORKERS, "")) if text]
    if len(found) == 1 and found[0].isdigit():
        raise ValueError(f"{WORKERS}={found[0]}: a number of gpu workers to rent is wpt-manager's setting, which "
                         "rents them and gives their keys; here, list the workers' keys")
    return [Key.parse(text) for text in found]


def machines(found: list[Key]) -> list[dict]:
    """Where each of the run's gpu workers (their keys, in order) stands:
    spec["machines"] of its job (wpt_worker.training.placement); see the
    module's docstring. ValueError when worker 0's address is this
    machine's own and another worker's is not."""
    count = len(found)
    if count < 2:
        return [ONE_MACHINE]
    places = {key.place for key in found}
    first = found[0]
    if (None not in places and len(places) == 1) or (places == {None} and all(is_private(key.host)
                                                                              for key in found)):
        privates = [key.private or key.host for key in found]
        _reached(privates, "private address", "private=")
        return [{"index": index, "count": count, "transport": NCCL, "master": privates[0],
                 "port": first.port + 1, "private": privates[index]} for index in range(count)]
    _reached([key.host for key in found], "address", "the address")
    secret = secrets.token_urlsafe(32)
    hub = str(Key(first.host, first.port, first.pin, secret))
    return [{"index": index, "count": count, "transport": LINK, "secret": secret, **({"hub": hub} if index else {})}
            for index in range(count)]


def _reached(hosts, what, fix):
    """ValueError when hosts[0], what every other worker reaches gpu worker
    0 at, is this machine's own while another worker's is not."""
    if is_loopback(hosts[0]) and not all(is_loopback(host) for host in hosts):
        raise ValueError(f"gpu worker 0's {what} is {hosts[0]}, this machine's own, but other gpu workers run "
                         f"elsewhere and would reach themselves there: give gpu worker 0's key {fix} the address "
                         "the other workers reach it at")


def start_workers(environ=None, log=print) -> list["Worker"]:
    """The workers the settings (environ) name, each reached; with none, a
    local worker. ValueError for a key that is not one; LinkError for a
    worker that cannot be reached or refuses the link (the ones reached
    before it are closed)."""
    environ = os.environ if environ is None else environ
    found = keys(environ)
    if not found:
        return [Worker.local_worker(os.environ)]  # this process's own environment, whatever else the settings hold
    workers = []
    try:
        for key in found:
            log(f"gpu worker: {key.shown()}")
            workers.append(Worker(key))
    except BaseException:
        for worker in workers:
            worker.close()
        raise
    return workers


def run_jobs(workers, specs, handlers) -> int:
    """Start the job of each spec on its worker and answer their requests
    with handlers(worker) ({kind: handler(header, body)}) until every job
    ends; the exit code of the first that failed, else 0. A job that fails
    stops the others; a link that breaks stops the others, then raises
    LinkError, naming its worker. A stop signal meanwhile is passed on to
    every job."""
    for worker in workers:
        worker.exit = None
        worker.link.handlers.update(handlers(worker))
    started = []
    try:
        for worker, spec in zip(workers, specs):
            worker.link.call("job.start", json.dumps(spec).encode())
            started.append(worker)
    except LinkError as error:
        for worker in started:
            worker.stop()
        raise LinkError(f"{worker}: {error}") from None
    failed, broken = [], []

    def serve(worker):
        worker.link.serving = threading.current_thread()
        try:
            worker.link.serve(until=lambda: worker.exit is not None)
        except LinkError as error:
            broken.append(LinkError(f"{worker}: {error}"))
        else:
            if worker.exit == 0:
                return
            failed.append(worker.exit)
        for other in workers:
            if other is not worker:
                other.stop()

    with _Passed(workers):
        if len(workers) == 1:
            serve(workers[0])
        else:
            threads = [threading.Thread(target=serve, args=(worker,), name=f"serve-{index}", daemon=True)
                       for index, worker in enumerate(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                while thread.is_alive():  # in steps, so a signal reaches this thread's handlers
                    thread.join(0.2)
    if broken:
        raise broken[0]
    return failed[0] if failed else 0


def code_root() -> Path:
    """The root of the checkout this code runs from: the repository whose
    packages/wpt-worker holds this package."""
    root = Path(wpt_worker.__file__).resolve().parents[4]
    if not (root / "packages" / "wpt-worker" / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise NoCheckout(f"{Path(wpt_worker.__file__).parent} is not inside a checkout of the WPT repository: the "
                         "wpt manager runs from one, which it sends to its gpu worker")
    return root


def checkout(root, left_out: str) -> bytes:
    """The checkout at root as a zstd tar: every file git tracks and every
    file it does not ignore, as the working tree holds it now (uncommitted
    changes included, deleted files left out), but those under the folder
    left_out (a path from root, with /), which the job gets over the link;
    in path order, with no dates or owners, so the same tree gives the same
    bytes."""
    prefix = left_out.strip("/") + "/"
    try:
        listed = subprocess.run([program("git"), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                cwd=root, capture_output=True, check=True).stdout
    except (MissingTool, subprocess.CalledProcessError) as error:
        raise NoCheckout(f"{root}: git cannot list the checkout's files: {error}") from None
    paths = sorted({path for path in listed.decode("utf-8", "surrogateescape").split("\0")
                    if path and not path.startswith(prefix)})
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            local = Path(root) / path
            if not local.is_symlink() and not local.is_file():
                continue  # deleted from the working tree, or a submodule's folder
            info = archive.gettarinfo(str(local), arcname=path)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o755 if info.mode & 0o111 else 0o644
            if info.isreg():
                with open(local, "rb") as file:
                    archive.addfile(info, file)
            else:
                archive.addfile(info)
    return zstd.compress(data.getvalue())


class Files:
    """The files the manager offers its job (offer: an id the job reads
    through file.range, part by part), and those the job sends it
    (upload.*: each into a file of folder, complete once its sha256
    matches; received(id) gives its path, which the caller then owns)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.offered, self.uploads, self.done = {}, {}, {}
        self._numbers = itertools.count(1)

    def offer(self, path, sha256=None) -> dict:
        number = f"f{next(self._numbers)}"
        path = Path(path)
        self.offered[number] = path
        if sha256 is None:
            from wpt_utils.files import sha256_file
            sha256 = sha256_file(path)
        return {"file": number, "size": path.stat().st_size, "sha256": sha256}

    def handlers(self) -> dict:
        return {"file.range": self._range, "upload.start": self._start, "upload.part": self._part,
                "upload.end": self._end}

    def received(self, upload_id) -> Path:
        if upload_id not in self.done:
            raise ValueError(f"no upload {upload_id} arrived whole")
        return self.done.pop(upload_id)

    def _range(self, header, body):
        path = self.offered.get(header["file"])
        if path is None:
            raise ValueError(f"no file {header['file']} is offered")
        with open(path, "rb") as file:
            file.seek(header["start"])
            return {}, file.read(min(header["length"], PART))

    def _start(self, header, body):
        self.folder.mkdir(parents=True, exist_ok=True)
        number = f"u{next(self._numbers)}"
        fd, name = tempfile.mkstemp(dir=self.folder, prefix=f".upload-{number}-")
        os.close(fd)
        self.uploads[number] = (Path(name), hashlib.sha256())
        return {"upload": number}, b""

    def _part(self, header, body):
        path, digest = self.uploads[header["upload"]]
        with open(path, "ab") as out:
            out.write(body)
        digest.update(body)

    def _end(self, header, body):
        path, digest = self.uploads.pop(header["upload"])
        if digest.hexdigest() != header["sha256"]:
            path.unlink(missing_ok=True)
            raise ValueError(f"upload {header['upload']} did not arrive whole: its sha256 differs")
        self.done[header["upload"]] = path

    def close(self):
        for path, _ in self.uploads.values():
            path.unlink(missing_ok=True)
        for path in self.done.values():
            path.unlink(missing_ok=True)
        self.uploads, self.done = {}, {}


class Worker:
    """The manager's link to one gpu worker; see the module's docstring."""

    def __init__(self, key, process=None, local=False):
        self.key, self.process, self.local = key, process, local
        self.exit = None
        self.link = Connection(dial(key), notices={"job.exit": self._exited, "print": self._print},
                               name="manager")

    def __str__(self):
        return "a local gpu worker" if self.local else f"the gpu worker at {self.key.host}:{self.key.port}"

    @classmethod
    def local_worker(cls, environ) -> "Worker":
        clean = {name: value for name, value in environ.items() if not SECRETS.fullmatch(name)}
        # in a process group of its own: a Ctrl+C reaches the manager only, which passes it on to the job
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"process_group": 0})
        process = subprocess.Popen([sys.executable, "-m", "wpt_worker", "--local"], env=clean, stdout=subprocess.PIPE,
                                   text=True, encoding="utf-8", **group)
        line = process.stdout.readline()
        try:
            key = Key.parse(line)
        except ValueError:
            process.kill()
            process.wait()
            raise LinkError(f"the local gpu worker did not start (exit code {process.returncode})") from None
        try:
            return cls(key, process, local=True)
        except BaseException:
            process.kill()
            process.wait()
            raise

    def send_checkout(self, left_out: str, root=None) -> int:
        """Send the checkout at root (default: the one this code runs from),
        left_out left out (checkout()); the number of its files."""
        data = checkout(root or code_root(), left_out)
        for index, start in enumerate(range(0, max(len(data), 1), PART)):
            self.link.call("checkout.part", data[start:start + PART], first=index == 0)
        header, _ = self.link.call("checkout.end", sha256=hashlib.sha256(data).hexdigest())
        return header["files"]

    def run(self, spec: dict, handlers: dict) -> int:
        """Start the job of spec (JSON) and answer its requests with
        handlers ({kind: handler(header, body)}, on this thread) until it
        ends; its exit code (run_jobs, for this worker alone)."""
        return run_jobs([self], [spec], lambda worker: handlers)

    def call(self, channel, kind, body=b"", **fields):
        """A request to one process of the job."""
        return self.link.call(kind, body, channel=channel, **fields)

    def stop(self):
        """Ask the job to stop: its processes get SIGTERM."""
        try:
            self.link.tell("job.stop")
        except LinkError:
            pass

    def close(self):
        self.link.close()
        if self.process is not None:  # a local worker ends once its manager is gone
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _exited(self, header, body):
        self.exit = header["code"]

    def _print(self, header, body):
        stream = sys.stderr if header.get("stream") == "err" else sys.stdout
        stream.write(header["text"])
        stream.flush()


class _Passed:
    """While active, a stop signal to the manager is passed on to every
    worker's job (Worker.stop), each time; signals reach Python's main
    thread only."""

    NAMES = ("SIGINT", "SIGTERM", "SIGBREAK")

    def __init__(self, workers):
        self.workers, self.previous = workers, {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in self.NAMES:
                number = getattr(signal, name, None)
                if number is not None and signal.getsignal(number) is not signal.SIG_IGN:
                    self.previous[number] = signal.signal(number, lambda *_: self._stop())
        return self

    def _stop(self):
        for worker in self.workers:
            worker.stop()

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        return False
