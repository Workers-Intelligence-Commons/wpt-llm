"""One fresh Docker container: for one attempt, or for scoring one.

The container has no network (--network none) and a read-only root. It
can write only /work, its working folder, and /tmp: two folders kept in
memory, each holding at most `disk` MiB. Its memory (those folders' files
included), processes and CPU time are limited too (Limits). Its first
process is docker's init (--init), which collects every finished process,
so background commands that ended never count against the process limit.
Its commands run as `user`; it starts with the given files in /work, which
root unpacks with that user as their owner, and is removed, with every
process in it, when the sandbox closes. A sandbox whose user is not root
(an attempt's) keeps only the capabilities root needs to unpack files with
their owners (CHOWN, DAC_OVERRIDE, FOWNER), which a non-root user's
commands do not hold, and nothing in it can gain privileges (no setuid);
the check's, as root, keeps docker's default capabilities.

Every command is started by a shell that first prints STARTED, so a run
tells docker's own failure to start it (no such container, a container
that is not running, a refusal of the daemon or of the runtime) from the
command's exit code, and docker's text never passes for the command's
output. When docker cannot start a command because the model's own
processes or files filled the container (its process limit, or its
memory), run() raises SandboxFull; for any other cause, SandboxError.
Memory that runs out may also end the container's first process, and so
the container, whose /work is then lost: it counts as full too, and
freeze() accepts it.

Every command runs under `timeout`, so it stops after its seconds even when
it ignores signals; it counts as stopped only when the host saw it run that
long and timeout's codes came back (124 after TERM, 137 after KILL), never
from its exit code alone. Its output is read as it comes and only its start and
its end are kept, so a command that prints without end never fills the
host's memory. Docker is started through docker.py.
"""

import io
import posixpath
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass
from typing import NamedTuple

from wpt_dataset.manifest import EXECUTABLE, SYMLINK

from . import docker
from .docker import DockerFailed

WORK = "/work"
AGENT, ROOT = "1000:1000", "0:0"  # user:group of the model's commands, and of the check's
GRACE = 5  # seconds between timeout's TERM and its KILL
WAIT = 30  # seconds docker itself may take beyond a command's own limit
STOPPED = {124, 137}  # timeout's exit codes when it stopped the command: by TERM, or by KILL after GRACE
CHUNK = 1 << 16
ERRORS = 4096  # bytes kept of what docker cp prints on its standard error
STARTED = b"+"  # the first byte of every command's output, printed before the command runs, then left out
LAUNCH = f'printf {STARTED.decode()}; exec "$@"'
SPAWN = 16  # free processes and threads docker's runtime needs to start a command (about 6), with room to spare
# the only capabilities an attempt's container keeps: root's, to unpack files with their owners
CONFINED = ["--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "DAC_OVERRIDE", "--cap-add", "FOWNER"]


class SandboxError(Exception):
    """Docker is missing, or refused to create, fill, run in, copy from or
    remove a container. The message names the container when one was left
    behind."""


class SandboxFull(Exception):
    """Docker could not start a command because the processes or files the
    sandbox's own commands left fill it: its process limit, or its memory
    (the files in /work and /tmp included). What /work holds can still be
    copied: all of it, or nothing when running out of memory stopped the
    container."""


@dataclass(frozen=True)
class Limits:
    memory: int  # MiB, the files in /work and /tmp included
    disk: int  # MiB that /work and /tmp may each hold
    processes: int  # processes and threads at once
    cpus: float  # CPUs' worth of time


class Ran(NamedTuple):
    code: int
    stdout: bytes  # its start and end, with a line between saying how much was cut
    stderr: bytes  # the same; empty when the command ran with merged output
    timed_out: bool  # ran its whole time on the host and timeout stopped it
    cut: int  # bytes left out of stdout


class Sandbox:
    """A running container of `image` holding `files` ({path under /work:
    (manifest mode, content)}), whose commands run as `user`. Use it in a
    with block: leaving it removes the container."""

    def __init__(self, image: str, files: dict[str, tuple[str, bytes]], *, user: str, limits: Limits):
        self.image, self.user, self.limits, self.container = image, user, limits, None
        confined = [*CONFINED, "--security-opt", "no-new-privileges"] if user != ROOT else []
        created = self._docker(
            "create", "--init", "--network", "none", "--read-only", "--workdir", WORK, *confined,
            "--mount", f"type=volume,dst={WORK},volume-driver=local,volume-opt=type=tmpfs,volume-opt=device=tmpfs,"
                       f"volume-opt=o=size={limits.disk}m",
            "--tmpfs", f"/tmp:rw,exec,size={limits.disk}m",
            "--memory", f"{limits.memory}m", "--memory-swap", f"{limits.memory}m",
            "--pids-limit", str(limits.processes), "--cpus", str(limits.cpus),
            image, "sleep", "infinity")
        self.container = created.decode().strip()
        try:  # started first: the folders in memory exist only in a running container
            self._docker("start", self.container)
            work = {posixpath.join(WORK, path): entry for path, entry in files.items()}
            self._unpack(work, owner=user, folders={WORK.strip("/")} | {f for path in work for f in _parents(path)})
        except BaseException as error:  # a failing close is added to it as a note, as in __exit__
            try:
                self.close()
            except SandboxError as failed:
                error.add_note(str(failed))
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        """Close; when an exception is already leaving the block, a failing
        close is added to it as a note instead of replacing it."""
        try:
            self.close()
        except SandboxError as failed:
            if error is None:
                raise
            error.add_note(str(failed))

    def run(self, argv: list[str], *, seconds: float, limit: int, stdin: bytes = b"", merged: bool = False) -> Ran:
        """Run argv in /work as the sandbox's user; with merged, stderr goes
        into stdout, in the order the command wrote them. Of an output
        longer than limit bytes, the first and last limit / 2 are kept.
        SandboxFull when docker could not start it in a container the
        sandbox's commands filled; SandboxError for docker's other
        failures."""
        started = time.monotonic()
        process = self._start("exec", "-i", "-u", self.user, self.container, "sh", "-c", LAUNCH, "sh",
                              "timeout", "-k", str(GRACE), str(seconds), *argv,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT if merged else subprocess.PIPE)
        outputs = [_Kept(limit), _Kept(limit)]
        threads = [threading.Thread(target=_feed, args=(process.stdin, stdin), daemon=True),
                   threading.Thread(target=_drain, args=(process.stdout, outputs[0], STARTED), daemon=True)]
        if not merged:
            threads.append(threading.Thread(target=_drain, args=(process.stderr, outputs[1]), daemon=True))
        for thread in threads:
            thread.start()
        try:
            code = process.wait(timeout=seconds + GRACE + WAIT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise SandboxError(f"container {self.container}: docker exec did not return") from None
        finally:
            for thread in threads:
                thread.join()
        if not outputs[0].opened:  # docker printed its own error instead of starting the command
            error = docker.failed("exec", code, outputs[0].bytes() + outputs[1].bytes())
            if self._full():
                raise SandboxFull(f"container {self.container} is full: {error}")
            raise SandboxError(f"container {self.container}: {error}")
        stopped = code in STOPPED and time.monotonic() - started >= seconds
        return Ran(code, outputs[0].bytes(), outputs[1].bytes(), stopped, outputs[0].cut)

    def put(self, path: str, content: bytes, mode: str):
        """Write one file (an absolute path under /work or /tmp), owned by
        root, into the container. A missing folder above it is created,
        owned by root; an existing one keeps its owner and mode."""
        self._unpack({path: (mode, content)}, owner=ROOT, folders=set())

    def freeze(self):
        """Stop every process in the container where it stands; they stay
        stopped until the container is removed. A container that running
        out of memory stopped has none left; one stopped otherwise is a
        SandboxError."""
        running, oom = self._state()
        if running:
            self._docker("pause", self.container)
        elif not oom:
            raise SandboxError(f"container {self.container} is not running")

    def copy_work(self, into: "Sandbox"):
        """Copy /work, as it is now, into another sandbox's /work; freeze
        first, so no process changes it on the way."""
        reading = self._start("cp", f"{self.container}:{WORK}", "-", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        read = _Kept(ERRORS)
        drain = threading.Thread(target=_drain, args=(reading.stderr, read), daemon=True)  # a full pipe would block it
        drain.start()
        try:
            writing = self._start("exec", "-i", "-u", ROOT, into.container, "tar", "-x", "-f", "-", "-C", "/",
                                  stdin=reading.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except SandboxError:  # nothing reads the archive: stop the copy, so no process is left behind
            reading.kill()
            reading.wait()
            drain.join()
            raise
        finally:
            reading.stdout.close()  # only `writing` reads it now
        written = writing.communicate()[1]
        reading.wait()
        drain.join()
        for process, errors in ((reading, read.bytes()), (writing, written)):
            if process.returncode:
                raise SandboxError(f"copying /work from {self.container} into {into.container}: "
                                   f"{errors.decode(errors='replace').strip()}")

    def close(self):
        """Remove the container and its processes; SandboxError when docker refuses."""
        if self.container:
            container, self.container = self.container, None
            try:
                self._docker("rm", "-f", "-v", container)
            except SandboxError as error:
                raise SandboxError(f"container {container} is left behind: {error}") from None

    def _full(self) -> bool:
        """Whether the container is full: memory ran out in it (docker
        records an OOM kill), or it runs and its processes and threads
        leave fewer than SPAWN free."""
        try:
            running, oom = self._state()
            if oom or not running:
                return oom
            return int(docker.run("stats", "--no-stream", "--format", "{{.PIDs}}", self.container)) + SPAWN \
                >= self.limits.processes
        except (SandboxError, DockerFailed, ValueError):
            return False

    def _state(self) -> tuple[bool, bool]:
        """(whether the container runs, whether memory ran out in it)."""
        running, oom = self._docker("inspect", "-f", "{{.State.Running}} {{.State.OOMKilled}}",
                                    self.container).split()
        return running == b"true", oom == b"true"

    def _unpack(self, files, owner, folders):
        """Copy files given by absolute path into the container, with
        `folders` (paths without the leading /, mode 755), through one tar
        stream that root unpacks: every entry belongs to owner, and a listed
        folder that exists takes that owner and mode. A missing folder that
        is not listed is created owned by root; one that exists is left as
        it is."""
        uid, gid = map(int, owner.split(":"))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for folder in sorted(folders):
                info = tarfile.TarInfo(folder)
                info.type, info.mode, info.uid, info.gid = tarfile.DIRTYPE, 0o755, uid, gid
                archive.addfile(info)
            for path, (mode, content) in sorted(files.items()):
                info = tarfile.TarInfo(path.lstrip("/"))
                info.uid, info.gid = uid, gid
                if mode == SYMLINK:
                    info.type, info.linkname = tarfile.SYMTYPE, content.decode()
                    archive.addfile(info)
                else:
                    info.size, info.mode = len(content), 0o755 if mode == EXECUTABLE else 0o644
                    archive.addfile(info, io.BytesIO(content))
        self._docker("exec", "-i", "-u", ROOT, self.container, "tar", "-x", "-f", "-", "-C", "/",
                     stdin=buffer.getvalue())

    def _docker(self, *args, stdin=b"") -> bytes:
        try:
            return docker.run(*args, stdin=stdin)
        except DockerFailed as error:
            raise SandboxError(str(error)) from None

    def _start(self, *args, **streams):
        try:
            return docker.start(*args, **streams)
        except DockerFailed as error:
            raise SandboxError(str(error)) from None


class _Kept:
    """The first and last limit / 2 bytes of a stream, and how many bytes
    between them were left out."""

    def __init__(self, limit):
        self.head, self.tail = limit - limit // 2, limit // 2
        self.start, self.end, self.total = bytearray(), bytearray(), 0
        self.opened = True  # the stream began with the bytes _drain was told to leave out

    def add(self, chunk):
        self.total += len(chunk)
        room = self.head - len(self.start)
        if room > 0:
            self.start += chunk[:room]
            chunk = chunk[room:]
        self.end += chunk
        if len(self.end) > self.tail:
            del self.end[:len(self.end) - self.tail]

    @property
    def cut(self):
        return self.total - len(self.start) - len(self.end)

    def bytes(self):
        middle = f"\n[cut: {self.cut} bytes]\n".encode() if self.cut else b""
        return bytes(self.start) + middle + bytes(self.end)


def _feed(stream, data):
    try:
        stream.write(data)
        stream.close()
    except (BrokenPipeError, OSError):  # the command stopped reading
        pass


def _drain(stream, kept, opening=b""):
    """Keep what stream gives until it closes; when it begins with opening,
    those bytes are left out, and kept.opened says whether it did."""
    if opening:
        first = stream.read(len(opening))
        kept.opened = first == opening
        if not kept.opened:
            kept.add(first)
    while chunk := stream.read1(CHUNK):
        kept.add(chunk)
    stream.close()


def _parents(path):
    """The folders above an absolute path, without the root, outermost first."""
    parts = path.strip("/").split("/")[:-1]
    return ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
