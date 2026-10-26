"""The gpu worker: a process that does only GPU work for one wpt manager at
a time, on the port it opens (docs/machines.md).

    python -m wpt_worker            a worker, as its container starts it
    python -m wpt_worker --local    the manager's own local worker: this machine only, any free port,
                                    a new token, one manager, then it ends; its folder is
                                    WPT_WORKER_FOLDER when set, else a temporary one it removes

It prints its key, wpt-worker://TOKEN@HOST:PORT#PIN, on a line of its own:
what the manager needs to reach it (WPT_GPU_WORKERS). A token set for it
(WPT_WORKER_TOKEN, as the manager sets one for a worker it rents) is
printed as its first 4 characters and stars, never whole. It makes its own
certificate (certificate.py), whose sha256 is the key's PIN, and refuses
any connection whose first frame does not carry the token (link.py).

Settings, from the environment:

    WPT_WORKER_ADDRESS  the address the port listens on (default 0.0.0.0, every address)
    WPT_WORKER_PORT     the port (default 8470)
    WPT_WORKER_TOKEN    the secret token; unset, the worker makes one and prints it in its key
                        (a token set here is never printed whole)
    WPT_WORKER_FOLDER   where it keeps its certificate, the manager's checkout and its jobs'
                        files (default: wpt-worker in the current folder)
    WPT_WORKER_WAIT     seconds it waits with no manager connected, then ends (default 3600: 1 hour);
                        a rented machine is stopped once its worker ends
    WPT_WORKER_INSTALL  1: install the libraries the checkout's uv.lock names for its job with
                        uv, into the folder's own environment (what its container does);
                        otherwise its job runs on the Python that runs the worker
    WPT_WORKER_TORCH    with WPT_WORKER_INSTALL, the PyTorch build: cuda (default), rocm, xpu or cpu
    WPT_WORKER_PLACE    the place the machine is in (one provider's cluster), and
    WPT_WORKER_PRIVATE  its private address there: both go into the key it prints (?place=
                        and &private=), which workers of one place use to meet through NCCL

One manager at a time: a second one is refused while the first is
connected. What the manager asks (link.py frames without a channel):

    checkout.part, checkout.end   its checkout (data/ left out), a zstd tar in parts, unpacked
                                  into FOLDER/checkout/ once its sha256 matches
    job.start                     run its job (the spec, JSON) from that checkout:
                                  python -m wpt_worker.job, in a process group of its own
    job.stop                      SIGTERM to the job's processes (again: again)

Every process of the job connects to this server, on this machine only
(127.0.0.1, with a key of the job's own), and gets a channel: the frames
between it and the manager go through here, its channel added and taken
off. The server tells the manager when a channel closes (channel.closed)
and when the job ends (job.exit, with its exit code). A manager that
disconnects stops its job.

When a run spans gpu workers in different places, their jobs exchange
gradients over this same port (wpt_worker.peers): the job of another
worker of the run sends, in its first frame, "peer": its worker's number,
and the run's own secret, which the manager made for that run and put in
every worker's job spec (spec["machines"]["secret"]), never this worker's
token; while this worker runs that run's job, the server keeps that
connection until a process of the job asks for it (peer.accept on the
job's own socket), then joins the two and copies the bytes both ways
(PEER_WAIT seconds at most for it to come). Every connection to the port
notices a machine that is gone without closing it (link.keep_alive).
"""

import argparse
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from compression import zstd
from pathlib import Path

from . import certificate
from .link import PROTOCOL, Key, LinkError, keep_alive, pin, receive, send, server_context

PORT = 8470
WAIT = 3600.0
GRACE = 30.0  # seconds the job gets to stop once its manager is gone, before it is killed
PEER_WAIT = 900.0  # seconds a job waits for another gpu worker's job to reach it (its libraries may still install)
TORCH = ("cuda", "rocm", "xpu", "cpu")
SOURCES = ("packages", "apps")  # the checkout's folders whose members' src/ folders the job imports from


class Settings:
    def __init__(self, environ, local=False):
        self.local = local
        self.address = "127.0.0.1" if local else environ.get("WPT_WORKER_ADDRESS") or "0.0.0.0"
        self.port = 0 if local else int(environ.get("WPT_WORKER_PORT") or PORT)
        self.given = bool(environ.get("WPT_WORKER_TOKEN"))  # a token set for it, by a person or the manager
        self.token = environ.get("WPT_WORKER_TOKEN") or secrets.token_urlsafe(32)
        self.folder = Path(environ.get("WPT_WORKER_FOLDER") or "wpt-worker")
        self.wait = float(environ.get("WPT_WORKER_WAIT") or WAIT)
        self.install = environ.get("WPT_WORKER_INSTALL", "") in ("1", "true", "yes")
        self.place = environ.get("WPT_WORKER_PLACE") or None
        self.private = environ.get("WPT_WORKER_PRIVATE") or None
        self.torch = environ.get("WPT_WORKER_TORCH") or "cuda"
        if self.torch not in TORCH:
            raise ValueError(f"WPT_WORKER_TORCH={self.torch!r}: one of {', '.join(TORCH)}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wpt_worker", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="the manager's own worker on this machine")
    args = parser.parse_args(argv)
    environ = dict(os.environ)
    temporary = args.local and not environ.get("WPT_WORKER_FOLDER")
    if temporary:  # its folder is its own, gone when it ends
        environ["WPT_WORKER_FOLDER"] = tempfile.mkdtemp(prefix="wpt-worker-")
    try:
        settings = Settings(environ, local=args.local)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        return Server(settings).run()
    finally:
        if temporary:
            shutil.rmtree(settings.folder, ignore_errors=True)


class Server:
    def __init__(self, settings):
        self.settings = settings
        self.folder = settings.folder.absolute()
        cert, key, der = certificate.ensure(self.folder / "tls")
        self.context = server_context(cert, key)
        self.pin = pin(der)
        self.session = None
        self.lock = threading.Lock()
        self.peers, self.peer_arrived = {}, threading.Condition()  # gpu worker number -> its job's TLS socket

    def run(self) -> int:
        listener = socket.create_server((self.settings.address, self.settings.port))
        listener.settimeout(1.0)
        host, port = listener.getsockname()[:2]
        shown = "127.0.0.1" if host == "0.0.0.0" else host
        key = Key(shown, port, self.pin, self.settings.token, self.settings.place, self.settings.private)
        # a token set for it is never printed whole: who set it knows it
        print(key.shown() if self.settings.given else key, flush=True)
        if not self.settings.local:
            print(f"gpu worker listening on {host}:{port}: give the line above to the wpt manager (WPT_GPU_WORKERS), "
                  f"with this machine's address in place of {shown} when it runs elsewhere"
                  + (", and the token WPT_WORKER_TOKEN holds in place of the stars" if self.settings.given else "")
                  + f"; it ends after {self.settings.wait:.0f} s with no manager connected", flush=True)
        idle_since = time.monotonic()
        try:
            while True:
                with self.lock:
                    session = self.session
                if session is not None and session.alive:
                    idle_since = time.monotonic()
                elif self.settings.local and session is not None:  # its one manager is gone
                    return 0
                elif time.monotonic() - idle_since > self.settings.wait:
                    print(f"no manager for {self.settings.wait:.0f} s: the gpu worker ends", flush=True)
                    return 0
                try:
                    raw, _ = listener.accept()
                except OSError:  # TimeoutError included: measure again
                    continue
                threading.Thread(target=self._greet, args=(raw,), daemon=True).start()
        finally:
            listener.close()
            with self.lock:
                if self.session is not None:
                    self.session.end()

    def _greet(self, raw):
        """TLS, the token and the protocol version, then a session; a
        second manager is refused while one is connected."""
        try:
            raw.settimeout(30)
            keep_alive(raw)
            sock = self.context.wrap_socket(raw, server_side=True)
            header, _ = receive(sock)
        except (OSError, ssl.SSLError, LinkError):
            raw.close()
            return
        refusal = None
        peer = header.get("peer")
        if header.get("kind") == "hello" and peer is not None:  # another worker's job, with its run's secret
            self._peer(sock, peer, str(header.get("token", "")), header.get("protocol"))
            return
        if header.get("kind") != "hello" or not hmac.compare_digest(str(header.get("token", "")),
                                                                    self.settings.token):
            refusal = "wrong token"
        elif header.get("protocol") != PROTOCOL:
            refusal = f"this worker speaks protocol {PROTOCOL}, the manager {header.get('protocol')}"
        session = None
        with self.lock:
            if refusal is None and self.session is not None and self.session.alive:
                refusal = "another manager is connected"
            if refusal is None:
                session = self.session = Session(self, sock)
        if refusal is not None:
            try:
                send(sock, {"kind": "refused", "message": refusal})
            except OSError:
                pass
            sock.close()
            return
        sock.settimeout(None)
        try:
            send(sock, {"kind": "welcome", "protocol": PROTOCOL})
        except OSError:
            session.end()
            return
        session.run()

    def _peer(self, sock, number, secret, protocol):
        """The job of the run's gpu worker number, which knows the secret of
        the run whose job runs here (spec["machines"]["secret"], never this
        worker's token): kept until this worker's job takes it
        (Session._accept_peer), then answered; refused when no job of a run
        with that secret runs here."""
        with self.lock:
            session = self.session
        refusal = None
        expected = None if session is None else session.peer_secret
        if not isinstance(number, int) or session is None or not session.alive or session.job is None \
                or session.job.poll() is not None or expected is None \
                or not hmac.compare_digest(secret, expected):
            refusal = "no job of that run runs on this gpu worker"
        elif protocol != PROTOCOL:
            refusal = f"this worker speaks protocol {PROTOCOL}, the other {protocol}"
        if refusal is not None:
            try:
                send(sock, {"kind": "refused", "message": refusal})
            except OSError:
                pass
            sock.close()
            return
        sock.settimeout(None)
        with self.peer_arrived:
            previous = self.peers.pop(number, None)
            if previous is not None:  # a later try of the same peer wins
                previous.close()
            self.peers[number] = sock
            self.peer_arrived.notify_all()

    def take_peer(self, number, wait) -> ssl.SSLSocket | None:
        """The TLS socket of the job of gpu worker number, waiting up to wait
        seconds for it; None when it did not come."""
        with self.peer_arrived:
            self.peer_arrived.wait_for(lambda: number in self.peers, timeout=wait)
            return self.peers.pop(number, None)


class Session:
    """One manager's connection: its checkout, its job, the job's channels."""

    def __init__(self, server, sock):
        self.server, self.sock, self.alive = server, sock, True
        self.sending = threading.Lock()
        self.parts = []  # the checkout's parts received so far
        self.job = None  # the job's process
        self.peer_secret = None  # the running job's run secret, which other workers' jobs reach it with
        self.channels, self.next_channel = {}, 1
        self.readers = []  # one thread per process of the job, reading what it sends
        self.job_key = secrets.token_hex(16)
        self.jobs = socket.create_server(("127.0.0.1", 0))  # where the job's processes connect
        threading.Thread(target=self._accept_jobs, daemon=True).start()

    def run(self):
        try:
            while True:
                header, body = receive(self.sock)
                if "channel" in header:
                    self._to_channel(header, body)
                elif "id" in header:
                    self._request(header, body)
                elif header.get("kind") == "job.stop":  # a notice, so a signal handler never waits
                    self._stop_job(wait=False)
        except LinkError:
            pass
        finally:
            self.end()

    def end(self):
        if not self.alive:
            return
        self.alive = False
        self._stop_job(wait=True)
        for channel in list(self.channels.values()):
            channel.close()
        self.jobs.close()
        self.sock.close()

    def _send(self, header, body=b""):
        with self.sending:
            send(self.sock, header, body)

    def _request(self, header, body):
        reply = {"reply": header["id"]}
        try:
            fields = getattr(self, "_" + header["kind"].replace(".", "_"))(header, body)
            self._send({**reply, **(fields or {})})
        except Exception as error:
            self._send({**reply, "error": type(error).__name__, "message": str(error)})

    def _checkout_part(self, header, body):
        if header.get("first"):
            self.parts = []
        self.parts.append(body)

    def _checkout_end(self, header, body):
        data = b"".join(self.parts)
        self.parts = []
        if hashlib.sha256(data).hexdigest() != header["sha256"]:
            raise ValueError("the checkout did not arrive whole: its sha256 differs")
        target = self.server.folder / "checkout"
        partial = self.server.folder / "checkout.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(zstd.decompress(data)), mode="r:") as archive:
            archive.extractall(partial, filter="data")  # nothing outside the folder, no device files
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
        return {"files": sum(1 for _ in target.rglob("*"))}

    def _job_start(self, header, body):
        if self.job is not None and self.job.poll() is None:
            raise ValueError("a job is running already")
        checkout = self.server.folder / "checkout"
        if not checkout.is_dir():
            raise ValueError("no checkout: send it before the job")
        work = self.server.folder / "job"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        (work / "spec.json").write_bytes(body)
        self.peer_secret = (json.loads(body).get("machines") or {}).get("secret")
        python = self._environment(checkout) if self.server.settings.install else sys.executable
        paths = [str(src) for root in SOURCES for src in sorted((checkout / root).glob("*/src"))]
        environ = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "PYTHONSAFEPATH": "1",
                   "WPT_WORKER_JOB": f"127.0.0.1:{self.jobs.getsockname()[1]}", "WPT_WORKER_JOB_KEY": self.job_key,
                   "WPT_WORKER_SPEC": str(work / "spec.json"), "WPT_WORKER_WORK": str(work)}
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"process_group": 0})
        self.job = subprocess.Popen([python, "-m", "wpt_worker.job"], cwd=checkout, env=environ, **group)
        threading.Thread(target=self._watch, args=(self.job,), daemon=True).start()
        return {"pid": self.job.pid}

    def _job_stop(self, header, body):
        self._stop_job(wait=False)

    def _environment(self, checkout) -> str:
        """The Python of the folder's own environment, with the libraries
        the checkout's uv.lock names for the job and the PyTorch build of
        WPT_WORKER_TORCH installed by uv (the checkout's own packages come
        from its folders)."""
        uv = shutil.which("uv")
        if uv is None:
            raise ValueError("uv is not installed: the worker's container installs it when it starts")
        venv = self.server.folder / "venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            subprocess.run([uv, "venv", "--python", "3.14", str(venv)], check=True)
        requirements = self.server.folder / "requirements.txt"
        # the job's libraries, then the PyTorch build with the indexes it comes from
        exported = [subprocess.run([uv, "export", "--frozen", "--no-emit-workspace", "--no-hashes", "--no-header",
                                    *selection], cwd=checkout, check=True, capture_output=True, text=True).stdout
                    for selection in (["--package", "wpt-worker", "--no-dev"],
                                      ["--only-group", self.server.settings.torch, "--emit-index-url"])]
        requirements.write_text("\n".join(exported), encoding="utf-8")
        # a build such as torch==2.14.1+cu130 is on its own index only: every index is searched for it
        subprocess.run([uv, "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match",
                        "-r", str(requirements)], check=True)
        return str(python)

    def _stop_job(self, wait):
        job = self.job
        if job is None or job.poll() is not None:
            return
        _signal(job)
        if wait:
            try:
                job.wait(GRACE)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()

    def _watch(self, job):
        code = job.wait()
        # what its processes sent before they ended reaches the manager before their end does
        for thread in list(self.readers):
            thread.join(GRACE)
        try:
            self._send({"kind": "job.exit", "code": code})
        except OSError:
            pass

    def _accept_jobs(self):
        while self.alive:
            try:
                raw, _ = self.jobs.accept()
            except OSError:
                return
            reader = threading.Thread(target=self._job_channel, args=(raw,), daemon=True)
            self.readers.append(reader)
            reader.start()

    def _job_channel(self, sock):
        try:
            header, _ = receive(sock)
        except LinkError:
            sock.close()
            return
        if header.get("kind") not in ("hello", "peer.accept") \
                or not hmac.compare_digest(str(header.get("key", "")), self.job_key):
            sock.close()
            return
        if header["kind"] == "peer.accept":
            self._accept_peer(sock, header.get("peer"))
            return
        with self.server.lock:
            number, self.next_channel = self.next_channel, self.next_channel + 1
            self.channels[number] = sock
        send(sock, {"kind": "welcome", "channel": number})
        try:
            while True:
                header, body = receive(sock)
                self._send({**header, "channel": number}, body)
        except (LinkError, OSError):
            pass
        finally:
            self.channels.pop(number, None)
            sock.close()
            try:
                self._send({"kind": "channel.closed", "channel": number})
            except OSError:
                pass

    def _accept_peer(self, sock, number):
        """Join a process of this job to the job of gpu worker number, which
        reaches this worker over TLS with its token (Server._peer): each
        side gets "welcome", then the bytes go both ways as they are."""
        peer = self.server.take_peer(number, PEER_WAIT)
        if peer is None:
            try:
                send(sock, {"kind": "refused", "message": f"gpu worker {number} did not reach this one within "
                                                          f"{PEER_WAIT:.0f} s"})
            except OSError:
                pass
            sock.close()
            return
        try:
            send(peer, {"kind": "welcome", "protocol": PROTOCOL})
            send(sock, {"kind": "welcome", "peer": number})
        except OSError:
            peer.close()
            sock.close()
            return
        threading.Thread(target=_pipe, args=(peer, sock), daemon=True).start()
        _pipe(sock, peer)

    def _to_channel(self, header, body):
        sock = self.channels.get(header["channel"])
        if sock is None:
            if "id" in header:
                self._send({"reply": header["id"], "channel": header["channel"], "error": "LinkError",
                            "message": f"no process of the job on channel {header['channel']}"})
            return
        try:
            send(sock, {key: value for key, value in header.items() if key != "channel"}, body)
        except OSError:
            pass


def _pipe(source, target):
    """Copy what source sends to target until either closes; then both are
    closed, which ends the other direction's copy too."""
    try:
        while data := source.recv(1 << 20):
            target.sendall(data)
    except (OSError, ssl.SSLError):
        pass
    finally:
        for sock in (source, target):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


def _signal(job):
    try:
        if os.name == "nt":
            job.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(job.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
