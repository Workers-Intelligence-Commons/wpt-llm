"""The link between the wpt manager and a gpu worker, and between the gpu
worker and the processes of its job: frames over a socket, with requests
that the other side answers.

A frame is a header, a JSON object, and a body, bytes:

    4 bytes  the header's length, big-endian
    header   JSON, UTF-8
    8 bytes  the body's length, big-endian
    body     bytes (a file's part, a checkpoint's part, the checkout)

A request carries "kind" and "id"; its answer carries "reply" (the
request's id) and, when the handler failed, "error" (the name of the
exception's type) and "message". A frame with "kind" and no "id" is a
notice: nobody answers it. Every value is what JSON holds, so neither side
ever runs what the other sends: no pickle crosses a link.

Between the wpt manager and a gpu worker, the frames go over TLS. The gpu
worker makes its own certificate (certificate.py); the manager accepts only
the certificate whose sha256 it was given (pinned), and sends the secret
token in its first frame; the worker closes a connection whose first frame
does not carry it. Several processes of one job share the manager's
connection: each frame between them names its process's channel (a
number), which the worker's server adds and strips (server.py). The job
of another gpu worker of the same run connects the same way, with "peer"
in its first frame and the run's own secret in place of the token, to
exchange gradients with this worker's job (server.py, peers.py).

A Connection is safe to use from several threads. One thread reads every
frame; an answer goes to the thread waiting for it, a request waits for
the serving thread (the one that calls serve(), or waits in call()). So a
request that arrives while the serving thread waits for its own answer is
handled meanwhile: the manager can ask the job to write tokens while the
job waits for the manager's answer to "play this update".
"""

import hashlib
import hmac
import ipaddress
import json
import queue
import socket
import ssl
import struct
import threading
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

PROTOCOL = 1  # the link's version: a manager and a worker of different versions refuse each other
PART = 16 << 20  # bytes of a file sent in one frame
MOST_HEADER = 16 << 20  # a header longer than this is not one of ours
MOST_BODY = 1 << 30
SCHEME = "wpt-worker"
# a machine that vanishes without closing its connections (power cut, network cut) is noticed: TCP asks the
# other side after IDLE quiet seconds, every INTERVAL seconds, COUNT times, and data left unacknowledged
# for UNACKNOWLEDGED seconds breaks the connection too
IDLE, INTERVAL, COUNT, UNACKNOWLEDGED = 30, 10, 6, 90


class LinkError(Exception):
    """The link broke (the other side closed it, or sent something that is
    not a frame), or the other side refused it (wrong token, another
    certificate than the pinned one, another protocol version)."""


class RemoteError(LinkError):
    """The other side's handler failed: kind is the name of its exception's
    type (such as "StepFailed", "MissingObject", "RecordError"), message
    its text."""

    def __init__(self, kind, message):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind, self.message = kind, message


def send(sock, header: dict, body: bytes = b""):
    data = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(data)) + data + struct.pack(">Q", len(body)))
    if body:
        sock.sendall(body)


def _exactly(sock, count) -> bytes:
    parts, left = [], count
    while left:
        try:
            part = sock.recv(min(left, 1 << 20))
        except (OSError, ssl.SSLError) as error:
            raise LinkError(f"the link broke: {error}") from None
        if not part:
            raise LinkError("the other side closed the link")
        parts.append(part)
        left -= len(part)
    return b"".join(parts)


def receive(sock) -> tuple[dict, bytes]:
    """The next frame: (header, body). LinkError when the link closed or the
    bytes are not a frame."""
    (length,) = struct.unpack(">I", _exactly(sock, 4))
    if length > MOST_HEADER:
        raise LinkError(f"a header of {length} bytes: not a frame of this link")
    try:
        header = json.loads(_exactly(sock, length))
    except ValueError:
        raise LinkError("a header that is not JSON: not a frame of this link") from None
    (size,) = struct.unpack(">Q", _exactly(sock, 8))
    if size > MOST_BODY:
        raise LinkError(f"a body of {size} bytes: not a frame of this link")
    return header, _exactly(sock, size) if size else b""


class Key:
    """How the manager reaches one gpu worker: its address, its port, the
    sha256 of its certificate (pin) and the secret token, written as
    wpt-worker://TOKEN@HOST:PORT#PIN (what a worker prints when it starts,
    and what WPT_GPU_WORKERS lists). Two optional parts say where the
    worker is, for a run on several machines (docs/machines.md, "Several
    machines"): ?place=NAME, the place it is in (one home network, one
    provider's cluster), and &private=ADDRESS, its private address there,
    which its peers in the same place reach it at:
    wpt-worker://TOKEN@HOST:PORT?place=NAME&private=ADDRESS#PIN."""

    def __init__(self, host, port, pin, token, place=None, private=None):
        self.host, self.port, self.pin, self.token = host, int(port), pin, token
        self.place, self.private = place or None, private or None

    def __str__(self):
        return self._text(quote(self.token, safe=""))

    def shown(self) -> str:
        """The key with its token hidden but its first 4 characters: what
        the manager prints."""
        return self._text(self.token[:4] + "*" * 8)

    def _text(self, token):
        query = urlencode([(name, value) for name, value in (("place", self.place), ("private", self.private))
                           if value])
        return f"{SCHEME}://{token}@{self.host}:{self.port}{'?' + query if query else ''}#{self.pin}"

    @classmethod
    def parse(cls, text) -> "Key":
        parts = urlsplit(text.strip())
        if parts.scheme != SCHEME or not parts.username or not parts.hostname or not parts.port \
                or len(parts.fragment) != 64:
            raise ValueError(f"not a gpu worker's key: {Key._hidden(text)!r}; a worker prints its key, "
                             f"{SCHEME}://TOKEN@HOST:PORT#PIN, when it starts")
        query = dict(parse_qsl(parts.query))
        unknown = sorted(set(query) - {"place", "private"})
        if unknown:
            raise ValueError(f"a gpu worker's key with {', '.join(unknown)}: a key knows only place and private")
        return cls(parts.hostname, parts.port, parts.fragment, unquote(parts.username), query.get("place"),
                   query.get("private"))

    @staticmethod
    def _hidden(text):
        """text with what could be a token hidden, for an error message."""
        head, at, rest = text.strip().partition("@")
        return f"{head[:len(SCHEME) + 7]}{'*' * 8}@{rest}" if at else text.strip()[:len(SCHEME) + 3] + "..."


def addresses(host) -> set:
    """Every address host stands for (ipaddress objects); none when it
    stands for none."""
    try:
        return {ipaddress.ip_address(info[4][0].split("%")[0]) for info in socket.getaddrinfo(host, None)}
    except OSError:
        return set()


def is_private(host) -> bool:
    """Whether every address host stands for is a private one (a home or
    a cluster's network, this machine's own included): what a gpu worker
    joins NCCL peers at, since NCCL's ports have no password."""
    found = addresses(host)
    return bool(found) and all(address.is_private for address in found)


def is_loopback(host) -> bool:
    """Whether every address host stands for is this machine's own
    (127.0.0.1, ::1): another machine reaching it reaches itself."""
    found = addresses(host)
    return bool(found) and all(address.is_loopback for address in found)


def pin(der: bytes) -> str:
    """The sha256 of a certificate in DER form: what the manager pins."""
    return hashlib.sha256(der).hexdigest()


def keep_alive(sock):
    """Make sock notice a machine that is gone (IDLE, INTERVAL, COUNT,
    UNACKNOWLEDGED): its reads and writes then fail, a LinkError for the
    link, after about 90 seconds at most. A machine that is only slow, or
    busy, still answers TCP and is never cut off."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", IDLE), ("TCP_KEEPALIVE", IDLE), ("TCP_KEEPINTVL", INTERVAL),
                        ("TCP_KEEPCNT", COUNT), ("TCP_USER_TIMEOUT", UNACKNOWLEDGED * 1000)):
        option = getattr(socket, name, None)  # each system has its own set
        if option is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
            except OSError:
                pass
    if hasattr(sock, "ioctl") and hasattr(socket, "SIO_KEEPALIVE_VALS"):  # Windows
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, IDLE * 1000, INTERVAL * 1000))


def server_context(certificate, key) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certificate, key)
    return context


def _tls(key: Key, timeout) -> ssl.SSLSocket:
    """A TLS connection to the worker of key, its certificate checked
    against the pin before anything is sent."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = False  # the worker's certificate names no host: the pin is what identifies it
    context.verify_mode = ssl.CERT_NONE
    try:
        raw = socket.create_connection((key.host, key.port), timeout=timeout)
    except OSError as error:
        raise LinkError(f"the gpu worker at {key.host}:{key.port} cannot be reached: {error}") from None
    keep_alive(raw)
    try:
        sock = context.wrap_socket(raw)
    except (OSError, ssl.SSLError) as error:
        raw.close()
        raise LinkError(f"no TLS with the gpu worker at {key.host}:{key.port}: {error}") from None
    shown = pin(sock.getpeercert(binary_form=True))
    if not hmac.compare_digest(shown, key.pin):
        sock.close()
        raise LinkError(f"the gpu worker at {key.host}:{key.port} shows the certificate {shown}, not the pinned "
                        f"{key.pin}: nothing was sent")
    return sock


def dial(key: Key, timeout=30.0, peer=None, answer=None) -> ssl.SSLSocket:
    """A TLS connection to the worker of key (_tls), and the token sent in
    the first frame; the worker's answer read, within answer seconds
    (default timeout). LinkError when the worker cannot be reached, shows
    another certificate, or refuses the token or the protocol version.
    peer: this side is not a manager but the job of the run's gpu worker
    number peer, which reaches the job of the worker of key (server.py);
    key's token is then the run's secret."""
    sock = _tls(key, timeout)
    try:
        send(sock, {"kind": "hello", "token": key.token, "protocol": PROTOCOL,
                    **({"peer": peer} if peer is not None else {})})
        sock.settimeout(answer or timeout)
        header, _ = receive(sock)
    except (OSError, LinkError) as error:
        sock.close()
        raise LinkError(f"the gpu worker at {key.host}:{key.port} did not answer: {error}") from None
    if header.get("kind") != "welcome":
        sock.close()
        raise LinkError(f"the gpu worker at {key.host}:{key.port} refused the link: {header.get('message')}")
    sock.settimeout(None)
    return sock


def probe(key: Key, timeout=30.0) -> bool:
    """Whether the worker of key is there: it answers on its port and shows
    the pinned certificate. Nothing is sent, so a worker busy with another
    manager counts as there too."""
    try:
        _tls(key, timeout).close()
    except LinkError:
        return False
    return True


class Connection:
    """One side of a link (see the module's docstring). handlers: {kind:
    handler(header, body) -> (fields, body) or None}, called on the
    serving thread; an exception a handler raises is answered as an error
    of its type's name. notices: {kind: handler(header, body)}, called on
    the reading thread for a frame nobody answers. serving: the thread that
    handles requests (by default the thread that makes the Connection)."""

    def __init__(self, sock, handlers=None, notices=None, serving=None, name="link"):
        self.sock, self.name = sock, name
        self.handlers, self.notices = dict(handlers or {}), dict(notices or {})
        self.serving = serving or threading.current_thread()
        self._sending = threading.Lock()
        self._ids, self._ids_lock = 0, threading.Lock()
        self._waiting = {}  # request id -> queue of its answer
        self._requests = queue.Queue()  # (header, body) of requests for the serving thread
        self._broken = None
        self._reader = threading.Thread(target=self._read, name=f"{name}-reader", daemon=True)
        self._reader.start()

    @property
    def broken(self) -> LinkError | None:
        return self._broken

    def call(self, kind, body=b"", channel=None, **fields) -> tuple[dict, bytes]:
        """Send a request and wait for its answer: (its header, its body).
        RemoteError when the other side's handler failed, LinkError when the
        link broke. On the serving thread, requests that arrive meanwhile are
        handled while it waits."""
        with self._ids_lock:
            self._ids += 1
            number = self._ids
        answer = queue.Queue()
        self._waiting[number] = answer
        try:
            self._send({"kind": kind, "id": number, **({"channel": channel} if channel is not None else {}),
                        **fields}, body)
            while True:
                if threading.current_thread() is self.serving:
                    try:
                        header, body = answer.get(timeout=0.05)
                    except queue.Empty:
                        self._handle_waiting()
                        continue
                else:
                    header, body = answer.get()
                if header is None:
                    raise self._broken
                if "error" in header:
                    raise RemoteError(header["error"], header.get("message", ""))
                return header, body
        finally:
            self._waiting.pop(number, None)

    def tell(self, kind, body=b"", channel=None, **fields):
        """Send a notice, which nobody answers."""
        self._send({"kind": kind, **({"channel": channel} if channel is not None else {}), **fields}, body)

    def serve(self, until, poll=0.1):
        """Handle requests on this thread (the serving thread) until until()
        is true; LinkError when the link breaks first."""
        while not until():
            try:
                header, body = self._requests.get(timeout=poll)
            except queue.Empty:
                if self._broken is not None and self._requests.empty():
                    raise self._broken
                continue
            self._handle(header, body)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def _send(self, header, body=b""):
        if self._broken is not None:
            raise self._broken
        with self._sending:
            try:
                send(self.sock, header, body)
            except OSError as error:
                self._break(LinkError(f"the link broke: {error}"))
                raise self._broken from None

    def _handle_waiting(self):
        while True:
            try:
                header, body = self._requests.get_nowait()
            except queue.Empty:
                return
            self._handle(header, body)

    def _handle(self, header, body):
        handler = self.handlers.get(header["kind"])
        reply = {"reply": header["id"], **({"channel": header["channel"]} if "channel" in header else {})}
        try:
            if handler is None:
                raise LinkError(f"{self.name}: no request {header['kind']!r} here")
            answered = handler(header, body)
            fields, data = answered if answered is not None else ({}, b"")
            self._send({**reply, **fields}, data)
        except LinkError as error:
            if self._broken is not None:
                return  # nobody to answer
            self._send({**reply, "error": type(error).__name__, "message": str(error)})
        except BaseException as error:  # the other side hears of it; this side's own error goes on
            if self._broken is None:
                self._send({**reply, "error": type(error).__name__, "message": str(error)})
            if not isinstance(error, Exception):
                raise

    def _read(self):
        while True:
            try:
                header, body = receive(self.sock)
            except LinkError as error:
                self._break(error)
                return
            if "reply" in header:
                waiting = self._waiting.get(header["reply"])
                if waiting is not None:
                    waiting.put((header, body))
            elif "id" in header:
                self._requests.put((header, body))
            else:
                notice = self.notices.get(header.get("kind"))
                if notice is not None:
                    notice(header, body)

    def _break(self, error):
        if self._broken is None:
            self._broken = error
        for waiting in list(self._waiting.values()):
            waiting.put((None, b""))
