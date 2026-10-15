"""Serve one local web interface: its built files, and the JSON routes its
pages declare.

The server listens on 127.0.0.1 and answers any client, like the commands:
no login and no header to send. Its routes change only what the commands
change, through the same library functions.
"""

import json
import mimetypes
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

@dataclass(frozen=True)
class Route:
    """method and pattern ("/api/sources/{id}") select the handler, which
    gets the pattern's values and the request's JSON body (None for GET) and
    returns (HTTP status, JSON payload)."""
    method: str
    pattern: str
    handler: Callable[[dict, object], tuple[int, object]]

    def match(self, method, path):
        if method != self.method:
            return None
        regex = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(self.pattern))
        found = re.fullmatch(regex, path)
        return {k: unquote(v) for k, v in found.groupdict().items()} if found else None


def make_server(routes: list[Route], static: Path, port: int = 0) -> ThreadingHTTPServer:
    """A server on 127.0.0.1:port (0: any free port), not started yet:
    call serve_forever(). OSError if the port is taken."""
    static = Path(static).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._answer("GET")

        def do_PUT(self):
            self._answer("PUT")

        def do_POST(self):
            self._answer("POST")

        def _answer(self, method):
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                return self._api(method, path)
            if method != "GET":
                return self._json(405, {"problems": ["only GET outside /api/"]})
            self._file(path)

        def _api(self, method, path):
            body = None
            if method != "GET":
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
                except (ValueError, UnicodeDecodeError) as error:
                    return self._json(400, {"problems": [f"the body is not JSON: {error}"]})
            for route in routes:
                values = route.match(method, path)
                if values is not None:
                    return self._json(*route.handler(values, body))
            self._json(404, {"problems": [f"no route {method} {path}"]})

        def _file(self, path):
            target = (static / path.lstrip("/")).resolve()
            if target == static or target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(static) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), kind)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

        def _send(self, status, data, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_request(self, code="-", size="-"):  # the terminal shows only refused requests
            if str(code).isdigit() and int(code) >= 400:
                super().log_request(code, size)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
