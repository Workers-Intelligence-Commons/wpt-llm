"""Serve one local web interface: its built files, the JSON routes its pages
declare, and their OpenAPI description at /api/openapi.json.

The server listens on 127.0.0.1 and answers any client, like the commands:
no login and no header to send; it checks neither the Host header nor a
request's Content-Type (what that leaves open: docs/architecture.md, "Two
interfaces"). Its routes change
only what the commands change, through the same service layer
(wpt_dataset.repo). Each route declares the JSON Schema of the body it
takes and of its answer: the server refuses a body that does not match
with 400, and the OpenAPI description is made from the same declarations,
so it cannot drift from the routes.
"""

import json
import mimetypes
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

import jsonschema

PROBLEMS = {"type": "object", "required": ["problems"],
            "properties": {"problems": {"type": "array", "items": {"type": "string"}}}}
REFUSALS = {"400": "the body does not match the route's schema", "404": "no such file, source or route",
            "409": "the file changed since the version sent (read it again)",
            "422": "the rules of `wpt-dataset check` refuse it; nothing was written",
            "500": "the machine failed (out of memory, a GPU driver error), or the fault is unknown (the message says which)",
            "503": "what the route needs is missing on this machine (the store, a model folder, PyTorch)"}


@dataclass(frozen=True)
class Route:
    """method and pattern ("/api/sources/{id}") select the handler, which
    gets the pattern's values and the request's JSON body (None for GET) and
    returns (HTTP status, JSON payload). summary says what it does; body is
    the JSON Schema of the body it takes (None: no body); answer the JSON
    Schema of a successful answer."""
    method: str
    pattern: str
    handler: Callable[[dict, object], tuple[int, object]]
    summary: str = ""
    body: dict | None = None
    answer: dict | None = None

    def match(self, method, path):
        if method != self.method:
            return None
        regex = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(self.pattern))
        found = re.fullmatch(regex, path)
        return {k: unquote(v) for k, v in found.groupdict().items()} if found else None


def openapi(routes: list[Route]) -> dict:
    """The OpenAPI 3.1 description of the routes."""
    paths = {}
    for route in routes:
        operation = {"summary": route.summary,
                     "responses": {"200": {"description": "done",
                                           "content": {"application/json": {"schema": route.answer or {}}}},
                                   **{status: {"description": why,
                                               "content": {"application/json": {"schema": PROBLEMS}}}
                                      for status, why in REFUSALS.items()}}}
        names = re.findall(r"\{(\w+)\}", route.pattern)
        if names:
            operation["parameters"] = [{"name": name, "in": "path", "required": True, "schema": {"type": "string"},
                                        "description": "percent-encoded; a path's '/' is %2F"} for name in names]
        if route.body is not None:
            operation["requestBody"] = {"required": True, "content": {"application/json": {"schema": route.body}}}
        paths.setdefault(route.pattern, {})[route.method.lower()] = operation
    return {"openapi": "3.1.0",
            "info": {"title": "wpt-web", "version": "1",
                     "description": "The repository's data, read and written through the service layer "
                                    "(wpt_dataset.repo), as the `wpt-dataset` commands do. Every write applies the "
                                    "rules of `wpt-dataset check` and is refused when the file changed since the "
                                    "version sent. docs/architecture.md, \"The service layer\"."},
            "paths": paths}


def make_server(routes: list[Route], static: Path, port: int = 0) -> ThreadingHTTPServer:
    """A server on 127.0.0.1:port (0: any free port), not started yet:
    call serve_forever(). OSError if the port is taken."""
    static = Path(static).resolve()
    routes = [*routes, Route("GET", "/api/openapi.json", lambda values, body: (200, description),
                             "this description of every route", answer={"type": "object"})]
    description = openapi(routes)

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
                    if route.body is not None:
                        error = jsonschema.exceptions.best_match(
                            jsonschema.Draft202012Validator(route.body).iter_errors(body))
                        if error is not None:
                            where = "/".join(map(str, error.absolute_path)) or "(top level)"
                            return self._json(400, {"problems": [f"the body, at {where}: {error.message}"]})
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
