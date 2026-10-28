"""Builders for throwaway repositories, upstream projects and HTTP servers."""

import functools
import http.server
import io
import json
import os
import secrets
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path

import pytest

from wpt_dataset import download as downloading
from wpt_dataset import license_check, manifest, protocols, sources
from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_store import open_store

# The same small project, as git, mercurial and tar would publish it.
FILES = {
    "README.md": b"the cat sat on the mat and the dog sat on the log\n",
    "docs/guide.txt": b"le chat est sur le tapis et le chien dort\n" * 3,
    "run.sh": b"#!/bin/sh\necho hello\n",
}
EXECUTABLE = {"run.sh"}
SYMLINKS = {"latest.txt": "docs/guide.txt"}

GIT_ENV = {"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def run(*args, cwd=None, env=None, input=None):
    return subprocess.run(args, cwd=cwd, env={**os.environ, **GIT_ENV, **(env or {})}, input=input,
                          capture_output=True, check=True).stdout


def project_diff(files=FILES, executable=EXECUTABLE, symlinks=SYMLINKS) -> bytes:
    """The project as one git-format diff that creates it. Modes are written in
    the diff, not on a disk, because Windows files carry no executable bit."""
    def added(path, mode, content: bytes):
        lines = content.splitlines(keepends=True)
        body = b"".join(b"+" + line for line in lines)
        if not content.endswith(b"\n"):
            body += b"\n\\ No newline at end of file\n"
        head = f"diff --git a/{path} b/{path}\nnew file mode {mode}\n--- /dev/null\n+++ b/{path}\n"
        return head.encode() + f"@@ -0,0 +1,{len(lines)} @@\n".encode() + body

    parts = [(path, added(path, "100755" if path in executable else "100644", content)) for path, content in files.items()]
    parts += [(path, added(path, "120000", target.encode())) for path, target in symlinks.items()]
    return b"".join(part for _, part in sorted(parts))


def parquet_bytes(rows: list[dict]) -> bytes:
    """A Parquet file of rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def git_project(folder: Path, files=FILES, executable=EXECUTABLE, symlinks=SYMLINKS):
    """(address, commit) of a new local git repository holding the files."""
    folder.mkdir()
    run("git", "init", "-q", cwd=folder)
    run("git", "apply", "--cached", "-", cwd=folder, input=project_diff(files, executable, symlinks))
    run("git", "commit", "-q", "-m", "first", cwd=folder)
    return str(folder), run("git", "rev-parse", "HEAD", cwd=folder).decode().strip()


@pytest.fixture
def git_upstream(tmp_path):
    """(address, commit) of a local git repository holding the project."""
    return git_project(tmp_path / "upstream-git")


@pytest.fixture
def hg_upstream(tmp_path):
    folder = tmp_path / "upstream-hg"
    folder.mkdir()
    run("hg", "init", "-q", cwd=folder)
    run("hg", "import", "-q", "--bypass", "-u", "T", "-m", "first", "-", cwd=folder, input=project_diff())
    return str(folder), run("hg", "log", "-r", "tip", "-T", "{node}", cwd=folder).decode()


def tar_bytes(top="project-1.0", files=FILES, executable=EXECUTABLE, symlinks=SYMLINKS) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in files.items():
            info = tarfile.TarInfo(f"{top}/{path}")
            info.size, info.mode = len(content), 0o755 if path in executable else 0o644
            archive.addfile(info, io.BytesIO(content))
        for path, target in symlinks.items():
            info = tarfile.TarInfo(f"{top}/{path}")
            info.type, info.linkname = tarfile.SYMTYPE, target
            archive.addfile(info)
        folder = tarfile.TarInfo(f"{top}/empty-folder")
        folder.type = tarfile.DIRTYPE
        archive.addfile(folder)
    return buffer.getvalue()


@pytest.fixture
def http_folder(tmp_path):
    """(folder, base URL) of a local HTTP server; put files in folder."""
    folder = tmp_path / "www"
    folder.mkdir()
    handler = functools.partial(QuietHandler, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield folder, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class RangeHandler(QuietHandler):
    """Serves a file whole or one range of it (`bytes=START-` or
    `bytes=START-END`), says it serves ranges unless `announce` is false,
    and adds (path, its Range header or None) of every GET to `asked`."""

    asked: list
    announce = True

    def do_GET(self):
        self.asked.append((self.path, self.headers.get("Range")))
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        wanted = self.headers.get("Range")
        if wanted:
            first, _, last = wanted.removeprefix("bytes=").partition("-")
            start, end = int(first), int(last) if last else len(data) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            data = data[start:end + 1]
        else:
            self.send_response(200)
        if self.announce:
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):  # the reader had what it needed
            pass


def _ranged(folder, announce):
    """(folder, base URL, asked, the server) of a local HTTP server serving byte ranges (RangeHandler)."""
    asked = []
    folder.mkdir()
    handler = functools.partial(type("Handler", (RangeHandler,), {"asked": asked, "announce": announce}),
                                directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return folder, f"http://127.0.0.1:{server.server_address[1]}", asked, server


@pytest.fixture
def ranged_folder(tmp_path):
    """(folder, base URL, asked) of a local HTTP server that serves byte
    ranges (RangeHandler); put files in folder."""
    *found, server = _ranged(tmp_path / "ranged", True)
    yield tuple(found)
    server.shutdown()


@pytest.fixture
def unannounced_ranged_folder(tmp_path):
    """ranged_folder's, from a server that serves byte ranges without
    saying so (no Accept-Ranges header)."""
    *found, server = _ranged(tmp_path / "unannounced", False)
    yield tuple(found)
    server.shutdown()


@pytest.fixture
def store(tmp_path):
    return open_store((tmp_path / "store").as_uri())


DECISION = {"decision": "admitted", "by": "Test Reviewer", "date": "2026-10-14"}  # an admission needs no reason
ATTRIBUTION = "Copyright (c) 2026 Test Author, https://example.org/project"
LICENSE_TEXT = "Copyright (c) 2026 Test Author\n\nPermission is hereby granted, free of charge, to any person.\n"
# the license text LICENSE_TEXT is (MIT, admitted) in every test repository, and the place a license entry
# finds it at unless a test says otherwise
REVIEWED = license_check.text_id(LICENSE_TEXT.encode())
TEXTS = {REVIEWED: {"decision": DECISION, "kind": "license", "license": "MIT", "files": [],
                    "explanation": "a test's license text"}}
PLACE = {"place": {"file": "LICENSE"}, "expected": [REVIEWED]}
ITSELF = {"place": {"entry": True}}  # the place of an empty source's entry: the entry itself


def stated(repo, tree, staged) -> dict:
    """The place and expected text of a test source's default entry: its
    LICENSE file (PLACE) when the upstream tree holds one; else its first
    file (not a history file nor a link), whose text is written into the
    repository as an admitted license text."""
    if "LICENSE" in tree:
        return PLACE
    path = next((p for p, e in sorted(tree.items()) if not p.startswith(".git/") and e.mode != manifest.SYMLINK), None)
    if path is None:  # nothing to state it: the test's entry is the entry itself
        return ITSELF
    return statement_in(repo, path, manifest.content(staged, tree[path]))


def statement_in(repo, path: str, content: bytes) -> dict:
    """The place and expected text of an entry whose statement is the file
    at path, holding content: its text is written into the repository as
    an admitted license text, so the build lets the entry's files train."""
    text = license_check.text_id(content)
    write_texts(repo.root, {text: {**TEXTS[REVIEWED], "explanation": f"a test's statement, in {path}"}})
    return {"place": {"file": path}, "expected": [text]}


def write_texts(root: Path, texts=None):
    """data/license-texts/<id>/license-text.json of the repository at root, for each of TEXTS, or of the
    texts given ({id: item})."""
    for text, item in (TEXTS if texts is None else texts).items():
        (root / "data" / "license-texts" / text).mkdir(parents=True, exist_ok=True)
        (root / "data" / "license-texts" / text / "license-text.json").write_text(
            json.dumps(item, indent=2) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "wpt"
    for folder in ("sources", "recipes/mixtures", "recipes/tokenizers", "recipes/models"):
        (root / "data" / folder).mkdir(parents=True)
    write_texts(root)
    return Repo(root)


def covering(paths) -> list[str]:
    """Coverage patterns in the allowed shapes for these paths, history files
    aside: each top folder as `<folder>/**`, each file at the top by its path."""
    paths = [path for path in paths if not path.startswith(".git/")]
    return sorted({sources.exact_pattern(path.split("/", 1)[0]) + "/**" if "/" in path else
                   sources.exact_pattern(path) for path in paths})


def entry(paths, license="MIT", **more):
    """A license entry covering paths, admitted (DECISION), its statement at
    PLACE unless more gives its place."""
    return {"license": license, "paths": list(paths), "attribution": ATTRIBUTION,
            **({} if "place" in more else PLACE), "text": LICENSE_TEXT, "decision": DECISION, **more}


# the project's files (FILES and SYMLINKS), as one license entry covers them
PROJECT = covering([*FILES, *SYMLINKS])


class Objects:
    """A store as a fetch stores into it (manifest.Extractor): each kept
    entry an object under its sha256."""

    def __init__(self, store):
        self.store = store

    def keep(self, data):
        return self.store.put_bytes(data), None

    def put_file(self, path):
        return self.store.put_file(path)


def upstream(data, address, store):
    """The upstream tree at address, as it serves it, each file's
    content put into store as an object (protocols.fetch); store is a
    wpt_store store, or a download's Staging."""
    with tempfile.TemporaryDirectory(prefix="wpt-test-fetch-") as tmp:
        sink = store if isinstance(store, downloading.Staging) else Objects(store)
        return protocols.fetch(data, address, sink, Path(tmp)).tree


def session(store, command="test"):
    """A new session of store, named like a run record folder."""
    return store.session(f"20261025T120000Z-{command}-{secrets.token_hex(4)}", command, "abc123", False)


def download(repo, store, *ids):
    """The download step on ids, in a session of its own, as `wpt-dataset
    download` runs it: each source's error raised, the first first."""
    with session(store, "download") as running:
        done = downloading.download(repo, list(ids), store, running, log=lambda *_: None)
    for result in done:
        if result.error is not None:
            raise result.error
    return done


def add_source(repo, store, origin, item, data, patches=(), fetch=True):
    """Write data/sources/<id>/source.json and, with fetch, download it into
    the store. Without `licenses`, one admitted MIT entry covers every file
    the upstream gives (covering), read from it first, after an entry of
    its own for the history files when it has any; its statement is the
    upstream's LICENSE file (PLACE), or without one its first file, whose
    text becomes an admitted license text of the repository (stated), so
    the build lets its files train. Returns the source id."""
    data = {"formats": [{"format": "text", "paths": ["**"]}], "decision": DECISION, **data}
    if "licenses" not in data:
        paths, place = [], ITSELF
        if data["type"] != "empty":
            with tempfile.TemporaryDirectory(prefix="wpt-test-staging-") as staging:
                staged = downloading.Staging(staging)
                tree = upstream(data, data["address"], staged)
                paths, place = list(tree), stated(repo, tree, staged)
        history = [entry([".git/history/**"], **place)] if any(path.startswith(".git/history/") for path in paths) \
            else []
        data["licenses"] = history + [entry(covering(paths) or ["README.md"], **place)]
    sid = "-".join([origin] + ([item] if item else []) + [address_hash(data["address"])])
    folder = repo.root / "data" / "sources" / sid
    folder.mkdir()
    (folder / "source.json").write_text(dump_json(data), encoding="utf-8", newline="\n")
    for i, patch in enumerate(patches, start=1):
        (folder / f"{i:04d}-change.patch").write_text(patch, encoding="utf-8", newline="\n")
    if fetch:
        download(repo, store, sid)
    return sid


MODEL = {"n_layer": 2, "global_attention": ["CSA"], "d_model": 16, "kda_heads": 2, "kda_head_dim": 8,
         "attention_heads": 2, "attention_head_dim": 128, "query_rank": 8, "output_groups": 1, "output_rank": 8,
         "index_heads": 2, "index_head_dim": 128, "index_top_k": 2, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
         "shared_width": 16, "engram_layers": [1], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
         "engram_table_size": 31}
STEPS = [{"step": "read-text", "optimizer": {"name": "adamw", "settings": {}},
          "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 4}], "eval_every": 1,
                       "eval_batches": 1}, "locked": []}]


def write_model_recipe(repo, name, data):
    """A model recipe whose data part is data; its model and steps are the
    smallest that train."""
    return write_recipe(repo, "model", name, {"data": data, "model": MODEL, "seed": 1337, "steps": STEPS,
                                              "drafter": {}})


def write_recipe(repo, kind, name, document):
    path = repo.root / "data" / "recipes" / f"{kind}s" / f"{name}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def kit():
    """This module's builders and constants (test files cannot import
    conftest under --import-mode=importlib)."""
    import sys

    return sys.modules[__name__]
