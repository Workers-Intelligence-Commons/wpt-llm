"""A source's file list and the bytes of the few files asked for, without a
full download: what the one-off migration of sources reads (migrate.py),
from a source.json of the shape that pinned a `revision` and a `checksum`.
The migration asks for it only for a source that needs it.

    the store holds the tree under its   -> its paths and bytes from there (the refs upstream/<checksum>
    checksum                                and upstream-files/<checksum> that shape named it by)
    otherwise, by the source's type:
        git                             -> the commit `revision` without file contents (a partial
                                           fetch), each kept submodule at the commit it pins; a file's
                                           bytes are its blob, asked for by id when it is read, or, for
                                           a file git stores with LFS (its blob a pointer), the LFS
                                           object, through the repository's LFS batch request (lfs.py)
        huggingface                     -> the dataset as the git repository it is, read as git is
        http-archive, a zip             -> its table of contents through range requests; a member's
                                           bytes the same way
        http-file                       -> its one file; its bytes are the download
        empty                           -> no upstream file
    then                                -> the paths the patches name added (a path a patch removes
                                           stays listed: listing more paths only adds shapes that
                                           cover nothing)

The `extract` rules apply as a download applies them (manifest.Extractor),
and `.git/history/` files are not listed: no history entry is narrowed.
Everything fetched goes into a temporary folder the caller owns.

Failures: NotListed when the list cannot be read without a full download
(a tar archive, a server that serves no byte ranges, an archive in a
Hugging Face dataset, a mercurial source); the protocol failure
(protocols.base.named) when a file's bytes cannot be read (its LFS object
included); what a fetch raises otherwise (the protocol failures,
UnsupportedEntry).
"""

import io
import itertools
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from . import archives, lfs, manifest, robots
from .errors import DatasetError, UnsupportedEntry
from .manifest import SUBMODULE, Extractor
from .protocols.base import UNREACHABLE, USER_AGENT, named
from .protocols.git import fetch_blobs as _git_fetch_blobs, fetch_commit as _git_fetch
from .protocols.http import is_archive
from .sources import _touched, stored_upstream, temporary_git


class NotListed(DatasetError):
    """A source's file list cannot be read without downloading it whole."""


@dataclass
class Files:
    """A source's final paths (patches included), and read(path): the bytes
    of one of them, or None when no upstream file has that path."""
    paths: list[str]
    read: Callable[[str], bytes | None]


class _Paths:
    """What an Extractor stores into when only paths are wanted: nothing."""

    def keep(self, data):
        return "", None


def files(sid: str, document: dict, patches, store, tmp: Path) -> Files:
    """The final paths of a source of the pinned shape and a reader of their
    bytes (see the module's docstring); tmp: an empty folder."""
    kind = document["type"]
    if kind == "empty":
        found = Files([], lambda path: None)
    elif (held := stored_upstream(document["checksum"].removeprefix("sha256:"), store)) is not None:
        found = Files(list(held), lambda path: manifest.content(store, held[path]) if path in held else None)
    else:
        try:
            found = _listed(document, tmp)
        except UNREACHABLE as error:
            raise named(error, sid) from error
    if not patches:
        return found
    (tmp / "patches").mkdir()
    git, _ = temporary_git(tmp / "patches")
    added = sorted(_touched(git, patches) - set(found.paths))
    return Files(found.paths + added, found.read)


def _listed(document: dict, tmp: Path) -> Files:
    kind, address = document["type"], document["address"]
    extract = document.get("extract", {})
    if "addresses" in document:
        raise NotListed(f"{address}: a source of several addresses is read whole")
    robots.check(address, kind, USER_AGENT)
    if kind == "git":
        return _git(document, tmp)[0]
    if kind == "huggingface":
        return _huggingface(document, tmp)
    if kind == "http-archive":
        return _zip(address, Extractor(_Paths(), extract))
    if kind == "http-file":
        name = extract.get("name") or unquote(urlparse(address).path.rsplit("/", 1)[-1])
        kept = Extractor(_Paths(), extract).kept(name)

        def read(path):
            with archives.get(address, USER_AGENT) as answer:
                return answer.read() if path == kept else None

        return Files([kept] if kept is not None else [], read)
    raise NotListed(f"{address}: a {kind} source's file list is read only by downloading it")


def _git(document: dict, tmp: Path) -> tuple[Files, dict[str, str]]:
    """The kept paths of the commit and of each kept submodule (sources.py
    fetches them the same way), each with its blob (an LFS pointer read as
    the object it names), and {kept path: its path in the commit}."""
    extractor = Extractor(_Paths(), document.get("extract", {}))
    listed, blobs, names = document.get("submodules", {}), {}, {}
    pending = [("", document["address"], document["revision"])]
    for number in itertools.count():
        if not pending:
            break
        prefix, address, revision = pending.pop(0)
        repo = tmp / f"repo-{number}"
        _git_fetch(address, revision, repo)
        for raw, mode, kind, object_id in manifest._git_tree(repo, revision, prefix):
            path = extractor.kept(raw)
            if path is None:
                continue
            if mode == SUBMODULE:
                name = raw.decode()
                if name not in listed:
                    raise UnsupportedEntry(f"{name}: a submodule with no address in `submodules`")
                robots.check(listed[name], "git", USER_AGENT)
                pending.append((name + "/", listed[name], object_id))
            else:
                blobs[path], names[path] = (address, repo, object_id), raw.decode()

    def read(path):
        if path not in blobs:
            return None
        address, repo, object_id = blobs[path]
        try:
            _git_fetch_blobs(address, repo, [object_id])
            blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", object_id], capture_output=True,
                                  check=True, env=manifest.PLAIN_GIT).stdout
            stored = lfs.pointer(blob)
            return blob if stored is None else lfs.fetch(address, *stored, USER_AGENT)
        except UNREACHABLE as error:
            raise named(error, f"{address}: {path}") from error

    return Files(sorted(blobs), read), names


def _huggingface(document: dict, tmp: Path) -> Files:
    """The dataset's git commit (_git); an archive it holds is unpacked by a
    download, its members read only by downloading it, so NotListed."""
    listed, names = _git(document, tmp)
    if archive := next((name for name in names.values() if is_archive(name)), None):
        raise NotListed(f"{document['address']}: {archive} is an archive, whose members are read only by "
                        f"downloading it")
    return listed


def _zip(address: str, extractor: Extractor) -> Files:
    """A zip's table of contents and members, through range requests."""
    with archives.get(address, USER_AGENT) as answer:
        start, size = answer.read(4), answer.headers.get("Content-Length", "")
        address, validator = answer.geturl(), archives._validator(answer.headers)
    if start not in archives.ZIP_STARTS:
        raise NotListed(f"{address}: not a zip; a tar's members are read only by downloading it")
    if not size.isdigit():
        raise NotListed(f"{address}: no size given, so no range of it can be asked for")
    ranged = archives.Ranged(address, int(size), USER_AGENT, validator)
    try:
        archive = zipfile.ZipFile(io.BufferedReader(ranged, archives.BLOCK))
    except archives.WholeAnswer as whole:
        whole.answer.close()
        raise NotListed(f"{address}: the server serves no byte ranges") from None
    members = {}
    for info in archive.infolist():
        path = extractor.kept(info.filename)
        if path is not None and not info.is_dir():
            members[path] = info

    def read(path):
        return archive.read(members[path]) if path in members else None

    return Files(sorted(members), read)

