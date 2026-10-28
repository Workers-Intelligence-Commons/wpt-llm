"""The git protocol: a repository on any host, at the commit its address
serves when asked (its HEAD).

    robots.txt of the address's site   -> read first, for the address and the two paths a fetch asks for
                                          (robots.py); a refusal is CrawlForbidden
    the commit, without file contents  -> a partial fetch (a server that offers none sends the contents
                                          too); with `history.since`, the commits since then as well
    the kept files' contents           -> only those the store does not hold: the git objects the store
                                          database's map knows (what the extractor stores into gives
                                          them, known_objects) are taken from the store, their sha256 with
                                          them; the others are asked for by id, in one request
    a file git stores with LFS (its    -> the LFS object it names, through the repository's LFS batch
    blob a pointer)                       request (lfs.py), streamed to the disk: the file's bytes are the
                                          object's, and the pointer's id is what the map learns
    `history.since`                    -> its commits as the files .git/history/<commit> and
                                          .git/history/order (manifest.read_git_history)
    each kept submodule                -> fetched the same way, at the commit it pins, from the address
                                          `submodules` lists for it, recursively

A Hugging Face dataset is such a repository: its file list comes from git,
and only the kept files are fetched, its large files through LFS.

What it got, for the record: {"commit": the commit the address served}.

Failures: what protocols.base.named makes of what git and urllib raise
(SourceGone, TooManyRequests, ServerError, AccessRefused, HostUnreachable,
Incomplete, FetchFailed); CrawlForbidden, RobotsUnreadable (robots.py);
the failures of lfs.py; UnsupportedEntry for a kept submodule with no
address in `submodules`, an address `submodules` lists for no kept
submodule, or a tree entry that is neither a file nor a symlink; BadPath
for a path the tree cannot hold.
"""

import itertools
import os
import subprocess
from pathlib import Path

from .. import lfs, manifest, retry, robots
from ..errors import UnsupportedEntry
from ..manifest import Extractor
from .base import USER_AGENT

# git's messages in English whatever the machine's language: they name the failure (protocols.base.named) and
# whether a try may pass (retry.py); and never a prompt, from git, a credential manager or ssh: a server that asks
# for a password fails at once, and protocols.base.named makes it AccessRefused
ENGLISH = {**os.environ, "LC_ALL": "C", "LANGUAGE": "", "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "",
           "GCM_INTERACTIVE": "never", "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh") + " -o BatchMode=yes"}


def fetch(address: str, data: dict, extractor: Extractor, tmp: Path) -> dict:
    """The files of the commit address serves added to extractor (see the
    module's docstring), its repositories in the empty folder tmp."""
    robots.check(address, "git", USER_AGENT)  # before anything is downloaded
    since = data["history"]["since"] if data.get("history") else None
    listed, found, served = data.get("submodules", {}), set(), None
    pending = [("", address, None, since)]
    store = extractor.store
    for number in itertools.count():
        if not pending:
            break
        prefix, address, revision, since = pending.pop(0)
        repo = tmp / f"repo-{number}"
        fetch_commit(address, revision, repo, full=since is not None)
        revision = revision or _head(repo)
        served = served or revision
        history = manifest.git_history_blobs(repo, revision, extractor, since)
        in_tree = manifest.git_blobs_needed(repo, revision, extractor, None, prefix)
        needed = list(dict.fromkeys(in_tree + history))
        known = store.known_objects(needed) if hasattr(store, "known_objects") else {}
        fetch_blobs(address, repo, [object_id for object_id in needed if object_id not in known])
        # the blobs only the history's patches read are kept too, so a later download asks for none of them again
        held = set(in_tree) | set(known)
        extractor.keep_blobs(repo, [object_id for object_id in history if object_id not in held])
        if history:  # the patches diff both sides of each change: git must hold them
            lacking = manifest.missing_objects(repo, [object_id for object_id in history if object_id in known])
            written = manifest.put_blobs(repo, [store.file_path(known[object_id]) for object_id in lacking])
            # a stored copy that is not the blob itself (an LFS file's, which is the object it points to) is
            # asked for upstream
            if unlike := [object_id for object_id, put in zip(lacking, written) if put != object_id]:
                fetch_blobs(address, repo, unlike)
        submodules = manifest.read_git(repo, revision, extractor, prefix, known, _large(address, tmp / "lfs"))
        if since:
            manifest.read_git_history(repo, since, revision, extractor)
        for path, commit in submodules.items():
            if path not in listed:
                raise UnsupportedEntry(f"{path}: a submodule with no address in `submodules`; list it or exclude it")
            robots.check(listed[path], "git", USER_AGENT)
            found.add(path)
            pending.append((path + "/", listed[path], commit, None))
    if unused := sorted(set(listed) - found):
        raise UnsupportedEntry(f"`submodules` lists {', '.join(unused)}, not a kept submodule of the commit")
    return {"commit": served}


def _large(address: str, folder: Path):
    """What read_git asks for a blob: for an LFS pointer, a local file
    holding the object it names (lfs.download, into folder); None for any
    other blob, which is the file itself."""
    def large(blob: bytes) -> Path | None:
        stored = lfs.pointer(blob)
        if stored is None:
            return None
        folder.mkdir(exist_ok=True)
        return lfs.download(address, *stored, folder / stored[0], USER_AGENT)

    return large


def fetch_commit(address: str, revision: str | None, repo: Path, full=False):
    """The commit `revision` (None: the one the address's HEAD names, kept as
    the repository's FETCH_HEAD) without file contents (git's partial
    fetch); with `full`, its whole history too. A server that does not offer
    partial fetches sends the contents as well."""
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], capture_output=True, check=True)
    git = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo)]
    for setting in (["remote", "add", "origin", address], ["config", "remote.origin.promisor", "true"],
                    ["config", "remote.origin.partialclonefilter", "blob:none"]):
        subprocess.run([*git, *setting], capture_output=True, check=True)
    depth = [] if full else ["--depth", "1"]

    def fetch(*what):
        retry.call(lambda: subprocess.run([*git, "fetch", "-q", "--no-tags", "--filter=blob:none", *what],
                                          capture_output=True, check=True, env=ENGLISH), address)

    if revision is None:
        fetch(*depth, "origin", "HEAD")
        return
    try:
        fetch(*depth, "origin", revision)
    except subprocess.CalledProcessError:  # some servers only serve branch and tag tips
        fetch("origin", "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True,
                   check=True, env=ENGLISH)


def _head(repo: Path) -> str:
    """The commit a fetch of the address's HEAD gave (fetch_commit)."""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "FETCH_HEAD^{commit}"], capture_output=True,
                          check=True).stdout.decode().strip()


def fetch_blobs(address: str, repo: Path, object_ids: list[str]):
    """The file contents a partial fetch left out, asked for by id in one request."""
    missing = manifest.missing_objects(repo, object_ids)
    if not missing:
        return
    command = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo), "fetch", "-q", "--no-tags",
               "--no-write-fetch-head", "--filter=blob:none", "--stdin", "origin"]
    wanted = "".join(object_id + "\n" for object_id in missing).encode()
    retry.call(lambda: subprocess.run(command, input=wanted, capture_output=True, check=True, env=ENGLISH), address)
