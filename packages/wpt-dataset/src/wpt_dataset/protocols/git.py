"""The git protocol: a repository on any host, at the commit the source's
`revision` names.

    robots.txt of the address's site   -> read first, for the address and the two paths a fetch asks for
                                          (robots.py); a refusal is CrawlForbidden
    the commit `revision`, without     -> a partial fetch of that commit (a server that serves only branch
    file contents                         and tag tips is asked for those, and the commit must be among
                                          what they hold; a server that offers no partial fetch sends the
                                          contents too); with `history.since`, the commits since then as
                                          well, `since` an ancestor of `revision`
    the kept files' contents           -> only those the store does not hold: the git objects the store
                                          database's map knows (what the extractor stores into gives
                                          them, known_objects) are taken from the store, their sha256 with
                                          them; the others are asked for by id, in one request
    the same, when the store lacks    -> a second fetch of the commit, and of its history when the
    half or more of those contents (a     source has one, with every content, into a new repository
    first download)                       that replaces the first: git spends some twenty times more per
                                          blob on a request by id; what the source does not keep comes
                                          with it and is never stored; a server that cannot send it all
                                          is asked for the missing contents by id instead
    a file git stores with LFS (its    -> the LFS object it names, through the repository's LFS batch
    blob a pointer)                       request (lfs.py), streamed to the disk: the file's bytes are the
                                          object's, and the pointer's id is what the map learns
    `history.since`                    -> its commits as the files .git/history/<commit> and
                                          .git/history/order (manifest.read_git_history)
    each kept submodule                -> fetched the same way, at the commit it pins, from the address
                                          `submodules` lists for it, recursively

A Hugging Face dataset is such a repository: its file list comes from git,
its large files through LFS. A first download brings the whole history with
every content, and what the source does not keep is never stored; an update
asks only for the files the store lacks.

What it got, for the record: {"commit": `revision`}.

Failures: what protocols.base.named makes of what git and urllib raise
(SourceGone, TooManyRequests, ServerError, AccessRefused, HostUnreachable,
Incomplete, FetchFailed); FetchFailed for a commit the address does not
serve (the source's `revision`, or the one the parent names for a
submodule); CrawlForbidden, RobotsUnreadable (robots.py);
the failures of lfs.py; UnsupportedEntry for a kept submodule with no
address in `submodules`, an address `submodules` lists for no kept
submodule, or a tree entry that is neither a file nor a symlink; BadPath
for a path the tree cannot hold.
"""

import itertools
import os
import shutil
import subprocess
from pathlib import Path

from .. import lfs, manifest, retry, robots
from ..errors import FetchFailed, UnsupportedEntry
from ..manifest import Extractor
from .base import USER_AGENT

# git's messages in English whatever the machine's language: they name the failure (protocols.base.named) and
# whether a try may pass (retry.py); and never a prompt, from git, a credential manager or ssh: a server that asks
# for a password fails at once, and protocols.base.named makes it AccessRefused
ENGLISH = {**os.environ, "LC_ALL": "C", "LANGUAGE": "", "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "",
           "GCM_INTERACTIVE": "never", "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh") + " -o BatchMode=yes"}


def fetch(address: str, data: dict, extractor: Extractor, tmp: Path) -> dict:
    """The files of the commit data's `revision` names added to extractor
    (see the module's docstring), its repositories in the empty folder tmp."""
    robots.check(address, "git", USER_AGENT)  # before anything is downloaded
    since = data["history"]["since"] if data.get("history") else None
    listed, found = data.get("submodules", {}), set()
    pending = [("", address, data["revision"], since, "`revision`")]
    store = extractor.store
    for number in itertools.count():
        if not pending:
            break
        prefix, address, revision, since, named = pending.pop(0)
        repo = tmp / f"repo-{number}"
        fetch_commit(address, revision, repo, named, full=since is not None)
        history = manifest.git_history_blobs(repo, revision, extractor, since)
        in_tree = manifest.git_blobs_needed(repo, revision, extractor, None, prefix)
        needed = list(dict.fromkeys(in_tree + history))
        known = store.known_objects(needed) if hasattr(store, "known_objects") else {}
        missing = manifest.missing_objects(repo, [object_id for object_id in needed if object_id not in known])
        # a first download misses nearly every blob it needs and an update nearly none, so one half of them
        # missing tells the two apart; git spends far less per blob when one fetch brings every content than when
        # they are asked for by id, so a first download takes that fetch: what the source does not keep comes with
        # it and is deleted with tmp, never stored
        if missing and 2 * len(missing) >= len(needed):
            shutil.rmtree(repo)  # the whole fetch makes a new repository
            whole = repo.with_name(f"{repo.name}-whole")
            try:
                fetch_commit(address, revision, whole, named, full=since is not None, contents=True)
                repo = whole
            except (FetchFailed, subprocess.CalledProcessError):
                # a server that cannot send some blob of the history (one deleted upstream) fails the whole fetch;
                # the request by id asks only for the blobs the source needs, and raises its own failure if it fails
                shutil.rmtree(whole, ignore_errors=True)
                fetch_commit(address, revision, repo, named, full=since is not None)
                fetch_blobs(address, repo, missing)
        else:
            fetch_blobs(address, repo, missing)
        # the blobs only the history's patches read are kept too, so a later download asks for none of them again
        held = set(in_tree) | set(known)
        extractor.keep_blobs(repo, [object_id for object_id in history if object_id not in held])
        if history:  # the patches diff both sides of each change: git must hold them
            lacking = manifest.missing_objects(repo, [object_id for object_id in history if object_id in known])
            # in the order the files lie in the bundles, so each bundle read serves the next files too
            lacking.sort(key=lambda object_id: (known[object_id].bundle, known[object_id].start))
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
            pending.append((path + "/", listed[path], commit, None, f"the commit its parent names for {path}"))
    if unused := sorted(set(listed) - found):
        raise UnsupportedEntry(f"`submodules` lists {', '.join(unused)}, not a kept submodule of the commit")
    return {"commit": data["revision"]}


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


def fetch_commit(address: str, revision: str, repo: Path, named: str, full=False, contents=False):
    """The commit `revision`, into a new repository at repo: without file
    contents (git's partial fetch), or with them with `contents`; with
    `full`, its whole history too. A server that does not offer partial
    fetches sends the contents either way. FetchFailed when the address
    does not serve that commit, its message saying what names it (named)."""
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], capture_output=True, check=True)
    git = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo)]
    partial = [] if contents else [["config", "remote.origin.promisor", "true"],
                                   ["config", "remote.origin.partialclonefilter", "blob:none"]]
    for setting in (["remote", "add", "origin", address], *partial):
        subprocess.run([*git, *setting], capture_output=True, check=True)
    depth = [] if full else ["--depth", "1"]
    filtered = [] if contents else ["--filter=blob:none"]

    def fetch(*what):
        retry.call(lambda: subprocess.run([*git, "fetch", "-q", "--no-tags", *filtered, *what],
                                          capture_output=True, check=True, env=ENGLISH), address)

    try:
        fetch(*depth, "origin", revision)
    except subprocess.CalledProcessError:  # some servers only serve branch and tag tips
        fetch("origin", "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")
    # GIT_NO_LAZY_FETCH: a commit the fetches did not bring is not asked for again
    held = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True,
                          env={**ENGLISH, "GIT_NO_LAZY_FETCH": "1"})
    if held.returncode:
        raise FetchFailed(f"{address}: does not serve the commit {revision} ({named})")


def fetch_blobs(address: str, repo: Path, object_ids: list[str]):
    """The file contents a partial fetch left out, asked for by id in one request."""
    missing = manifest.missing_objects(repo, object_ids)
    if not missing:
        return
    command = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo), "fetch", "-q", "--no-tags",
               "--no-write-fetch-head", "--filter=blob:none", "--stdin", "origin"]
    wanted = "".join(object_id + "\n" for object_id in missing).encode()
    retry.call(lambda: subprocess.run(command, input=wanted, capture_output=True, check=True, env=ENGLISH), address)
