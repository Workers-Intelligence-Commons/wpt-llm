"""The outside programs WPT runs (mercurial, GnuPG, Bun, git), the commit
of the WPT repository a command runs from, and what identifies the code
itself.

program(name) is how every WPT package finds such a program: a missing one
raises MissingTool, which names it, instead of an error from deep inside a
command. code() is the one place that reads the repository commit: what
a fingerprint holds in place of a version number, the commit of the
checkout the code runs from and a hash of its uncommitted changes.
code_outside(folder) identifies the same checkout without one of its
top-level folders: given the data folder, what the fingerprint of a
result computed from one source's own inputs alone holds instead, so a
commit that changes only the data leaves that result's fingerprint the
same. repository() gives the commit, and whether there are changes, in
the shape provenance.json and each run's record name them.
"""

import functools
import shutil
import subprocess
from pathlib import Path


class MissingTool(Exception):
    """A program the work needs (mercurial for mercurial sources, GnuPG for
    GPG-signed commits, Bun for wpt-web) is not installed."""


def program(name: str) -> str:
    """The absolute path of a program on the search path. Handing git an
    absolute path also matters on Windows, where Git for Windows would
    otherwise run its own bundled GnuPG, which rejects Windows paths."""
    path = shutil.which(name)
    if path is None:
        raise MissingTool(f"{name} is not installed (or not on the search path)")
    return path


def repository() -> dict:
    """{"repository_commit": the commit of the checkout this code runs
    from, "repository_dirty": whether it has uncommitted changes}, both
    from code(), so they never disagree with it; each None when the code
    is not in a git checkout."""
    found = code()
    if found["commit"] is None:
        return {"repository_commit": None, "repository_dirty": None}
    return {"repository_commit": found["commit"], "repository_dirty": found["changes"] is not None}


# code()'s answer, and code_outside()'s for each folder asked, found once per process, or given by the process
# that started this one (set_code), which then gives every folder this one may ask
_CODE = None
_OUTSIDE = {}
_GIVEN = False


def code() -> dict:
    """{"commit": the commit of the WPT checkout this code runs from,
    "changes": the sha256 of its uncommitted changes (every tracked file's
    difference from that commit and every untracked file git does not
    ignore), None when there are none}: what a fingerprint holds instead
    of a version number, so any commit or change to the checkout, its data
    included, makes a new one. Code that is not in a git checkout
    (installed as packages) has commit None and, as changes, the sha256 of
    every file of the installed wpt-* packages. Found once per process
    (set_code gives a worker process its starter's)."""
    global _CODE
    if _CODE is None:
        top = _checkout()
        commit = _git(top, "rev-parse", "HEAD").decode().strip() if top else ""
        _CODE = {"commit": commit, "changes": _changes(top)} if commit else \
            {"commit": None, "changes": _installed_code()}
    return dict(_CODE)


def code_outside(folder: str) -> str:
    """The sha256 of the WPT checkout this code runs from without its
    top-level folder named folder (checkout_outside). Code that is not in
    a git checkout has the sha256 of every file of the installed wpt-*
    packages, as in code(). Found once per process and folder (set_code
    gives a worker process its starter's). In a process given its codes
    by set_code, a folder they do not hold raises KeyError, since finding
    it there could disagree with the starter's."""
    if folder not in _OUTSIDE:
        if _GIVEN:
            raise KeyError(f"code_outside({folder!r}) was not given by the starting process (codes)")
        top = _checkout()
        found = top and _git(top, "rev-parse", "HEAD")
        _OUTSIDE[folder] = checkout_outside(top, folder) if found else _installed_code()
    return _OUTSIDE[folder]


def codes(*folders: str) -> dict:
    """{"code": code(), "outside": {folder: code_outside(folder)} of each
    of folders}: what set_code gives a worker process, folders being every
    folder it may ask code_outside for."""
    return {"code": code(), "outside": {folder: code_outside(folder) for folder in folders}}


def set_code(found: dict):
    """Use found (codes() of another process of the same run) as this
    process's code() and code_outside(), so a worker process's
    fingerprints are its starter's, even when the checkout changed
    meanwhile."""
    global _CODE, _OUTSIDE, _GIVEN
    _CODE, _OUTSIDE, _GIVEN = dict(found["code"]), dict(found["outside"]), True


def code_name(found=None) -> str:
    """code() as one line: the commit, then +<the first 12 hex digits of
    the changes> when there are some."""
    found = found or code()
    return (found["commit"] or "packages") + (f"+{found['changes'][:12]}" if found["changes"] else "")


def checkout_outside(top, folder: str) -> str:
    """The sha256 of the git checkout at top without its top-level folder
    named folder: the top-level entries of its commit but that folder,
    each with the object id that names its whole content, then its
    uncommitted changes outside that folder (tracked files' differences
    and untracked files git does not ignore)."""
    import hashlib

    listed = _git(top, "ls-tree", "-z", "HEAD")
    entries = [entry for entry in listed.split(b"\0") if entry and entry.split(b"\t", 1)[-1] != folder.encode()]
    digest = hashlib.sha256(b"\0".join(entries))
    digest.update(b"\0changes\0" + (_changes(top, folder) or "").encode())
    return digest.hexdigest()


def _checkout() -> Path | None:
    """The top folder of the git checkout this code runs from, or None."""
    top = _git(Path(__file__).parent, "rev-parse", "--show-toplevel").decode().strip()
    return Path(top) if top else None


def _changes(top, *outside: str) -> str | None:
    """The sha256 of the uncommitted changes of the git checkout at top,
    every tracked file's difference from its commit and every untracked
    file git does not ignore, the top-level folders outside left out; None
    when there are none."""
    import hashlib

    top = Path(top)
    paths = ["--", ".", *(f":(top,exclude){folder}" for folder in outside)]
    digest, changed = hashlib.sha256(), False
    difference = _git(top, "diff", "HEAD", "--binary", "--no-color", "--no-ext-diff", "--no-textconv", *paths)
    if difference:
        digest.update(difference)
        changed = True
    for path in sorted(filter(None, _git(top, "ls-files", "--others", "--exclude-standard", "-z", *paths)
                              .split(b"\0"))):
        try:
            content = (top / path.decode()).read_bytes()
        except OSError:
            continue  # gone meanwhile, or a folder git lists
        digest.update(b"\0" + path + b"\0" + hashlib.sha256(content).digest())
        changed = True
    return digest.hexdigest() if changed else None


def _git(top, *command) -> bytes:
    """What a git command run in the folder top prints; b"" when it fails."""
    try:
        return subprocess.run([program("git"), "-C", str(top), *command], capture_output=True, check=True).stdout
    except (MissingTool, OSError, subprocess.CalledProcessError):
        return b""


@functools.cache
def _installed_code() -> str:
    import hashlib
    import importlib.metadata

    digest = hashlib.sha256()
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"] or ""):
        name = distribution.metadata["Name"] or ""
        if not name.startswith("wpt-"):
            continue
        for file in sorted(distribution.files or [], key=str):
            if file.suffix == ".pyc":
                continue
            try:
                content = file.locate().read_bytes()
            except OSError:
                continue
            digest.update(f"\0{name}/{file}\0".encode() + hashlib.sha256(content).digest())
    return digest.hexdigest()
