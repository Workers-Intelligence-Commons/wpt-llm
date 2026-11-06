"""The outside programs WPT runs (mercurial, GnuPG, Bun, git), the commit
of the WPT repository a command runs from, and what identifies the code
itself.

program(name) is how every WPT package finds such a program: a missing one
raises MissingTool, which names it, instead of an error from deep inside a
command. code() is the one place that reads the repository commit: what
a run's record and a model's provenance name as the code that made
something, the commit of the checkout the code runs from and a hash of
its uncommitted changes. code_name() gives it as one line, which a stored
result of the data pipeline keeps beside its ref as what made it.
repository() gives the commit, and whether there are changes, in the shape
provenance.json and each run's record name them.
"""

import functools
import shutil
import subprocess
from pathlib import Path


class MissingTool(Exception):
    """A program the work needs (mercurial for mercurial sources, GnuPG for
    GPG-signed commits, Bun for wpt web) is not installed."""


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


_CODE = None  # code()'s answer, found once per process


def code() -> dict:
    """{"commit": the commit of the WPT checkout this code runs from,
    "changes": the sha256 of its uncommitted changes (every tracked file's
    difference from that commit and every untracked file git does not
    ignore), None when there are none}: what a run's record and a model's
    provenance name as the code that made it, and what a fingerprint holds
    when any change to the checkout, its data included, must make a new
    one. Code that is not in a git checkout (installed as packages) has
    commit None and, as changes, the sha256 of every file of the installed
    wpt-* packages. Found once per process."""
    global _CODE
    if _CODE is None:
        top = _checkout()
        commit = _git(top, "rev-parse", "HEAD").decode().strip() if top else ""
        _CODE = {"commit": commit, "changes": _changes(top)} if commit else \
            {"commit": None, "changes": _installed_code()}
    return dict(_CODE)


def fingerprint(inputs) -> str:
    """The sha256 of inputs as canonical JSON: the name of a saved result,
    inputs being everything that decides it."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_name(found=None) -> str:
    """code() as one line: the commit, then +<the first 12 hex digits of
    the changes> when there are some."""
    found = found or code()
    return (found["commit"] or "packages") + (f"+{found['changes'][:12]}" if found["changes"] else "")


def _checkout() -> Path | None:
    """The top folder of the git checkout this code runs from, or None."""
    top = _git(Path(__file__).parent, "rev-parse", "--show-toplevel").decode().strip()
    return Path(top) if top else None


def _changes(top) -> str | None:
    """The sha256 of the uncommitted changes of the git checkout at top,
    every tracked file's difference from its commit and every untracked
    file git does not ignore; None when there are none."""
    import hashlib

    top = Path(top)
    digest, changed = hashlib.sha256(), False
    difference = _git(top, "diff", "HEAD", "--binary", "--no-color", "--no-ext-diff", "--no-textconv")
    if difference:
        digest.update(difference)
        changed = True
    for path in sorted(filter(None, _git(top, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))):
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
