"""The outside programs WPT runs (mercurial, GnuPG, Bun, git), the commit
of the WPT repository a command runs from, and what identifies the code
itself.

program(name) is how every WPT package finds such a program: a missing one
raises MissingTool, which names it, instead of an error from deep inside a
command. code() is the one place that reads the repository commit: what
every fingerprint holds in place of a version number, the commit of the
checkout the code runs from and a hash of its uncommitted changes.
repository() gives the same commit, and whether there are changes, in
the shape provenance.json and each run's record name them.
"""

import shutil
import subprocess


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


_CODE = None  # code()'s answer, found once per process (or given by the process that started this one)


def code() -> dict:
    """{"commit": the commit of the WPT checkout this code runs from,
    "changes": the sha256 of its uncommitted changes (every tracked file's
    difference from that commit and every untracked file git does not
    ignore), None when there are none}: what every fingerprint holds
    instead of a version number, so any commit or change to the code makes
    a new one. Code that is not in a git checkout (installed as packages)
    has commit None and, as changes, the sha256 of every file of the
    installed wpt-* packages. Found once per process (set_code gives a
    worker process its starter's)."""
    global _CODE
    if _CODE is None:
        _CODE = _found_code()
    return dict(_CODE)


def set_code(found: dict):
    """Use found (code() of another process of the same run) as this
    process's code(), so a worker process's fingerprints are its
    starter's, even when the checkout changed meanwhile."""
    global _CODE
    _CODE = dict(found)


def code_name(found=None) -> str:
    """code() as one line: the commit, then +<the first 12 hex digits of
    the changes> when there are some."""
    found = found or code()
    return (found["commit"] or "packages") + (f"+{found['changes'][:12]}" if found["changes"] else "")


def _found_code() -> dict:
    import hashlib
    from pathlib import Path

    def git(*command, text=True):
        try:
            return subprocess.run([program("git"), "-C", str(Path(__file__).parent), *command], capture_output=True,
                                  text=text, check=True).stdout
        except (MissingTool, OSError, subprocess.CalledProcessError):
            return None

    top = git("rev-parse", "--show-toplevel")
    commit = git("rev-parse", "HEAD")
    if top is None or commit is None:
        return {"commit": None, "changes": _installed_code()}
    top = Path(top.strip())
    digest, changed = hashlib.sha256(), False
    difference = git("-C", str(top), "diff", "HEAD", "--binary", "--no-color", "--no-ext-diff", "--no-textconv",
                     text=False) or b""
    if difference:
        digest.update(difference)
        changed = True
    for path in sorted(filter(None, (git("-C", str(top), "ls-files", "--others", "--exclude-standard", "-z",
                                         text=False) or b"").split(b"\0"))):
        try:
            content = (top / path.decode()).read_bytes()
        except OSError:
            continue  # gone meanwhile, or a folder git lists
        digest.update(b"\0" + path + b"\0" + hashlib.sha256(content).digest())
        changed = True
    return {"commit": commit.strip(), "changes": digest.hexdigest() if changed else None}


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
