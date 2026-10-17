"""The outside programs WPT runs (mercurial, GnuPG, Bun, git), and the
commit of the WPT repository a command runs from.

program(name) is how every WPT package finds such a program: a missing one
raises MissingTool, which names it, instead of an error from deep inside a
command. repository() is the one place that reads the repository commit,
which provenance.json and each run's record name.
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
    """{"repository_commit": the commit checked out in the current folder,
    "repository_dirty": whether its working tree has changes}; each None
    when there is no git repository or no git."""
    def git(*command):
        try:
            return subprocess.run([program("git"), *command], capture_output=True, text=True, check=True).stdout.strip()
        except (MissingTool, OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"repository_commit": git("rev-parse", "HEAD"),
            "repository_dirty": bool(status) if status is not None else None}
