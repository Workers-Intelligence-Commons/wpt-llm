"""The outside programs wpt-dataset runs besides git: mercurial and GnuPG."""

import shutil

from .errors import MissingTool


def program(name: str) -> str:
    """The absolute path of a program on the search path. Handing git an
    absolute path also matters on Windows, where Git for Windows would
    otherwise run its own bundled GnuPG, which rejects Windows paths."""
    path = shutil.which(name)
    if path is None:
        raise MissingTool(f"{name} is not installed (or not on the search path)")
    return path
