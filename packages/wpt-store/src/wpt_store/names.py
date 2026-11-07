"""The names a store allows for session IDs, the stages and results of its
cache, the things a session holds, and run records and their files."""

import re
from dataclasses import dataclass

NAME_CHARACTERS = "A-Za-z0-9._-"  # what a name may hold, as a regular expression class
NAME = re.compile(f"[{NAME_CHARACTERS}]+")
ROOT = ("index", "cache", "cut-key")  # what a store's root holds besides its sessions: no session takes these names


@dataclass(frozen=True)
class File:
    """One stored file, as whatever names it knows it: its key, without the
    .zst the store adds when it keeps the file compressed, and the sha256
    of its bytes, which every read checks."""

    key: str
    sha256: str

    @classmethod
    def of(cls, reference) -> "File":
        """The File a record's reference to a file names: {"key", "sha256"}."""
        return cls(reference["key"], reference["sha256"])

    def reference(self) -> dict:
        """The one shape a record refers to a file by: {"key", "sha256"}."""
        return {"key": self.key, "sha256": self.sha256}


def check_name(part):
    """ValueError unless part can name a stage, a result or something a
    session holds: letters, digits, '.', '_' and '-' only, not starting
    with '.'."""
    if not NAME.fullmatch(part) or part.startswith("."):
        raise ValueError(f"names use letters, digits, '.', '_' and '-' only, and do not start with '.': {part!r}")


def check_session(session):
    """ValueError unless session can be a session ID: a name (check_name)
    that is not one of the store's root files or folders (ROOT)."""
    check_name(session)
    if session in ROOT:
        raise ValueError(f"{session!r} is a name the store keeps for itself, not a session ID")


def check_record_name(part):
    """ValueError unless part can name a record file: what check_name allows."""
    check_name(part)


def as_record_name(text) -> str:
    """text made into a part of a run's or record file's name that
    check_record_name accepts: '-' for each character it does not allow,
    without the leading '.'s, and '-' for nothing left."""
    return re.sub(f"[^{NAME_CHARACTERS}]", "-", str(text)).lstrip(".") or "-"
