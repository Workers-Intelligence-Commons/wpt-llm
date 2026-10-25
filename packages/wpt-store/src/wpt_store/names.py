"""The names a store allows for refs and record folders and files."""

import re

NAME_CHARACTERS = "A-Za-z0-9._-"  # what a ref or record name may hold, as a regular expression class
NAME = re.compile(f"[{NAME_CHARACTERS}]+")


def check_ref(kind, name):
    for part in (kind, name):
        if not NAME.fullmatch(part):
            raise ValueError(f"ref parts use letters, digits, '.', '_' and '-' only: {part!r}")


def check_record_name(part):
    """ValueError unless part can name a record folder or file: letters,
    digits, '.', '_' and '-' only, not starting with '.'."""
    if not NAME.fullmatch(part) or part.startswith("."):
        raise ValueError(f"record folders and files use letters, digits, '.', '_' and '-' only, "
                         f"and do not start with '.': {part!r}")


def as_record_name(text) -> str:
    """text made into a part of a record folder or file name that
    check_record_name accepts: '-' for each character it does not allow,
    without the leading '.'s, and '-' for nothing left."""
    return re.sub(f"[^{NAME_CHARACTERS}]", "-", str(text)).lstrip(".") or "-"
