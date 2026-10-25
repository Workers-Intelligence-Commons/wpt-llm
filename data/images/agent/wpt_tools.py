"""What the programs that answer tools share (docs/training.md, "The
agent image"). A program reads the call's arguments as a JSON object on
its standard input, prints its result, and exits 1 with its message on
standard error when the call fails.

A program runs in group 1001, which reads the task's hidden files under
TOOL_FILES, while the model's own tools cannot. A tool that reads the
model's files must therefore never be led there: in_work() opens a path
only inside WORK, one folder at a time and never through a symbolic link,
so neither a path such as ../tmp nor a link the model made (even one it
swaps while the program runs) reaches anything outside /work.
"""

import errno
import json
import os
import posixpath
import sys

WORK = "/work"
TOOL_FILES = "/tmp/wpt-tools"  # the files only the tools' programs read (wpt_agent.sandbox)


def fail(message: str):
    """End the call as failed, with message as its result."""
    print(message, file=sys.stderr)
    sys.exit(1)


def arguments(required=(), optional=()) -> dict:
    """The call's arguments; the call fails unless each required one, and
    each optional one given, is a string."""
    try:
        given = json.load(sys.stdin)
    except ValueError:
        fail("the arguments are not a JSON object")
    if not isinstance(given, dict):
        fail("the arguments are not a JSON object")
    wrong = [name for name in required if not isinstance(given.get(name), str)]
    wrong += [name for name in optional if name in given and not isinstance(given[name], str)]
    if wrong:
        fail(f"needs the string arguments {', '.join(required)}; missing or not a string: {', '.join(wrong)}")
    return given


def relative(path: str, root: str = WORK) -> str | None:
    """path (relative to root, or absolute) as a path inside root: "" for
    root itself, None when it leads outside."""
    full = posixpath.normpath(posixpath.join(root, path))
    if full == root:
        return ""
    return full[len(root) + 1:] if full.startswith(root + "/") else None


def in_work(path: str, folder: bool = False, root: str = WORK) -> int:
    """A file descriptor of path inside root (/work), a folder's when
    folder; the call fails when the path leads outside root, goes through
    a symbolic link, or names nothing."""
    inside = relative(path, root)
    if inside is None:
        fail(f"{path}: outside {root}")
    parts = inside.split("/") if inside else []
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    for number, part in enumerate(parts, start=1):
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK  # a pipe never blocks the opening
        flags |= os.O_DIRECTORY if number < len(parts) or folder else 0
        try:
            opened = os.open(part, flags, dir_fd=descriptor)
        except FileNotFoundError:
            fail(f"{path}: no such file or folder")
        except NotADirectoryError:
            fail(f"{path}: not a folder")
        except PermissionError:
            fail(f"{path}: permission denied")
        except OSError as error:
            if error.errno == errno.ELOOP:  # O_NOFOLLOW met a symbolic link
                fail(f"{path}: goes through a symbolic link, which this tool does not follow")
            fail(f"{path}: {error.strerror}")
        finally:
            os.close(descriptor)
        descriptor = opened
    return descriptor


def text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")
