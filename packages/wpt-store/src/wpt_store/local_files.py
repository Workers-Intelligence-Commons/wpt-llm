"""Files on this machine's disk that one live process owns: a bundle being
filled (wpt_store.sessions), a temporary copy of the store database and a
session's rows (wpt_store.database), all in the store's cache folder.

Each one lies beside a lock file (its name, then LOCK) that the owning
process holds; the system releases the lock when the process ends, crash
included. A file whose lock can be taken is one no live process owns
(unowned). Removing one removes its files first and its lock file last,
so a lock file left alone means a removal that did not finish. A lock is
held only once its lock file is still the one at that name: a process that
waited for a lock while another removed its file tries again on the new
one, so two processes never both hold one file.
"""

import os
import threading
from pathlib import Path

from wpt_utils.files import lock_file, try_lock_file, unlock_file

LOCK = ".lock"  # the end of the name of the lock file beside a local file
_held, _held_lock = set(), threading.Lock()  # the local files this process holds the lock of, by path()


class LocalFile:
    """A local file (path(), and the files beside it whose names add one
    of endings to its name) and its lock, held by this process: taken here
    (waiting while another process holds it), or the descriptor lock that
    unowned took."""

    def __init__(self, path, endings=("",), lock=None):
        self.base, self.endings = Path(path), endings
        self.base.parent.mkdir(parents=True, exist_ok=True)
        with _held_lock:
            _held.add(self.base)
        try:
            self.lock = lock if lock is not None else _take(self.path(LOCK), wait=True)
        except BaseException:
            with _held_lock:
                _held.discard(self.base)
            raise

    def path(self, ending="") -> Path:
        return self.base.with_name(self.base.name + ending)

    def release(self):
        """The lock released, the files left."""
        if self.lock is not None:
            unlock_file(self.lock)
            self.lock = None
            with _held_lock:
                _held.discard(self.base)

    def remove(self):
        """The files removed, then the lock released and its file removed."""
        if self.lock is None:
            return
        try:
            for ending in self.endings:
                self.path(ending).unlink(missing_ok=True)
        finally:
            self.release()
            self.path(LOCK).unlink(missing_ok=True)


def unowned(folder, start, endings=("",)) -> list[LocalFile]:
    """The local files in folder whose name starts with start that no live
    process owns, their locks taken: those found by any of their files or
    by their lock file alone. Never one this process holds (the system's
    lock does not keep it from taking its own on every system)."""
    folder = Path(folder)
    bases = set()
    for path in folder.glob(start + "*") if folder.is_dir() else ():
        name = path.name.removesuffix(LOCK) if path.name.endswith(LOCK) else path.name
        for ending in sorted(endings, key=len, reverse=True):
            if ending and name.endswith(ending):
                name = name.removesuffix(ending)
                break
        bases.add(folder / name)
    taken = []
    for base in sorted(bases):
        with _held_lock:
            if base in _held:
                continue
        descriptor = _take(base.with_name(base.name + LOCK), wait=False)
        if descriptor is not None:
            taken.append(LocalFile(base, endings, descriptor))
    return taken


def _take(lock, wait) -> int | None:
    """The lock of the lock file at lock, held: its descriptor, once the
    file locked is still the one at that name; None, without wait, while
    another process holds it or after it removed it."""
    while True:
        try:
            descriptor = lock_file(lock) if wait else try_lock_file(lock)
        except FileNotFoundError:  # its folder removed meanwhile
            if wait:
                raise
            return None
        if descriptor is None:
            return None
        try:
            current = os.path.samestat(os.fstat(descriptor), os.stat(lock))
        except FileNotFoundError:
            current = False
        if current:
            return descriptor
        unlock_file(descriptor)  # its owner removed it while this process waited
        if not wait:
            return None
