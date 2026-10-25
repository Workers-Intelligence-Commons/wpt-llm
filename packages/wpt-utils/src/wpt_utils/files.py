"""Files: the sha256 of a file, the one way WPT hashes a file's bytes, so
the store, the model folders and the run records name the same file alike;
and a lock the operating system holds on a file for one process, which the
service layer and a local store share."""

import errno
import hashlib
import os

CHUNK = 1 << 20  # bytes read at once: a file of any size is hashed in little memory


def sha256_file(path) -> str:
    """The sha256 of the file at path, as 64 lowercase hex digits. OSError
    when it cannot be read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def lock_file(path) -> int:
    """An open descriptor of path (created when missing), locked for this
    process (fcntl.flock on Linux and macOS, msvcrt.locking on Windows);
    waits while another process holds it. The system releases the lock
    when the process ends, crash included, so none is left behind. Locks
    between processes only: threads of one process need their own lock."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # gives up after 10 seconds: try again
                    break
                except OSError as error:
                    if error.errno != errno.EDEADLOCK:
                        raise
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def unlock_file(descriptor: int):
    """Release a lock_file lock and close its descriptor."""
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)  # closing releases the lock too
