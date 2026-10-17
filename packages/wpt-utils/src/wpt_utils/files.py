"""The sha256 of a file: the one way WPT hashes a file's bytes, so the
store, the model folders and the run records name the same file alike."""

import hashlib

CHUNK = 1 << 20  # bytes read at once: a file of any size is hashed in little memory


def sha256_file(path) -> str:
    """The sha256 of the file at path, as 64 lowercase hex digits. OSError
    when it cannot be read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()
