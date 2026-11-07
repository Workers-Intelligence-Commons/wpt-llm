"""A source's files as one normal tar: `tar` lists and unpacks it by hand.

Tar writes the members into any object with write() (a blocks.Writer, which
makes it the source's .tar.zst), in the POSIX format that holds a path or a
link of any length: a regular file's bytes follow its header, a symlink is
a header naming its target, and a hard link a header naming an earlier
member, so a content a source holds under several paths is written once.
Every header is the same on any machine: no time, no owner.

Placed says where a file's bytes lie, as a range of the tar's own bytes,
so a reader takes them without the rest.

Failures: what out.write raises; OSError for a local file shorter than its
stated length.
"""

import tarfile
from dataclasses import dataclass

CHUNK = 1 << 20  # bytes of a local file read at once
BLOCK = tarfile.BLOCKSIZE
MODE = {"100644": 0o644, "100755": 0o755, "120000": 0o777}  # a tree's modes (wpt_dataset.manifest) as a tar holds them


@dataclass(frozen=True)
class Placed:
    """Where one file of a source lies: bytes [start, start + length) of
    the original bytes of its .tar.zst, the store file archive, as the
    version of that file its sha256 `version` names; sha256 is the file's
    own. A symlink has no bytes in a tar: text holds its target."""

    archive: str
    version: str
    start: int
    length: int
    sha256: str
    text: bytes | None = None


class Tar:
    """Members written into out as they come; close() ends the tar. size
    is the tar's bytes so far."""

    def __init__(self, out):
        self.out, self.size = out, 0

    def _header(self, path, mode, kind=tarfile.REGTYPE, size=0, link="") -> int:
        """A member's header written: where its bytes start."""
        info = tarfile.TarInfo(path)
        info.mode, info.type, info.size, info.linkname = MODE[mode], kind, size, link
        self._write(info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape"))
        return self.size

    def _write(self, data):
        self.out.write(data)
        self.size += len(data)

    def _pad(self):
        if left := -self.size % BLOCK:
            self._write(b"\0" * left)

    def add(self, path: str, mode: str, data: bytes) -> int:
        """A regular file: where its bytes start."""
        start = self._header(path, mode, size=len(data))
        self._write(data)
        self._pad()
        return start

    def add_file(self, path: str, mode: str, file, length: int) -> int:
        """A regular file of length bytes from the local file `file`, never
        whole in memory: where its bytes start."""
        start = self._header(path, mode, size=length)
        left = length
        with open(file, "rb") as source:
            while left:
                block = source.read(min(CHUNK, left))
                if not block:
                    raise OSError(f"{file}: shorter than its {length} bytes")
                self._write(block)
                left -= len(block)
        self._pad()
        return start

    def link(self, path: str, mode: str, target: str):
        """A hard link to the earlier member target."""
        self._header(path, mode, tarfile.LNKTYPE, link=target)

    def symlink(self, path: str, target: bytes):
        self._header(path, "120000", tarfile.SYMTYPE, link=target.decode("utf-8", "surrogateescape"))

    def close(self):
        """The tar's end: two empty blocks, then zeros up to tar's record size."""
        self._write(b"\0" * (2 * BLOCK))
        if left := -self.size % tarfile.RECORDSIZE:
            self._write(b"\0" * left)
