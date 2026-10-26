"""Compressed files a range of which can be read without the rest: zstd's
seekable format.

The bytes are cut into blocks of FRAME bytes (the last one shorter), each
compressed alone as one zstd frame, so any byte range is read by reading
and decompressing only the frames that hold it. After the frames comes
the seek table, a skippable frame that plain zstd ignores: each frame's
compressed and original sizes (two little-endian 32-bit numbers), then the
number of frames, a descriptor byte (0: no per-frame checksums) and the
format's magic number. The whole file is ordinary zstd: `zstd -d` gives
back the original bytes.

Writer compresses while the bytes come, several frames at once as
wpt_utils.parallel allows, the frames written in order. SeekTable reads a
file's table and says which compressed bytes hold a range; read() reads
a range of a local file through its table.
"""

import bisect
import struct
from compression import zstd
from concurrent.futures import ThreadPoolExecutor

from wpt_utils.parallel import Sizing, in_order

FRAME = 1 << 20  # original bytes per frame: the unit a range read decompresses
LEVEL = 3  # zstd's own default
SKIPPABLE = 0x184D2A5E  # the magic number of the skippable frame holding the seek table
SEEKABLE = 0x8F92EAB1  # the seekable format's magic number, last in the file
FOOTER = 9  # number of frames (4 bytes), descriptor (1), magic (4)


class BadTable(ValueError):
    """The end of a file is not a seek table this module writes."""


class Writer:
    """Bytes written into out (a binary file open for writing) as seekable
    zstd, compressed as they come: write() any number of times, then
    close(), which writes the seek table and gives (original size, stored
    size). flush() ends the frame being filled early, so every byte
    written so far is in out and in `sizes`. Up to `batch` frames wait to be compressed together; with an
    executor and a sizing they are compressed side by side
    (wpt_utils.parallel.in_order), otherwise one after the other."""

    def __init__(self, out, executor=None, sizing=None, batch=64):
        self.out, self.executor, self.sizing, self.batch = out, executor, sizing, batch
        self.sizes = []  # (compressed, original) of every frame written
        self._block = bytearray()  # bytes not yet a whole frame
        self._waiting = []  # whole frames not yet compressed

    def write(self, data):
        self._block += data
        while len(self._block) >= FRAME:
            self._waiting.append(bytes(self._block[:FRAME]))
            del self._block[:FRAME]
            if len(self._waiting) >= self.batch:
                self._compress()

    def flush(self):
        if self._block:
            self._waiting.append(bytes(self._block))
            self._block.clear()
        self._compress()

    def close(self):
        self.flush()
        table = b"".join(struct.pack("<II", *sizes) for sizes in self.sizes)
        table += struct.pack("<IBI", len(self.sizes), 0, SEEKABLE)
        self.out.write(struct.pack("<II", SKIPPABLE, len(table)) + table)
        return sum(original for _, original in self.sizes), sum(stored for stored, _ in self.sizes) + 8 + len(table)

    def _compress(self):
        blocks, self._waiting = self._waiting, []
        if self.executor is not None and len(blocks) > 1:
            frames = in_order(compress, blocks, self.executor, self.sizing)
        else:
            frames = map(compress, blocks)
        for block, frame in zip(blocks, frames):
            self.out.write(frame)
            self.sizes.append((len(frame), len(block)))


def compress(block: bytes) -> bytes:
    return zstd.compress(block, LEVEL)


def compressed(data: bytes, executor=None, sizing=None) -> bytes:
    """data as a whole seekable zstd file, in memory; its frames compressed
    as Writer does with that executor and sizing."""
    import io

    out = io.BytesIO()
    writer = Writer(out, executor, sizing)
    writer.write(data)
    writer.close()
    return out.getvalue()


def read(file, table: "SeekTable", start: int, length: int) -> bytes:
    """Original bytes [start, start + length) of a seekable zstd file open
    for reading (file), whose frames table describes; fewer when it ends
    first."""
    frames = table.frames(start, length)
    if not frames:
        return b""
    file.seek(table.stored[frames[0]])
    data = file.read(table.stored[frames[-1] + 1] - table.stored[frames[0]])
    at = table.stored[frames[0]]
    joined = b"".join(zstd.decompress(data[table.stored[i] - at:table.stored[i + 1] - at]) for i in frames)
    offset = start - table.original[frames[0]]
    return joined[offset:offset + length]


def parallel():
    """An executor and a sizing for Writer: as many frames at once as
    wpt_utils.parallel allows."""
    sizing = Sizing("compressions")
    return ThreadPoolExecutor(sizing.limit), sizing


class SeekTable:
    """Where each frame of a seekable zstd file lies: its compressed bytes
    and the original bytes it holds."""

    def __init__(self, sizes):
        self.sizes = sizes  # (compressed, original) per frame
        self.stored, self.original = [0], [0]  # where each frame starts, in the file and in the original bytes
        for stored, original in sizes:
            self.stored.append(self.stored[-1] + stored)
            self.original.append(self.original[-1] + original)

    @property
    def size(self):
        """The original bytes' length."""
        return self.original[-1]

    @staticmethod
    def tail(frames: int) -> int:
        """The bytes at the end of a file that hold a table of that many frames."""
        return 8 + 8 * frames + FOOTER

    @classmethod
    def parse(cls, end: bytes) -> "SeekTable | int":
        """The table at the end of a file, from its last bytes (at least
        FOOTER of them), or the number of last bytes needed when end is
        too short. BadTable when they are not a seek table."""
        if len(end) < FOOTER:
            raise BadTable("shorter than a seek table's footer")
        frames, descriptor, magic = struct.unpack("<IBI", end[-FOOTER:])
        if magic != SEEKABLE or descriptor != 0:
            raise BadTable("no seek table at the end")
        need = cls.tail(frames)
        if len(end) < need:
            return need
        skippable, length = struct.unpack("<II", end[-need:][:8])
        if skippable != SKIPPABLE or length != need - 8:
            raise BadTable("the seek table's frame is not whole")
        body = end[-need + 8:-FOOTER]
        return cls([struct.unpack_from("<II", body, 8 * i) for i in range(frames)])

    def frames(self, start: int, length: int) -> range:
        """The frames holding original bytes [start, start + length)."""
        if length <= 0 or start >= self.size:
            return range(0)
        first = bisect.bisect_right(self.original, start) - 1
        last = bisect.bisect_left(self.original, min(start + length, self.size)) - 1
        return range(first, last + 1)
