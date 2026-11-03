"""Reads of a stored compressed file with several requests in flight, for a
kind whose every request waits (S3): the bytes of a read are fetched in
blocks at once, and while a reader works on one block the blocks after it
are already on their way.

A file's frames are grouped in blocks of FRAMES frames; a block is one
request for its compressed bytes. Fetch(key, table, first, last, read)
gives the compressed bytes of frames first to last, the blocks that hold
them requested together on REQUESTS threads at most. When a read starts
where the one before it ended on the same file, or at the start of a file (a reader going through a
file or a bundle in order), the blocks that follow are requested too, as
long as the compressed bytes of the blocks requested and not yet taken
stay within LIMIT (also when a read takes frames of the last block taken, past its middle); a reader that has gone elsewhere leaves its blocks to
the next reader that needs room (they are dropped first). The bytes a
read gets are the stored bytes, whatever the timing.

A request that fails raises at the read that takes its block, with the
error the kind raised; a block requested ahead and never taken costs
nothing more than its bytes held.
"""

import collections
import threading
from concurrent.futures import ThreadPoolExecutor

# requests in flight at once for the reads of one process: R2 gives each connection a few MiB/s, so a reader's
# link fills only with many connections at once
REQUESTS = 8
# compressed bytes of blocks requested ahead and not yet taken, for the reads of one process: about 8 blocks of
# 2 to 8 MB, enough to keep REQUESTS requests busy while a reader works, small enough for many worker processes
LIMIT = 64 << 20
FRAMES = 8  # frames in one block: a request of about 1 MB of text compressed 3 to 4 times per frame, with the wait paid once


class Ahead:
    def __init__(self, requests=REQUESTS, limit=LIMIT, frames=FRAMES):
        self.requests, self.limit, self.frames = requests, limit, frames
        self._executor = None
        self._lock = threading.Lock()
        self._held = collections.OrderedDict()  # (key, block) -> (future, compressed bytes), oldest first
        self._next = {}  # key -> the block after the last one a read took

    def cover(self, table, first, last):
        """The first and last frame of the blocks that hold frames first to
        last: what fetch gives."""
        return first // self.frames * self.frames, min((last // self.frames + 1) * self.frames, len(table.sizes)) - 1

    def fetch(self, key, table, first, last, read) -> bytes:
        """The stored bytes of the frames cover() gives for first and last
        of the file key, which table describes. read(start, length) reads
        stored bytes."""
        wanted = range(first // self.frames, last // self.frames + 1)
        with self._lock:
            sequential = wanted[0] == 0 or self._next.get(key) == wanted[0]  # a read from the start of a file goes through it
            futures = []
            for block in wanted:
                held = self._held.pop((key, block), None)
                sequential = sequential or held is not None
                futures.append(held[0] if held is not None else self._request(table, block, read))
            self._next[key] = wanted[-1] + 1
            if sequential:
                self._ahead(key, table, wanted[-1] + 1, read)
        try:
            data = b"".join(future.result() for future in futures)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return data

    def reached(self, key, table, frame, read):
        """A read took frame from the frames kept, not from a request: when
        that is in the second half of the last block a read took, the reader
        is going through the file, and the blocks after it are requested."""
        with self._lock:
            block = frame // self.frames
            if self._next.get(key) == block + 1 and frame % self.frames >= self.frames // 2:
                self._ahead(key, table, block + 1, read)

    def forget(self, key):
        """Drop what is held about a file that changed or went."""
        with self._lock:
            self._next.pop(key, None)
            for held in [held for held in self._held if held[0] == key]:
                self._held.pop(held)[0].cancel()

    def _span(self, table, block):
        start = block * self.frames
        return table.stored[start], table.stored[min(start + self.frames, len(table.sizes))]

    def _request(self, table, block, read):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.requests, thread_name_prefix="read-ahead")
        start, end = self._span(table, block)
        return self._executor.submit(read, start, end - start)

    def _ahead(self, key, table, block, read):
        """Request the blocks from block on, while the bytes held allow."""
        blocks = -(-len(table.sizes) // self.frames)
        held = sum(size for _, size in self._held.values())
        for block in range(block, blocks):
            if (key, block) in self._held:
                continue
            start, end = self._span(table, block)
            for other in [other for other in self._held if other[0] != key]:
                if held + end - start <= self.limit:
                    break
                gone = self._held.pop(other)  # a reader that went elsewhere
                gone[0].cancel()
                held -= gone[1]
            if held + end - start > self.limit:
                return
            self._held[(key, block)] = (self._request(table, block, read), end - start)
            held += end - start
