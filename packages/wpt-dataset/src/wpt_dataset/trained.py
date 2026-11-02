"""The documents a vocabulary was trained on, as one object of the store.

The list is one line per document, the JSON array `["<id>", "<sha256 of its
Markdown>"]` (JSON escapes any line break in an id), sorted by the line's bytes. A build writes the lines while it reads the documents,
into runs of a temporary folder that are sorted one at a time and merged, so
the list is never held in memory whole, and puts the sorted file in the
store. The lock names the object by its sha256 and the number of its lines.
"""

import heapq
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .errors import TrainedListMissing

RUN_LINES = 1 << 20  # lines sorted in memory at once


def line(document_id: str, sha256: str) -> bytes:
    """The list's line of a document."""
    return (json.dumps([document_id, sha256], ensure_ascii=False) + "\n").encode()


class TrainedList:
    """Collects (document id, sha256) pairs in any order, then puts them in the store as the sorted list:

        with TrainedList() as trained:
            trained.add(document_id, sha256)
            list_sha256, count = trained.put(store)
    """

    def __enter__(self):
        self._folder = tempfile.TemporaryDirectory(prefix="wpt-trained-")
        self._runs: list[Path] = []
        self._lines: list[bytes] = []
        self.count = 0
        return self

    def __exit__(self, *_):
        self._folder.cleanup()

    def add(self, document_id: str, sha256: str):
        self._lines.append(line(document_id, sha256))
        self.count += 1
        if len(self._lines) >= RUN_LINES:
            self._run()

    def _run(self):
        path = Path(self._folder.name) / f"run-{len(self._runs)}"
        self._lines.sort()
        with open(path, "wb") as out:
            out.writelines(self._lines)
        self._runs.append(path)
        self._lines = []

    def put(self, store) -> tuple[str, int]:
        """(the sha256 of the sorted list, the number of its lines), the list now an object of the store."""
        if self._lines:
            self._run()
        merged = Path(self._folder.name) / "list"
        files = [open(path, "rb") for path in self._runs]
        try:
            with open(merged, "wb") as out:
                out.writelines(heapq.merge(*files))
        finally:
            for file in files:
                file.close()
        return store.put_file(merged), self.count


def put_pairs(store, pairs) -> tuple[str, int]:
    """(sha256, count) of the list of (document id, sha256) pairs, which fit in memory, put in the store."""
    lines = sorted(line(document_id, sha256) for document_id, sha256 in pairs)
    return store.put_bytes(b"".join(lines)), len(lines)


def read(store, sha256: str) -> Iterator[tuple[str, str]]:
    """(document id, sha256) of every line of a list in the store, in the list's order; TrainedListMissing
    when the store does not hold it."""
    if not store.has(sha256):
        raise TrainedListMissing(f"the list of trained documents {sha256} is not in {store}")
    with open(store.path(sha256), "rb") as file:
        for raw in file:
            document_id, document_sha256 = json.loads(raw)
            yield document_id, document_sha256
