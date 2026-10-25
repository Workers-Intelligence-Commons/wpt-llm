"""The token shard format shared by the dataset build (writer) and training
(reader).

A shard is a Parquet file with one row per document pass:

    tokens    list<uint32>   the document's tokens, ending with END
    source    string         source id
    document  string         document id (<source id>/<path>)
    pool      string         the mixture pool it was drawn for ("held-out" for evaluation)
    pass      int32          which repetition of the pool this row belongs to

A shard is identified twice: by the sha256 of its file (the store object)
and by its token hash, which covers only the rows' content. A newer pyarrow
may write different file bytes for the same rows; the token hash stays the
same, so it is what repeatability checks compare.
"""

import bisect
import concurrent.futures
import hashlib
import json
import os
import struct

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wpt_store import CorruptObject
from wpt_utils.files import sha256_file

SCHEMA = pa.schema([
    ("tokens", pa.list_(pa.uint32())),
    ("source", pa.string()),
    ("document", pa.string()),
    ("pool", pa.string()),
    ("pass", pa.int32()),
])
ROW_GROUP_ROWS = 1024


class TokenHash:
    """sha256 over rows: each field length-prefixed, tokens as little-endian uint32."""

    def __init__(self):
        self._h = hashlib.sha256()

    def add(self, tokens, source: str, document: str, pool: str, pass_: int):
        for text in (source, document, pool):
            data = text.encode()
            self._h.update(struct.pack("<Q", len(data)) + data)
        array = np.asarray(tokens, dtype="<u4")
        self._h.update(struct.pack("<iQ", pass_, len(array)) + array.tobytes())

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def write_shard(path, rows) -> str:
    """rows: (tokens, source, document, pool, pass) tuples. Returns the token hash."""
    digest = TokenHash()
    columns = {name: [] for name in SCHEMA.names}
    for tokens, source, document, pool, pass_ in rows:
        digest.add(tokens, source, document, pool, pass_)
        for name, value in zip(SCHEMA.names, (np.asarray(tokens, dtype=np.uint32), source, document, pool, pass_)):
            columns[name].append(value)
    table = pa.table(columns, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd", row_group_size=ROW_GROUP_ROWS)
    return digest.hexdigest()


def read_shard(path):
    """Yield (tokens as numpy uint32, source, document, pool, pass), in order."""
    file = pq.ParquetFile(path)
    for i in range(file.num_row_groups):
        group = file.read_row_group(i)
        tokens = group.column("tokens")
        others = [group.column(n).to_pylist() for n in ("source", "document", "pool", "pass")]
        for j in range(group.num_rows):
            yield (tokens[j].values.to_numpy(zero_copy_only=False), *(column[j] for column in others))


def read_verified(store, shard) -> list:
    """The rows of a shard recorded as {"object": sha256, "token_sha256": ...},
    in order. The store checks the file; this checks the rows against the
    token hash. Raises CorruptObject when they differ."""
    rows, digest = [], TokenHash()
    for row in read_shard(store.path(shard["object"])):
        digest.add(*row)
        rows.append(row)
    if digest.hexdigest() != shard["token_sha256"]:
        raise CorruptObject(f"shard {shard['object']}: its rows do not match their recorded token hash")
    return rows


def documents(store, shards, passes=None):
    """Yield (document id, source id, tokens) of every document in these
    shards once, in shard order (a document the mixture repeats comes
    once), each shard checked (read_verified) before its documents are
    given; one shard is in memory at a time, and for a remote store each
    shard's local copy is released once read. The disk then holds at most
    one shard only when the shards were not already local (wpt-train
    --stream, whose ShardStream released each one, or a wpt-train run
    whose steps read no text windows); copies already there, as after
    wpt-train without --stream, go one by one as they are read. passes, a collections.Counter when given, counts
    each document's rows, the times the shards hold it: whole once every
    document was given."""
    seen = set()
    for shard in shards:
        rows = read_verified(store, shard)
        store.release(shard["object"])  # its rows are in memory now
        for tokens, source, document, _, _ in rows:
            if passes is not None:
                passes[document] += 1
            if document not in seen:
                seen.add(document)
                yield document, source, tokens


def token_file(store, shards) -> Path:
    """A local file holding the tokens of these shards' rows, in order, as
    little-endian uint32, for reading from disk on demand (np.memmap).

    It is built from the shards, each checked against its token hash, into
    the store's scratch folder, and named after the shards. A later call
    reuses it only if it still matches the sha256 recorded when it was built;
    otherwise it is built again. Raises CorruptObject (leaving no file) when
    a shard's rows do not match their token hash."""
    key = hashlib.sha256(json.dumps([[s["object"], s["token_sha256"]] for s in shards]).encode()).hexdigest()
    folder = store.scratch / "tokens"
    path, mark = folder / f"{key}.u32", folder / f"{key}.sha256"
    if path.is_file() and mark.is_file() and sha256_file(path) == mark.read_text().strip():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f"{key}.{os.getpid()}.partial"
    try:
        with open(partial, "wb") as out:
            for shard in shards:
                digest = TokenHash()
                for row in read_shard(store.path(shard["object"])):
                    digest.add(*row)
                    out.write(np.asarray(row[0], dtype="<u4").tobytes())
                if digest.hexdigest() != shard["token_sha256"]:
                    raise CorruptObject(f"shard {shard['object']}: its rows do not match their recorded token hash")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    mark.write_text(sha256_file(path) + "\n")
    return path


class ShardStream:
    """Training tokens read a few shards at a time, for a dataset larger than
    the disk of the machine that trains.

    The shards are walked in order at the speed training uses tokens: before
    a step that has used `consumed` tokens, the shard holding token
    `consumed` (counted over every shard, starting again after the last) and
    the next window - 1 shards are the ones in use, and windows are drawn
    from them only. The shard after them is prepared in the background, and
    a shard that leaves the set loses its token file and, for a remote
    store, its local copy. Which shards are in use depends only on
    `consumed`, so a resumed run and any number of processes use the same
    ones. Each shard's rows are checked against its token hash when its
    token file is built (see token_file)."""

    def __init__(self, store, shards, window: int):
        if not shards:
            raise ValueError("no shards to stream")
        self.store, self.shards = store, shards
        self.window = max(1, min(window, len(shards)))
        self.ends = []
        for shard in shards:
            self.ends.append((self.ends[-1] if self.ends else 0) + shard["tokens"])
        self._open = {}
        self._ahead = None
        self._pool = concurrent.futures.ThreadPoolExecutor(1)

    def __len__(self):
        return self.ends[-1]

    def first(self, consumed: int) -> int:
        """The index of the first shard in use after `consumed` tokens."""
        return bisect.bisect_right(self.ends, consumed % self.ends[-1])

    def at(self, consumed: int) -> "Joined":
        """The tokens of the shards in use after `consumed` tokens, as one array."""
        first = self.first(consumed)
        wanted = [(first + i) % len(self.shards) for i in range(self.window)]
        for index in list(self._open):
            if index not in wanted:
                self._close(index)
        for index in wanted:
            if index not in self._open:
                if self._ahead is not None and self._ahead[0] == index:
                    path, self._ahead = self._ahead[1].result(), None
                else:
                    path = self._prepare(index)
                self._open[index] = np.memmap(path, dtype="<u4", mode="r")
        upcoming = (first + self.window) % len(self.shards)
        if upcoming not in wanted and (self._ahead is None or self._ahead[0] != upcoming):
            if self._ahead is not None:
                self._ahead[1].result()
            self._ahead = (upcoming, self._pool.submit(self._prepare, upcoming))
        return Joined([self._open[i] for i in wanted])

    def close(self):
        self._pool.shutdown(wait=True)
        for index in list(self._open):
            self._close(index)
        if self._ahead is not None:
            _remove(self._ahead[1].result())
            self._ahead = None

    def _prepare(self, index):
        shard = self.shards[index]
        path = token_file(self.store, [shard])
        self.store.release(shard["object"])  # its tokens are in the token file now
        return path

    def _close(self, index):
        array = self._open.pop(index)
        path = Path(array.filename)
        del array
        _remove(path)


def _remove(path: Path):
    """A token file and its mark."""
    for file in (path, path.with_suffix(".sha256")):
        try:
            file.unlink(missing_ok=True)
        except PermissionError:
            pass  # another process still reads it (Windows); the scratch folder can be cleared later


class Joined:
    """Several token arrays read as one: a slice may span two of them."""

    def __init__(self, parts):
        self.parts = parts
        self.ends = []
        for part in parts:
            self.ends.append((self.ends[-1] if self.ends else 0) + len(part))

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Joined reads contiguous slices only")
        start, stop, _ = key.indices(len(self))
        pieces = []
        while start < stop:
            i = bisect.bisect_right(self.ends, start)
            base = self.ends[i - 1] if i else 0
            end = min(stop, self.ends[i])
            pieces.append(np.asarray(self.parts[i][start - base:end - base]))
            start = end
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype="<u4")


def sources_of(store, shards) -> set[str]:
    """The source ids in these shards' rows: from each shard's record when
    the build wrote them there, else by reading only that column."""
    found = set()
    for shard in shards:
        if "sources" in shard:
            found |= set(shard["sources"])
        else:
            found |= set(pq.read_table(store.path(shard["object"]), columns=["source"]).column("source").to_pylist())
    return found


def token_hash(path) -> str:
    digest = TokenHash()
    for row in read_shard(path):
        digest.add(*row)
    return digest.hexdigest()
