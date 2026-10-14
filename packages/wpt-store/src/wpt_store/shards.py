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

import hashlib
import json
import os
import struct

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wpt_store import CorruptObject, sha256_file

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


def sources_of(store, shards) -> set[str]:
    """The source ids in these shards' rows (reads only that column)."""
    return {source for shard in shards for source in
            pq.read_table(store.path(shard["object"]), columns=["source"]).column("source").to_pylist()}


def token_hash(path) -> str:
    digest = TokenHash()
    for row in read_shard(path):
        digest.add(*row)
    return digest.hexdigest()
