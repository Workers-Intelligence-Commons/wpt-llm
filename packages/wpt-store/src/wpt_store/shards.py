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
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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


def token_hash(path) -> str:
    digest = TokenHash()
    for row in read_shard(path):
        digest.add(*row)
    return digest.hexdigest()
