"""Files that hold many documents, one per record, each with its own license.

A `records` format entry names the fields of a record: `text` (the
document), `license`, and optionally `id` and `credit`. A field with dots
goes into nested objects (`metadata.license`); a value on the way that is a
string holding a JSON object is read as that object, as some datasets store
their metadata.

A record enters only when its license value is one of the `values` of a
license entry covering its file (see sources.check_license_map); any other
record is left out and counted. An optional `keep` list narrows this
further: each rule names a field and the values it may hold (`in`), and a
record enters only when every rule holds, for example to keep the records
of the one publisher a license covers. A record whose text is missing or
empty gives no document.

The file kind comes from its name: `.jsonl`, `.json` or `.ndjson` (JSON
Lines: one object per line), each optionally compressed as `.gz` or `.zst`,
or `.parquet`. Records are read one at a time from the store's file, so a
file of gigabytes is never held in memory whole.
"""

import gzip
import io
import json
from collections.abc import Iterator
from compression import zstd
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import pyarrow.parquet as pq

from .errors import InvalidFile

JSON_LINES = (".jsonl", ".json", ".ndjson")
COMPRESSED = {".gz": gzip.open, ".zst": zstd.open}
PARQUET_BATCH = 1024


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    license: str  # the SPDX expression of the license entry that lists its value
    credit: dict  # credit field -> value, as the record holds it


@dataclass
class Skipped:
    """Records left out of one file, by reason."""
    unlisted: dict  # license value -> count
    empty: int = 0
    outside: dict = dataclass_field(default_factory=dict)  # (field, value) a `keep` rule refused -> count


def rows(file: Path, name: str) -> Iterator[dict]:
    """Every record of a records file, in order."""
    lower = name.lower()
    if lower.endswith(".parquet"):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=PARQUET_BATCH):
            yield from batch.to_pylist()
        return
    opener = next((open_ for suffix, open_ in COMPRESSED.items() if lower.endswith(suffix)), None)
    base = lower.removesuffix(".gz").removesuffix(".zst")
    if not base.endswith(JSON_LINES):
        raise InvalidFile(f"{name}: a records file is .jsonl, .json, .ndjson (optionally .gz or .zst) or .parquet")
    with (opener(file, "rb") if opener else open(file, "rb")) as raw:
        for number, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InvalidFile(f"{name} line {number}: {error}") from error
            if not isinstance(row, dict):
                raise InvalidFile(f"{name} line {number}: a record is a JSON object")
            yield row


def field(row: dict, name: str):
    """The value of a dotted field, or None when any part is missing."""
    value = row
    for part in name.split("."):
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _text(value) -> str:
    """A field's value as listed in a source.json: strings as they are, any
    other value as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def records(file: Path, name: str, entry: dict, licenses: dict[str, str], skipped: Skipped) -> Iterator[Record]:
    """The records of one file that enter: licenses maps each listed license
    value to its entry's SPDX expression. Ids repeat at no point in a file."""
    seen = set()
    for position, row in enumerate(rows(file, name), start=1):
        refused = next(((rule["field"], value) for rule in entry.get("keep", [])
                        for value in [_text(field(row, rule["field"]))] if value not in rule["in"]), None)
        if refused:
            skipped.outside[refused] = skipped.outside.get(refused, 0) + 1
            continue
        value = _text(field(row, entry["license"]))
        if value not in licenses:
            skipped.unlisted[value] = skipped.unlisted.get(value, 0) + 1
            continue
        text = field(row, entry["text"])
        if not isinstance(text, str) or not text.strip():
            skipped.empty += 1
            continue
        record_id = field(row, entry["id"]) if "id" in entry else position
        record_id = str(record_id) if record_id is not None else str(position)
        if record_id in seen:
            raise InvalidFile(f"{name}: record id {record_id!r} appears twice; name another `id` field")
        seen.add(record_id)
        credit = {name_: field(row, name_) for name_ in entry.get("credit", [])}
        yield Record(record_id, text, licenses[value], credit)
