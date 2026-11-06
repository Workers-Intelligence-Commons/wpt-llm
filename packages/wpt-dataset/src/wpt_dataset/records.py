"""Files that hold many documents, one per record, each with its own license.

A `records` format entry names the fields of a record: `text` (the
document), optionally `id`, `title`, `newest` and `credit`. A field with dots goes
into nested objects (`metadata.license`); a value on the way that is a
string holding a JSON object is read as that object, as some datasets store
their metadata.

A record enters only when a license entry covers it (license_check.py):
when the entries covering its file have a field place, the entry that
expects the value of that field of the record; any other record is left
out and counted. A file whose one license entry has no field place is a
dataset that states one license for all its records, and every record
takes it. An optional `keep` list narrows
this further: each rule names a field and the values it may hold (`in`),
and a record enters only when every rule holds, for example to keep the
records of the one publisher a license covers. A record whose text is
missing or empty gives no document. A `title` field's value, when the
record holds one, becomes the document's first line, a `#` heading.

Ids repeat at no point in a file, unless the entry names a date field in
`newest`, which needs an `id` field every record holds: of the records that share an id in all the files of the source
the entry covers, only the one whose date (ISO 8601) is the latest is read,
and when dates are equal the later one (in path order, then in its file),
as a dataset that holds several revisions of one document needs; the
others are left out and counted, before any other rule.

That choice reads every file the entry covers, so it is made once and saved
in the store: the positions of the records to read, file by file, a 4-byte
number each, under the ref newest-index/<newest_key: the entry, the path and
sha256 of each file it covers and of its joined file, in path order, and
NEWEST_VERSION>. Nothing else enters that fingerprint, so any read of the same
files finds the index, whatever build or machine saved it.

The file kind comes from its name: `.jsonl`, `.json` or `.ndjson` (JSON
Lines: one object per line), each optionally compressed as `.gz` or `.zst`,
`.parquet`, or `.xml`. Records are read one at a time from the store's
file, so a file of gigabytes is never held in memory whole. Two kinds are
one document instead:

- With `records`, a `.json` file (optionally `.gz` or `.zst`) is one JSON
  document, and that dotted field of it holds the list of records, as a
  page scan holds its "full articles".
- An `.xml` file is one record, and each field is an XPath 1.0 expression
  read at the document, its prefixes those of the entry's `namespaces`: an
  expression giving nodes gives the string value of the first one (none
  when there is no node), any other value as it is. With a `join`, the
  record also reads one other file of the source's tree, the one of the
  join's `paths` whose path holds the same key: `key` is a regular
  expression searched in the record file's path and `joined_key` in each
  joined file's path, and a key is the expression's groups joined by one
  space (its whole match when it has none). The joined file's document is
  the XPath variable named by the join's `name`, an empty node set when no
  file holds the key; the joined files are read only that way and give no
  document of their own. A newspaper issue can take its text from the
  issue's full-text file and its license and credit from the metadata
  file of the same issue, fetched from another address into another
  folder (sources.py). XML is read without loading anything from
  outside the file (no external entity or DTD, no network): only the
  entities its own DTD declares are expanded.
"""

import gzip
import io
import json
import re
from collections.abc import Callable, Iterator
from compression import zstd
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lxml import etree

from wpt_telemetry import Telemetry, get_result_ref, set_result_ref
from wpt_utils.programs import fingerprint

from .errors import InvalidFile
from .manifest import Stored, glob_match_any

NEWEST_VERSION = 1  # raised by hand when a change to the code changes which records a `newest` index reads
JSON_LINES = (".jsonl", ".json", ".ndjson")
XML = ".xml"
COMPRESSED = {".gz": gzip.open, ".zst": zstd.open}
PARQUET_BATCH = 1024


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    license: str  # the SPDX expression of the license entry that lists its value
    credit: dict  # credit field -> value, as the record holds it
    title: str | None = None  # the `title` field's value, when the entry names one and the record holds it


@dataclass
class Skipped:
    """Records left out of one file, by reason."""
    unlisted: dict  # license value -> count
    empty: int = 0
    outside: dict = dataclass_field(default_factory=dict)  # (field, value) a `keep` rule refused -> count
    older: int = 0  # records replaced by a newer record of the same id (`newest`)
    untrained: dict = dataclass_field(default_factory=dict)  # license value of an entry that does not train -> count


def rows(file, name: str, within: str | None = None) -> Iterator[dict]:
    """Every record of a records file, in order; within: the dotted field
    of the file's one JSON document that holds them (`records`). file is a
    local Path or a file of the store (manifest.Stored), read as a stream
    from the store, never copied to the disk, but for Parquet, which is
    read from a local file (Stored.local)."""
    lower = name.lower()
    if lower.endswith(".parquet") and within is None:
        local = file.local() if isinstance(file, Stored) else file
        for batch in pq.ParquetFile(local).iter_batches(batch_size=PARQUET_BATCH):
            yield from batch.to_pylist()
        return
    opener = next((open_ for suffix, open_ in COMPRESSED.items() if lower.endswith(suffix)), None)
    base = lower.removesuffix(".gz").removesuffix(".zst")
    if within is not None:
        if not base.endswith(".json"):
            raise InvalidFile(f"{name}: a records file with `records` is one JSON document, .json "
                              f"(optionally .gz or .zst)")
        with file.open("rb") as stored, (opener(stored, "rb") if opener else stored) as raw:
            try:
                document = json.load(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise InvalidFile(f"{name}: {error}") from error
        found = field(document, within)
        if not isinstance(found, list) or not all(isinstance(row, dict) for row in found):
            raise InvalidFile(f"{name}: its field {within!r} is not a list of JSON objects")
        yield from found
        return
    if not base.endswith(JSON_LINES):
        raise InvalidFile(f"{name}: a records file is .jsonl, .json, .ndjson (optionally .gz or .zst), .parquet "
                          f"or .xml")
    with file.open("rb") as stored, (opener(stored, "rb") if opener else stored) as raw:
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


def joined_by(entry: dict, path: str) -> bool:
    """Whether a path is a file the entry's `join` reads, which gives no
    document of its own."""
    return "join" in entry and glob_match_any(path, entry["join"]["paths"])


def key_of(pattern: str, path: str) -> str | None:
    """A join's key in a path: the groups of pattern, searched in it,
    joined by one space, or its whole match when it has none; None when
    it is not found."""
    found = re.search(pattern, path)
    if found is None:
        return None
    return " ".join(group or "" for group in found.groups()) if found.groups() else found[0]


def join_index(entry: dict, paths) -> dict[str, str]:
    """{key: path} of the files among paths that the entry's join reads."""
    index = {}
    for path in paths:
        if joined_by(entry, path) and (key := key_of(entry["join"]["joined_key"], path)) is not None:
            if key in index:
                raise InvalidFile(f"{index[key]} and {path} hold the same join key {key!r}")
            index[key] = path
    return index


def joined_file(entry: dict, path: str, tree, indexes: dict):
    """The tree entry of the file the entry's `join` reads with this one (a
    tree's {path: entry}), or None when the entry has no join or no file
    holds the key. indexes keeps each join's index for the next file of the
    same tree, so a tree of many files is read once per join."""
    if "join" not in entry:
        return None
    rule = json.dumps(entry["join"], sort_keys=True)
    if rule not in indexes:
        indexes[rule] = join_index(entry, tree)
    partner = indexes[rule].get(key_of(entry["join"]["key"], path))
    return tree[partner] if partner else None


def records(file: "Path | Stored", name: str, entry: dict, license_field: str | None, licenses: dict, skipped: Skipped,
            joined: "Path | Stored | None" = None, newest: dict[str, np.ndarray] | None = None) -> Iterator[Record]:
    """The records of one file that enter: license_field is the field the
    license entries covering the file read (their field place), and
    licenses maps each value they expect to its entry's SPDX expression
    (with no field: None to the expression of the file's one entry), or to
    None when that entry does not train (its records are left out and
    counted as untrained); joined is the file the entry's `join` reads with
    this one, if any. Ids repeat at no
    point in a file, except under `newest`, where only the newest record of
    each id is read: newest is saved_newest_index over every file of the
    source the entry covers."""
    if ("newest" in entry) != (newest is not None):
        raise ValueError(f"{name}: an entry with `newest` is read with its saved_newest_index, and only such an entry")
    seen = set()
    wanted = iter(() if newest is None else newest.get(name, ()))  # the positions still to come, ascending
    following = next(wanted, None)
    for position, value_of in enumerate(_records_of(file, name, entry, joined), start=1):
        if newest is not None:
            if position != following:
                skipped.older += 1  # a newer record of its id is read instead
                continue
            following = next(wanted, None)
        refused = next(((rule["field"], value) for rule in entry.get("keep", [])
                        for value in [_text(value_of(rule["field"]))] if value not in rule["in"]), None)
        if refused:
            skipped.outside[refused] = skipped.outside.get(refused, 0) + 1
            continue
        value = _text(value_of(license_field)) if license_field is not None else None
        if value not in licenses:
            skipped.unlisted[value] = skipped.unlisted.get(value, 0) + 1
            continue
        if licenses[value] is None:  # its entry does not train (kept.py)
            skipped.untrained[value] = skipped.untrained.get(value, 0) + 1
            continue
        text = value_of(entry["text"])
        if not isinstance(text, str) or not text.strip():
            skipped.empty += 1
            continue
        record_id = _id(entry, value_of, position, name)
        if record_id in seen:
            raise InvalidFile(f"{name}: record id {record_id!r} appears twice; name another `id` field")
        seen.add(record_id)
        credit = {name_: value_of(name_) for name_ in entry.get("credit", [])}
        title = value_of(entry["title"]) if "title" in entry else None
        yield Record(record_id, text, licenses[value], credit,
                     _text(title).strip() or None if title is not None else None)


def _id(entry: dict, value_of, position: int, name: str) -> str:
    """A record's id: its `id` field, or its position when the entry names
    none or the record holds none. Under `newest`, ids are compared across
    files, where positions mean nothing: a record without an id is refused."""
    record_id = value_of(entry["id"]) if "id" in entry else None
    if record_id is not None:
        return str(record_id)
    if "newest" in entry:
        raise InvalidFile(f"{name}: record {position} has no id; under `newest` every record needs its `id` field")
    return str(position)


def newest_index(entry: dict, paths, read: Callable) -> dict[str, tuple[datetime, str, int]]:
    """{id: (date, path, position)} of the record to read for each id under
    `newest`, over every file among paths (read gives a path's file and its
    joined file): the record whose date is the latest, and when dates are
    equal the later one, in path order, then in its file."""
    best = {}
    for path in sorted(paths, key=str.encode):
        file, joined = read(path)
        for position, value_of in enumerate(_records_of(file, path, entry, joined), start=1):
            record_id = _id(entry, value_of, position, path)
            date = _date(value_of(entry["newest"]), path, record_id, entry["newest"])
            if record_id not in best or date >= best[record_id][0]:
                best[record_id] = date, path, position
    return best


def saved_newest_index(entry: dict, paths, read: Callable, store, telemetry=None) -> dict[str, np.ndarray]:
    """{path: the ascending positions of the records to read} of newest_index
    (paths and read as there, read giving manifest.Stored files), found under
    its ref, or scanned once and saved there. A path none of whose records is
    read is not in it. The ref is recorded before it is set
    (wpt_telemetry.set_result_ref)."""
    telemetry = telemetry or Telemetry.off()
    key = newest_key(entry, paths, read)
    known = get_result_ref(store, "newest-index", key, telemetry)
    if not known:
        known = store.put_bytes(_packed(newest_index(entry, paths, read)))
        set_result_ref(store, "newest-index", key, known, telemetry)
    return _unpacked(store.get_bytes(known))


def newest_files(paths, read) -> list:
    """[path, its sha256, its joined file's] of each of paths (read as in
    saved_newest_index), in path order."""
    files = []
    for path in sorted(paths, key=str.encode):
        file, joined = read(path)
        files.append([path, file.entry.sha256, joined.entry.sha256 if joined else None])
    return files


def newest_key(entry: dict, paths, read) -> str:
    """The fingerprint of a `newest` index: the entry, its files
    (newest_files) and NEWEST_VERSION."""
    return fingerprint({"kind": "newest-index", "version": NEWEST_VERSION, "entry": entry,
                        "files": newest_files(paths, read)})


def _packed(index: dict) -> bytes:
    """A newest_index as an object: the length of a JSON object {path: how
    many records of it are read} in 8 bytes, that object, then the positions
    of each path, ascending, in 4 bytes each. Its dates and ids are left out:
    a reader of the file needs only the positions."""
    positions = {}
    for _, path, position in index.values():
        positions.setdefault(path, []).append(position)
    counts = json.dumps({path: len(found) for path, found in positions.items()}, ensure_ascii=False).encode()
    return b"".join([len(counts).to_bytes(8, "little"), counts,
                     *(np.sort(np.array(found, dtype="<u4")).tobytes() for found in positions.values())])


def _unpacked(data: bytes) -> dict[str, np.ndarray]:
    """The positions of each path of a _packed index, read in place."""
    at = 8 + int.from_bytes(data[:8], "little")
    found = {}
    for path, count in json.loads(data[8:at]).items():
        found[path] = np.frombuffer(data, "<u4", count, at)
        at += 4 * count
    return found


def _date(value, name: str, record_id: str, field_name: str) -> datetime:
    """An ISO 8601 date or time, in UTC when it names no offset."""
    try:
        found = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise InvalidFile(f"{name}: record {record_id!r}: its {field_name} is not an ISO 8601 date: {value!r}") \
            from None
    return found if found.tzinfo else found.replace(tzinfo=timezone.utc)


def _records_of(file: "Path | Stored", name: str, entry: dict, joined: "Path | Stored | None") -> Iterator[Callable]:
    """For each record of a file, in order, a function giving the value of
    one of its fields."""
    if not name.lower().endswith(XML):
        for row in rows(file, name, entry.get("records")):
            yield lambda name_, row=row: field(row, name_)
        return
    if "records" in entry:
        raise InvalidFile(f"{name}: an .xml records file is one record; `records` is for a JSON document")
    root = _xml(file, name)
    variables = {entry["join"]["name"]: [_xml(joined, str(joined))] if joined else []} if "join" in entry else {}
    namespaces = entry.get("namespaces", {})

    def value_of(expression):
        try:
            found = etree.XPath(expression, namespaces=namespaces)(root, **variables)
        except etree.XPathError as error:
            raise InvalidFile(f"{name}: field {expression!r}: {error}") from error
        if isinstance(found, list):
            if not found:
                return None
            first = found[0]
            return "".join(first.itertext()) if isinstance(first, etree._Element) else str(first)
        return str(found) if isinstance(found, str) else found

    yield value_of


def _xml(file, name: str):
    """The root element of an XML file (a local Path or a manifest.Stored),
    read without loading anything from outside it: only the entities its
    own DTD declares are expanded."""
    parser = etree.XMLParser(resolve_entities="internal", load_dtd=False, no_network=True)
    try:
        return etree.fromstring(file.read_bytes(), parser)
    except etree.XMLSyntaxError as error:
        raise InvalidFile(f"{name}: {error}") from error
