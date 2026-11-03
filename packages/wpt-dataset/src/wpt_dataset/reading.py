"""A build's third step: only the chosen files of a source are read, in
bundle order, and their Markdown is kept so the vocabulary and the tokens
read it once.

A source's chosen files are read in batches of BATCH_BYTES. A batch's small
files are fetched together (wpt_store's fetch_files): each frame that holds
one is fetched once, so a block fetched is used whole. A larger file, and a
records file, is read as a stream, in order. The text check runs on the
batch's files (kept.py), and the batch's documents go into one pack: a
zstd file of a JSON line and the Markdown's bytes per document, with a
JSON file beside it, written last, naming the files it covers, those that
are not text and the records left out. The index of a records entry with
`newest` covers every file of that entry, so it is built once per read of
a source and every batch shares it.

Packs live in a folder of the cache folder named by the fingerprint of what
decides a conversion: the key of the source's Kept (kept.kept_key), its
type and the code of this module (wpt_utils.programs.code_of). A file found in
a pack of that folder is not read again, by this build or the next. Which
packs a read finds depends on the reads before it, so a Read gives its
documents in file path order whatever packs hold them, one open file per
pack: what a build makes of them never depends on the cache.

A source's tokens are one store object per set of chosen files, kept under
the ref source-tokens/<that fingerprint, the vocabulary and the files>, so a
build of the same recipe finds them without reading a file.
"""

import hashlib
import heapq
import json
import os
import tempfile
from compression import zstd
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wpt_telemetry import Telemetry, set_result_ref
from wpt_tokenizer import Tokenizer
from wpt_utils.programs import code_of

from . import checks, manifest, markdown, workers
from .kept import Kept, kept_files, skipped_values
from .sources import final_tree, fingerprint

BATCH_BYTES = 256 << 20  # file bytes of one batch: what its fetched files take on the disk at most
FETCHED_AT_MOST = 64 << 20  # the size from which a file is read alone, as a stream


@dataclass(frozen=True)
class Pack:
    path: Path  # the zstd file of its documents
    files: frozenset[str]  # the files it covers
    not_text: dict[str, str]  # {path: why} of those the text check refused
    records: dict  # kept.skipped_values of its records files

    def documents(self, files):
        """(entry, Markdown) of each document of these files: entry says
        its path, file, sha256 and record."""
        if not self.path.stat().st_size:
            return  # no document: zstd wrote nothing, and reads nothing as a cut file
        with zstd.open(self.path, "rb") as found:
            while line := found.readline():
                entry = json.loads(line)
                text = found.read(entry["length"])
                if entry["file"] in files:
                    yield entry, text


def _packs(folder: Path) -> list[Pack]:
    """Every whole pack of a folder, in the order of their names."""
    found = []
    for described in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        pack = json.loads(described.read_text(encoding="utf-8"))
        found.append(Pack(described.with_suffix(".zst"), frozenset(pack["files"]), pack["not_text"], pack["records"]))
    return found


def _fetched(source, batch: dict, store):
    """Fetch the files of a batch that are read whole and are small enough."""
    store.fetch_files([entry.placed for path, entry in batch.items()
                       if entry.placed is not None and entry.placed.length < FETCHED_AT_MOST
                       and checks.asks_text(source, path, entry)])


def _write_pack(source, batch: dict, held: Kept, store, folder: Path, newest_indexes: dict) -> Pack:
    """Read one batch of a source's files, {path: entry}, into a new pack of
    folder; store is a view of copies_in, newest_indexes what the read's
    batches share (markdown.documents)."""
    _fetched(source, batch, store)
    not_text = {path: why for path, entry in batch.items() if checks.asks_text(source, path, entry)
                and (why := checks.not_text(source, path, manifest.content(store, entry))) is not None}
    folder.mkdir(parents=True, exist_ok=True)
    name = fingerprint({"files": sorted(batch)})
    skipped = {}
    descriptor, partial = tempfile.mkstemp(dir=folder, prefix=".partial-")
    os.close(descriptor)
    try:
        with zstd.open(partial, "wb") as out:
            for converted in markdown.documents(source, held.less(not_text).tree, store, workers.repo(), skipped,
                                                only=batch.keys() - not_text.keys(), trains=held.trains,
                                                newest_indexes=newest_indexes):
                entry = {"path": converted.path, "file": converted.file, "length": len(converted.markdown),
                         "sha256": _sha256(converted.markdown)}
                if converted.record:
                    entry["record"] = {"license": converted.record.license, "credit": converted.record.credit}
                out.write(json.dumps(entry, ensure_ascii=False).encode() + b"\n" + converted.markdown)
        os.replace(partial, folder / f"{name}.zst")
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    records = skipped_values(skipped)
    described = {"files": sorted(batch), "not_text": not_text, "records": records}
    (folder / f".{name}").write_text(json.dumps(described, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(folder / f".{name}", folder / f"{name}.json")
    for entry in batch.values():
        store.release(entry.sha256)
    return Pack(folder / f"{name}.zst", frozenset(batch), not_text, records)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _batches(paths: list[str], tree: dict):
    """The paths, in their order, as {path: entry} of BATCH_BYTES at most,
    or one file alone."""
    batch, size = {}, 0
    for path in paths:
        length = tree[path].placed.length if tree[path].placed is not None else 0
        if batch and size + length > BATCH_BYTES:
            yield batch
            batch, size = {}, 0
        batch[path] = tree[path]
        size += length
    if batch:
        yield batch


@dataclass(frozen=True)
class Read:
    """A source's chosen files, read."""
    tree: str  # its final tree's manifest object
    files: frozenset[str]  # the chosen files it kept
    packs: tuple[Pack, ...]  # the packs that hold their documents
    check: dict  # its check result, as a dataset's record keeps it

    def documents(self):
        """(entry, Markdown) of each document, in file path order (a records
        file's in the file's order), whatever packs hold them: a pack gives
        its files in that order and a file lies whole in one pack."""
        files, parts = set(self.files), []
        for pack in self.packs:
            mine = files & pack.files
            files -= mine
            parts.append(pack.documents(mine))
        yield from heapq.merge(*parts, key=lambda found: found[0]["file"].encode())


def _opened(source, store, positions, telemetry):
    """(final tree, Kept before the text check, the chosen paths it keeps,
    the fingerprint of what decides their conversion)."""
    tree, known = final_tree(source, store, telemetry)
    held = kept_files(source, tree, known, store, workers.license_texts())
    names = list(tree)
    paths = [names[position] for position in positions if names[position] in held.tree]
    return tree, held, paths, fingerprint({"kind": "markdown", "code": code_of(__name__), "kept": held.key,
                                           "type": source.data["type"]})


def _read(source, store, tree, held, paths, key, cache) -> Read:
    """The chosen paths' documents, each file not in a pack of the folder of
    key read into a new one."""
    folder, wanted = Path(cache) / key, set(paths)
    used, covered, not_text, records = [], set(), {}, {}

    def use(pack):
        mine = (wanted & pack.files) - covered
        if mine:
            used.append(pack)
            covered.update(mine)
            not_text.update({path: why for path, why in pack.not_text.items() if path in mine})
            records.update({path: found for path, found in pack.records.items() if path in mine})

    for pack in _packs(folder):
        use(pack)
    newest_indexes = {}
    for batch in _batches([path for path in paths if path not in covered], tree):
        use(_write_pack(source, batch, held, store, folder, newest_indexes))
    check = held.less(not_text).result()
    if records:
        check["records"] = records
    return Read(held.final, frozenset(wanted - not_text.keys()), tuple(used), check)


def read_markdown(source, store, positions, cache, telemetry) -> Read:
    """The Read of a source's chosen files (their final tree's positions),
    in a worker process."""
    with tempfile.TemporaryDirectory(prefix="wpt-read-") as tmp:
        copies = store.copies_in(tmp)
        return _read(source, copies, *_opened(source, copies, positions, telemetry), cache)


_TOKENIZERS = {}  # in a worker process: each vocabulary object it read, as a Tokenizer


def tokenizer_of(vocabulary: str, store) -> Tokenizer:
    """The Tokenizer of a vocabulary object, read once per worker process."""
    if vocabulary not in _TOKENIZERS:
        _TOKENIZERS[vocabulary] = Tokenizer.from_bytes(store.get_bytes(vocabulary))
    return _TOKENIZERS[vocabulary]


def read_tokens(source, store, positions, vocabulary: str, cache, telemetry) -> dict:
    """A source's chosen files as tokens, in a worker process: {"tree": its
    final manifest object, "tokens": one object holding every document's
    tokens (each ending with END) one after the other, "documents": [{path,
    file, sha256 of its Markdown, start (byte offset), tokens, token_sha256,
    record}], "check": its check result}, its files and records left out
    as kept.py decides. Found under its ref when it is known. The ref is
    recorded before it is set (wpt_telemetry.set_result_ref)."""
    telemetry = telemetry or Telemetry.off()
    tokenizer = tokenizer_of(vocabulary, store)
    with tempfile.TemporaryDirectory(prefix="wpt-read-") as tmp:
        copies = store.copies_in(Path(tmp) / "copies")
        tree, held, paths, key = _opened(source, copies, positions, telemetry)
        ref = fingerprint({"kind": "source-tokens", "code": code_of(__name__), "markdown": key,
                           "vocabulary": vocabulary, "files": sorted(paths)})
        if known := store.get_ref("source-tokens", ref):
            return json.loads(store.get_bytes(known))
        read = _read(source, copies, tree, held, paths, key, cache)
        documents, offset = [], 0
        with open(Path(tmp) / "tokens.u32", "wb") as out:
            for entry, text in read.documents():
                data = np.asarray(tokenizer.encode(text) + [Tokenizer.END], dtype="<u4").tobytes()
                out.write(data)
                documents.append({**{name: value for name, value in entry.items() if name != "length"},
                                  "start": offset, "tokens": len(data) // 4, "token_sha256": _sha256(data)})
                offset += len(data)
        summary = {"tree": read.tree, "tokens": copies.put_file(Path(tmp) / "tokens.u32"), "documents": documents,
                   "check": read.check}
    known = store.put_bytes(json.dumps(summary, ensure_ascii=False, sort_keys=True).encode())
    set_result_ref(store, "source-tokens", ref, known, telemetry)
    return summary
