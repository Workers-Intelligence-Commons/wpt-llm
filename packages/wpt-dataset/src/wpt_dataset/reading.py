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

While a batch is converted and its documents turned into tokens, one thread
fetches the small files of the next batch, and never of the one after it. A
worker's fetched files thus take two batches on the disk at most, and what
fetching adds to its memory is what fetch_files holds: REQUESTS runs of
frames of about RUN_BYTES each, never more than the batch's own bytes
(wpt_store). A file read as a stream is not fetched: wpt_store.ahead reads
ahead of it.

A source of more than PART_BATCHES batches is read in parts (parts_of), each
a run of whole batches in bundle order, so several workers read and turn it
into tokens at once. The parts follow from the chosen files and their sizes
alone, never from the machine. A part opens the source's whole final tree, so
a note in another file reads as it does in one read. A source with a records
entry under `newest` stays one part, whatever its size: its index needs every
file of the entry. A file is never split, so a single file larger than a part
stays one long part.

Packs live in a folder of the cache folder named by the fingerprint of what
decides a conversion: the key of the source's Kept (kept.kept_key), its
type and the code of this module (wpt_utils.programs.code_of). A file found in
a pack of that folder is not read again, by this build or the next. Which
packs a read finds depends on the reads before it, so a Read gives its
documents in file path order whatever packs hold them, one open file per
pack: what a build makes of them never depends on the cache. The Reads of a
source's parts give theirs in that same order together (documents_of).

A part's tokens are one store object per set of chosen files, kept under the
ref source-tokens/<that fingerprint, the vocabulary and the files>, so a build
of the same recipe finds them without reading a file, and one stopped halfway
through a source reads only the parts it has no ref for. The object holds a
document's tokens where reading reached it; its summary lists the documents in
file path order.
"""

import hashlib
import heapq
import json
import os
import tempfile
from compression import zstd
from concurrent.futures import ThreadPoolExecutor
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
# batches of one part: enough for the reading ahead to pay and for a part's own opening of the source to stay small
# next to its reading, few enough that a round's last part ends soon after the others
PART_BATCHES = 4


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


def _fetchable(source, batch: dict) -> list:
    """The places of the files of a batch that are read whole and are small
    enough to fetch."""
    return [entry.placed for path, entry in batch.items()
            if entry.placed is not None and entry.placed.length < FETCHED_AT_MOST
            and checks.asks_text(source, path, entry)]


def _write_pack(source, batch: dict, held: Kept, store, folder: Path, newest_indexes: dict) -> Pack:
    """Read one batch of a source's files, {path: entry}, fetched already,
    into a new pack of folder; store is a view of copies_in, newest_indexes
    what the read's batches share (markdown.documents)."""
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


def _cuts(sizes: list[int]):
    """(start, stop) of each run of the sizes, in their order, of BATCH_BYTES
    at most, or one size alone."""
    start, total = 0, 0
    for at, size in enumerate(sizes):
        if at > start and total + size > BATCH_BYTES:
            yield start, at
            start, total = at, 0
        total += size
    if sizes:
        yield start, len(sizes)


def _batches(paths: list[str], tree: dict):
    """The paths, in their order, as {path: entry} of BATCH_BYTES at most,
    or one file alone."""
    sizes = [tree[path].placed.length if tree[path].placed is not None else 0 for path in paths]
    for start, stop in _cuts(sizes):
        yield {path: tree[path] for path in paths[start:stop]}


@dataclass(frozen=True)
class Part:
    """What one worker process reads of a source."""
    source: object
    positions: list[int] = ()  # its chosen files, by their positions in the final tree
    size: int = 0  # their bytes
    of: int = 1  # how many parts the source has
    number: int = 0  # its place among them, from 0


def parts_of(source, positions: list[int], sizes: list[int]) -> list[Part]:
    """The Parts of a source's chosen files, in their order (bundle order,
    so that a frame is fetched by one part), sizes being their bytes: every
    PART_BATCHES batches, or one for a source that needs its files together
    (see the module's docstring)."""
    together = any("newest" in entry for entry in source.data["formats"])
    starts = [0] if together else [start for start, _ in _cuts(sizes)][::PART_BATCHES]
    return [Part(source, positions[start:stop], sum(sizes[start:stop]), len(starts), number)
            for number, (start, stop) in enumerate(zip(starts, [*starts[1:], len(positions)]))]


def _in_order(found):
    """(entry, Markdown) of each document of found, [(pack, its files)], in
    file path order (a records file's in the file's order): a pack gives its
    files in that order and a file lies whole in one pack."""
    return heapq.merge(*(pack.documents(files) for pack, files in found), key=lambda one: one[0]["file"].encode())


def documents_of(reads):
    """_in_order of the documents of several Reads of one source, its parts,
    whatever packs hold them: as one Read of their files gives them."""
    found = []
    for read in reads:
        files = set(read.files)
        for pack in read.packs:
            mine = files & pack.files
            files -= mine
            found.append((pack, mine))
    return _in_order(found)


@dataclass(frozen=True)
class Read:
    """A source's chosen files, read."""
    tree: str  # its final tree's manifest object
    files: frozenset[str]  # the chosen files it kept
    packs: tuple[Pack, ...]  # the packs that hold their documents
    check: dict  # its check result, as a dataset's record keeps it

    def documents(self):
        """(entry, Markdown) of each document (documents_of)."""
        return documents_of([self])


def _opened(source, store, positions, telemetry):
    """(final tree, Kept before the text check, the chosen paths it keeps,
    the fingerprint of what decides their conversion)."""
    tree, known = final_tree(source, store, telemetry)
    held = kept_files(source, tree, known, store, workers.license_texts())
    names = list(tree)
    paths = [names[position] for position in positions if names[position] in held.tree]
    return tree, held, paths, fingerprint({"kind": "markdown", "code": code_of(__name__), "kept": held.key,
                                           "type": source.data["type"]})


def _read(source, store, tree, held, paths, key, cache, ready=lambda documents: None) -> Read:
    """The chosen paths' documents, each file not in a pack of the folder of
    key read into a new one, the next batch fetched meanwhile. ready gets
    each lot of documents as soon as it can be read: those of the packs found,
    then those of each new pack. An error leaves once the fetch under way
    ended, so nothing writes into store's folder after it."""
    folder, wanted = Path(cache) / key, set(paths)
    used, covered, not_text, records = [], set(), {}, {}

    def use(pack):
        mine = (wanted & pack.files) - covered
        if mine:
            used.append(pack)
            covered.update(mine)
            not_text.update({path: why for path, why in pack.not_text.items() if path in mine})
            records.update({path: found for path, found in pack.records.items() if path in mine})
        return mine

    found = [(pack, mine) for pack in _packs(folder) if (mine := use(pack))]
    waiting, newest_indexes = list(_batches([path for path in paths if path not in covered], tree)), {}
    with ThreadPoolExecutor(1, thread_name_prefix="fetch-ahead") as fetching:
        fetched = fetching.submit(store.fetch_files, _fetchable(source, waiting[0])) if waiting else None
        ready(_in_order(found))
        for batch, following in zip(waiting, [*waiting[1:], None]):
            fetched.result()
            if following is not None:
                fetched = fetching.submit(store.fetch_files, _fetchable(source, following))
            pack = _write_pack(source, batch, held, store, folder, newest_indexes)
            ready(pack.documents(use(pack)))
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
    """A source's chosen files, or a part's, as tokens, in a worker process:
    {"tree": its final manifest object, "tokens": one object holding every
    document's tokens (each ending with END) one after the other, "documents":
    [{path, file, sha256 of its Markdown, start (byte offset), tokens,
    token_sha256, record}] in file path order, "check": its check result},
    its files and records left out as kept.py decides. Each lot of documents
    becomes tokens as soon as it is read, so the next batch is fetched
    meanwhile. Found under its ref when it is known. The ref is
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
        documents = []
        with open(Path(tmp) / "tokens.u32", "wb") as out:
            def encoded(found):
                for entry, text in found:
                    data = np.asarray(tokenizer.encode(text) + [Tokenizer.END], dtype="<u4").tobytes()
                    documents.append({**{name: value for name, value in entry.items() if name != "length"},
                                      "start": out.tell(), "tokens": len(data) // 4, "token_sha256": _sha256(data)})
                    out.write(data)

            read = _read(source, copies, tree, held, paths, key, cache, encoded)
        documents.sort(key=lambda entry: entry["file"].encode())  # stable: a records file's stay in the file's order
        summary = {"tree": read.tree, "tokens": copies.put_file(Path(tmp) / "tokens.u32"), "documents": documents,
                   "check": read.check}
    known = store.put_bytes(json.dumps(summary, ensure_ascii=False, sort_keys=True).encode())
    set_result_ref(store, "source-tokens", ref, known, telemetry)
    return summary
