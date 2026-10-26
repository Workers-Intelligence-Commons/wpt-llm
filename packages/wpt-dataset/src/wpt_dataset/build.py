"""Build vocabularies and datasets from recipes.

    documents      the Markdown of every file of every admitted source's final
                   tree (see markdown.py); exact duplicates removed (the first by
                   source id, then path, keeps it)
    held-out       documents whose id hashes below the recipe's held_out share;
                   a document of a source with held_out_by hashes its held-out
                   group instead, so documents of any source with the same
                   group are held out together (eBible's chapters, by book
                   and chapter)
    pools          each training document goes to the first pool, in mixture
                   order, whose expression matches its tags
    shares         a pool fills weight x total_tokens by whole passes over its
                   documents, then a partial pass; more passes than max_epochs
                   (default 4) is refused
    shards         Parquet files in the store (see wpt_store.shards), rows in a
                   fixed pseudo-random order derived from the dataset id

The dataset id is the fingerprint of every input, the code
(wpt_utils.programs.code: its commit and a hash of its uncommitted
changes) among them, so building the same recipe on the same data with
the same code again is a lookup. A dataset whose record has
another shape than the current one (current_record) is built again, its record replaced.

Beside its record, every dataset keeps its document list (write_document_list): each document's id, the sha256 of its
Markdown and its tag, training or held-out, under a ref named by the
dataset id, so a model's provenance finds it. A dataset built
before builds wrote one never gets one.

The build never downloads: it reads the upstream files the download step
put in the store (download.py). A source that is not in the store
(NotDownloaded) is left out, and the record lists it with its error under
`skipped`. Any other source that fails (a patch no longer applies, a file
is broken) stops the build, unless the build is asked to skip such
sources: the record then lists each one the same way. A missing program
stops it anyway, since it fails every source that needs it. A new
vocabulary is never trained without a source: it is frozen for good. A
locked one rebuilt on a store that lacks it leaves out, and reports, any
admitted source that fails: only the documents its lock lists decide.

The record also names, for each source, the session (its run record's
folder) and the time of the download that put the files it used in the
store, from the store database (`downloads`; None for a source with no
such download, an empty one or one downloaded before the store kept
sessions). It is not part of the dataset id: the same files downloaded
again give the same dataset. `wpt-dataset build` runs as a session of its
own, which holds a `reused` line for each source whose upstream tree the
dataset holds.

Sources are converted and their conversations and tasks read in worker
processes (workers.py), several at once, and documents turned into tokens
on several threads at once, as many as wpt_utils.parallel allows, each
item's memory measured as it runs and each source's size counted on the
disk its local copies go to; the results are put together in the sources'
and the documents' order, so the dataset id and the shards are the ones a
build of one at a time gives.

With stream=True the build keeps no document's Markdown in the store and
no source's files on this machine: each source's documents are turned
into tokens (one store object per source, see source_tokens), and only
the files its conversion needs as files (a records file, a message tree,
a file its patches change) are copied, into a temporary folder removed
once the source is tokens; every other file is read from the store by
range (wpt_store's copies_in), so a conversion never opens a file of the
store itself. Sources are turned into tokens several at once, as above,
each in its own temporary folder. The conversations and tasks the build
keeps (conversations_and_tasks) are read the same way: each source, each
source a task points into and each task's repository has its local copies
in its own temporary folder, removed once it is read. The documents, the
dataset id and the shards are the same as without it; only what the store
and the machine keep differs.

With a telemetry (wpt_telemetry), the build records each source (fetched,
converted, its documents, or why it was skipped), the vocabulary, each
pool, each shard written, the contamination check and the dataset record.
"""

import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wpt_store import CorruptObject, MissingObject
from wpt_store.shards import write_shard
from wpt_telemetry import Telemetry, set_ref
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_utils.parallel import Sizing, each, in_order
from wpt_utils.programs import code

from . import conversations as chat_files
from . import download, manifest, markdown, pools, store_history, workers
from . import tasks as task_files
from .errors import DatasetError, LockedDocumentsMissing, LockMismatch, NotDownloaded, PoolError, TokenizerVersionNeeded
from .repo import Repo
from .sources import (check_license_map, copied_as_records, copy_size, fingerprint, held_out_group, source_tree,
                      tags_of, tree_key)

DEFAULT_MAX_EPOCHS = 4
SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per shard
NGRAM = 13  # contamination check: runs of 13 words, as in the GPT-3 paper
STREAM_READERS = 16  # documents read at once from a streamed build's token objects
DOCUMENT_LIST = "dataset-documents"  # the kind of the ref, named by the dataset id, of a dataset's document list
TRAINING, HELD_OUT = "training", "held-out"  # a listed document's tag
# the top-level fields every record this code writes holds (conversations and tasks only when the sources hold
# some): a record without one is of another shape, which its readers refuse and a build writes again
RECORD_FIELDS = ("id", "recipe_name", "recipe", "mixture", "tokenizer", "sources", "downloads", "skipped", "documents",
                 "pools", "total_tokens", "shards", "held_out_shards", "contamination", "fingerprints", "tools",
                 "tool_differences")


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    sha256: str
    tags: frozenset[str]
    group: str | None = None  # its held-out group (sources.held_out_group), hashed instead of its id

    @property
    def held_out_key(self) -> str:
        """What the held-out hash reads: its group, or its id without one."""
        return self.group or self.id


def gather(repo: Repo, store, log=print, skipped=None, telemetry=None, skip_failed=False):
    """(documents, sources info, number of duplicates removed). A source with
    records files gets `credits` in its info: the store object listing, one
    JSON line per document it gives, the document id, its license and the
    credit fields of its record. With a list as skipped, a source that is
    not in the store is left out and added to it, and so is any other
    failing source with skip_failed (see _skip); otherwise its error stops.
    A source whose license text is empty is left out, and log is called
    with a note naming it. Sources are converted several at once
    (each_source), and their documents put together in the sources' order."""
    telemetry = telemetry or Telemetry.off()
    documents, info, seen, duplicates = [], {}, set(), 0
    admitted = [source for source in repo.sources(log) if source.admitted]
    # closed at once when a source stops the others, so the sources still running end first
    with contextlib.closing(each_source(_converted, admitted, repo, store, telemetry)) as converted:
        for done, span in converted:
            source = done.item
            with telemetry.finish(span):
                if done.error is not None:
                    _skip(source, done.error, skipped, skip_failed, log)
                    span.set(skipped=f"{type(done.error).__name__}: {done.error}")
                    continue
                tree_object, found, repeated = done.value
                credits, kept = [], 0
                for document, credit in found:
                    if document.sha256 in seen:
                        repeated += 1
                        continue
                    seen.add(document.sha256)
                    documents.append(document)
                    kept += 1
                    if credit is not None:
                        credits.append(credit)
                info[source.id] = source_info(source, tree_object)
                duplicates += repeated
                if credits:
                    info[source.id]["credits"] = store.put_bytes("".join(f"{line}\n" for line in credits).encode())
                span.set(tree=tree_object, documents=kept, duplicates=repeated, credits=info[source.id].get("credits"))
    log(f"documents: {len(documents):,} from {len(info)} admitted sources ({duplicates:,} exact duplicates removed)")
    return documents, info, duplicates


def each_source(function, sources, repo, store, telemetry, args=(), stream=False):
    """(wpt_utils.parallel.Done of function(source, store, *args,
    telemetry=...), the source's "source" span) for each source, in the
    sources' order, each source in a worker process (workers.py), as many
    at once as wpt_utils.parallel allows, reserving on the disk its copies
    go to the size of the files its conversion will copy (copy_size, the
    conversion stating them beforehand): the temporary folder's
    with stream, the store's cache folder otherwise (the event
    "sources.at_once" says how many and why). A DatasetError, CorruptObject
    or MissingObject the function raises is the source's error; anything
    else (a missing program, a store that refuses a write) stops every
    source.

    The span opens under the caller's current span when the source's work
    starts, so what the worker records lands under it, and it says how
    many seconds the work took (converting_seconds); the caller ends it
    (telemetry.finish) once it has put the source's result together. The
    spans of the sources whose turn never came, because something stopped
    the sources, end with an error saying so."""
    folder = tempfile.gettempdir() if stream else store.cache
    with workers.Pool(repo, "sources at once", disk=lambda source: copy_size(source, store, copied_as_records(source)),
                      folder=folder) as pool:
        telemetry.event("sources.at_once", limit=pool.sizing.limit, why=pool.sizing.why)
        parent, spans = telemetry.current(), {}  # spans: each source begun and not yet given to the caller

        def run(source):
            with telemetry.within(parent):
                spans[source.id] = span = telemetry.begin("source", source=source.id, checksum=source.data["checksum"])
            started = time.monotonic()
            try:
                with telemetry.within(span):
                    return pool.call(function, source, store, *args, telemetry=telemetry)
            finally:
                span.set(converting_seconds=round(time.monotonic() - started, 3))

        try:
            for done in each(run, sources, (DatasetError, CorruptObject, MissingObject), pool.sizing):
                yield done, spans.pop(done.item.id)
        finally:
            stopped = RuntimeError("the sources stopped before this source's turn")
            for span in spans.values():
                telemetry.end(span, stopped)


def _converted(source, store, telemetry):
    """(manifest object of its final tree, [(document, its credit line or
    None)], exact duplicates inside the source) of one source, each
    document's Markdown put in the store, in a worker process; duplicates
    of other sources are gather's to remove."""
    tree, tree_object = source_tree(source, store, telemetry)
    found, mine, repeated = [], set(), 0
    for converted in markdown.documents(source, tree, store, workers.repo()):  # links are not documents; their targets are
        sha256 = store.put_bytes(converted.markdown)
        if sha256 in mine:
            repeated += 1
            continue
        mine.add(sha256)
        document_id = f"{source.id}/{converted.path}"
        credit = None
        if converted.record:
            credit = json.dumps({"document": document_id, "license": converted.record.license,
                                 "credit": converted.record.credit}, ensure_ascii=False, sort_keys=True)
        found.append((Document(document_id, source.id, sha256, tags_of(source, converted.file),
                               held_out_group(source, converted.path)), credit))
    return tree_object, found, repeated


def source_info(source, tree: str) -> dict:
    """What a dataset's record keeps of one source: where it comes from
    (each of its `addresses` too, when it has several), its revision,
    checksum, final tree and licenses."""
    where = {"addresses": source.data["addresses"]} if "addresses" in source.data else {}
    return {"address": source.data["address"], **where, "revision": source.data["revision"],
            "checksum": source.data["checksum"], "tree": tree, "licenses": source.data["licenses"]}


def _skip(source, error, skipped, skip_failed, log):
    """Record a source that failed, or raise its error: with a list as
    skipped, a source that is not in the store (NotDownloaded) is always
    recorded, any other failure only with skip_failed. A missing program
    (wpt_utils.programs.MissingTool), which would fail every source using
    it, is not caught here and always stops the build."""
    if skipped is None or not (skip_failed or isinstance(error, NotDownloaded)):
        raise error
    skipped.append({"source": source.id, "error": f"{type(error).__name__}: {error}"})
    log(f"skipped {source.id}: {type(error).__name__}: {error}")


def assign(documents, mixture) -> dict[str, list[Document]]:
    """Pool name -> its documents; each document goes to the first matching pool."""
    trees = _pool_trees(mixture)
    assigned = {name: [] for name, _ in trees}
    for document in documents:
        name = _pool_of(document, trees)
        if name is not None:
            assigned[name].append(document)
    return assigned


def _pool_trees(mixture):
    return [(pool["name"], pools.parse(pool["select"])) for pool in mixture["pools"]]


def _pool_of(document, trees) -> str | None:
    return next((name for name, tree in trees if pools.matches(tree, document.tags)), None)


# -- streaming --------------------------------------------------------------------

def streamed_documents(repo: Repo, store, telemetry=None, log=print, skipped=None, used=None):
    """(document, Markdown) of every admitted source, exact duplicates left
    out as in gather, in the sources' order, no Markdown kept in the store.
    Several sources are converted at once, each in a worker process
    (each_source), every local copy its conversion needs in a temporary
    folder of its own, removed once it is converted; its Markdown waits in
    a file of a temporary folder of this call until its turn, removed then.
    A source whose license text is empty is left out, and log is called
    with a note naming it. With a list as skipped, a source that fails
    (DatasetError, CorruptObject, MissingObject: one not in the store among
    them) gives the documents it gave before it failed, is added to it as
    gather adds one and logged, and the next source goes on; otherwise its
    error stops. With a set as used, the id of each source read goes into it."""
    telemetry = telemetry or Telemetry.off()
    seen = set()
    admitted = [source for source in repo.sources(log) if source.admitted]
    with tempfile.TemporaryDirectory(prefix="wpt-streamed-") as folder, \
            contextlib.closing(each_source(_streamed, admitted, repo, store, telemetry, (folder,),
                                           stream=True)) as converted:  # as in gather
        for done, span in converted:
            source = done.item
            with telemetry.finish(span):
                written, tree_object, error = done.value if done.error is None else (None, None, done.error)
            count = 0
            if written is not None:
                if used is not None:
                    used.add(source.id)
                with open(written, "rb") as found:
                    for line in found:
                        entry = json.loads(line)
                        text = found.read(entry["length"])
                        if entry["sha256"] in seen:
                            continue
                        seen.add(entry["sha256"])
                        count += 1
                        # no held-out group: its only reader, tokenizer training, never splits off a held-out share
                        yield Document(f"{source.id}/{entry['path']}", source.id, entry["sha256"],
                                       tags_of(source, entry["file"])), text
                Path(written).unlink()
            if error is not None:
                _skip(source, error, skipped, True, log)
                telemetry.event("source.skipped", source=source.id, error=f"{type(error).__name__}: {error}")
                continue
            # an event, not a span: the reader of this generator runs between its documents
            telemetry.event("source.streamed", source=source.id, checksum=source.data["checksum"], tree=tree_object,
                            documents=count)


def _streamed(source, store, folder, telemetry):
    """(a file of folder holding one source's documents, its final tree's
    manifest object or None, the error it failed with or None), in a worker
    process: for each document, a JSON line {path, file, sha256, length}
    then its Markdown's bytes, up to the failure."""
    descriptor, written = tempfile.mkstemp(dir=folder, prefix=f"{source.id}-")
    tree_object, error = None, None
    with os.fdopen(descriptor, "wb") as out, tempfile.TemporaryDirectory(prefix="wpt-stream-") as tmp:
        copies = store.copies_in(tmp)
        try:
            tree, tree_object = source_tree(source, copies, telemetry)
            for converted in markdown.documents(source, tree, copies, workers.repo()):
                entry = {"path": converted.path, "file": converted.file,
                         "sha256": hashlib.sha256(converted.markdown).hexdigest(), "length": len(converted.markdown)}
                out.write(json.dumps(entry, ensure_ascii=False).encode() + b"\n" + converted.markdown)
        except (DatasetError, CorruptObject, MissingObject) as failed:
            error = failed
    return written, tree_object, error


def source_tokens(repo, source, store, tokenizer, vocabulary: str, telemetry=None) -> dict:
    """A source as tokens, its Markdown not kept: {"tree": its final manifest
    object, "tokens": one object holding every document's tokens (each
    ending with END) one after the other, "documents": [{path, file, sha256
    of its Markdown, start (byte offset), tokens, token_sha256, record}]}.
    Every local copy its conversion needs, and its tokens until they are
    stored, are in a temporary folder, removed before it returns.

    Kept under a ref named after what decides it (the final tree, formats,
    licenses, code (wpt_utils.programs.code), vocabulary), so a source that
    did not change is never converted again. With a telemetry, the ref is recorded
    before it is set (wpt_telemetry.set_ref)."""
    key = fingerprint({"kind": "source-tokens", "tree": tree_key(source), "formats": source.data["formats"],
                       "licenses": [[e["license"], e["paths"], e.get("values")] for e in source.data["licenses"]],
                       "code": code(), "vocabulary": vocabulary})
    known = store.get_ref("source-tokens", key)
    if known:
        return json.loads(store.get_bytes(known))
    with tempfile.TemporaryDirectory(prefix="wpt-stream-") as tmp:
        tmp = Path(tmp)
        copies = store.copies_in(tmp / "copies")
        tree, tree_object = source_tree(source, copies, telemetry)
        documents, offset = [], 0
        with open(tmp / "tokens.u32", "wb") as out:
            for converted in markdown.documents(source, tree, copies, repo):
                data = np.asarray(tokenizer.encode(converted.markdown) + [Tokenizer.END], dtype="<u4").tobytes()
                out.write(data)
                entry = {"path": converted.path, "file": converted.file,
                         "sha256": hashlib.sha256(converted.markdown).hexdigest(),
                         "start": offset, "tokens": len(data) // 4, "token_sha256": hashlib.sha256(data).hexdigest()}
                if converted.record:
                    entry["record"] = {"license": converted.record.license, "credit": converted.record.credit}
                documents.append(entry)
                offset += len(data)
        summary = {"tree": tree_object, "tokens": copies.put_file(tmp / "tokens.u32"), "documents": documents}
    known = store.put_bytes(json.dumps(summary, ensure_ascii=False, sort_keys=True).encode())
    set_ref(store, "source-tokens", key, known, telemetry or Telemetry.off())
    return summary


def gather_streamed(repo: Repo, store, vocabulary: str, log=print, skipped=None, telemetry=None,
                    skip_failed=False):
    """gather's (documents, sources info, duplicates), plus every document's
    tokens (PackedDocuments), from source_tokens with the vocabulary object
    vocabulary: no document's Markdown
    stays in the store, and no source's files on this machine. Several
    sources at once (each_source), put together in the sources' order.
    skipped, skip_failed: as in gather."""
    telemetry = telemetry or Telemetry.off()
    documents, info, seen, duplicates = [], {}, set(), 0
    tokens = PackedDocuments(store)
    admitted = [source for source in repo.sources(log) if source.admitted]
    with contextlib.closing(each_source(_tokenized, admitted, repo, store, telemetry, (vocabulary,),
                                        stream=True)) as converted:  # as in gather
        for done, span in converted:
            source = done.item
            with telemetry.finish(span):
                if done.error is not None:
                    _skip(source, done.error, skipped, skip_failed, log)
                    span.set(skipped=f"{type(done.error).__name__}: {done.error}")
                    continue
                summary = done.value
                info[source.id] = source_info(source, summary["tree"])
                credits, before = [], duplicates
                for entry in summary["documents"]:
                    if entry["sha256"] in seen:
                        duplicates += 1
                        continue
                    seen.add(entry["sha256"])
                    document_id = f"{source.id}/{entry['path']}"
                    documents.append(Document(document_id, source.id, entry["sha256"], tags_of(source, entry["file"]),
                                              held_out_group(source, entry["path"])))
                    tokens.add(document_id, summary["tokens"], entry)
                    if "record" in entry:
                        credits.append(json.dumps({"document": document_id, **entry["record"]}, ensure_ascii=False,
                                                  sort_keys=True))
                if credits:
                    info[source.id]["credits"] = store.put_bytes("".join(f"{line}\n" for line in credits).encode())
                span.set(tree=summary["tree"], tokens_object=summary["tokens"],
                         documents=len(summary["documents"]) - (duplicates - before), duplicates=duplicates - before,
                         credits=info[source.id].get("credits"))
    log(f"documents: {len(documents):,} from {len(info)} admitted sources ({duplicates:,} exact duplicates removed)")
    return documents, info, duplicates, tokens


_TOKENIZERS = {}  # in a worker process: each vocabulary object it read, as a Tokenizer


def _tokenized(source, store, vocabulary, telemetry):
    """source_tokens of one source, its license map checked, in a worker
    process; the tokenizer is the vocabulary object's."""
    if vocabulary not in _TOKENIZERS:
        _TOKENIZERS[vocabulary] = Tokenizer.from_bytes(store.get_bytes(vocabulary))
    summary = source_tokens(workers.repo(), source, store, _TOKENIZERS[vocabulary], vocabulary, telemetry)
    check_license_map(source, manifest.parse_manifest(store.get_bytes(summary["tree"])))
    return summary


class PackedDocuments:
    """Documents' tokens read from the token objects of source_tokens, a
    part of the object each, checked against the document's own token
    sha256; nothing is copied to disk."""

    def __init__(self, store):
        self.store, self._where = store, {}

    def add(self, document_id, tokens_object, entry):
        self._where[document_id] = (tokens_object, entry["start"], entry["tokens"], entry["token_sha256"])

    def length(self, document_id) -> int:
        return self._where[document_id][2]

    def __getitem__(self, document_id) -> np.ndarray:
        tokens_object, start, count, sha256 = self._where[document_id]
        data = self.store.read_range(tokens_object, start, count * 4)
        if hashlib.sha256(data).hexdigest() != sha256:
            raise CorruptObject(f"tokens of {document_id} in {tokens_object} do not match their sha256")
        return np.frombuffer(data, dtype="<u4")

    def many(self, document_ids) -> dict[str, np.ndarray]:
        with concurrent.futures.ThreadPoolExecutor(STREAM_READERS) as pool:
            return dict(zip(document_ids, pool.map(self.__getitem__, document_ids)))


def _rank(*parts) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def is_held_out(key: str, share: float) -> bool:
    """Whether a document (key: its id, or its held-out group, Document.held_out_key) or a task (key: its id)
    is in the held-out share."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()) < share * 2**256


def write_document_list(store, dataset_id: str, documents, held_ids, telemetry) -> str:
    """Store a dataset's document list and return its object: one JSON line
    per document, in the build's order, holding its id, the sha256 of its
    Markdown (the one the dataset id's fingerprint reads) and its tag,
    HELD_OUT for the ids in held_ids (the split the build cut its shards
    by), TRAINING otherwise, kept under the ref DOCUMENT_LIST/<dataset id>.
    The ref is recorded in telemetry's record before it is set
    (wpt_telemetry.set_ref)."""
    lines = (json.dumps({"id": d.id, "sha256": d.sha256, "tag": HELD_OUT if d.id in held_ids else TRAINING},
                        ensure_ascii=False, sort_keys=True) for d in documents)
    listed = store.put_bytes("".join(f"{line}\n" for line in lines).encode())
    telemetry.output(dataset_documents=listed)
    set_ref(store, DOCUMENT_LIST, dataset_id, listed, telemetry)
    return listed


def document_list(store, dataset_id: str) -> list[dict] | None:
    """[{id, sha256, tag}] of every document of a dataset, as
    write_document_list stored them; None for a dataset built before builds
    wrote its list."""
    listed = store.get_ref(DOCUMENT_LIST, dataset_id)
    if listed is None:
        return None
    return [json.loads(line) for line in store.get_bytes(listed).splitlines()]


# -- tokenizer ------------------------------------------------------------------

def build_tokenizer(repo: Repo, name: str, store, log=print, stream=False, telemetry=None, session=None):
    """(tokenizer, vocabulary object sha256, fingerprint) of a tokenizer recipe.

    The lock file freezes one vocabulary per recipe version, so adding or
    changing sources never retrains it:

    - no lock, or the recipe's `version` was raised: train on the documents
      the mixture selects now, and write the lock (replacing the old one);
    - the same version, but the recipe (other than its note) changed:
      TokenizerVersionNeeded;
    - otherwise the locked vocabulary, from the store, or rebuilt from the
      documents the lock lists (LockedDocumentsMissing when some are not in
      the store, LockMismatch when the result differs). While it is
      rebuilt, an admitted source that fails, one not in the store among
      them, is reported (logged and recorded) and left out, never a stop:
      only the documents the lock lists decide.

    With stream, training reads each source's documents in the sources'
    order (streamed_documents) instead of keeping them in the store; the
    vocabulary is the same, since training counts words and its result
    does not depend on the order of the documents.

    With a session (wpt_store.sessions), a `reused` line for each source
    whose upstream tree the training read (none when the vocabulary was
    found in the store).
    """
    telemetry = telemetry or Telemetry.off()
    recipe = repo.recipe("tokenizer", name)
    lock = repo.generated("lock", name)
    read = set()  # the sources whose upstream tree the training read

    locked = lock is not None and lock["recipe"]["version"] == recipe["version"]
    if locked:
        if _training_part(lock["recipe"]) != _training_part(recipe):
            raise TokenizerVersionNeeded(f"tokenizer {name}: the recipe changed since version {recipe['version']} was "
                                         f"locked; raise `version` in {name}.json")
        trained = lock["documents"]
    elif stream:
        trees = _pool_trees(repo.recipe("mixture", recipe["mixture"]))
        trained = []

        def selected_texts():
            for document, text in streamed_documents(repo, store, telemetry, log, used=read):
                if _pool_of(document, trees) is not None:
                    trained.append([document.id, document.sha256])
                    yield text

        log(f"training tokenizer {name} version {recipe['version']} on the documents its mixture selects, "
            f"streamed from the sources (vocab_size {recipe['vocab_size']:,})")
        streamed = Tokenizer.train(selected_texts(), recipe["vocab_size"])
        if not trained:
            raise PoolError(f"tokenizer {name}: mixture {recipe['mixture']} selects no documents")
        trained.sort()
    else:
        mixture = repo.recipe("mixture", recipe["mixture"])
        documents, info, _ = gather(repo, store, log, telemetry=telemetry)
        read |= set(info)
        selected = sorted((d for pool in assign(documents, mixture).values() for d in pool), key=lambda d: d.id)
        if not selected:
            raise PoolError(f"tokenizer {name}: mixture {recipe['mixture']} selects no documents")
        trained = [[d.id, d.sha256] for d in selected]

    key = fingerprint({"kind": "vocabulary", "code": code(), "recipe": recipe, "documents": trained})
    vocabulary = lock["vocabulary_sha256"] if locked else store.get_ref("vocabularies", key)
    if vocabulary is not None and store.has(vocabulary):
        tokenizer = Tokenizer.from_bytes(store.get_bytes(vocabulary))
    elif stream:
        if locked:
            wanted, found, left_out = {tuple(d) for d in trained}, set(), []

            def locked_texts():
                for document, text in streamed_documents(repo, store, telemetry, log, skipped=left_out, used=read):
                    if (document.id, document.sha256) in wanted:
                        found.add((document.id, document.sha256))
                        yield text

            log(f"training tokenizer {name} version {recipe['version']} again on the {len(trained):,} documents "
                f"its lock lists, streamed from the sources")
            streamed = Tokenizer.train(locked_texts(), recipe["vocab_size"])
            missing = sorted(wanted - found)
            if missing:
                raise LockedDocumentsMissing(f"tokenizer {name} version {recipe['version']}: {len(missing):,} of the "
                                             f"documents it was trained on are not in the admitted sources any "
                                             f"more, such as {missing[0][0]}{_left_out(left_out)}")
        tokenizer = streamed
        built = store.put_bytes(tokenizer.to_bytes())
        if vocabulary is not None and built != vocabulary:
            raise LockMismatch(f"tokenizer {name}: built vocabulary {built} differs from the locked {vocabulary} "
                               "for the same documents (training is not repeatable here)")
        vocabulary = built
        set_ref(store, "vocabularies", key, vocabulary, telemetry)
    else:
        left_out = []
        if locked:  # puts the current documents in the store
            read |= set(gather(repo, store, log, skipped=left_out, telemetry=telemetry, skip_failed=True)[1])
        missing = [doc_id for doc_id, sha256 in trained if not store.has(sha256)]
        if missing:
            raise LockedDocumentsMissing(f"tokenizer {name} version {recipe['version']}: its vocabulary is not in "
                                         f"{store}, and {len(missing):,} of the documents it was trained on are not "
                                         f"either, such as {missing[0]}{_left_out(left_out)}")
        log(f"training tokenizer {name} version {recipe['version']} on {len(trained):,} documents "
            f"(vocab_size {recipe['vocab_size']:,})")
        tokenizer = Tokenizer.train((store.get_bytes(sha256) for _, sha256 in trained), recipe["vocab_size"])
        built = store.put_bytes(tokenizer.to_bytes())
        if vocabulary is not None and built != vocabulary:
            raise LockMismatch(f"tokenizer {name}: built vocabulary {built} differs from the locked {vocabulary} "
                               "for the same documents (training is not repeatable here)")
        vocabulary = built
        set_ref(store, "vocabularies", key, vocabulary, telemetry)

    if session is not None:
        for sid in sorted(read):
            download.used(repo.source(sid), store, session)
    if not locked:
        # recorded, and in the store, before the lock changes: a record that cannot be written leaves it as it was
        telemetry.irreversible("tokenizer.lock", path=repo.relative("lock", name), vocabulary=vocabulary)
        written = repo.write_generated("lock", name, {"recipe": recipe, "documents": trained,
                                                      "vocabulary_sha256": vocabulary})
        log(f"wrote {written}: commit it so every machine uses this vocabulary")
    return tokenizer, vocabulary, key


def _left_out(skipped) -> str:
    """The end of a LockedDocumentsMissing message: the sources left out, when some were."""
    return f" (left out: {', '.join(failed['source'] for failed in skipped)})" if skipped else ""


def _training_part(recipe) -> dict:
    """What trains the vocabulary: the recipe without its note."""
    return {k: v for k, v in recipe.items() if k != "note"}


# -- dataset ----------------------------------------------------------------------

def build_dataset(repo: Repo, name: str, store, log=print, shard_tokens=SHARD_TOKENS, stream=False,
                  skip_failed=False, telemetry=None, session=None) -> str:
    """Build (or find) the dataset of a model recipe's data part and return
    its id; with stream, keeping no document's Markdown and no source's
    files on this machine; leaving out the sources not in the store, and
    with skip_failed every source that fails (see the module's docstring).
    With a session (wpt_store.sessions), a `reused` line for each source
    whose upstream tree the dataset holds. Only the data part counts:
    recipes with the same data share one dataset."""
    telemetry = telemetry or Telemetry.off()
    recipe = repo.recipe("model", name)["data"]
    mixture = repo.recipe("mixture", recipe["mixture"])
    telemetry.event("dataset.recipe", recipe_name=name, recipe=recipe, mixture=mixture)
    with telemetry.span("tokenizer", recipe=recipe["tokenizer"], stream=stream) as span:
        tokenizer, vocabulary, tokenizer_key = build_tokenizer(repo, recipe["tokenizer"], store, log, stream=stream,
                                                               telemetry=telemetry)
        span.set(vocabulary=vocabulary, fingerprint=tokenizer_key, vocab_size=tokenizer.vocab_size)
    telemetry.input(recipe=name, tokenizer=recipe["tokenizer"], vocabulary=vocabulary)
    skipped = []
    with telemetry.span("documents", stream=stream, skip_failed=skip_failed) as span:
        if stream:
            documents, sources_info, duplicates, packed = gather_streamed(repo, store, vocabulary,
                                                                           log=lambda _: None, skipped=skipped,
                                                                           telemetry=telemetry,
                                                                           skip_failed=skip_failed)
        else:
            documents, sources_info, duplicates = gather(repo, store, log=lambda _: None, skipped=skipped,
                                                         telemetry=telemetry, skip_failed=skip_failed)
        span.set(documents=len(documents), sources=len(sources_info), duplicates=duplicates, skipped=skipped)
    telemetry.input(sources={sid: info["checksum"] for sid, info in sorted(sources_info.items())},
                    skipped_sources=[failed["source"] for failed in skipped])
    if session is not None:
        for sid in sorted(sources_info):
            download.used(repo.source(sid), store, session)
    for failed in skipped:
        log(f"skipped {failed['source']}: {failed['error']}")
    held_ids = {d.id for d in documents if is_held_out(d.held_out_key, recipe["held_out"])}
    steps_read = conversations_and_tasks(repo, store, set(sources_info), recipe["held_out"],
                                         {d.sha256 for d in documents if d.id in held_ids}, telemetry,
                                         temporary_copies(store) if stream else None)
    text_inputs = {
        "kind": "dataset", "code": code(), "recipe": recipe, "mixture": mixture, "vocabulary": vocabulary,
        # a document's held-out group only when it has one, so earlier ids stay the same
        "documents": [[d.id, d.sha256, sorted(d.tags), *([d.group] if d.group is not None else [])] for d in documents],
        # the record copies their licenses and attribution: a change rebuilds it
        "sources": sources_info,
        # each by its id and the kind of its error alone: the message may name this machine's store. Absent when
        # none, so earlier ids stay the same
        **({"skipped": [{"source": failed["source"], "error": failed["error"].split(":", 1)[0]} for failed in skipped]}
           if skipped else {}),
    }
    # the text's own id orders and picks its rows. It names the sources that hold documents only, no part and no
    # skipped source, so a change to the conversations or tasks (their sources included), or a source left out that
    # held no documents, never changes the text
    text_id = fingerprint({**{k: v for k, v in text_inputs.items() if k != "skipped"},
                           "sources": {sid: info for sid, info in sources_info.items()
                                       if sid in {d.source for d in documents}}})
    # each part by its object only when it holds something, so earlier ids stay the same
    dataset_id = fingerprint({**text_inputs, **{part: found["object"] for part, found in steps_read.items()}})
    known = store.get_ref("datasets", dataset_id)
    # a record of another shape is built again: its readers read this shape only
    if known and current_record(json.loads(store.get_bytes(known))):
        telemetry.output(dataset=dataset_id, dataset_record=known)
        log(f"dataset {dataset_id} is already built")
        return dataset_id

    held_out = [d for d in documents if d.id in held_ids]
    training = [d for d in documents if d.id not in held_ids]
    with tempfile.TemporaryDirectory(prefix="wpt-build-") as tmp:
        with telemetry.span("tokens", documents=len(documents)):
            tokens = packed if stream else EncodedDocuments(Path(tmp) / "tokens", tokenizer, documents, store)
        with telemetry.span("pools"):
            rows, pool_report = _fill_pools(text_id, recipe, mixture, training, tokens, log)
            for report in pool_report:
                telemetry.event("pool", **report)
        with telemetry.span("shards"):
            shards = _write_shards(rows, tokens, store, Path(tmp), shard_tokens, telemetry, "training")
            held_shards = _write_shards([(d, "held-out", 0) for d in held_out], tokens, store, Path(tmp), shard_tokens,
                                        telemetry, "held_out")

    def texts(chosen):
        """The Markdown of these documents, in order."""
        if not stream:
            yield from (store.get_bytes(d.sha256) for d in chosen)
            return
        for i in range(0, len(chosen), 1024):
            part = packed.many([d.id for d in chosen[i:i + 1024]])
            yield from (tokenizer.decode(part[d.id].tolist()) for d in chosen[i:i + 1024])

    record = {
        "id": dataset_id,
        "recipe_name": name,
        "recipe": recipe,
        "mixture": mixture,
        "tokenizer": {"recipe": recipe["tokenizer"], "recipe_version": repo.recipe("tokenizer", recipe["tokenizer"])["version"],
                      "fingerprint": tokenizer_key, "vocabulary": vocabulary,
                      "vocab_size": tokenizer.vocab_size},
        "sources": sources_info,
        "downloads": store_history.downloads_used(store, {sid: info["checksum"] for sid, info in sources_info.items()}),
        "skipped": skipped,
        "documents": {"total": len(documents), "duplicates_removed": duplicates,
                      "held_out": len(held_out), "training": len(training)},
        "pools": pool_report,
        "total_tokens": sum(s["tokens"] for s in shards),
        "shards": shards,
        "held_out_shards": held_shards,
        "contamination": _recorded_contamination(training, held_out, texts, telemetry),
        **steps_read,  # conversations and tasks, each part only when the sources hold some
        "fingerprints": part_fingerprints(vocabulary, shards, held_shards, steps_read),
        "tools": tool_versions(),
        "tool_differences": tool_differences(repo),
    }
    for difference in record["tool_differences"]:
        log(f"tool version differs from the repository's reference: {difference}")
    if record["contamination"]["overlapping"]:
        log(f"contamination: {record['contamination']['overlapping']} held-out documents share a run of "
            f"{NGRAM} words with training documents (reported only)")
    with telemetry.span("dataset.record"):
        record_object = store.put_bytes(json.dumps(record, indent=2).encode() + b"\n")
        # recorded, and in the store, before the ref is set: a record that cannot be written leaves no dataset
        # under this id
        telemetry.output(dataset=dataset_id, dataset_record=record_object)
        # before the dataset's ref: a dataset is never there without its list
        write_document_list(store, dataset_id, documents, held_ids, telemetry)
        set_ref(store, "datasets", dataset_id, record_object, telemetry)
    log(f"dataset {dataset_id}: {record['total_tokens']:,} training tokens in {len(shards)} shards")
    return dataset_id


def current_record(record) -> bool:
    """Whether a dataset's record holds every field this code writes
    (RECORD_FIELDS)."""
    return isinstance(record, dict) and all(field in record for field in RECORD_FIELDS)


def part_fingerprints(vocabulary: str, shards: list, held_shards: list, steps_read: dict) -> dict:
    """{part: sha256} of what each kind of training step reads, so that a
    step's fingerprint changes only with its own part: "text" (the training
    and held-out shards, by the hashes of their tokens), and "conversations"
    and "tasks" (their objects), each only when the dataset holds the part.
    Each holds the vocabulary too, since every part becomes tokens with it."""
    parts = {"text": {"shards": [s["token_sha256"] for s in shards],
                      "held_out_shards": [s["token_sha256"] for s in held_shards]},
             **{part: {"object": found["object"]} for part, found in steps_read.items()}}
    return {part: fingerprint({"part": part, "vocabulary": vocabulary, **content}) for part, content in parts.items()}


def temporary_copies(store):
    """copies (sources.each_read) for a streamed build: each call gives a
    context manager of a store whose local copies go into a new temporary
    folder, removed when it ends."""
    @contextlib.contextmanager
    def opened():
        with tempfile.TemporaryDirectory(prefix="wpt-stream-") as tmp:
            yield store.copies_in(tmp)
    return opened


def conversations_and_tasks(repo: Repo, store, kept: set[str], share: float, held: set[str], telemetry,
                            copies=None) -> dict:
    """The parts of a dataset that the training steps read besides its
    shards, each only when the sources hold some: {"conversations": {object,
    count, sources}, "tasks": {object, count, sources}}, sources the sorted
    ids of the sources the part holds or points into, each described in the
    record's own `sources`. Each object holds
    one JSON line per conversation ({id, tags, conversation}: every chat
    file, whatever its tags) or per task (tasks.to_json: every task, its
    pointers filled in, but the measure-only ones, which share and held
    decide: the recipe's held_out and the sha256 of every held-out
    document's Markdown). Only the sources in kept, those the build kept,
    are read; one that breaks a rule here stops the build, and so does a
    task pointing into a source the build did not keep, such as one not
    downloaded (RuleViolation naming the task and the source).
    Several sources are read at once, each through its own store from
    copies (sources.each_read), and their results put together in the
    sources' order."""
    parts = {}
    with telemetry.span("conversations") as span:
        found, sources = chat_files.tagged(repo, store, None, telemetry, log=lambda _: None, only=kept,
                                           copies=copies)
        if found:
            lines = (json.dumps({"id": cid, "tags": tags, "conversation": conversation}, ensure_ascii=False,
                                sort_keys=True) for cid, tags, conversation in found)
            parts["conversations"] = {"object": store.put_bytes("".join(f"{line}\n" for line in lines).encode()),
                                      "count": len(found), "sources": sorted(sources)}
        span.set(**parts.get("conversations", {"count": 0}))
    with telemetry.span("tasks") as span:
        found, sources = task_files.read(repo, store, None, share, log=lambda _: None, held=held, only=kept,
                                        copies=copies, pointable=kept)
        if found:
            lines = (json.dumps(task_files.to_json(task), ensure_ascii=False, sort_keys=True) for task in found)
            parts["tasks"] = {"object": store.put_bytes("".join(f"{line}\n" for line in lines).encode()),
                              "count": len(found), "sources": sorted(sources)}
        span.set(**parts.get("tasks", {"count": 0}))
    return parts


def _recorded_contamination(training, held_out, texts, telemetry):
    with telemetry.span("contamination") as span:
        found = contamination(training, held_out, texts)
        span.set(**{k: v for k, v in found.items() if k != "examples"})
    return found


def _fill_pools(text_id, recipe, mixture, training, tokens, log):
    """(rows, pool report): each pool filled to its share of the recipe's tokens."""
    rows, pool_report = [], []
    assigned = assign(training, mixture)
    for pool in mixture["pools"]:
        members = assigned[pool["name"]]
        available = sum(tokens.length(d.id) for d in members)
        share = round(pool["weight"] * recipe["total_tokens"])
        if available == 0:
            raise PoolError(f"pool {pool['name']} ({pool['select']}) selects no training documents")
        epochs = share / available
        limit = pool.get("max_epochs", DEFAULT_MAX_EPOCHS)
        if epochs > limit:
            raise PoolError(f"pool {pool['name']}: filling {share:,} tokens from {available:,} needs {epochs:.2f} "
                            f"passes, more than the {limit} allowed (set max_epochs in the mixture to allow more)")
        full = math.floor(epochs)
        for pass_ in range(full):
            rows += [(d, pool["name"], pass_) for d in members]
        remainder, taken = share - full * available, 0
        for d in sorted(members, key=lambda d: _rank(text_id, pool["name"], full, d.id)):
            if taken >= remainder:
                break
            rows.append((d, pool["name"], full))
            taken += tokens.length(d.id)
        pool_report.append({"name": pool["name"], "documents": len(members), "tokens": available,
                            "share": share, "epochs": round(epochs, 6)})
        log(f"pool {pool['name']}: {len(members):,} documents, {available:,} tokens, {epochs:.2f} passes for {share:,}")
    rows.sort(key=lambda r: _rank(text_id, "order", r[2], r[0].id))
    return rows, pool_report


class EncodedDocuments:
    """Every document's tokens (ending with END), encoded once into its own
    file in folder, several documents at once as wpt_utils.parallel allows
    (each document's memory measured as it is encoded); only their lengths
    stay in memory."""

    def __init__(self, folder: Path, tokenizer, documents, store):
        folder.mkdir()
        self.folder, self._lengths = folder, {}

        def encode(item):
            i, document = item
            tokens = np.asarray(tokenizer.encode(store.get_bytes(document.sha256)) + [Tokenizer.END], dtype="<u4")
            tokens.tofile(folder / f"{i}.u32")
            return document.id, (i, len(tokens))

        sizing = Sizing("documents encoded at once")
        with ThreadPoolExecutor(sizing.limit) as executor:
            self._lengths.update(in_order(encode, enumerate(documents), executor, sizing))

    def length(self, document_id) -> int:
        return self._lengths[document_id][1]

    def __getitem__(self, document_id) -> np.ndarray:
        return np.fromfile(self.folder / f"{self._lengths[document_id][0]}.u32", dtype="<u4")

    def many(self, document_ids) -> dict[str, np.ndarray]:
        return {document_id: self[document_id] for document_id in document_ids}


def _write_shards(rows, tokens, store, folder: Path, shard_tokens, telemetry, part):
    shards, batch, size = [], [], 0

    def flush():
        path = folder / f"shard-{len(shards):05d}.parquet"
        read = tokens.many(list(dict.fromkeys(d.id for d, _, _ in batch)))
        digest = write_shard(path, [(read[d.id], d.source, d.id, pool, p) for d, pool, p in batch])
        shards.append({"object": store.put_file(path), "token_sha256": digest, "rows": len(batch), "tokens": size,
                       "sources": sorted({d.source for d, _, _ in batch})})
        telemetry.event("shard", part=part, index=len(shards) - 1, **shards[-1])
        path.unlink()

    for row in rows:
        batch.append(row)
        size += tokens.length(row[0].id)
        if size >= shard_tokens:
            flush()
            batch, size = [], 0
    if batch:
        flush()
    return shards


def contamination(training, held_out, texts) -> dict:
    """Held-out documents sharing a run of NGRAM words with training text.
    texts(documents) yields their Markdown, in order."""
    def grams(data: bytes):
        words = data.split()
        return {hashlib.blake2b(b" ".join(words[i:i + NGRAM]), digest_size=8).digest()
                for i in range(len(words) - NGRAM + 1)}

    owner = {}
    for document, text in zip(held_out, texts(held_out)):
        for gram in grams(text):
            owner.setdefault(gram, document.id)
    found = {}
    if owner:
        for document, text in zip(training, texts(training)):
            for gram in grams(text):
                if gram in owner:
                    found.setdefault(owner[gram], document.id)
    return {"ngram_words": NGRAM, "held_out_documents": len(held_out), "overlapping": len(found),
            "examples": [{"held_out": h, "training": t} for h, t in sorted(found.items())[:20]]}


# -- tool versions ------------------------------------------------------------------

def tool_versions() -> dict:
    """The code (wpt_utils.programs.code) and the outside tools it ran with."""
    versions = {"code": code(), "python": platform.python_version(), "platform": platform.platform(),
                "rustc": RUSTC_VERSION}
    for package in ("pyarrow", "numpy", "jsonschema"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        versions["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        versions["git"] = None
    return versions


def tool_differences(repo: Repo) -> list[str]:
    """Where the running tools differ from the reference pinned in the
    repository (.python-version, uv.lock, the tokenizer's rust-toolchain.toml).
    Reported, never refused."""
    differences = []
    pinned = repo.root / ".python-version"
    if pinned.exists():
        want = pinned.read_text().strip().split(".")
        if platform.python_version().split(".")[:len(want)] != want:
            differences.append(f"python {platform.python_version()} (reference {'.'.join(want)})")
    toolchain = repo.root / "packages" / "wpt-tokenizer" / "rust-toolchain.toml"
    if toolchain.exists():
        want = tomllib.loads(toolchain.read_text())["toolchain"]["channel"]
        have = RUSTC_VERSION.split()[1]  # "rustc 1.98.0 (88d9e12ae 2026-08-18)"
        if have != want:
            differences.append(f"rustc {have} built the tokenizer (reference {want})")
    lock = repo.root / "uv.lock"
    if lock.exists():
        locked = {p["name"]: p.get("version") for p in tomllib.loads(lock.read_text()).get("package", [])}
        for package in ("pyarrow", "numpy", "jsonschema"):
            try:
                have = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
            if locked.get(package) and locked[package] != have:
                differences.append(f"{package} {have} (reference {locked[package]})")
    return differences
