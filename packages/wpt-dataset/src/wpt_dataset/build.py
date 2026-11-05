"""Build vocabularies and datasets from recipes, in three steps.

    index          every downloaded file that can give a document, from the
                   store's records alone, each content once (choice.py)
    choose         the recipe's mixture draws its documents from the index,
                   by bytes, with a fixed seed per pool (choice.py)
    read           only the chosen files are read, in bundle order, and
                   their Markdown is kept so the vocabulary and the tokens
                   read it once (reading.py)

The choice comes in rounds: a pool whose training tokens fall short of its
share draws again, at the bytes per token measured, until it holds its
share or the index has no more of it. What a round reads becomes:

    documents      the Markdown of the chosen files, as kept.py keeps them;
                   exact duplicates removed (the first by source id, then
                   path, keeps it, whichever round reads it)
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
                   fixed pseudo-random order derived from the dataset id; the
                   token objects the rows lie in are copied to this machine
                   first (PackedDocuments.local), several at once, and the
                   copies removed when both parts are written, so no row is
                   read by range from the store
    contamination  runs of NGRAM words that held-out documents share with
                   training documents, checked before those copies are removed:
                   the held-out runs are listed once, and batches of training
                   documents are searched in worker processes
                   (_recorded_contamination)

The dataset id is the fingerprint of every input, the code of this module
(wpt_utils.programs.code_of: the source files of the WPT modules it
imports) among them, so building the same recipe on the same data with
the same code again reads no file, and a change to a file of the checkout
this module does not import (a document, a test, another application)
leaves it the same. A dataset whose record has
another shape than the current one (current_record) is built again, its record replaced.

Beside its record, every dataset keeps its document list (write_document_list): each document's id, the sha256 of its
Markdown and its tag, training or held-out, under a ref named by the
dataset id, so a model's provenance finds it. A dataset built
before builds wrote one never gets one.

A dataset recipe names its vocabulary. A build that finds no lock for it
trains it on a seeded sample of the first round's chosen documents, in the
same pool proportions (the recipe's sample_bytes, Choice.sample), and
writes the lock; every later dataset that names it reuses the lock
(build_tokenizer).

The index applies every source's format map to its file list, through the
function the conversion calls later (markdown.format_of), so the two cannot
disagree: a path that matches no format entry or several fails there, in
the minutes the lists take, and every failing source is named at once. A
build that is not strict leaves those out and lists them.

With plan, a build stops after the first choice and says what it would
read and compute.

The build never contacts upstream: it reads the upstream files and the
license pages the download step put in the store (download.py). A
failure is judged per item. A file no license entry covers, a file that
is not text, every file and record of a license entry that does not let
them train, and a task whose pointer no longer holds its passage, are
each left out alone; the record says which and why (each source's
`check`, and `left_out_tasks`). The record speaks of the files the build
read: a file it did not choose was never checked for text. A whole source
that fails (it is not in the store, a cut or a patch of it does not apply,
its format map fails, a file is broken) is left out, and the record lists
it with its error under `skipped`; a build that trains a vocabulary lists
each one in the lock too (build_tokenizer). A strict build stops at a
failing source instead, and names every source the index could not take
before any text is read. A missing program stops a build in either case,
since it fails every source that needs it.

The record also names, for each source, the session (its run record's
folder) and the time of the download that put the files it used in the
store, from the store database (`downloads`; None for a source with no
such download, an empty one or one downloaded before the store kept
sessions). It is not part of the dataset id: the same files downloaded
again give the same dataset. `wpt dataset build` runs as a session of its
own, which holds a `reused` line for each source whose upstream tree the
dataset holds.

Sources are read, turned into tokens, and their conversations and tasks
read in worker processes (workers.py), several at once, as many as
wpt_utils.parallel allows, each item's memory measured as it runs and each
item's chosen bytes counted on the disk its local copies go to; the
results are put together in the documents' order, so the dataset id and the
shards are the ones a build of one source at a time gives. A large source is
read in parts (reading.parts_of), each an item of its own, and the build
takes a source only once all its parts are read, as if one item had read it:
a part that fails takes the whole source out, and the parts after it are not
read. On a store whose reads wait, downloaders copy the files of the coming
parts to this machine ahead of the workers (downloaders.py).

No document's Markdown goes to the store, and no source's files stay on this
machine: only the files a conversion needs as files (a Parquet file, whose
reader seeks) and those the downloaders fetch ahead are copied, into temporary
folders removed once the part is read; every other file, records and message trees included, is read from
the store as a stream or by range (wpt_store's copies_in). The Markdown
of the chosen documents stays in the cache folder of the store, for the
next build. With stream=True it goes to a temporary folder instead, removed
when the build ends, and the machine keeps nothing; the documents, the
dataset id and the shards are the same.

The conversations and tasks the build keeps (conversations_and_tasks) do
not depend on the choice: every indexed source that holds some is read, each
through its own temporary folder of local copies when streaming.

With a telemetry (wpt_telemetry), the build records each source (read,
its documents, or why it was skipped), the vocabulary, each
pool, each shard written, the contamination check and the dataset record,
and a long stage says how far it is every few minutes (Telemetry.progress).
"""

import collections
import concurrent.futures
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wpt_store import CorruptObject, MissingObject
from wpt_store.shards import write_shard
from wpt_telemetry import Telemetry, set_ref
from wpt_tokenizer import BASE_SIZE, RUSTC_VERSION, Tokenizer
from wpt_utils.lines import WARNING, say
from wpt_utils.parallel import Done, Sizing, each, in_order
from wpt_utils.programs import code, code_of

from . import choice as choosing
from . import conversations as chat_files
from . import download, downloaders, license_check, pools, reading, store_history, trained, workers
from . import tasks as task_files
from .errors import (DatasetError, PoolError, TokenizerVersionNeeded,
                     TrainedListMissing, VocabularyMissing)
from .kept import kept, skipped_values
from .repo import SIZE_OF, Repo
from .sources import fingerprint, held_manifest, held_out_group, tags_of

DEFAULT_MAX_EPOCHS = 4
SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per shard
NGRAM = 13  # contamination check: runs of 13 words, as in the GPT-3 paper
TOKEN_READERS = 16  # documents read at once from the token objects
# token objects made local at once for the shards. Each copy already fetches up to Sizing("downloads").limit parts at
# once (Store._copy), and an S3 store holds UPLOADS + that limit + REQUESTS connections: 3 objects at once stay within
# them up to 20 parts each
OBJECTS_AT_ONCE = 3
CONTAMINATION_BATCH_TOKENS = 1 << 22  # tokens of the training documents one worker checks in one call (a longer document alone)
DOCUMENT_LIST = "dataset-documents"  # the kind of the ref, named by the dataset id, of a dataset's document list
TRAINING, HELD_OUT = "training", "held-out"  # a listed document's tag
# the top-level fields every record this code writes holds (conversations and tasks only when the sources hold
# some): a record without one is of another shape, which its readers refuse and a build writes again
RECORD_FIELDS = ("id", "recipe_name", "recipe", "mixture", "tokenizer", "sources", "downloads", "skipped",
                 "left_out_tasks", "documents",
                 "pools", "total_tokens", "shards", "held_out_shards", "contamination", "fingerprints", "tools",
                 "tool_differences")
SOURCE_ERRORS = (DatasetError, CorruptObject, MissingObject)  # what fails one source, and not the others


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


def each_source(function, parts, repo, store, telemetry, progress, args, disk, folder, download=None):
    """(wpt_utils.parallel.Done of the source with [function(source, store,
    *args(part), telemetry=...) of each of its parts], the source's "source"
    span) for each source, in the parts' order, a source's parts (reading.Part)
    next to each other and given together once the last ends. Each part runs
    in a worker process (workers.py), as many at once as wpt_utils.parallel
    allows, reserving on the disk of folder disk(part) bytes, the conversion
    stating them beforehand (the event "sources.at_once" says how many and
    why). progress (Telemetry.progress) advances as each part's work ends, in
    whatever order they end. A DatasetError, CorruptObject or MissingObject
    the function raises is the source's error, the first of its parts' in
    their order; anything else (a missing program, a store that refuses a
    write) stops every source. A part whose turn comes after an earlier part
    of its source failed does not run and gives None: nothing of the source
    counts any more. The parts before a failed one still run, so the source's
    error does not depend on which part failed first in time.

    download, when given, says what the downloaders copy ahead of each part:
    the positions, vocabulary and cache folder of downloaders.download. A
    part then waits for a download under way, runs on the folder it filled,
    and that folder goes when the part ends.

    A part that saves its source's `newest` indexes (Part.indexes) runs
    reading.save_newest instead of function, as a part in every other way:
    the source's other parts start once it saved them or failed with the
    source's error, its failure is the source's, other sources are read
    meanwhile, and it adds nothing to the list given. An error that stops
    every source leaves those parts unstarted, so none scans the files itself.

    The span opens under the caller's current span when the source's first
    part starts, so what the workers record lands under it, and it says how
    many seconds the work took, its parts' added (converting_seconds); the
    caller ends it (telemetry.finish) once it has put the source's result
    together. The spans of the sources whose turn never came, because
    something stopped the sources, end with an error saying so."""
    with (workers.Pool(repo, "sources at once", disk=disk, folder=folder) as pool,
          downloaders.ahead_of(parts, download, repo, store, telemetry) as ahead):
        telemetry.event("sources.at_once", limit=pool.sizing.limit, why=pool.sizing.why)
        parent, spans = telemetry.current(), {}  # spans: each source begun and not yet given to the caller
        seconds, lock = collections.Counter(), threading.Lock()  # lock: a source's parts run on several threads
        failed = {}  # {source id: the number of its first part that failed so far}
        indexing = {part.source.id for part in parts if part.indexes}  # the sources whose indexes are not saved yet

        def run(part):
            source = part.source
            with lock:
                passed = failed.get(source.id, part.number) < part.number
            if passed:
                ahead.release(part)
                progress.advance()
                return None
            with lock:
                if source.id not in spans:
                    with telemetry.within(parent):
                        spans[source.id] = telemetry.begin("source", source=source.id)
                span = spans[source.id]
            started = time.monotonic()
            try:
                with telemetry.within(span):
                    if not part.indexes:
                        return pool.call(function, source, ahead.take(part), *args(part), telemetry=telemetry)
                    pool.call(reading.save_newest, source, store, part.positions, telemetry=telemetry)
                with lock:
                    indexing.discard(source.id)
                return None
            except SOURCE_ERRORS:
                with lock:
                    failed[source.id] = min(part.number, failed.get(source.id, part.number))
                    indexing.discard(source.id)
                raise
            finally:
                ahead.release(part)
                with lock:
                    seconds[source.id] += time.monotonic() - started
                    span.set(converting_seconds=round(seconds[source.id], 3))
                progress.advance()

        try:
            gathered = []  # what the parts of the source under way gave
            for done in each(run, parts, SOURCE_ERRORS, pool.sizing,
                             ready=lambda part: part.indexes or part.source.id not in indexing):
                gathered.append(done)
                if len(gathered) == done.item.of:
                    source = done.item.source
                    error = next((one.error for one in gathered if one.error is not None), None)
                    yield Done(source, [one.value for one in gathered if not one.item.indexes], error), \
                        spans.pop(source.id)
                    gathered = []
        finally:
            stopped = RuntimeError("the sources stopped before this source's turn")
            for span in spans.values():
                telemetry.end(span, stopped)


def checked(held, skipped: dict) -> dict:
    """A source's check result as a dataset's record keeps it: which
    statement was found at which place and which license text matched it,
    with that text's decision (kept.Kept.statements), what was left out and
    why (kept.Kept.left_out), and, for each records file that left records
    out, the values no entry lists (`unlisted`) and those of an entry that
    does not train (`untrained`), each with its count (markdown.documents'
    skipped)."""
    result = held.result()
    if values := skipped_values(skipped):
        result["records"] = values
    return result


def source_info(source, tree: str, store, check: dict) -> dict:
    """What a dataset's record keeps of one source: where it comes from
    (each of its `addresses` too, when it has several), the manifest of
    the upstream tree it was built from (sources.held_manifest), its final
    tree, licenses and its check result (checked)."""
    where = {"addresses": source.data["addresses"]} if "addresses" in source.data else {}
    return {"address": source.data["address"], **where, "manifest": held_manifest(source, store), "tree": tree,
            "licenses": source.data["licenses"], "check": check}


def _skip(source, error, skipped, strict, log):
    """Record a source that failed in skipped, or with strict raise its
    error. A missing program (wpt_utils.programs.MissingTool), which would
    fail every source using it, is not caught here and always stops the
    build."""
    if strict:
        raise error
    skipped.append({"source": source.id, "error": f"{type(error).__name__}: {error}"})
    log(f"skipped {source.id}: {type(error).__name__}: {error}", WARNING)


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


# -- the chosen documents -------------------------------------------------------------

def markdown_cache(store, stream: bool):
    """A context manager of the folder the chosen documents' Markdown goes
    to: the store's cache folder, or with stream a temporary one."""
    if stream:
        return tempfile.TemporaryDirectory(prefix="wpt-markdown-")
    return contextlib.nullcontext(store.cache / "markdown")


def chosen_first(repo: Repo, recipe: dict, mixture: dict, store, log, skipped, telemetry, strict):
    """(the Choice of a dataset recipe, the rows of its first round). A
    source the index could not take goes into skipped; with strict,
    DatasetError names every such source, with no text read."""
    index = choosing.build(repo, store, telemetry, log)
    if strict and index.failed:
        failing = [f"{type(error).__name__}: {error}" for _, error in index.failed]
        raise DatasetError(f"{len(failing)} of {len(index.sources) + len(index.failed)} admitted sources fail in the "
                           f"index; nothing was read:\n" + "\n".join(failing))
    for source, error in index.failed:
        _skip(source, error, skipped, False, log)
        telemetry.event("source.skipped", source=source.id, error=f"{type(error).__name__}: {error}")
    found = choosing.Choice(index, mixture)
    rows = found.take(choosing.first_wanted(mixture, recipe))
    log(f"index: {len(index.rows):,} documents of {len(index.sources):,} sources ({index.duplicates:,} repeated files "
        f"left out); chosen in the first round: {len(rows):,}")
    return found, rows


def _chosen_parts(found, rows, without=frozenset()) -> list[reading.Part]:
    """The parts of the sources of the rows, in bundle order."""
    return [part for source, its in choosing.by_source(found.index, rows, without)
            for part in reading.parts_of(source, its["position"].tolist(), its["bytes"].tolist())]


def _merged_check(known: dict, new: dict) -> dict:
    """One check result out of those of two parts or rounds of a source: the
    files left out of either, the records left out of either."""
    paths = {item["path"]: item for item in new["left_out"] + known["left_out"] if "path" in item}
    files = sorted(paths.values(), key=lambda item: item["path"].encode())
    merged = {**new, "left_out": files + [item for item in new["left_out"] if "path" not in item]}
    if records := {**known.get("records", {}), **new.get("records", {})}:
        merged["records"] = records
    return merged


@dataclass
class Gathered:
    """What the rounds of a build read."""
    documents: list
    info: dict  # {source id: source_info}
    duplicates: int
    tokens: "PackedDocuments"


def read_rounds(repo, recipe, mixture, found, rows, vocabulary, store, cache, skipped, strict, failed, log,
                telemetry) -> Gathered:
    """Read a build's rows, then draw and read until every pool holds its
    share in training tokens or the index has no more of it (see the
    module's docstring). rows: the first round's. failed: the ids of the
    sources left out so far, which a source that fails here joins, its earlier
    rounds' documents gone."""
    documents, info, credits, seen = {}, {}, {}, {}  # seen: {sha256 of a Markdown: the document that keeps it}
    duplicates, trees, tokens, round_ = found.index.duplicates, _pool_trees(mixture), PackedDocuments(store), 0
    while len(rows):
        round_ += 1
        parts = _chosen_parts(found, rows, failed)
        # closed at once when a source stops the others, so the sources still running end first
        with (telemetry.progress(f"read, round {round_}", len(parts), "parts") as progress,
              contextlib.closing(each_source(reading.read_tokens, parts, repo, store, telemetry, progress,
                                             lambda part: (part.positions, vocabulary, str(cache)),
                                             lambda part: part.size, tempfile.gettempdir(),
                                             lambda part: (part.positions, vocabulary, str(cache)))) as converted):
            for done, span in converted:
                source = done.item
                with telemetry.finish(span):
                    if done.error is not None:
                        _skip(source, done.error, skipped, strict, log)
                        span.set(skipped=f"{type(done.error).__name__}: {done.error}")
                        failed.add(source.id)
                        continue
                    summaries = done.value
                    check = functools.reduce(_merged_check, [summary["check"] for summary in summaries])
                    if source.id in info:
                        check = _merged_check(info[source.id]["check"], check)
                    tree = summaries[0]["tree"]
                    info[source.id] = source_info(source, tree, store, check)
                    before = duplicates
                    for summary, entry in ((summary, entry) for summary in summaries for entry in summary["documents"]):
                        document_id, held = f"{source.id}/{entry['path']}", seen.get(entry["sha256"])
                        if held is not None:
                            duplicates += 1
                            if _place(held) < _place(document_id):
                                continue
                            del documents[held]
                            credits.pop(held, None)
                        seen[entry["sha256"]] = document_id
                        documents[document_id] = Document(document_id, source.id, entry["sha256"],
                                                          tags_of(source, entry["file"]),
                                                          held_out_group(source, entry["path"]))
                        tokens.add(document_id, summary["tokens"], entry)
                        if "record" in entry:
                            credits[document_id] = json.dumps({"document": document_id, **entry["record"]},
                                                              ensure_ascii=False, sort_keys=True)
                    span.set(tree=tree, tokens_objects=[summary["tokens"] for summary in summaries],
                             documents=sum(len(summary["documents"]) for summary in summaries)
                             - (duplicates - before), duplicates=duplicates - before)
        kept_tokens = dict.fromkeys(found.names, 0)
        for document in documents.values():
            if document.source not in failed and not is_held_out(document.held_out_key, recipe["held_out"]) \
                    and (pool := _pool_of(document, trees)) is not None:
                kept_tokens[pool] += tokens.length(document.id)
        read = {name: found.bytes_of(name)[0] for name in found.names}
        rows = found.take(choosing.next_wanted(mixture, recipe, read, kept_tokens))
    chosen = sorted((d for d in documents.values() if d.source not in failed), key=lambda d: (d.source, d.id.encode()))
    info = {sid: info[sid] for sid in sorted(info) if sid not in failed}
    lines = {}
    for document in chosen:
        if document.id in credits:
            lines.setdefault(document.source, []).append(credits[document.id])
    for sid, found_lines in lines.items():
        info[sid]["credits"] = store.put_bytes("".join(f"{line}\n" for line in found_lines).encode())
    log(f"documents: {len(chosen):,} from {len(info)} sources in {round_} rounds ({duplicates:,} exact duplicates "
        f"removed)")
    return Gathered(chosen, info, duplicates, tokens)


def _place(document_id: str) -> tuple[str, bytes]:
    """What decides which of two documents with the same Markdown stays:
    the first by source id, then by path."""
    source, _, path = document_id.partition("/")
    return source, path.encode()


def _reads(rows) -> str:
    """What reading these rows takes, for a plan."""
    numbers = choosing.in_numbers(rows)
    return (f"reads {numbers['files']:,} files, {numbers['bytes']:,} bytes, of {numbers['sources']:,} sources in "
            f"{numbers['bundles']:,} bundles")


def plan_lines(found, mixture, recipe, rows, vocabulary: str) -> list[str]:
    """What a dataset's build would read and compute, from its first round's
    rows."""
    lines = []
    for pool in mixture["pools"]:
        taken, held = found.bytes_of(pool["name"])
        lines.append(f"pool {pool['name']}: weight {pool['weight']}, {found.taken[pool['name']]:,} of "
                     f"{len(found.orders[pool['name']]):,} documents, {taken:,} of {held:,} bytes, for "
                     f"{round(pool['weight'] * recipe['total_tokens']):,} tokens")
    return [*lines, f"vocabulary: {vocabulary}",
            f"{_reads(rows)}, and computes their Markdown and tokens; a pool that falls short of its share then "
            f"draws again"]


def vocabulary_plan_lines(mixture, sample: dict, name: str) -> list[str]:
    """What training the vocabulary name would read and compute, from each
    pool's rows of its sample (choice.Choice.sample_of_pools)."""
    rows = np.concatenate(list(sample.values()))
    lines = [f"pool {pool['name']}: weight {pool['weight']}, {len(sample[pool['name']]):,} files, "
             f"{int(sample[pool['name']]['bytes'].sum()):,} bytes of the sample" for pool in mixture["pools"]]
    return [*lines, f"vocabulary: {_sample_line(name, rows)}",
            f"{_reads(rows)}, computes their Markdown and trains the vocabulary on it; it computes no token"]


class PackedDocuments:
    """Documents' tokens read from the token objects of reading.read_tokens, a
    part of the object each, checked against the document's own token
    sha256. Each document is read by range from the store, unless its token
    object has a local copy (local), which the store's read_range reads."""

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

    @contextlib.contextmanager
    def local(self, document_ids, telemetry):
        """A with block in which every token object these documents lie in
        has a local copy in the store's cache folder (Store.path: checked
        against its sha256; a local store's uncompressed object is its own
        file and gets none), made several at once, so reading the documents
        takes no range from the store. The copies are released when the
        block ends, also when it or the copying fails. The disk the copies
        take is counted as wpt_utils.parallel.Sizing counts it: the copies
        in flight stay within what the free disk of the cache folder allows,
        and the first always starts."""
        objects = sorted({self._where[document_id][0] for document_id in document_ids})
        try:
            sizes = {sha: self.store.size(sha) for sha in objects}
            sizing = Sizing("token objects made local at once", disk=sizes.__getitem__, folder=self.store.cache,
                            network=OBJECTS_AT_ONCE)
            progress = telemetry.progress("token objects made local", sum(sizes.values()), "bytes")

            def make(sha):
                self.store.path(sha)
                return sha

            with progress, concurrent.futures.ThreadPoolExecutor(sizing.limit) as executor:
                for sha in in_order(make, objects, executor, sizing):
                    progress.advance(sizes[sha])
            yield
        finally:
            for sha in objects:
                self.store.release(sha)

    def of(self, document_ids) -> "PackedDocuments":
        """The documents of document_ids alone, over the same store, small enough to send to a worker process."""
        part = PackedDocuments(self.store)
        part._where = {document_id: self._where[document_id] for document_id in document_ids}
        return part

    def many(self, document_ids) -> dict[str, np.ndarray]:
        with concurrent.futures.ThreadPoolExecutor(TOKEN_READERS) as pool:
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

def locked_vocabulary(repo: Repo, name: str, store):
    """(tokenizer, vocabulary object sha256, fingerprint) of the lock of the
    tokenizer recipe name, or None when it has none for this `version`.

    The lock file freezes one vocabulary per recipe version, so adding or
    changing sources never retrains it:

    - no lock, or the recipe's `version` was raised: None, and the next
      build that names it trains it;
    - the same version, but the recipe (other than its note) changed:
      TokenizerVersionNeeded;
    - otherwise the locked vocabulary, shared through the store, checked
      against the sha256 the lock keeps: it is never trained again from the
      documents the lock lists, here or on another machine. A store that
      does not hold it gives VocabularyMissing: a machine without access to
      the store that holds it cannot use it. A store without its list of
      documents gives TrainedListMissing."""
    recipe = repo.recipe("tokenizer", name)
    lock = repo.generated("lock", name)
    if lock is None or lock["recipe"]["version"] != recipe["version"]:
        return None
    if _training_part(lock["recipe"]) != _training_part(recipe):
        raise TokenizerVersionNeeded(f"tokenizer {name}: the recipe changed since version {recipe['version']} was "
                                     f"locked; raise `version` in {name}.json")
    vocabulary = lock["vocabulary_sha256"]
    if not store.has(vocabulary):
        raise VocabularyMissing(f"tokenizer {name} version {recipe['version']}: its locked vocabulary {vocabulary} "
                                f"is not in {store}; a locked vocabulary is shared through the store that holds "
                                f"it, never trained again")
    if not store.has(lock["documents_sha256"]):
        raise TrainedListMissing(f"tokenizer {name} version {recipe['version']}: the list of the documents it "
                                 f"was trained on, {lock['documents_sha256']}, is not in {store}; it is shared "
                                 f"through the store that holds the vocabulary")
    key = fingerprint({"kind": "vocabulary", "code": code_of(__name__), "recipe": recipe,
                       "documents": lock["documents_sha256"]})
    return Tokenizer.from_bytes(store.get_bytes(vocabulary)), vocabulary, key


def train_vocabulary(repo: Repo, name: str, mixture, found, store, cache, skipped, strict, failed, log, telemetry,
                     session=None):
    """(tokenizer, vocabulary object sha256, fingerprint) of the tokenizer
    recipe name, trained on the sample its sample_bytes draws out of the
    rows found took, and locked. The fingerprint, the key of the ref
    vocabularies/<fingerprint>, holds the code of this module, the recipe and
    the sha256 of the list of the documents it is trained on.

    The lock names those documents by one object of the store, their sorted
    list (wpt_dataset.trained), with its sha256 and its number of lines, and
    lists the sources left out (skipped) by their ids and the kinds of
    their errors. Training reads every document of the
    sample and counts chunks in a table of a fixed number of slots
    (wpt_tokenizer): while the documents have no more distinct chunks than
    that, the vocabulary is the same whatever their order; beyond it, it
    depends on the order, the sources' by bundle and each source's documents
    by path (reading.documents_of).

    With a session (wpt_store.sessions), a `reused` line for each source
    the training read."""
    recipe = repo.recipe("tokenizer", name)
    parts = _chosen_parts(found, found.sample(mixture, recipe["sample_bytes"]), failed)
    reads, seen = [], {}  # reads: each source's, its parts' Reads; seen: {sha256 of a Markdown: the document keeping it}
    with trained.TrainedList() as listed:
        # closed at once when a source stops the others, so the sources still running end first
        with (telemetry.progress("vocabulary sample", len(parts), "parts") as progress,
              contextlib.closing(each_source(reading.read_markdown, parts, repo, store, telemetry, progress,
                                             lambda part: (part.positions, str(cache)),
                                             lambda part: part.size, tempfile.gettempdir(),
                                             lambda part: (part.positions, None, str(cache)))) as converted):
            for done, span in converted:
                with telemetry.finish(span):
                    if done.error is not None:
                        _skip(done.item, done.error, skipped, strict, log)
                        span.set(skipped=f"{type(done.error).__name__}: {done.error}")
                        failed.add(done.item.id)
                        continue
                    reads.append(done.value)
                    for entry, _ in reading.documents_of(done.value):
                        document_id, held = f"{done.item.id}/{entry['path']}", seen.get(entry["sha256"])
                        if held is None or _place(document_id) < _place(held):
                            seen[entry["sha256"]] = document_id
        for sha256, document_id in seen.items():
            listed.add(document_id, sha256)
        if not listed.count:
            raise PoolError(f"tokenizer {name}: the mixture's sample holds no documents")
        documents_sha256, document_count = listed.put(store)

    def texts():
        repeated = set()
        for read in reads:
            for entry, text in reading.documents_of(read):
                if entry["sha256"] not in repeated:
                    repeated.add(entry["sha256"])
                    yield text

    key = fingerprint({"kind": "vocabulary", "code": code_of(__name__), "recipe": recipe,
                       "documents": documents_sha256})
    vocabulary = store.get_ref("vocabularies", key)
    if vocabulary is not None and store.has(vocabulary):
        tokenizer = Tokenizer.from_bytes(store.get_bytes(vocabulary))
    else:
        log(f"training tokenizer {name} version {recipe['version']} on {document_count:,} documents "
            f"(vocab_size {recipe['vocab_size']:,})")
        merges = telemetry.progress("vocabulary merges", recipe["vocab_size"] - BASE_SIZE, "merges")
        with contextlib.ExitStack() as stages:
            def counted():
                with telemetry.progress("vocabulary chunks", document_count, "documents") as progress:
                    yield from _counted(texts(), progress)
                stages.enter_context(merges)  # the merges start where the last document is counted

            tokenizer = Tokenizer.train(counted(), recipe["vocab_size"],
                                        merged=lambda done: merges.advance(done - merges.done))
        vocabulary = store.put_bytes(tokenizer.to_bytes())
    set_ref(store, "vocabularies", key, vocabulary, telemetry)

    if session is not None:
        for source in sorted({part.source.id for part in parts} - failed):
            download.used(repo.source(source), store, session)
    # recorded, and in the store, before the lock changes: a record that cannot be written leaves it as it was
    telemetry.irreversible("tokenizer.lock", path=repo.relative("lock", name), vocabulary=vocabulary)
    left_out = [{"source": one["source"], "error": one["error"].split(":", 1)[0]} for one in skipped]
    written = repo.write_generated("lock", name, {"recipe": recipe, "documents_sha256": documents_sha256,
                                                  "document_count": document_count,
                                                  "vocabulary_sha256": vocabulary,
                                                  **({"left_out": left_out} if left_out else {})})
    log(f"wrote {written}: commit it so every machine uses this vocabulary, and keep the store that holds it")
    return tokenizer, vocabulary, key


def build_tokenizer(repo: Repo, name: str, store, log=say, stream=False, strict=False, telemetry=None,
                    session=None, plan=False):
    """(tokenizer, vocabulary object sha256, fingerprint) of the vocabulary
    a dataset recipe names (name: a model recipe's, or SERIES:SIZE,
    Repo.model_recipe): the locked one (locked_vocabulary), or with no lock
    the one it indexes, chooses, samples, trains and locks
    (train_vocabulary), stopping before any token. With plan, None after
    logging what it would read and compute (vocabulary_plan_lines)."""
    telemetry = telemetry or Telemetry.off()
    recipe = repo.model_recipe(name)["data"]
    locked = locked_vocabulary(repo, recipe["tokenizer"], store)
    if locked is not None:
        if plan:
            log(f"vocabulary: {recipe['tokenizer']} is locked; nothing to read or compute")
            return None
        return locked
    mixture, skipped = repo.recipe("mixture", recipe["mixture"]), []
    found, rows = chosen_first(repo, recipe, mixture, store, log, skipped, telemetry, strict)
    if plan:
        sample = found.sample_of_pools(mixture, repo.recipe("tokenizer", recipe["tokenizer"])["sample_bytes"])
        for line in vocabulary_plan_lines(mixture, sample, recipe["tokenizer"]):
            log(line)
        return None
    with markdown_cache(store, stream) as cache:
        return train_vocabulary(repo, recipe["tokenizer"], mixture, found, store, cache, skipped, strict, set(),
                                log, telemetry, session)


def _sample_line(name: str, sample) -> str:
    numbers = choosing.in_numbers(sample)
    return f"{name} has no lock: it trains on a sample of {numbers['files']:,} files, {numbers['bytes']:,} bytes"


def _training_part(recipe) -> dict:
    """What trains the vocabulary: the recipe without its note."""
    return {k: v for k, v in recipe.items() if k != "note"}


# -- dataset ----------------------------------------------------------------------

def build_dataset(repo: Repo, name: str, store, log=say, shard_tokens=SHARD_TOKENS, stream=False,
                  strict=False, telemetry=None, session=None, plan=False) -> str | None:
    """Build (or find) the dataset of a model recipe's data part (name: a
    model recipe's, or SERIES:SIZE, Repo.model_recipe) and return its id;
    with stream, keeping no document's Markdown on this machine; leaving out
    the sources that fail, unless
    strict (see the module's docstring). With a session
    (wpt_store.sessions), a `reused` line for each source whose upstream
    tree the dataset holds. Only the data part counts: recipes with the same
    data share one dataset, as every size of a series does. With plan, None
    after logging what it would read and compute (plan_lines)."""
    telemetry = telemetry or Telemetry.off()
    recipe = repo.model_recipe(name)["data"]
    mixture = repo.recipe("mixture", recipe["mixture"])
    telemetry.event("dataset.recipe", recipe_name=name, recipe=recipe, mixture=mixture)
    skipped, failed = [], set()
    found, rows = chosen_first(repo, recipe, mixture, store, log, skipped, telemetry, strict)
    locked = locked_vocabulary(repo, recipe["tokenizer"], store)
    if plan:
        sample = found.sample(mixture, repo.recipe("tokenizer", recipe["tokenizer"])["sample_bytes"])
        state = f"{recipe['tokenizer']} is locked" if locked else _sample_line(recipe["tokenizer"], sample)
        for line in plan_lines(found, mixture, recipe, rows, state):
            log(line)
        return None
    with markdown_cache(store, stream) as cache:
        with telemetry.span("tokenizer", recipe=recipe["tokenizer"], stream=stream) as span:
            tokenizer, vocabulary, tokenizer_key = locked or train_vocabulary(
                repo, recipe["tokenizer"], mixture, found, store, cache, skipped, strict, failed, log, telemetry)
            span.set(vocabulary=vocabulary, fingerprint=tokenizer_key, vocab_size=tokenizer.vocab_size)
        telemetry.input(recipe=name, tokenizer=recipe["tokenizer"], vocabulary=vocabulary)
        with telemetry.span("documents", stream=stream, strict=strict) as span:
            gathered = read_rounds(repo, recipe, mixture, found, rows, vocabulary, store, cache, skipped, strict,
                                   failed, lambda *_: None, telemetry)
            span.set(documents=len(gathered.documents), sources=len(gathered.info), duplicates=gathered.duplicates,
                     skipped=skipped)
    documents, sources_info, duplicates, packed = (gathered.documents, gathered.info, gathered.duplicates,
                                                   gathered.tokens)
    for failed_one in skipped:
        log(f"skipped {failed_one['source']}: {failed_one['error']}", WARNING)
    for sid, info in sorted(sources_info.items()):
        if info["check"]["left_out"] or info["check"].get("records"):
            log(f"{sid}: left out {_left_out_summary(info['check'])}", WARNING)
    held_ids = {d.id for d in documents if is_held_out(d.held_out_key, recipe["held_out"])}
    built = {source.id for source in found.index.sources} - failed
    steps_read, left_out_tasks = conversations_and_tasks(repo, store, built, recipe["held_out"],
                                         {d.sha256 for d in documents if d.id in held_ids}, telemetry,
                                         temporary_copies(store) if stream else None)
    sources_info = _with_parts(repo, store, sources_info, steps_read, telemetry)
    telemetry.input(sources={sid: info["manifest"] for sid, info in sorted(sources_info.items())},
                    skipped_sources=[failed_one["source"] for failed_one in skipped])
    if session is not None:
        for sid in sorted(sources_info):
            download.used(repo.source(sid), store, session)
    text_inputs = {
        "kind": "dataset", "code": code_of(__name__), "recipe": recipe, "mixture": mixture, "vocabulary": vocabulary,
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
    holding = {d.source for d in documents}
    text_id = fingerprint({**{k: v for k, v in text_inputs.items() if k != "skipped"},
                           "sources": {sid: info for sid, info in sources_info.items() if sid in holding}})
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

    def texts(chosen):
        """The Markdown of these documents, in order, decoded from their tokens."""
        for i in range(0, len(chosen), 1024):
            part = packed.many([d.id for d in chosen[i:i + 1024]])
            yield from (tokenizer.decode(part[d.id].tolist()) for d in chosen[i:i + 1024])

    with tempfile.TemporaryDirectory(prefix="wpt-build-") as tmp:
        with telemetry.span("pools"):
            rows, pool_report = _fill_pools(text_id, recipe, mixture, training, packed, log)
            for report in pool_report:
                telemetry.event("pool", **report)
        with telemetry.span("shards"):
            held_rows = [(d, "held-out", 0) for d in held_out]
            with packed.local([d.id for d, _, _ in rows + held_rows], telemetry):
                shards = _write_shards(rows, packed, store, Path(tmp), shard_tokens, telemetry, "training")
                held_shards = _write_shards(held_rows, packed, store, Path(tmp), shard_tokens, telemetry, "held_out")
                # from the same local copies; the worker processes read them through the same cache folder
                checked = _recorded_contamination(repo, training, held_out, packed, texts, vocabulary, Path(tmp),
                                                  telemetry)

    record = {
        "id": dataset_id,
        "recipe_name": name,
        "recipe": recipe,
        "mixture": mixture,
        "tokenizer": {"recipe": recipe["tokenizer"], "recipe_version": repo.recipe("tokenizer", recipe["tokenizer"])["version"],
                      "fingerprint": tokenizer_key, "vocabulary": vocabulary,
                      "vocab_size": tokenizer.vocab_size},
        "sources": sources_info,
        "downloads": store_history.downloads_used(store, {sid: info["manifest"] for sid, info in sources_info.items()}),
        "skipped": skipped,
        "left_out_tasks": left_out_tasks,
        "documents": {"total": len(documents), "duplicates_removed": duplicates,
                      "held_out": len(held_out), "training": len(training)},
        "pools": pool_report,
        "total_tokens": sum(s["tokens"] for s in shards),
        "shards": shards,
        "held_out_shards": held_shards,
        "contamination": checked,
        **steps_read,  # conversations and tasks, each part only when the sources hold some
        "fingerprints": part_fingerprints(vocabulary, shards, held_shards, steps_read),
        "tools": tool_versions(),
        "tool_differences": tool_differences(repo),
    }
    for difference in record["tool_differences"]:
        log(f"tool version differs from the repository's reference: {difference}", WARNING)
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


def run(repo: Repo, name: str, store, arguments, stream=False, strict=False, plan=False) -> str | None:
    """`wpt dataset build`: build_dataset of the model recipe name, recorded
    in the store (wpt_telemetry: records/<UTC>-wpt-dataset-build-<name>-.../)
    with a session whose lines name each source it read (`reused`), the
    session ended before the record. Returns the dataset id, None with plan.
    The failures of build_dataset, RecordError when the record cannot be
    written, and StoreError when the session cannot."""
    with Telemetry.start(store, "wpt dataset build", arguments, name, inputs={"recipe": name},
                         env_file=repo.root / ".env") as telemetry:
        with download.session(store, telemetry) as session:
            dataset_id = build_dataset(repo, name, store, log=telemetry.log, stream=stream, strict=strict,
                                       telemetry=telemetry, session=session, plan=plan)
        if dataset_id is not None:
            kind, key = dataset_ref(name)
            set_ref(store, kind, key, store.get_ref("datasets", dataset_id), telemetry)
            telemetry.log(dataset_id)
    return dataset_id


def dataset_ref(name: str) -> tuple[str, str]:
    """The ref naming the record of the dataset last built for a model
    recipe's name: recipes/NAME, or series/SERIES for a size of a series,
    every size of which builds the same dataset."""
    series, separated, _ = name.partition(SIZE_OF)
    return ("series", series) if separated else ("recipes", name)


def last_dataset(store, name: str) -> str | None:
    """The id of the dataset last built for a model recipe's name; None
    when none was."""
    record = store.get_ref(*dataset_ref(name))
    return None if record is None else json.loads(store.get_bytes(record))["id"]


def run_tokenizer(repo: Repo, name: str, store, arguments, stream=False, strict=False,
                  plan=False) -> tuple[str, str] | None:
    """`wpt dataset build-tokenizer`: build_tokenizer of the dataset recipe
    name, recorded and with a session as run does. Returns (vocabulary
    object sha256, fingerprint), None with plan. The failures of
    build_tokenizer, and those of the record and the session, as run."""
    with Telemetry.start(store, "wpt dataset build-tokenizer", arguments, name, inputs={"recipe": name},
                         env_file=repo.root / ".env") as telemetry:
        with download.session(store, telemetry) as session:
            built = build_tokenizer(repo, name, store, log=telemetry.log, stream=stream, strict=strict,
                                    telemetry=telemetry, session=session, plan=plan)
        if built is None:
            return None
        _, vocabulary, key = built
        telemetry.log(f"vocabulary {vocabulary} (fingerprint {key})")
    return vocabulary, key


def _left_out_summary(check: dict) -> str:
    """What a source's check result left out, in one line, for the log."""
    counts = {}
    for item in check["left_out"]:
        counts[item["why"]] = counts.get(item["why"], 0) + (item["files"] if item["why"] == "entry" else 1)
    records = sum(count for found in check.get("records", {}).values() for values in found.values()
                  for count in values.values())
    parts = [f"{count:,} files ({why})" for why, count in sorted(counts.items())]
    return ", ".join(parts + ([f"{records:,} records"] if records else [])) + " (see the record's `check`)"


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


def conversations_and_tasks(repo: Repo, store, built: set[str], share: float, held: set[str], telemetry,
                            copies=None) -> tuple[dict, list]:
    """(the parts of a dataset that the training steps read besides its
    shards, the tasks left out). The parts, each only when the sources hold
    some: {"conversations": {object, count, sources}, "tasks": {object,
    count, sources}}, sources the sorted
    ids of the sources the part holds or points into, each described in the
    record's own `sources`. Each object holds
    one JSON line per conversation ({id, tags, conversation}: every chat
    file, whatever its tags) or per task (tasks.to_json: every task, its
    pointers filled in, but the measure-only ones, which share and held
    decide: the recipe's held_out and the sha256 of every held-out
    document's Markdown). Only the sources in built, those the build kept,
    are read, each as kept.py keeps it; one that breaks a rule here stops
    the build. A task that cannot be filled (tasks.fill: it points into a
    source the build did not keep, such as one not downloaded, or its
    passage is not in the stored bytes any more) is left out, each {task,
    why} in the list of the tasks left out.
    Several sources are read at once, each through its own store from
    copies (sources.each_read), and their results put together in the
    sources' order."""
    parts, left_out = {}, []
    with telemetry.span("conversations") as span:
        found, sources = chat_files.tagged(repo, store, None, telemetry, log=lambda *_: None, only=built,
                                           copies=copies)
        if found:
            lines = (json.dumps({"id": cid, "tags": tags, "conversation": conversation}, ensure_ascii=False,
                                sort_keys=True) for cid, tags, conversation in found)
            parts["conversations"] = {"object": store.put_bytes("".join(f"{line}\n" for line in lines).encode()),
                                      "count": len(found), "sources": sorted(sources)}
        span.set(**parts.get("conversations", {"count": 0}))
    with telemetry.span("tasks") as span:
        found, sources = task_files.read(repo, store, None, share, log=lambda *_: None, held=held, only=built,
                                        copies=copies, pointable=built, left_out=left_out, telemetry=telemetry)
        if found:
            lines = (json.dumps(task_files.to_json(task), ensure_ascii=False, sort_keys=True) for task in found)
            parts["tasks"] = {"object": store.put_bytes("".join(f"{line}\n" for line in lines).encode()),
                              "count": len(found), "sources": sorted(sources)}
        span.set(**parts.get("tasks", {"count": 0}), left_out=len(left_out))
    return parts, left_out


def _with_parts(repo: Repo, store, info: dict, parts: dict, telemetry) -> dict:
    """The sources' info with those a dataset's conversations and tasks
    name and no chosen document brought, each as kept.py keeps it."""
    named = {sid for part in parts.values() for sid in part["sources"]} - info.keys()
    with temporary_copies(store)() as here:
        for sid in sorted(named):
            source = repo.source(sid)
            held = kept(source, here, license_check.LicenseTexts(repo.license_texts()), telemetry)
            info = {**info, sid: source_info(source, held.final, here, checked(held, {}))}
    return dict(sorted(info.items()))


def _recorded_contamination(repo, training, held_out, packed, texts, vocabulary, folder: Path, telemetry):
    """contamination() with the training side searched in worker processes (workers.py), as many at once as
    wpt_utils.parallel allows. Each takes a batch of training documents (CONTAMINATION_BATCH_TOKENS tokens), reads their
    tokens through the store's cache folder, decodes them and gives back which held-out runs each holds. The held-out
    runs reach the workers as one file in folder, which each worker maps once."""
    with (telemetry.span("contamination") as span,
          telemetry.progress("contamination", len(training) + len(held_out), "documents") as progress,
          workers.Pool(repo, "contamination batches at once") as pool):
        def search(documents, runs):
            path = folder / "held-out-runs.npy"
            np.save(path, runs)

            def look(batch):
                ids = [d.id for d in batch]
                return pool.call(_runs_found, packed.of(ids), ids, vocabulary, str(path), telemetry=telemetry)

            # no error is an item's result: any stops the check
            for done in each(look, _batches(documents, packed), (), pool.sizing):
                yield from done.value
                progress.advance(len(done.item))

        found = contamination(training, held_out, lambda chosen: _counted(texts(chosen), progress), search)
        span.set(**{k: v for k, v in found.items() if k != "examples"})
    return found


def _batches(documents, tokens):
    """The documents in runs of CONTAMINATION_BATCH_TOKENS tokens at least (the last may hold fewer), in order."""
    batch, size = [], 0
    for document in documents:
        batch.append(document)
        size += tokens.length(document.id)
        if size >= CONTAMINATION_BATCH_TOKENS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


_HELD_OUT_RUNS = {}  # in a worker process: the held-out runs file it mapped, by path


def _runs_found(documents, ids, vocabulary: str, path: str, telemetry) -> list[np.ndarray]:
    """In a worker process: for each of ids (read from documents, a PackedDocuments of just those), the positions
    in the held-out runs (the file at path) of the runs its text holds."""
    if path not in _HELD_OUT_RUNS:
        _HELD_OUT_RUNS.clear()
        _HELD_OUT_RUNS[path] = np.load(path, mmap_mode="r")
    runs = _HELD_OUT_RUNS[path]
    tokenizer = reading.tokenizer_of(vocabulary, documents.store)
    found = []
    for document_id in ids:
        here = runs_of(tokenizer.decode(documents[document_id].tolist()))
        at = np.minimum(np.searchsorted(runs, here), len(runs) - 1)
        found.append(at[runs[at] == here])
    return found


def _counted(items, progress):
    """The items, a progress step for each."""
    for item in items:
        progress.advance()
        yield item


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


def _write_shards(rows, tokens, store, folder: Path, shard_tokens, telemetry, part):
    shards, batch, size = [], [], 0
    progress = telemetry.progress(f"{part} shards", sum(tokens.length(row[0].id) for row in rows), "tokens")

    def flush():
        path = folder / f"shard-{len(shards):05d}.parquet"
        read = tokens.many(list(dict.fromkeys(d.id for d, _, _ in batch)))
        digest = write_shard(path, [(read[d.id], d.source, d.id, pool, p) for d, pool, p in batch])
        shards.append({"object": store.put_file(path), "token_sha256": digest, "rows": len(batch), "tokens": size,
                       "sources": sorted({d.source for d, _, _ in batch})})
        telemetry.event("shard", part=part, index=len(shards) - 1, **shards[-1])
        path.unlink()
        progress.advance(size)

    with progress:
        for row in rows:
            batch.append(row)
            size += tokens.length(row[0].id)
            if size >= shard_tokens:
                flush()
                batch, size = [], 0
        if batch:
            flush()
    return shards


def runs_of(text: bytes) -> np.ndarray:
    """The distinct runs of NGRAM words of a text, each as the 8 bytes of its hash read as a uint64."""
    words = text.split()
    return np.frombuffer(b"".join({hashlib.blake2b(b" ".join(words[i:i + NGRAM]), digest_size=8).digest()
                                   for i in range(len(words) - NGRAM + 1)}), dtype="<u8")


def contamination(training, held_out, texts, search) -> dict:
    """Held-out documents sharing a run of NGRAM words with training text.
    texts(documents) yields their Markdown, in order. search(training, runs)
    gives, for each training document in order, the positions in runs (the
    sorted array of every run the held-out documents hold) of the runs its
    text holds. Each held-out document is paired with the first training
    document, in order, that shares a run with it."""
    pieces = [runs_of(text) for text in texts(held_out)]
    everything = np.concatenate(pieces) if pieces else np.empty(0, dtype="<u8")
    owners = np.concatenate([np.full(len(piece), i, dtype=np.int64) for i, piece in enumerate(pieces)]) \
        if pieces else np.empty(0, dtype=np.int64)
    runs, first = np.unique(everything, return_index=True)  # first: where each run first occurs, in held-out order
    owners = owners[first]
    found = {}
    if len(runs):
        for document, positions in zip(training, search(training, runs)):
            for owner in np.unique(owners[positions]):
                found.setdefault(held_out[owner].id, document.id)
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
