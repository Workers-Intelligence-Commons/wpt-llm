"""Build vocabularies and datasets from recipes.

    documents      the Markdown of every file of every admitted source's final
                   tree (see markdown.py); exact duplicates removed (the first by
                   source id, then path, keeps it)
    held-out       documents whose id hashes below the recipe's held_out share
    pools          each training document goes to the first pool, in mixture
                   order, whose expression matches its tags
    shares         a pool fills weight x total_tokens by whole passes over its
                   documents, then a partial pass; more passes than max_epochs
                   (default 4) is refused
    shards         Parquet files in the store (see wpt_store.shards), rows in a
                   fixed pseudo-random order derived from the dataset id

The dataset id is the fingerprint of every input, so building the same
recipe on the same data again is a lookup.

A source that fails (its checksum differs, every address fails, a file is
broken) stops the build, unless the build is asked to skip such sources:
the record then lists each one with its error under `skipped`. A missing
program stops it anyway, since it fails every source that needs it. A
vocabulary is never trained without a source: it is frozen for good.

With stream=True the build keeps no source's files: each source is fetched
into a throwaway store, turned into tokens (one store object per source,
see source_tokens) and the throwaway store removed before the next one.
The documents, the dataset id and the shards are the same as without it;
only what the store keeps differs.

With a telemetry (wpt_telemetry), the build records each source (fetched,
converted, its documents, or why it was skipped), the vocabulary, each
pool, each shard written, the contamination check and the dataset record.
"""

import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wpt_store import CorruptObject, MissingObject, open_store
from wpt_store.shards import write_shard
from wpt_telemetry import Telemetry, set_ref
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_tokenizer import __version__ as TOKENIZER_VERSION

from . import manifest, markdown, pools
from .errors import DatasetError, LockedDocumentsMissing, LockMismatch, PoolError, TokenizerVersionNeeded
from .repo import Repo, dump_json, load_json
from .sources import VERSION, check_license_map, fingerprint, source_tree, tags_of, tree_key

DEFAULT_MAX_EPOCHS = 4
SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per shard
NGRAM = 13  # contamination check: runs of 13 words, as in the GPT-3 paper
STREAM_READERS = 16  # documents read at once from a streamed build's token objects


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    sha256: str
    tags: frozenset[str]


def gather(repo: Repo, store, log=print, skipped=None, telemetry=None):
    """(documents, sources info, number of duplicates removed). A source with
    records files gets `credits` in its info: the store object listing, one
    JSON line per document it gives, the document id, its license and the
    credit fields of its record. With a list as skipped, a failing source is
    left out and added to it (see _skip); otherwise its error stops."""
    telemetry = telemetry or Telemetry.off()
    documents, info, seen, duplicates = [], {}, set(), 0
    for source in repo.sources():
        if not source.admitted:
            continue
        with telemetry.span("source", source=source.id, checksum=source.data["checksum"]) as span:
            try:
                tree, tree_object = source_tree(source, store, telemetry)
                found, mine, credits, repeated = [], set(), [], 0  # kept apart until the source succeeds
                for converted in markdown.documents(source, tree, store):  # links are not documents; their targets are
                    sha256 = store.put_bytes(converted.markdown)
                    if sha256 in seen or sha256 in mine:
                        repeated += 1
                        continue
                    mine.add(sha256)
                    document_id = f"{source.id}/{converted.path}"
                    found.append(Document(document_id, source.id, sha256, tags_of(source, converted.file)))
                    if converted.record:
                        credits.append(json.dumps({"document": document_id, "license": converted.record.license,
                                                   "credit": converted.record.credit}, ensure_ascii=False,
                                                  sort_keys=True))
            except (DatasetError, CorruptObject, MissingObject) as error:
                _skip(source, error, skipped, log)
                span.set(skipped=f"{type(error).__name__}: {error}")
                continue
            info[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                               "checksum": source.data["checksum"], "tree": tree_object,
                               "licenses": source.data["licenses"]}
            documents += found
            seen |= mine
            duplicates += repeated
            if credits:
                info[source.id]["credits"] = store.put_bytes("".join(f"{line}\n" for line in credits).encode())
            span.set(tree=tree_object, documents=len(found), duplicates=repeated, credits=info[source.id].get("credits"))
    log(f"documents: {len(documents):,} from {len(info)} admitted sources ({duplicates:,} exact duplicates removed)")
    return documents, info, duplicates


def _skip(source, error, skipped, log):
    """Record a source that failed, or raise its error when nothing asked to
    skip. A missing program (wpt_utils.programs.MissingTool), which would fail every
    source using it, is not caught here and always stops the build."""
    if skipped is None:
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

def _fetched(source, folder: Path):
    """(throwaway store, final tree, manifest object) of a source fetched
    into a new store in folder; removing folder removes every file of it."""
    scratch = open_store(folder.as_uri())
    tree, tree_object = source_tree(source, scratch)
    return scratch, tree, tree_object


def streamed_documents(repo: Repo, telemetry=None):
    """(document, Markdown) of every admitted source, exact duplicates left
    out as in gather, each source fetched into a throwaway store that is
    removed before the next one."""
    telemetry = telemetry or Telemetry.off()
    seen = set()
    for source in repo.sources():
        if not source.admitted:
            continue
        with tempfile.TemporaryDirectory(prefix="wpt-stream-") as tmp:
            scratch, tree, tree_object = _fetched(source, Path(tmp) / "store")
            count = 0
            for converted in markdown.documents(source, tree, scratch):
                sha256 = hashlib.sha256(converted.markdown).hexdigest()
                if sha256 in seen:
                    continue
                seen.add(sha256)
                count += 1
                document = Document(f"{source.id}/{converted.path}", source.id, sha256, tags_of(source, converted.file))
                yield document, converted.markdown
            # an event, not a span: the reader of this generator runs between its documents
            telemetry.event("source.streamed", source=source.id, checksum=source.data["checksum"], tree=tree_object,
                            documents=count)


def source_tokens(source, store, tokenizer, vocabulary: str, telemetry=None) -> dict:
    """A source as tokens, its files not kept: {"tree": its final manifest
    object, "tokens": one object holding every document's tokens (each
    ending with END) one after the other, "documents": [{path, file, sha256
    of its Markdown, start (byte offset), tokens, token_sha256, record}]}.

    Kept under a ref named after what decides it (the final tree, formats,
    licenses, wpt-dataset version, vocabulary), so a source that did not
    change is never fetched again. With a telemetry, the ref is recorded
    before it is set (wpt_telemetry.set_ref)."""
    key = fingerprint({"kind": "source-tokens", "tree": tree_key(source), "formats": source.data["formats"],
                       "licenses": [[e["license"], e["paths"], e.get("values")] for e in source.data["licenses"]],
                       "wpt_dataset": VERSION, "vocabulary": vocabulary})
    known = store.get_ref("source-tokens", key)
    if known:
        return json.loads(store.get_bytes(known))
    with tempfile.TemporaryDirectory(prefix="wpt-stream-") as tmp:
        tmp = Path(tmp)
        scratch, tree, tree_object = _fetched(source, tmp / "store")
        documents, offset = [], 0
        with open(tmp / "tokens.u32", "wb") as out:
            for converted in markdown.documents(source, tree, scratch):
                data = np.asarray(tokenizer.encode(converted.markdown) + [Tokenizer.END], dtype="<u4").tobytes()
                out.write(data)
                entry = {"path": converted.path, "file": converted.file,
                         "sha256": hashlib.sha256(converted.markdown).hexdigest(),
                         "start": offset, "tokens": len(data) // 4, "token_sha256": hashlib.sha256(data).hexdigest()}
                if converted.record:
                    entry["record"] = {"license": converted.record.license, "credit": converted.record.credit}
                documents.append(entry)
                offset += len(data)
        summary = {"tree": store.put_bytes(scratch.get_bytes(tree_object)), "tokens": store.put_file(tmp / "tokens.u32"),
                   "documents": documents}
    known = store.put_bytes(json.dumps(summary, ensure_ascii=False, sort_keys=True).encode())
    set_ref(store, "source-tokens", key, known, telemetry or Telemetry.off())
    return summary


def gather_streamed(repo: Repo, store, tokenizer, vocabulary: str, log=print, skipped=None, telemetry=None):
    """gather's (documents, sources info, duplicates), plus every document's
    tokens (PackedDocuments), from source_tokens: no source's files stay in
    the store. skipped: as in gather."""
    telemetry = telemetry or Telemetry.off()
    documents, info, seen, duplicates = [], {}, set(), 0
    tokens = PackedDocuments(store)
    for source in repo.sources():
        if not source.admitted:
            continue
        with telemetry.span("source", source=source.id, checksum=source.data["checksum"]) as span:
            try:
                summary = source_tokens(source, store, tokenizer, vocabulary, telemetry)
                check_license_map(source, manifest.parse_manifest(store.get_bytes(summary["tree"])))
            except (DatasetError, CorruptObject, MissingObject) as error:
                _skip(source, error, skipped, log)
                span.set(skipped=f"{type(error).__name__}: {error}")
                continue
            info[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                               "checksum": source.data["checksum"], "tree": summary["tree"],
                               "licenses": source.data["licenses"]}
            credits, before = [], duplicates
            for entry in summary["documents"]:
                if entry["sha256"] in seen:
                    duplicates += 1
                    continue
                seen.add(entry["sha256"])
                document_id = f"{source.id}/{entry['path']}"
                documents.append(Document(document_id, source.id, entry["sha256"], tags_of(source, entry["file"])))
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


def is_held_out(document_id: str, share: float) -> bool:
    return int.from_bytes(hashlib.sha256(document_id.encode()).digest()) < share * 2**256


# -- tokenizer ------------------------------------------------------------------

def build_tokenizer(repo: Repo, name: str, store, log=print, stream=False, telemetry=None):
    """(tokenizer, vocabulary object sha256, fingerprint) of a tokenizer recipe.

    The lock file freezes one vocabulary per recipe version, so adding or
    changing sources never retrains it:

    - no lock, or the recipe's `version` was raised: train on the documents
      the mixture selects now, and write the lock (replacing the old one);
    - the same version, but the recipe (other than its note) or the
      tokenizer code changed: TokenizerVersionNeeded;
    - otherwise the locked vocabulary, from the store, or rebuilt from the
      documents the lock lists (LockedDocumentsMissing when some are not in
      the store, LockMismatch when the result differs).

    With stream, training reads each source from a throwaway store
    (streamed_documents) instead of keeping the documents in the store; the
    vocabulary is the same, since training counts words and its result
    does not depend on the order of the documents.
    """
    telemetry = telemetry or Telemetry.off()
    recipe = repo.recipe("tokenizer", name)
    lock_path = repo.lock_path(name)
    lock = load_json(lock_path, "lock") if lock_path.exists() else None

    locked = lock is not None and lock["recipe"]["version"] == recipe["version"]
    if locked:
        if _training_part(lock["recipe"]) != _training_part(recipe) or lock["wpt_tokenizer"] != TOKENIZER_VERSION:
            changed = "the recipe" if _training_part(lock["recipe"]) != _training_part(recipe) else f"the tokenizer code ({lock['wpt_tokenizer']} -> {TOKENIZER_VERSION})"
            raise TokenizerVersionNeeded(f"tokenizer {name}: {changed} changed since version {recipe['version']} was "
                                         f"locked; raise `version` in {repo.recipe_path('tokenizer', name).name}")
        trained = lock["documents"]
    elif stream:
        trees = _pool_trees(repo.recipe("mixture", recipe["mixture"]))
        trained = []

        def selected_texts():
            for document, text in streamed_documents(repo, telemetry):
                if _pool_of(document, trees) is not None:
                    trained.append([document.id, document.sha256])
                    yield text

        log(f"training tokenizer {name} version {recipe['version']} on the documents its mixture selects, "
            f"one source at a time (vocab_size {recipe['vocab_size']:,})")
        streamed = Tokenizer.train(selected_texts(), recipe["vocab_size"])
        if not trained:
            raise PoolError(f"tokenizer {name}: mixture {recipe['mixture']} selects no documents")
        trained.sort()
    else:
        mixture = repo.recipe("mixture", recipe["mixture"])
        documents, _, _ = gather(repo, store, log, telemetry=telemetry)
        selected = sorted((d for pool in assign(documents, mixture).values() for d in pool), key=lambda d: d.id)
        if not selected:
            raise PoolError(f"tokenizer {name}: mixture {recipe['mixture']} selects no documents")
        trained = [[d.id, d.sha256] for d in selected]

    key = fingerprint({"kind": "vocabulary", "wpt_tokenizer": TOKENIZER_VERSION, "recipe": recipe, "documents": trained})
    vocabulary = lock["vocabulary_sha256"] if locked else store.get_ref("vocabularies", key)
    if vocabulary is not None and store.has(vocabulary):
        tokenizer = Tokenizer.from_bytes(store.get_bytes(vocabulary))
    elif stream:
        if locked:
            wanted, found = {tuple(d) for d in trained}, set()

            def locked_texts():
                for document, text in streamed_documents(repo, telemetry):
                    if (document.id, document.sha256) in wanted:
                        found.add((document.id, document.sha256))
                        yield text

            log(f"training tokenizer {name} version {recipe['version']} again on the {len(trained):,} documents "
                f"its lock lists, one source at a time")
            streamed = Tokenizer.train(locked_texts(), recipe["vocab_size"])
            missing = sorted(wanted - found)
            if missing:
                raise LockedDocumentsMissing(f"tokenizer {name} version {recipe['version']}: {len(missing):,} of the "
                                             f"documents it was trained on are not in the admitted sources any "
                                             f"more, such as {missing[0][0]}")
        tokenizer = streamed
        built = store.put_bytes(tokenizer.to_bytes())
        if vocabulary is not None and built != vocabulary:
            raise LockMismatch(f"tokenizer {name}: built vocabulary {built} differs from the locked {vocabulary} "
                               "for the same documents (training is not repeatable here)")
        vocabulary = built
        set_ref(store, "vocabularies", key, vocabulary, telemetry)
    else:
        if locked:
            gather(repo, store, log, telemetry=telemetry)  # puts the current documents in the store
        missing = [doc_id for doc_id, sha256 in trained if not store.has(sha256)]
        if missing:
            raise LockedDocumentsMissing(f"tokenizer {name} version {recipe['version']}: its vocabulary is not in "
                                         f"{store}, and {len(missing):,} of the documents it was trained on are not "
                                         f"either, such as {missing[0]}")
        log(f"training tokenizer {name} version {recipe['version']} on {len(trained):,} documents "
            f"(vocab_size {recipe['vocab_size']:,})")
        tokenizer = Tokenizer.train((store.get_bytes(sha256) for _, sha256 in trained), recipe["vocab_size"])
        built = store.put_bytes(tokenizer.to_bytes())
        if vocabulary is not None and built != vocabulary:
            raise LockMismatch(f"tokenizer {name}: built vocabulary {built} differs from the locked {vocabulary} "
                               "for the same documents (training is not repeatable here)")
        vocabulary = built
        set_ref(store, "vocabularies", key, vocabulary, telemetry)

    if not locked:
        # recorded, and in the store, before the lock changes: a record that cannot be written leaves it as it was
        telemetry.irreversible("tokenizer.lock", path=lock_path.relative_to(repo.root).as_posix(),
                               vocabulary=vocabulary)
        lock_path.write_text(dump_json({"schema": 2, "recipe": recipe, "wpt_tokenizer": TOKENIZER_VERSION,
                                        "documents": trained, "vocabulary_sha256": vocabulary}),
                             encoding="utf-8", newline="\n")
        log(f"wrote {lock_path.relative_to(repo.root)}: commit it so every machine uses this vocabulary")
    return tokenizer, vocabulary, key


def _training_part(recipe) -> dict:
    """What trains the vocabulary: the recipe without its shape number and note."""
    return {k: v for k, v in recipe.items() if k not in ("schema", "note")}


# -- dataset ----------------------------------------------------------------------

def build_dataset(repo: Repo, name: str, store, log=print, shard_tokens=SHARD_TOKENS, stream=False,
                  skip_failed=False, telemetry=None) -> str:
    """Build (or find) the dataset of a model recipe's data part and return
    its id; with stream, keeping no source's files; with skip_failed,
    leaving out the sources that fail (see the module's docstring). Only
    the data part counts: recipes with the same data share one dataset."""
    telemetry = telemetry or Telemetry.off()
    recipe = repo.recipe("model", name)["data"]
    mixture = repo.recipe("mixture", recipe["mixture"])
    telemetry.event("dataset.recipe", recipe_name=name, recipe=recipe, mixture=mixture)
    with telemetry.span("tokenizer", recipe=recipe["tokenizer"], stream=stream) as span:
        tokenizer, vocabulary, tokenizer_key = build_tokenizer(repo, recipe["tokenizer"], store, log, stream=stream,
                                                               telemetry=telemetry)
        span.set(vocabulary=vocabulary, fingerprint=tokenizer_key, vocab_size=tokenizer.vocab_size)
    telemetry.input(recipe=name, tokenizer=recipe["tokenizer"], vocabulary=vocabulary)
    skipped = [] if skip_failed else None
    with telemetry.span("documents", stream=stream, skip_failed=skip_failed) as span:
        if stream:
            documents, sources_info, duplicates, packed = gather_streamed(repo, store, tokenizer, vocabulary,
                                                                           log=lambda _: None, skipped=skipped,
                                                                           telemetry=telemetry)
        else:
            documents, sources_info, duplicates = gather(repo, store, log=lambda _: None, skipped=skipped,
                                                         telemetry=telemetry)
        span.set(documents=len(documents), sources=len(sources_info), duplicates=duplicates, skipped=skipped or [])
    telemetry.input(sources={sid: info["checksum"] for sid, info in sorted(sources_info.items())},
                    skipped_sources=[failed["source"] for failed in skipped or []])
    for failed in skipped or []:
        log(f"skipped {failed['source']}: {failed['error']}")
    dataset_id = fingerprint({
        "kind": "dataset", "wpt_dataset": VERSION, "recipe": recipe, "mixture": mixture, "vocabulary": vocabulary,
        "documents": [[d.id, d.sha256, sorted(d.tags)] for d in documents],
        "sources": sources_info,  # the record copies their licenses and attribution: a change rebuilds it
        **({"skipped": skipped} if skipped else {}),  # absent when none, so earlier ids stay the same
    })
    if store.get_ref("datasets", dataset_id):
        telemetry.output(dataset=dataset_id, dataset_record=store.get_ref("datasets", dataset_id))
        log(f"dataset {dataset_id} is already built")
        return dataset_id

    held_out = [d for d in documents if is_held_out(d.id, recipe["held_out"])]
    held_ids = {d.id for d in held_out}
    training = [d for d in documents if d.id not in held_ids]
    with tempfile.TemporaryDirectory(prefix="wpt-build-") as tmp:
        with telemetry.span("tokens", documents=len(documents)):
            tokens = packed if stream else EncodedDocuments(Path(tmp) / "tokens", tokenizer, documents, store)
        with telemetry.span("pools"):
            rows, pool_report = _fill_pools(dataset_id, recipe, mixture, training, tokens, log)
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
        "schema": 3,
        "id": dataset_id,
        "recipe_name": name,
        "recipe": recipe,
        "mixture": mixture,
        "tokenizer": {"recipe": recipe["tokenizer"], "recipe_version": repo.recipe("tokenizer", recipe["tokenizer"])["version"],
                      "fingerprint": tokenizer_key, "vocabulary": vocabulary,
                      "version": TOKENIZER_VERSION, "vocab_size": tokenizer.vocab_size},
        "sources": sources_info,
        "skipped": skipped or [],
        "documents": {"total": len(documents), "duplicates_removed": duplicates,
                      "held_out": len(held_out), "training": len(training)},
        "pools": pool_report,
        "total_tokens": sum(s["tokens"] for s in shards),
        "shards": shards,
        "held_out_shards": held_shards,
        "contamination": _recorded_contamination(training, held_out, texts, telemetry),
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
        set_ref(store, "datasets", dataset_id, record_object, telemetry)
    log(f"dataset {dataset_id}: {record['total_tokens']:,} training tokens in {len(shards)} shards")
    return dataset_id


def _recorded_contamination(training, held_out, texts, telemetry):
    with telemetry.span("contamination") as span:
        found = contamination(training, held_out, texts)
        span.set(**{k: v for k, v in found.items() if k != "examples"})
    return found


def _fill_pools(dataset_id, recipe, mixture, training, tokens, log):
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
        for d in sorted(members, key=lambda d: _rank(dataset_id, pool["name"], full, d.id)):
            if taken >= remainder:
                break
            rows.append((d, pool["name"], full))
            taken += tokens.length(d.id)
        pool_report.append({"name": pool["name"], "documents": len(members), "tokens": available,
                            "share": share, "epochs": round(epochs, 6)})
        log(f"pool {pool['name']}: {len(members):,} documents, {available:,} tokens, {epochs:.2f} passes for {share:,}")
    rows.sort(key=lambda r: _rank(dataset_id, "order", r[2], r[0].id))
    return rows, pool_report


class EncodedDocuments:
    """Every document's tokens (ending with END), encoded once into its own
    file in folder; only their lengths stay in memory."""

    def __init__(self, folder: Path, tokenizer, documents, store):
        folder.mkdir()
        self.folder, self._lengths = folder, {}
        for i, document in enumerate(documents):
            tokens = np.asarray(tokenizer.encode(store.get_bytes(document.sha256)) + [Tokenizer.END], dtype="<u4")
            tokens.tofile(folder / f"{i}.u32")
            self._lengths[document.id] = (i, len(tokens))

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
    versions = {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION}
    for package in ("wpt-dataset", "wpt-store", "wpt-tokenizer", "pyarrow", "numpy", "jsonschema"):
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
