"""Build vocabularies and datasets from recipes.

    documents      every file of every admitted source's final tree; exact
                   duplicates removed (the first by source id, then path, keeps it)
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
"""

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

from wpt_store.shards import write_shard
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_tokenizer import __version__ as TOKENIZER_VERSION

from . import pools
from .errors import LockMismatch, PoolError
from .repo import Repo, dump_json, load_json
from .manifest import SYMLINK
from .sources import VERSION, fingerprint, source_tree, tags_of

DEFAULT_MAX_EPOCHS = 4
SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per shard
NGRAM = 13  # contamination check: runs of 13 words, as in the GPT-3 paper


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    sha256: str
    tags: frozenset[str]


def gather(repo: Repo, store, log=print):
    """(documents, sources info, number of duplicates removed)."""
    documents, info, seen, duplicates = [], {}, set(), 0
    for source in repo.sources():
        if not source.admitted:
            continue
        tree, tree_object = source_tree(source, store)
        info[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                           "checksum": source.data["checksum"], "tree": tree_object,
                           "licenses": source.data["licenses"]}
        for path in sorted(tree, key=str.encode):
            entry = tree[path]
            if entry.mode == SYMLINK:
                continue  # a link is not a document; its target is
            if entry.sha256 in seen:
                duplicates += 1
                continue
            seen.add(entry.sha256)
            documents.append(Document(f"{source.id}/{path}", source.id, entry.sha256, tags_of(source, path)))
    log(f"documents: {len(documents):,} from {len(info)} admitted sources ({duplicates:,} exact duplicates removed)")
    return documents, info, duplicates


def assign(documents, mixture) -> dict[str, list[Document]]:
    """Pool name -> its documents; each document goes to the first matching pool."""
    trees = [(pool["name"], pools.parse(pool["select"])) for pool in mixture["pools"]]
    assigned = {name: [] for name, _ in trees}
    for document in documents:
        for name, tree in trees:
            if pools.matches(tree, document.tags):
                assigned[name].append(document)
                break
    return assigned


def _rank(*parts) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def is_held_out(document_id: str, share: float) -> bool:
    return int.from_bytes(hashlib.sha256(document_id.encode()).digest()) < share * 2**256


# -- tokenizer ------------------------------------------------------------------

def build_tokenizer(repo: Repo, name: str, store, log=print):
    """(tokenizer, vocabulary object sha256, fingerprint). Writes the lock
    file when it is missing or its fingerprint is outdated."""
    recipe = repo.recipe("tokenizer", name)
    mixture = repo.recipe("mixture", recipe["mixture"])
    documents, _, _ = gather(repo, store, log)
    selected = sorted((d for pool in assign(documents, mixture).values() for d in pool), key=lambda d: d.id)
    if not selected:
        raise PoolError(f"tokenizer {name}: mixture {recipe['mixture']} selects no documents")
    key = fingerprint({
        "kind": "vocabulary", "wpt_tokenizer": TOKENIZER_VERSION, "wpt_dataset": VERSION,
        "recipe": recipe, "mixture": mixture, "documents": [[d.id, d.sha256] for d in selected],
    })
    vocabulary = store.get_ref("vocabularies", key)
    if vocabulary is None:
        log(f"training tokenizer {name} on {len(selected):,} documents (vocab_size {recipe['vocab_size']:,})")
        tokenizer = Tokenizer.train((store.get_bytes(d.sha256) for d in selected), recipe["vocab_size"])
        vocabulary = store.put_bytes(tokenizer.to_bytes())
        store.set_ref("vocabularies", key, vocabulary)
    else:
        tokenizer = Tokenizer.from_bytes(store.get_bytes(vocabulary))

    lock_path = repo.lock_path(name)
    lock = load_json(lock_path, "lock") if lock_path.exists() else None
    if lock and lock["fingerprint"] == key:
        if lock["vocabulary_sha256"] != vocabulary:
            raise LockMismatch(f"tokenizer {name}: built vocabulary {vocabulary} differs from the locked "
                               f"{lock['vocabulary_sha256']} for the same inputs (training is not repeatable here)")
    else:
        lock_path.write_text(dump_json({"schema": 1, "fingerprint": key, "vocabulary_sha256": vocabulary}), encoding="utf-8", newline="\n")
        log(f"wrote {lock_path.relative_to(repo.root)}: commit it so every machine checks against it")
    return tokenizer, vocabulary, key


# -- dataset ----------------------------------------------------------------------

def build_dataset(repo: Repo, name: str, store, log=print, shard_tokens=SHARD_TOKENS) -> str:
    """Build (or find) the dataset of a recipe and return its id."""
    recipe = repo.recipe("dataset", name)
    mixture = repo.recipe("mixture", recipe["mixture"])
    tokenizer, vocabulary, tokenizer_key = build_tokenizer(repo, recipe["tokenizer"], store, log)
    documents, sources_info, duplicates = gather(repo, store, log=lambda _: None)
    dataset_id = fingerprint({
        "kind": "dataset", "wpt_dataset": VERSION, "recipe": recipe, "mixture": mixture, "vocabulary": vocabulary,
        "documents": [[d.id, d.sha256, sorted(d.tags)] for d in documents],
        "sources": sources_info,  # the record copies their licenses and attribution: a change rebuilds it
    })
    if store.get_ref("datasets", dataset_id):
        log(f"dataset {dataset_id} is already built")
        return dataset_id

    held_out = [d for d in documents if is_held_out(d.id, recipe["held_out"])]
    held_ids = {d.id for d in held_out}
    training = [d for d in documents if d.id not in held_ids]
    tokens = {}
    for document in documents:
        tokens[document.id] = tokenizer.encode(store.get_bytes(document.sha256)) + [Tokenizer.END]

    rows, pool_report = [], []
    assigned = assign(training, mixture)
    for pool in mixture["pools"]:
        members = assigned[pool["name"]]
        available = sum(len(tokens[d.id]) for d in members)
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
            taken += len(tokens[d.id])
        pool_report.append({"name": pool["name"], "documents": len(members), "tokens": available,
                            "share": share, "epochs": round(epochs, 6)})
        log(f"pool {pool['name']}: {len(members):,} documents, {available:,} tokens, {epochs:.2f} passes for {share:,}")
    rows.sort(key=lambda r: _rank(dataset_id, "order", r[2], r[0].id))

    with tempfile.TemporaryDirectory(prefix="wpt-build-") as tmp:
        shards = _write_shards(rows, tokens, store, Path(tmp), shard_tokens)
        held_shards = _write_shards([(d, "held-out", 0) for d in held_out], tokens, store, Path(tmp), shard_tokens)

    record = {
        "schema": 2,
        "id": dataset_id,
        "recipe_name": name,
        "recipe": recipe,
        "mixture": mixture,
        "tokenizer": {"recipe": recipe["tokenizer"], "fingerprint": tokenizer_key, "vocabulary": vocabulary,
                      "version": TOKENIZER_VERSION, "vocab_size": tokenizer.vocab_size},
        "sources": sources_info,
        "documents": {"total": len(documents), "duplicates_removed": duplicates,
                      "held_out": len(held_out), "training": len(training)},
        "pools": pool_report,
        "total_tokens": sum(s["tokens"] for s in shards),
        "shards": shards,
        "held_out_shards": held_shards,
        "contamination": contamination(training, held_out, store),
        "tools": tool_versions(),
        "tool_differences": tool_differences(repo),
    }
    for difference in record["tool_differences"]:
        log(f"tool version differs from the repository's reference: {difference}")
    if record["contamination"]["overlapping"]:
        log(f"contamination: {record['contamination']['overlapping']} held-out documents share a run of "
            f"{NGRAM} words with training documents (reported only)")
    store.set_ref("datasets", dataset_id, store.put_bytes(json.dumps(record, indent=2).encode() + b"\n"))
    log(f"dataset {dataset_id}: {record['total_tokens']:,} training tokens in {len(shards)} shards")
    return dataset_id


def _write_shards(rows, tokens, store, folder: Path, shard_tokens):
    shards, batch, size = [], [], 0

    def flush():
        path = folder / f"shard-{len(shards):05d}.parquet"
        digest = write_shard(path, [(tokens[d.id], d.source, d.id, pool, p) for d, pool, p in batch])
        shards.append({"object": store.put_file(path), "token_sha256": digest, "rows": len(batch), "tokens": size})
        path.unlink()

    for row in rows:
        batch.append(row)
        size += len(tokens[row[0].id])
        if size >= shard_tokens:
            flush()
            batch, size = [], 0
    if batch:
        flush()
    return shards


def contamination(training, held_out, store) -> dict:
    """Held-out documents sharing a run of NGRAM words with training text."""
    def grams(data: bytes):
        words = data.split()
        return {hashlib.blake2b(b" ".join(words[i:i + NGRAM]), digest_size=8).digest()
                for i in range(len(words) - NGRAM + 1)}

    owner = {}
    for document in held_out:
        for gram in grams(store.get_bytes(document.sha256)):
            owner.setdefault(gram, document.id)
    found = {}
    if owner:
        for document in training:
            for gram in grams(store.get_bytes(document.sha256)):
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
