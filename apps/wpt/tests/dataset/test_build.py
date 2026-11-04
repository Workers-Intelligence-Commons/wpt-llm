import dataclasses
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import re
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from wpt_dataset import (build, choice, conversations, kept, license_check, markdown, reading, records, sources, tasks,
                         trained, workers)
from wpt_dataset.build import build_dataset, document_list, is_held_out
from wpt.dataset import main
from wpt_dataset.errors import DatasetError, NotDownloaded, PoolError, SourceGone, TokenizerVersionNeeded, TrainedListMissing, VocabularyMissing
from wpt_dataset.sources import stored_files, upstream_key
from wpt_store import CorruptObject, Store, open_store
from wpt_store.sessions import file_list_bytes
from wpt_store.shards import read_shard, token_hash
import wpt_telemetry
from wpt_telemetry import Telemetry
from wpt_tokenizer import Tokenizer
from wpt_utils import programs
from wpt_utils.parallel import Sizing

FRENCH = ["le chat dort sur le tapis rouge", "la pluie tombe sur la ville grise", "un enfant lit un livre ancien",
          "le vent souffle dans les arbres", "une femme chante dans la rue"]
ENGLISH = ["the cat sleeps on the red mat", "rain falls on the grey town", "a child reads an old book",
           "the wind blows in the trees", "a woman sings in the street"]


@pytest.fixture
def corpus(kit, tmp_path):
    """A git project: fr/ and en/ texts, fr/lit/ also tagged literature."""
    folder = tmp_path / "corpus"
    files = {}
    for i, line in enumerate(FRENCH):
        files[f"fr/{'lit/' if i < 2 else ''}text{i}.txt"] = (f"{line}. " * 20).encode()
    for i, line in enumerate(ENGLISH):
        files[f"en/text{i}.txt"] = (f"{line}. " * 20).encode()
    return kit.git_project(folder, files=files, executable=set(), symlinks={})


def setup_recipes(kit, repo, store, corpus, total_tokens, held_out=0.0):
    """The corpus as a source, downloaded into store, and recipes to build it; returns its id."""
    sid = kit.add_source(repo, store, "example", "corpus", {
        "type": "git", "address": corpus[0], # the held-out hash reads the document's path, not its id: the id holds a hash of the address, which
        # holds the temporary folder's name, so the same test would hold other documents out in every run
        "held_out_by": {"pattern": r"(fr|en)/(?:lit/)?(text\d)"},
        "tags": [{"tags": ["fr"], "paths": ["fr/**"]}, {"tags": ["literature"], "paths": ["fr/lit/**"]},
                 {"tags": ["en"], "paths": ["en/**"]}]})
    kit.write_recipe(repo, "mixture", "main", {"pools": [
        {"name": "literature", "select": "literature", "weight": 0.25},
        {"name": "fr", "select": "fr && !(en)", "weight": 0.25},  # also matches fr/lit/: first pool wins
        {"name": "en", "select": "en", "weight": 0.5}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": 10**9, "vocab_size": 656})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main",
                                                "total_tokens": total_tokens, "held_out": held_out})
    return sid


def listed_documents(store, dataset_id) -> bytes:
    """The list of trained documents the lock must name when every document of the dataset trained the
    vocabulary: one JSON `[id, sha256]` line each, sorted, built from the dataset's own document list."""
    return b"".join(sorted((json.dumps([d["id"], d["sha256"]], ensure_ascii=False) + "\n").encode()
                          for d in document_list(store, dataset_id)))


def fresh_store(kit, repo, folder, sid, shared=None):
    """A new store (another machine) with the source downloaded into it,
    and the locked vocabularies of shared, with the lists of their documents,
    copied into it: a locked vocabulary reaches another machine through the
    store, never by training again."""
    fresh = open_store(folder.as_uri())
    kit.download(repo, fresh, sid)
    for name in repo.names("lock") if shared is not None else ():
        lock = repo.generated("lock", name)
        for locked in (lock["vocabulary_sha256"], lock["documents_sha256"]):
            fresh.put_bytes(shared.get_bytes(locked))
    return fresh


def record_of(store, dataset_id):
    return json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))


def rows_of(store, shards):
    return [row for shard in shards for row in read_shard(store.path(shard["object"]))]


def test_build_fills_pools_and_is_repeatable_on_another_machine(kit, repo, store, corpus, tmp_path):
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    dataset_id = build_dataset(repo, "small", store, log=lambda _: None)
    record = record_of(store, dataset_id)

    pools = {p["name"]: p for p in record["pools"]}
    assert pools["literature"]["documents"] == 2 and pools["fr"]["documents"] == 3 and pools["en"]["documents"] == 5
    rows = rows_of(store, record["shards"])
    assert {pool for _, _, doc, pool, _ in rows if "/fr/lit/" in doc} == {"literature"}  # first pool won
    for name, pool in pools.items():
        drawn = sum(len(tokens) for tokens, *_, p, _ in rows if p == name)
        assert pool["share"] <= drawn < pool["share"] + max(len(t) for t, *_ in rows)  # whole documents
        assert pool["epochs"] <= 4
    for shard in record["shards"]:
        assert token_hash(store.path(shard["object"])) == shard["token_sha256"]
    vocabulary = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))
    tokens, _, document, *_ = rows[0]
    assert tokens[-1] == Tokenizer.END
    original = kit.run("git", "show", f"{corpus[1]}:{document.split('/', 1)[1]}", cwd=corpus[0])
    # training reads the file's Markdown: this one-line text, without its last space, then a newline
    assert vocabulary.decode(tokens[:-1]) == original.strip() + b"\n"

    # same inputs: a lookup, not a rebuild
    assert build_dataset(repo, "small", store, log=lambda _: None) == dataset_id
    # a fresh store (another machine): the same id, vocabulary and tokens
    fresh = fresh_store(kit, repo, tmp_path / "other-store", sid, shared=store)
    assert build_dataset(repo, "small", fresh, log=lambda _: None) == dataset_id
    again = record_of(fresh, dataset_id)
    assert again["tokenizer"]["vocabulary"] == record["tokenizer"]["vocabulary"]
    assert [s["token_sha256"] for s in again["shards"]] == [s["token_sha256"] for s in record["shards"]]
    # each record names the download its files came from, which the dataset id leaves out
    for built, record_ in ((store, record), (fresh, again)):
        with built.database() as found:
            downloaded = found.execute("SELECT session, time FROM lines WHERE source = ? AND outcome = 'downloaded'",
                                       (sid,)).fetchall()
        assert [(record_["downloads"][sid]["session"], record_["downloads"][sid]["time"])] == downloaded
    assert again["downloads"][sid]["session"] != record["downloads"][sid]["session"]


def test_held_out_documents_never_reach_training(kit, repo, store, tmp_path):
    # ids contain the (random) temporary path, so which documents are held out
    # varies between runs: 40 documents at 50% make an empty side vanishingly rare
    folder, commit = kit.git_project(tmp_path / "many", files={
        f"t{i:02d}.txt": f"document number {i} says hello. ".encode() * 5 for i in range(40)}, executable=set(), symlinks={})
    kit.add_source(repo, store, "example", "many", {"type": "git", "address": folder, "tags": [{"tags": ["all"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "all", {"pools": [{"name": "all", "select": "all", "weight": 1, "max_epochs": 100}]})
    kit.write_recipe(repo, "tokenizer", "all", {"version": 1, "sample_bytes": 10**9, "vocab_size": 556})
    kit.write_model_recipe(repo, "split", {"mixture": "all", "tokenizer": "all",
                                                "total_tokens": 3000, "held_out": 0.5})

    record = record_of(store, build_dataset(repo, "split", store, log=lambda _: None))
    held = {doc for _, _, doc, _, _ in rows_of(store, record["held_out_shards"])}
    trained = {doc for _, _, doc, _, _ in rows_of(store, record["shards"])}
    assert held and trained and not held & trained
    assert record["documents"]["held_out"] == len(held)
    assert record["contamination"]["held_out_documents"] == len(held)

    # the build keeps, found from the
    # dataset id, every document's id, Markdown sha256 and tag, so a task
    # reader learns the held-out texts without converting any source
    dataset_id = record["id"]
    listed = document_list(store, dataset_id)
    assert {entry["id"] for entry in listed if entry["tag"] == "held-out"} == held
    training = {entry["id"] for entry in listed if entry["tag"] == "training"}
    assert len(training) == record["documents"]["training"] and trained <= training  # the shards draw some
    vocabulary = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))
    markdown = {doc: vocabulary.decode(tokens[:-1].tolist())
                for tokens, _, doc, _, _ in rows_of(store, record["shards"] + record["held_out_shards"])}
    for entry in listed:  # the sha256 of the document's Markdown, the text training reads
        if entry["id"] in markdown:
            assert hashlib.sha256(markdown[entry["id"]]).hexdigest() == entry["sha256"]


def test_documents_with_the_same_held_out_group_share_their_fate_in_every_source(kit, repo, store, tmp_path):
    # eBible's chapters are held out by
    # book and chapter, so the same chapter is held out in every translation
    # at once; a source's held_out_by pattern gives the group its hash reads
    pattern = {"pattern": r"_([A-Z0-9]{3})_0*([1-9][0-9]*)_read\.txt$"}
    sources = []
    for code, digits in (("aaa", 2), ("bbb", 3)):  # chapter numbers written with 2 and 3 digits
        files = {f"{code}_002_GEN_{n:0{digits}d}_read.txt": f"{code} genesis chapter {n} text. ".encode() * 5
                 for n in range(1, 21)}
        files["notes.txt"] = f"notes of {code}. ".encode() * 5
        folder, commit = kit.git_project(tmp_path / code, files=files, executable=set(), symlinks={})
        sources.append(kit.add_source(repo, store, "example", code, {
            "type": "git", "address": folder, "held_out_by": pattern,
            "tags": [{"tags": ["all"], "paths": ["**"]}]}))
    kit.write_recipe(repo, "mixture", "all", {"pools": [{"name": "all", "select": "all", "weight": 1, "max_epochs": 100}]})
    kit.write_recipe(repo, "tokenizer", "all", {"version": 1, "sample_bytes": 10**9, "vocab_size": 556})
    kit.write_model_recipe(repo, "split", {"mixture": "all", "tokenizer": "all", "total_tokens": 6000,
                                           "held_out": 0.5})

    dataset_id = build_dataset(repo, "split", store, log=lambda _: None)
    record = record_of(store, dataset_id)
    held = {doc for _, _, doc, _, _ in rows_of(store, record["held_out_shards"])}
    for n in range(1, 21):  # a chapter's fate is its group's, "GEN <n>", in both sources
        chapter = {f"{sources[0]}/aaa_002_GEN_{n:02d}_read.txt", f"{sources[1]}/bbb_002_GEN_{n:03d}_read.txt"}
        assert (chapter <= held) if is_held_out(f"GEN {n}", 0.5) else not chapter & held
    assert 0 < sum(is_held_out(f"GEN {n}", 0.5) for n in range(1, 21)) < 20  # both fates occur
    for sid in sources:  # a path the pattern is not found in keeps its id's fate
        assert (f"{sid}/notes.txt" in held) == is_held_out(f"{sid}/notes.txt", 0.5)
    listed = document_list(store, dataset_id)  # the document list says the same
    assert {entry["id"] for entry in listed if entry["tag"] == "held-out"} == held

    # the groups are part of the dataset id: the same documents without them are another dataset
    for sid in sources:
        path = repo.root / "data" / "sources" / sid / "source.json"
        data = json.loads(path.read_text())
        del data["held_out_by"]
        path.write_text(json.dumps(data, indent=2) + "\n")
    assert build_dataset(repo, "split", store, log=lambda _: None) != dataset_id


def test_too_many_passes_over_a_pool_is_refused(kit, repo, store, corpus):
    setup_recipes(kit, repo, store, corpus, total_tokens=10_000_000)
    with pytest.raises(PoolError, match="more than the 4 allowed"):
        build_dataset(repo, "small", store, log=lambda _: None)


def test_the_locked_vocabulary_stays_until_its_version_is_raised(kit, repo, store, corpus, tmp_path):
    # many datasets and models share one vocabulary, so new sources must not change it;
    # only a person raising the recipe's version trains a new one
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json")
    lock = json.loads(lock_path.read_text())
    assert lock["vocabulary_sha256"] == first["tokenizer"]["vocabulary"] and lock["document_count"] == 10
    assert "documents" not in lock
    assert store.get_bytes(lock["documents_sha256"]) == listed_documents(store, first["id"])
    assert first["tokenizer"]["recipe_version"] == 1

    folder, commit = kit.git_project(tmp_path / "more", files={"en/new.txt": b"zebras graze quietly. " * 30},
                                     executable=set(), symlinks={})
    more = kit.add_source(repo, store, "example", "more", {"type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]})
    second = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert second["documents"]["total"] == 11  # the new source is in the dataset...
    assert second["tokenizer"]["vocabulary"] == first["tokenizer"]["vocabulary"]  # ...but not in the vocabulary
    assert json.loads(lock_path.read_text()) == lock
    # the locked vocabulary is shared through the store, never trained again
    # from the documents the lock lists: a store without it refuses the build,
    # and the same store given the vocabulary builds with it
    fresh = fresh_store(kit, repo, tmp_path / "fresh", sid)
    kit.download(repo, fresh, more)
    with pytest.raises(VocabularyMissing, match=lock["vocabulary_sha256"]):
        build_dataset(repo, "small", fresh, log=lambda _: None)
    assert fresh.put_bytes(store.get_bytes(lock["vocabulary_sha256"])) == lock["vocabulary_sha256"]
    with pytest.raises(TrainedListMissing, match=lock["documents_sha256"]):  # the list of its documents is shared too
        build_dataset(repo, "small", fresh, log=lambda _: None)
    assert fresh.put_bytes(store.get_bytes(lock["documents_sha256"])) == lock["documents_sha256"]
    assert record_of(fresh, build_dataset(repo, "small", fresh, log=lambda _: None))["tokenizer"]["vocabulary"] \
        == lock["vocabulary_sha256"]

    recipe_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.json")
    recipe = json.loads(recipe_path.read_text())
    recipe_path.write_text(json.dumps({**recipe, "note": "a note changes nothing"}))
    build_dataset(repo, "small", store, log=lambda _: None)
    recipe_path.write_text(json.dumps({**recipe, "vocab_size": 706}))
    with pytest.raises(TokenizerVersionNeeded, match="the recipe changed since version 1"):
        build_dataset(repo, "small", store, log=lambda _: None)

    recipe_path.write_text(json.dumps({**recipe, "vocab_size": 706, "version": 2}))
    third = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    new_lock = json.loads(lock_path.read_text())
    assert third["tokenizer"]["recipe_version"] == 2 and third["tokenizer"]["vocabulary"] == new_lock["vocabulary_sha256"]
    assert new_lock["vocabulary_sha256"] != lock["vocabulary_sha256"] and new_lock["document_count"] == 11


def test_a_change_of_the_tokenizer_code_asks_for_no_version_and_leaves_the_lock_as_it_is(kit, repo, store, corpus,
                                                                                         monkeypatch):
    # the lock holds no version of the tokenizer code; the code's hash is in the fingerprints, so changed code
    # builds the locked vocabulary again, and only a change of the recipe asks a person for a version
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    first = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock_path = repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json"
    locked = lock_path.read_bytes()
    assert set(json.loads(locked)) == {"recipe", "documents_sha256", "document_count", "vocabulary_sha256"}
    assert "version" not in first["tokenizer"] and "recipe_version" in first["tokenizer"]

    monkeypatch.setattr(build, "code_of", lambda module: "another tokenizer code")
    second = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert second["tokenizer"]["vocabulary"] == first["tokenizer"]["vocabulary"] and lock_path.read_bytes() == locked
    assert second["fingerprints"] != first["fingerprints"]  # the code is part of what made the tokens


def test_tokenizer_check_round_trips_every_document_and_reports_each_source(kit, repo, store, corpus, tmp_path,
                                                                           monkeypatch, capsys):
    # shows whether text the vocabulary never saw still encodes compactly enough
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    build_dataset(repo, "small", store, log=lambda _: None)
    folder, commit = kit.git_project(tmp_path / "more", files={"new.txt": "zèbres ✓ broutent. ".encode() * 30},
                                     executable=set(), symlinks={})
    late = kit.add_source(repo, store, "example", "more", {"type": "git", "address": folder})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())

    assert main(["--root", str(repo.root), "tokenizer-check", "main"]) == 0
    lines = capsys.readouterr().out.splitlines()
    corpus_line = next(line for line in lines if line.startswith("example-corpus-"))
    assert ": 10 documents, " in corpus_line and "not in the vocabulary" not in corpus_line
    assert "source files: reading and validating 2" in lines
    assert next(line for line in lines if line.startswith(late)).endswith("(not in the vocabulary's training)")
    assert lines[-1].startswith("every document decodes back exactly: 11 documents, ")
    documents, _ = kit.converted(repo, store)
    assert len(documents) == 11 and not any(store.has(hashlib.sha256(d.markdown).hexdigest()) for d in documents)


def test_a_streamed_build_gives_the_same_dataset_and_keeps_no_markdown(kit, repo, store, corpus, tmp_path):
    # a streamed build keeps tokens in the store, never the documents'
    # Markdown, and the dataset is the one a normal build gives
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000, held_out=0.2)
    # one pool of all 10 documents, so the held-out share always leaves it some
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "all", "select": "fr || en", "weight": 1}]})
    streamed_store = fresh_store(kit, repo, tmp_path / "streamed", sid)
    streamed_id = build_dataset(repo, "small", streamed_store, log=lambda _: None, stream=True)  # trains the vocabulary
    lock_path = repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["document_count"] == 10 and "documents" not in lock
    assert streamed_store.get_bytes(lock["documents_sha256"]) == listed_documents(streamed_store, streamed_id)
    assert build_dataset(repo, "small", streamed_store, log=lambda _: None, stream=True) == streamed_id
    assert json.loads(lock_path.read_text()) == lock  # a later build reuses the lock
    normal_store = fresh_store(kit, repo, tmp_path / "normal", sid, shared=streamed_store)
    normal_id = build_dataset(repo, "small", normal_store, log=lambda _: None)  # with the vocabulary it locked

    assert streamed_id == normal_id
    streamed, normal = record_of(streamed_store, streamed_id), record_of(normal_store, normal_id)
    for part in ("tokenizer", "sources", "documents", "pools", "contamination"):
        assert streamed[part] == normal[part]
    assert [s["token_sha256"] for s in streamed["shards"] + streamed["held_out_shards"]] == \
           [s["token_sha256"] for s in normal["shards"] + normal["held_out_shards"]]
    vocabulary = Tokenizer.from_bytes(streamed_store.get_bytes(streamed["tokenizer"]["vocabulary"]))
    for tokens, *_ in rows_of(streamed_store, streamed["shards"]):
        assert not streamed_store.has(hashlib.sha256(vocabulary.decode(tokens.tolist())).hexdigest())  # its Markdown


def test_a_source_missing_from_the_store_is_left_out_listed_and_recorded(kit, repo, store, corpus, tmp_path,
                                                                         monkeypatch, capsys):
    # the build never downloads: a source the download step could not keep
    # must not block every build, and the record says it is missing
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = build_dataset(repo, "small", store, log=lambda _: None)  # also locks the vocabulary
    folder, _ = kit.git_project(tmp_path / "gone", files={"en/gone.txt": b"a text upstream removed. " * 30},
                                executable=set(), symlinks={})
    changed = kit.add_source(repo, store, "example", "changed", {
        "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]}, fetch=False)
    shutil.rmtree(folder)
    with pytest.raises(SourceGone, match=changed):  # upstream no longer serves it: nothing kept
        kit.download(repo, store, changed)

    for stream in (False, True):
        without = build_dataset(repo, "small", store, log=lambda _: None, stream=stream)  # no --strict: the build goes on
        record = record_of(store, without)
        assert [s["source"] for s in record["skipped"]] == [changed]
        assert record["skipped"][0]["error"].startswith("NotDownloaded: ")
        assert changed not in record["sources"] and record["documents"] == record_of(store, first)["documents"]
        assert without != first  # the id says something is missing

    # the build's record names the missing source and why
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert main(["--root", str(repo.root), "build", "small"]) == 0
    [folder] = [name for name in store.record_folders() if "-wpt-dataset-build-small-" in name]
    events = [json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()]
    [missing] = [e["attributes"]["error"] for e in events if e["kind"] == "event" and e["name"] == "source.skipped"]
    assert missing.startswith(f"NotDownloaded: {changed}: ")
    assert json.loads(store.read_record(folder, "ids.json"))["inputs"]["skipped_sources"] == [changed]

    # the check goes on past a failure, so one run names every source to fix
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    capsys.readouterr()
    assert main(["--root", str(repo.root), "check", "--fetch", *repo.source_ids()]) == 1
    out, err = capsys.readouterr()
    assert "Markdown documents" in out and f"1 of 2 fetched sources failed:\n{changed}: SourceGone: " in err


def test_each_file_that_fails_a_check_is_left_out_alone_and_reported_in_the_record(kit, repo, store, tmp_path):
    # a failure is judged per item: the build leaves out each file no entry
    # covers, each file that is not text, and every file of an entry that does
    # not let its files train (the entry is not admitted, or its statement does
    # not match the text it expects), and the dataset's record says which and
    # why; the rest of the source trains
    files = {"LICENSE": kit.LICENSE_TEXT.encode(), "kept/one.txt": b"the tide comes in at dusk. " * 20,
             "kept/bytes.txt": b"not text \xc3\x28 \x81" * 10, "loose.txt": b"no entry covers me. " * 20,
             "waiting/two.txt": b"an entry under review covers me. " * 20,
             "other/NOTICE": b"Some other words, not the expected license.\n",
             "other/three.txt": b"my statement does not match. " * 20}
    folder, commit = kit.git_project(tmp_path / "mixed", files=files, executable=set(), symlinks={})
    under_review = {"decision": "under-review", "by": "Test Reviewer", "date": "2026-10-27"}
    sid = kit.add_source(repo, store, "example", "mixed", {
        "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}],
        "licenses": [kit.entry(["LICENSE", "kept/**"]), kit.entry(["waiting/**"], decision=under_review),
                     kit.entry(["other/**"], place={"file": "other/NOTICE"}, expected=[kit.REVIEWED])]})
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": 10**9, "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 300,
                                           "held_out": 0})
    logged = []

    dataset_id = build_dataset(repo, "small", store, log=logged.append)
    record = record_of(store, dataset_id)

    assert [document["id"] for document in document_list(store, dataset_id)] == [f"{sid}/LICENSE",
                                                                                 f"{sid}/kept/one.txt"]
    check = record["sources"][sid]["check"]
    assert check["left_out"] == [
        {"why": "not-text", "path": "kept/bytes.txt", "detail": "kept/bytes.txt is not UTF-8 text"},
        {"why": "uncovered", "path": "loose.txt",
         "detail": "loose.txt matches no license entry; every path must match exactly 1"},
        {"why": "entry", "entry": 1, "license": "MIT", "files": 1, "detail": "the entry is under-review"},
        {"why": "entry", "entry": 2, "license": "MIT", "files": 2,
         "detail": f"no license text holds the text there (its id would be "
                   f"{license_check.text_id(files['other/NOTICE'])}; `wpt dataset add-license-text` adds it, "
                   f"under review)"}]
    assert [(found["place"], found["matched"]) for found in check["statements"]] == [
        ({"file": "LICENSE"}, True), ({"file": "LICENSE"}, True), ({"file": "other/NOTICE"}, False)]
    assert f"{sid}: left out 3 files (entry), 1 files (not-text), 1 files (uncovered) (see the record's `check`)" \
        in logged


def test_a_build_goes_on_past_a_failing_source_and_lists_it_unless_strict(kit, repo, store, corpus, tmp_path):
    # a source in the store whose files are broken is left out and the record says so;
    # a strict build stops at it
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = build_dataset(repo, "small", store, log=lambda _: None)  # also locks the vocabulary
    folder, commit = kit.git_project(tmp_path / "other", files={"en/other.txt": b"owls hunt at night. " * 30},
                                     executable=set(), symlinks={})
    broken = kit.add_source(repo, store, "example", "broken", {"type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]})
    [placed] = {placed for _, placed in stored_files(upstream_key(repo.source(broken)), store).values()}
    for bundle in (store.root / "bundles").glob(f"{placed.bundle}*"):
        bundle.write_bytes(bytes(b ^ 0xFF for b in bundle.read_bytes()))  # every byte changed

    with pytest.raises(CorruptObject):
        build_dataset(repo, "small", store, log=lambda _: None, strict=True)
    for stream in (False, True):
        skipped = build_dataset(repo, "small", store, log=lambda _: None, stream=stream)
        record = record_of(store, skipped)
        assert [s["source"] for s in record["skipped"]] == [broken]
        assert record["skipped"][0]["error"].startswith("CorruptObject: ")
        assert record["documents"] == record_of(store, first)["documents"]


def broken_sources(kit, repo, store, tmp_path, count=2):
    """The ids of sources in the store whose format map gives the files of en/ two entries."""
    ids = []
    for number in range(count):
        folder, _ = kit.git_project(tmp_path / f"twice{number}", files={"en/a.txt": b"owls hunt at night. " * 30},
                                    executable=set(), symlinks={})
        ids.append(kit.add_source(repo, store, "example", f"twice{number}", {
            "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}],
            "formats": [{"format": "text", "paths": ["**"]}, {"format": "text", "paths": ["en/**"]}]}))
    return ids


def nothing_converted(monkeypatch):
    """Make any conversion of a source (the first reading of a text) fail the test."""
    def converted(*arguments, **more):
        raise AssertionError("a source was converted")

    monkeypatch.setattr(build, "each_source", converted)


@pytest.mark.parametrize("stream", [False, True])
def test_a_strict_build_with_a_broken_format_map_stops_before_a_text_is_read_and_names_each_source(
        kit, repo, store, corpus, tmp_path, monkeypatch, stream):
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    twice = broken_sources(kit, repo, store, tmp_path)
    nothing_converted(monkeypatch)

    for run in (lambda: build.build_tokenizer(repo, "small", store, log=lambda _: None, stream=stream, strict=True),
                lambda: build_dataset(repo, "small", store, log=lambda _: None, stream=stream, strict=True)):
        with pytest.raises(DatasetError) as stopped:
            run()
        message = str(stopped.value)
        assert all(f"{sid}: en/a.txt matches 2 format entries" in message for sid in twice)
        assert message.count("matches 2 format entries") == 2  # the corpus is not named: its map is sound


class Machine:
    """A machine with plenty of everything, as wpt_utils.parallel measures it."""

    def __init__(self, cores):
        self.cores = cores

    def memory(self):
        return 1e12, 1e12

    def idle_cores(self):
        return float(self.cores)

    def free_disk(self, folder):
        return 1e15


def first_ends_last(monkeypatch, module, name, first, log):
    """Make the worker function module.name end last for the item first
    (first_ends_last.py), each item's end logged into log; returns the
    module's ended_out_of_order."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))  # where the workers import the stand-in from
    helper = importlib.import_module("first_ends_last")
    monkeypatch.setattr(module, name, helper.FirstEndsLast(module.__name__, name, first, log))
    return helper.ended_out_of_order


def on_cores(monkeypatch, cores):
    """Make the build's worker processes as many as a machine of cores, with plenty of memory and disk, allows."""
    monkeypatch.setattr(workers, "Sizing", lambda what, cores=cores, **more: Sizing(what, Machine(cores), **more))


def source_of(events, span):
    """The source of the "source" span an event's span lies in (through the parts a worker records under it)."""
    starts = {e["span"]: e for e in events if e["kind"] == "span_start"}
    while span in starts:
        if starts[span]["name"] == "source":
            return starts[span]["attributes"]["source"]
        span = starts[span]["parent"]
    return None


@pytest.mark.parametrize("stream", [False, True])
def test_sources_converted_at_once_give_the_dataset_converted_one_at_a_time(kit, repo, store, tmp_path, monkeypatch,
                                                                            stream):
    # sources are converted in worker processes, several at once, and
    # documents encoded several at once, finishing in any order; the dataset
    # id, the documents and the shards must not change, and the record still
    # ties each ref a source's conversion sets to that source
    ids = []
    for n in range(4):
        files = {f"en/{n}-{i}.txt": f"source {n} text {i}: the wind blows over hill {i}. ".encode() * 20
                 for i in range(3)}
        files["en/shared.txt"] = b"every source holds this same page. " * 20  # the first source keeps it
        folder, commit = kit.git_project(tmp_path / f"upstream-{n}", files=files, executable=set(), symlinks={})
        ids.append(kit.add_source(repo, store, "example", f"part-{n}", {
            "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]}))
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": 10**9, "vocab_size": 600})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 3000,
                                           "held_out": 0.2})
    build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    built, log = {}, tmp_path / "ends.txt"
    for cores in (1, 16):  # 1: one worker process at a time; 16: 12 at once, the first source ending last
        on_cores(monkeypatch, cores)
        if cores == 16:
            ended_out_of_order = first_ends_last(monkeypatch, reading, "read_tokens", sorted(ids)[0], log)
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", ids[0], shared=store)
        kit.download(repo, fresh, *ids[1:])
        with Telemetry.start(fresh, "wpt dataset build", [], "small", env_file=tmp_path / ".env",
                             environ={}) as telemetry:
            dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None, stream=stream, telemetry=telemetry)
        built[cores] = dataset_id, record_of(fresh, dataset_id), document_list(fresh, dataset_id)
        events = [json.loads(line) for line in fresh.read_record(telemetry.folder, "events.jsonl").splitlines()]
        at_once = [e["attributes"]["limit"] for e in events if e["kind"] == "event" and e["name"] == "sources.at_once"]
        assert at_once and set(at_once) == {max(1, int(cores * 0.75))}
        refs = [source_of(events, e["span"]) for e in events  # what each source's conversion stores once
                if e["kind"] == "event" and e["name"] == "store.ref_set"
                and e["attributes"]["ref"].startswith("source-tokens")]
        assert sorted(refs) == sorted(ids)  # recorded by the worker, under its source's span
    assert ended_out_of_order(log, sorted(ids)[0])
    (one_id, one, one_list), (many_id, many, many_list) = built[1], built[16]
    assert many_id == one_id and many_list == one_list
    assert one["documents"]["duplicates_removed"] == 3 and len(one["sources"]) == 4
    for part in ("sources", "documents", "pools", "contamination"):
        assert many[part] == one[part]
    assert [s["token_sha256"] for s in many["shards"] + many["held_out_shards"]] == \
           [s["token_sha256"] for s in one["shards"] + one["held_out_shards"]]


def test_a_source_s_kept_files_and_tokens_are_made_again_only_when_its_own_inputs_change(kit, repo, store, corpus,
                                                                                        tmp_path, monkeypatch):
    # a commit that changes only data/ converts again only the sources
    # whose inputs changed: another source, with a license text this
    # source does not match, leaves this source's source-kept and
    # source-tokens refs found again; a change to the decision of the text
    # this source matched, or to the code of the modules kept.py and
    # reading.py import (wpt_utils.programs.code_of), makes both again
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    _, vocabulary, _ = build.build_tokenizer(repo, "small", store, log=lambda _: None)
    monkeypatch.setattr(workers, "_REPO", repo)  # read_tokens runs here, as in a worker process
    monkeypatch.setattr(workers, "license_texts", lambda: license_check.LicenseTexts(repo.license_texts()))
    for ref in (store.root / "refs" / "source-tokens").glob("*"):
        ref.unlink()
    for name in ("_CODE", "_MODULES", "_GIVEN"):  # put back as they were when the test ends
        monkeypatch.setattr(programs, name, getattr(programs, name))
    keyed = programs.codes(*workers.KEYED)["modules"]

    def made() -> dict[str, int]:
        source = repo.source(sid)
        kept.kept(source, store, workers.license_texts())
        reading.read_tokens(source, store, [0, 1], vocabulary, tmp_path / "markdown", None)
        return {kind: len(list((store.root / "refs" / kind).iterdir())) for kind in ("source-kept", "source-tokens")}

    assert made() == {"source-kept": 1, "source-tokens": 1}
    folder, _ = kit.git_project(tmp_path / "other", files={"notes.txt": b"another source's own text. " * 20},
                                executable=set(), symlinks={})
    other = kit.add_source(repo, store, "example", "other", {"type": "git", "address": folder})
    matched = repo.source(sid).data["licenses"][0]["expected"][0]
    assert repo.source(other).data["licenses"][0]["expected"] != [matched]  # a text this source does not match
    # the commit that adds it: a new whole commit (code()), the same code of the modules
    programs.set_code({"code": {"commit": "0" * 40, "changes": None}, "modules": keyed})
    assert made() == {"source-kept": 1, "source-tokens": 1}

    item = repo.license_texts()[matched]
    kit.write_texts(repo.root, {matched: {**item, "decision": {**item["decision"], "decision": "refused",
                                                               "reason": "a test's refusal"}}})
    assert made() == {"source-kept": 2, "source-tokens": 2}
    programs.set_code({"code": {"commit": "1" * 40, "changes": None},
                       "modules": {**keyed, "wpt_dataset.kept": "1" * 64}})
    assert made() == {"source-kept": 3, "source-tokens": 3}
    programs.set_code({"code": {"commit": "1" * 40, "changes": None},
                       "modules": {**keyed, "wpt_dataset.kept": "1" * 64, "wpt_dataset.reading": "1" * 64}})
    assert made() == {"source-kept": 3, "source-tokens": 4}


def test_a_worker_process_finds_what_its_starter_kept_under_the_starter_s_codes(kit, repo, store, corpus,
                                                                              monkeypatch):
    # a source's conversion runs in a worker process, which must find the
    # results its starter keyed: given its starter's codes, the worker
    # computes the starter's key, even codes that differ from what it would
    # find from the files itself, and so finds the starter's result
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    for name in ("_CODE", "_MODULES", "_GIVEN"):  # put back as they were when the test ends
        monkeypatch.setattr(programs, name, getattr(programs, name))
    programs.set_code({"code": programs.code(), "modules": {module: "1" * 64 for module in workers.KEYED}})
    texts = license_check.LicenseTexts(repo.license_texts())
    here = kept.kept(repo.source(sid), store, texts)
    with workers.Pool(repo, "sources at once") as pool:
        there = pool.call(kept.kept, repo.source(sid), store, texts)
    assert there.key == here.key
    assert len(list((store.root / "refs" / "source-kept").iterdir())) == 1


def test_every_file_that_is_not_python_in_a_keyed_package_is_named_with_whether_it_enters_the_keys():
    # a key holds Python files and compiled modules only: a new file of another kind in a keyed package (a table
    # the code reads, say) would change results without changing a key, until someone says whether it enters
    packages = {}
    for module in workers.KEYED:
        for name in programs._imported(module)[0]:
            top = name.partition(".")[0]
            packages[top] = Path(importlib.util.find_spec(top).submodule_search_locations[0])
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    found = {file.relative_to(folder.parent).as_posix()
             for folder in packages.values() for file in folder.rglob("*")
             if file.is_file() and file.suffix != ".py" and not file.name.endswith(suffixes)
             and "__pycache__" not in file.parts}
    assert found == set(workers.NOT_PYTHON)


def test_a_streamed_build_leaves_no_copy_of_a_source_s_files_on_this_machine(kit, repo, store, tmp_path):
    # --stream is for data larger than the disk: what a source's conversion
    # copies out of the store (a Parquet records file, whose reader seeks)
    # goes before the next source
    records = kit.parquet_bytes([{"id": str(i), "text": f"record {i} tells of the sea. " * 10} for i in range(5)])
    folder, commit = kit.git_project(tmp_path / "records", files={"data/part.parquet": records}, executable=set(),
                                     symlinks={})
    sid = kit.add_source(repo, store, "example", "records", {
        "type": "git", "address": folder, "formats": [{"format": "records", "paths": ["**"], "text": "text", "id": "id"}],
        "tags": [{"tags": ["en"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": 10**9, "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 300,
                                           "held_out": 0})
    record_file = hashlib.sha256(records).hexdigest()

    build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    assert not list(store.cache.rglob(record_file))  # its records file was read from a copy, now gone
    assert not (store.cache / "markdown").exists()  # and its Markdown went with the build
    normal = fresh_store(kit, repo, tmp_path / "normal", sid, shared=store)
    build_dataset(repo, "small", normal, log=lambda _: None)
    assert not list(normal.cache.rglob(record_file))
    assert list((normal.cache / "markdown").rglob("*.zst"))  # without --stream, the Markdown stays for the next build


def test_a_streamed_build_keeps_the_conversations_and_leaves_no_copy_of_their_files(kit, repo, store, corpus,
                                                                                    tmp_path):
    # the build also keeps the conversations the training steps read, and a
    # streamed one still leaves none of the files it read on this machine
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    messages = "".join(json.dumps(row) + "\n" for row in [
        {"id": "q", "parent": None, "role": "prompter", "text": "Why is the sky blue?"},
        {"id": "a", "parent": "q", "role": "assistant", "text": "Air scatters blue light most."}]).encode()
    folder, commit = kit.git_project(tmp_path / "talks", files={"messages.jsonl": messages}, executable=set(),
                                     symlinks={})
    talks = kit.add_source(repo, store, "example", "talks", {
        "type": "git", "address": folder, "formats": [{"format": "chat", "paths": ["*.jsonl"], "tree": {
            "id": "id", "parent": "parent", "role": "role", "text": "text",
            "roles": {"prompter": "user", "assistant": "assistant"}}}],
        "tags": [{"tags": ["answer-as-assistant"], "paths": ["**"]}]})
    messages_file = hashlib.sha256(messages).hexdigest()

    streamed_id = build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    assert not list(store.cache.rglob(messages_file))  # read from a copy, now gone
    normal = fresh_store(kit, repo, tmp_path / "normal", sid, shared=store)
    kit.download(repo, normal, talks)
    assert build_dataset(repo, "small", normal, log=lambda _: None) == streamed_id
    streamed, built = record_of(store, streamed_id), record_of(normal, streamed_id)
    assert streamed["conversations"] == built["conversations"]
    assert streamed["conversations"]["count"] == 1 and streamed["conversations"]["sources"] == [talks]
    [line] = store.get_bytes(streamed["conversations"]["object"]).decode().splitlines()
    assert json.loads(line)["id"] == f"{talks}/messages.jsonl:a"


def chat_source(kit, repo, store, tmp_path, name, text="Air scatters blue light most.", fetch=True):
    """A source of one chat file holding one conversation, tagged answer-as-assistant; returns its id."""
    messages = "".join(json.dumps(row) + "\n" for row in [
        {"id": "q", "parent": None, "role": "prompter", "text": "Why is the sky blue?"},
        {"id": "a", "parent": "q", "role": "assistant", "text": text}]).encode()
    folder, commit = kit.git_project(tmp_path / name, files={"messages.jsonl": messages}, executable=set(),
                                     symlinks={})
    return kit.add_source(repo, store, "example", name, {
        "type": "git", "address": folder, "formats": [{"format": "chat", "paths": ["*.jsonl"], "tree": {
            "id": "id", "parent": "parent", "role": "role", "text": "text",
            "roles": {"prompter": "user", "assistant": "assistant"}}}],
        "tags": [{"tags": ["answer-as-assistant"], "paths": ["**"]}]}, fetch=fetch)


def test_a_source_left_out_gives_the_same_dataset_on_any_machine_and_never_moves_the_text(kit, repo, store, corpus,
                                                                                          tmp_path):
    # a chat source not downloaded is left out; its error names the store, yet
    # two machines missing it build the same dataset, and the text (its rows'
    # order) is the one of a build that holds the source
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000, held_out=0.2)
    talks = chat_source(kit, repo, store, tmp_path, "talks", fetch=False)
    whole = fresh_store(kit, repo, tmp_path / "whole", sid)
    kit.download(repo, whole, talks)
    held = record_of(whole, build_dataset(repo, "small", whole, log=lambda _: None))  # locks the vocabulary
    assert held["conversations"]["sources"] == [talks]
    other = fresh_store(kit, repo, tmp_path / "other", sid, shared=whole)  # given the vocabulary the first build locked
    lock = repo.generated("lock", "main")
    for locked in (lock["vocabulary_sha256"], lock["documents_sha256"]):
        store.put_bytes(whole.get_bytes(locked))  # and so is this machine

    here = build_dataset(repo, "small", store, log=lambda _: None)
    assert build_dataset(repo, "small", other, log=lambda _: None) == here
    missing = record_of(store, here)
    assert [s["source"] for s in missing["skipped"]] == [talks]
    assert held["fingerprints"]["text"] == missing["fingerprints"]["text"]
    assert held["shards"] == missing["shards"]


def test_a_dataset_whose_record_lacks_a_field_is_built_again(kit, repo, store, corpus):
    # a dataset whose record a build of another shape wrote is built again, so
    # wpt train, which reads the current shape only, can use it
    setup_recipes(kit, repo, store, corpus, total_tokens=500)
    dataset_id = build_dataset(repo, "small", store, log=lambda _: None)
    current = record_of(store, dataset_id)
    older = {k: v for k, v in current.items() if k != "fingerprints"}
    assert not build.current_record(older) and build.current_record(current)
    store.set_ref("datasets", dataset_id, store.put_bytes(json.dumps(older).encode()))

    assert build_dataset(repo, "small", store, log=lambda _: None) == dataset_id
    assert build.current_record(record_of(store, dataset_id))
    assert record_of(store, dataset_id)["fingerprints"] == current["fingerprints"]


@pytest.mark.parametrize("stream", [False, True])
def test_conversations_read_at_once_give_the_dataset_read_one_at_a_time(kit, repo, store, corpus, tmp_path,
                                                                        monkeypatch, stream):
    # chat sources are read in worker processes, several at once, finishing
    # in any order; the conversations and the dataset id must be the ones read
    # one at a time give
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    ids = [chat_source(kit, repo, store, tmp_path, f"talks-{n}", text=f"Answer number {n}.") for n in range(4)]
    built, log, first = {}, tmp_path / "ends.txt", None
    for cores in (1, 16):  # 1: one worker process at a time; 16: 12 at once, the first source ending last
        on_cores(monkeypatch, cores)
        if cores == 16:
            ended_out_of_order = first_ends_last(monkeypatch, conversations, "_of_source", sorted(ids)[0], log)
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", sid, shared=first)
        first = first or fresh
        kit.download(repo, fresh, *ids)
        dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None, stream=stream)
        record = record_of(fresh, dataset_id)
        built[cores] = dataset_id, record["conversations"], fresh.get_bytes(record["conversations"]["object"])
    assert ended_out_of_order(log, sorted(ids)[0])
    assert built[1] == built[16]
    assert built[1][1]["sources"] == sorted(ids) and built[1][1]["count"] == 4



TASK = {"instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 4, "max_tokens": 200, "command_seconds": 5}, "needs": [["write", "bash"]]}
CHECK = b"#!/bin/sh\ntest -f out.txt && echo 1 || echo 0\n"
BOOK = b"The lamp was lit at dusk. The keeper wrote the hour in his log. Then he slept.\n"


def book_source(kit, repo, store, tmp_path, fetch=True):
    """A source of one text, notes.txt, that tasks point into; returns (its id, its Markdown)."""
    folder, commit = kit.git_project(tmp_path / "book", files={"notes.txt": BOOK}, executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "book", {"type": "git", "address": folder},
                         fetch=fetch)
    return sid, BOOK.strip() + b"\n"  # the text's Markdown: its paragraph, ending with one newline


def task_source(kit, repo, store, tmp_path, name, book, text):
    """A source of two tasks tagged finish-tasks: one on its own files, one taking its file from book's
    notes.txt (a pointer); returns its id."""
    (repo.root / "data" / "images" / "tiny").mkdir(parents=True, exist_ok=True)
    (repo.root / "data" / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    end = text.index(b" The keeper")
    pointed = {**TASK, "instructions": f"Read the lamp's line, task {name}.", "pointers": [
        {"for": "files", "path": "lamp.txt", "source": book, "document": "notes.txt", "span": [0, end],
         "sha256": hashlib.sha256(text[:end]).hexdigest(), "removed": []}]}
    files = {"tasks/own/task.json": json.dumps({**TASK, "instructions": f"Task {name}."}).encode(),
             "tasks/own/check": CHECK, "tasks/own/files/README": f"start {name}\n".encode(),
             "tasks/pointed/task.json": json.dumps(pointed).encode(), "tasks/pointed/check": CHECK}
    folder, commit = kit.git_project(tmp_path / name, files=files,
                                     executable={path for path in files if path.endswith("/check")}, symlinks={})
    return kit.add_source(repo, store, "example", name, {
        "type": "git", "address": folder, "formats": [{"format": "task", "paths": ["tasks/**"]}],
        "tags": [{"tags": ["finish-tasks"], "paths": ["**"]}]})


def test_task_sources_read_at_once_give_the_dataset_read_one_at_a_time(kit, repo, store, corpus, tmp_path,
                                                                       monkeypatch):
    # task sources, and the source their pointers read, are read in worker
    # processes, several at once; the tasks kept and the dataset id must be the
    # ones a machine reading one at a time gives
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    book, text = book_source(kit, repo, store, tmp_path)
    ids = [task_source(kit, repo, store, tmp_path, f"tasks-{n}", book, text) for n in range(4)]
    built, log, first = {}, tmp_path / "ends.txt", None
    for cores in (1, 16):  # 1: one worker process at a time; 16: 12 at once, the first source ending last
        on_cores(monkeypatch, cores)
        if cores == 16:
            ended_out_of_order = first_ends_last(monkeypatch, tasks, "_tagged", sorted(ids)[0], log)
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", sid, shared=first)
        first = first or fresh
        kit.download(repo, fresh, book, *ids)
        dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None)
        record = record_of(fresh, dataset_id)
        built[cores] = dataset_id, record["tasks"], fresh.get_bytes(record["tasks"]["object"])
    assert ended_out_of_order(log, sorted(ids)[0])
    assert built[1] == built[16]
    assert built[1][1]["count"] == 8 and built[1][1]["sources"] == sorted([book, *ids])
    found = [json.loads(line) for line in built[1][2].decode().splitlines()]
    assert [task["id"] for task in found] == [f"{sid}/tasks/{folder}" for sid in sorted(ids)
                                              for folder in ("own", "pointed")]
    lamp = tasks.from_json(found[1]).files["lamp.txt"][1]
    assert lamp == b"The lamp was lit at dusk." and all(task["files"] == found[1]["files"] for task in found[1::2])


def test_a_task_pointing_into_a_source_not_downloaded_is_left_out_and_reported(kit, repo, store, corpus,
                                                                                     tmp_path):
    # the book a task reads is not in this store: the build leaves the book out,
    # and the task, which would give the model a text that is not there, is
    # left out alone and named in the dataset's record; the other task stays
    setup_recipes(kit, repo, store, corpus, total_tokens=500)
    build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    book, text = book_source(kit, repo, store, tmp_path, fetch=False)
    tasks_id = task_source(kit, repo, store, tmp_path, "tasks", book, text)
    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert record["left_out_tasks"] == [{"task": f"{tasks_id}/tasks/pointed",
                                         "why": f"RuleViolation: {tasks_id}/tasks/pointed: it points into {book}, "
                                                f"which the build left out"}]
    assert record["tasks"]["count"] == 1 and record["tasks"]["sources"] == [tasks_id]


def test_a_streamed_conversion_copies_only_the_files_it_needs_as_files(kit, repo, store, tmp_path):
    # a Parquet records file is read as a file (its reader seeks), so it is
    # copied; a JSON Lines records file, even patched (a store object), and
    # the text beside it are read from the store as streams and never copied
    records = "".join(json.dumps({"id": str(i), "text": f"record {i} tells of the sea. " * 10}) + "\n"
                      for i in range(3)).encode()
    table = kit.parquet_bytes([{"id": "t", "text": "a table row tells of the hills. " * 10}])
    folder, commit = kit.git_project(tmp_path / "mixed", files={"data/part.jsonl": records, "data/table.parquet": table,
                                                                "notes.txt": b"plain words on a page.\n" * 50},
                                     executable=set(), symlinks={})
    changed = records.replace(b"record 0 tells", b"record 0 says")
    lines = records.decode().splitlines()
    patch = ("diff --git a/data/part.jsonl b/data/part.jsonl\n--- a/data/part.jsonl\n+++ b/data/part.jsonl\n"
             f"@@ -1,3 +1,3 @@\n-{lines[0]}\n+{changed.decode().splitlines()[0]}\n {lines[1]}\n {lines[2]}\n")
    sid = kit.add_source(repo, store, "example", "mixed", {
        "type": "git", "address": folder, "formats": [{"format": "records", "paths": ["data/**"], "text": "text", "id": "id"},
                    {"format": "text", "paths": ["notes.txt"]}]}, patches=[patch])
    source = repo.source(sid)
    copies = tmp_path / "copies"
    view = store.copies_in(copies)
    tree, _ = sources.source_tree(source, view)
    found = [converted.path for converted in markdown.documents(source, tree, view, repo)]
    assert found == ["data/part.jsonl#0", "data/part.jsonl#1", "data/part.jsonl#2", "data/table.parquet#t",
                     "notes.txt"]
    assert sorted(path.name for path in copies.rglob("*") if path.is_file()) == [hashlib.sha256(table).hexdigest()]


def test_the_disk_a_conversion_reserves_is_the_size_of_the_files_it_will_copy_stated_beforehand(kit, repo, store, tmp_path):
    # Parquet records files are copied as files; JSON Lines records, text, and the history of a git source, are
    # read from the store as streams; a task's repository copies every file but the history's; pointers copy the
    # Parquet files they point into
    records = kit.parquet_bytes([{"id": str(i), "text": f"record {i} tells of the sea. " * 10} for i in range(3)])
    other = kit.parquet_bytes([{"id": str(i), "text": f"record {i} tells of the sky. " * 10} for i in range(3)])
    lines = "".join(json.dumps({"id": str(i), "text": f"line {i}. " * 10}) + "\n" for i in range(3)).encode()
    notes = b"plain words on a page.\n" * 50
    folder, commit = kit.git_project(tmp_path / "mixed", files={"data/a.parquet": records, "data/b.parquet": other,
                                                               "data/copy.parquet": records, "data/c.jsonl": lines,
                                                               "notes.txt": notes},
                                     executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "mixed", {
        "type": "git", "address": folder, "formats": [{"format": "records", "paths": ["data/**"], "text": "text", "id": "id"},
                    {"format": "text", "paths": ["notes.txt"]}]})
    source = repo.source(sid)

    # a content that several paths hold is copied once
    assert sources.copy_size(source, store, sources.copied_as_records(source)) == len(records) + len(other)
    assert sources.copy_size(source, store, sources.copied_by_pointers(source, ["data/b.parquet#1"])) == len(other)
    assert sources.copy_size(source, store, sources.copied_by_pointers(source, ["data/c.jsonl#1", "notes.txt"])) == 0
    assert sources.copy_size(source, store, sources.copied_for_repository(source)) \
        == len(records) + len(other) + len(lines) + len(notes)
    assert sources.copy_size(source, store, sources.copied_as_trees(source)) == 0  # it holds no chat file
    assert sources.copy_size(source, store, lambda path: False) == 0


def many_files(kit, repo, store, tmp_path, name="many", count=30, tag="en"):
    """A source of count different files of about 1 KB, tagged tag; returns its id."""
    files = {f"{tag}/{name}-{i:02d}.txt": f"{name} page {i} tells of a long road. ".encode() * 30 for i in range(count)}
    folder, _ = kit.git_project(tmp_path / name, files=files, executable=set(), symlinks={})
    return kit.add_source(repo, store, "example", name, {"type": "git", "address": folder,
                                                         "tags": [{"tags": [tag], "paths": ["**"]}]})


def one_pool(kit, repo, total_tokens, sample_bytes=10**9):
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": sample_bytes, "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": total_tokens,
                                           "held_out": 0})


@pytest.fixture
def two_kilobyte_groups(monkeypatch):
    """The choice groups the 1 KB files of a test source by frames of 2 KB, so that its groups are small."""
    monkeypatch.setattr(choice.blocks, "FRAME", 2048)


def reads_here(monkeypatch, repo):
    """Run reading's worker functions in this process, as in a worker process."""
    monkeypatch.setattr(workers, "_REPO", repo)
    monkeypatch.setattr(workers, "license_texts", lambda: license_check.LicenseTexts(repo.license_texts()))


def test_a_read_builds_a_source_s_newest_index_once_however_many_batches_it_takes(kit, repo, store, tmp_path,
                                                                                  monkeypatch):
    # the index reads every file of its entry: built per batch, a source of N large files would be read N times
    def lines(*rows):
        return "".join(json.dumps({"id": name, "date": date, "text": f"{name} at {date}"}) + "\n"
                       for name, date in rows).encode()

    files = {f"part-{i}.ndjson": lines((f"only-{i}", "2025-01-01"), ("shared", f"2025-0{i + 1}-01")) for i in range(4)}
    folder, _ = kit.git_project(tmp_path / "revisions", files=files, executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "revisions", {
        "type": "git", "address": folder,
        "formats": [{"format": "records", "paths": ["*.ndjson"], "text": "text", "id": "id", "newest": "date"}]})
    reads_here(monkeypatch, repo)
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)  # each file a batch alone
    indexes, index = [], records.newest_index
    monkeypatch.setattr(records, "newest_index", lambda *a: indexes.append(a) or index(*a))

    read = reading.read_markdown(repo.source(sid), store, list(range(4)), tmp_path / "markdown", Telemetry.off())

    assert len(read.packs) == 4 and len(indexes) == 1
    assert {entry["path"]: text for entry, text in read.documents()} == {
        **{f"part-{i}.ndjson#only-{i}": markdown.text(f"only-{i} at 2025-01-01") for i in range(4)},
        "part-3.ndjson#shared": markdown.text("shared at 2025-04-01")}


LF, CRLF = b"one line\nand another\n" * 20, b"one line\r\nand another\r\n" * 20  # other bytes, one Markdown


def lines_source(kit, repo, store, folder, name, files):
    upstream, _ = kit.git_project(folder, files=files, executable=set(), symlinks={})
    return kit.add_source(repo, store, "example", name, {"type": "git", "address": upstream,
                                                         "tags": [{"tags": ["en"], "paths": ["**"]}]})


def every_file_chosen(kit, repo):
    """Recipes whose first round takes every file of these small sources, one document left being enough."""
    one_pool(kit, repo, 1000)
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1, "max_epochs": 1000}]})


def test_a_large_source_is_cut_every_few_batches_from_its_sizes_alone(kit, repo, store, tmp_path, monkeypatch):
    monkeypatch.setattr(reading, "BATCH_BYTES", 10)
    sid = many_files(kit, repo, store, tmp_path)
    source = repo.source(sid)
    positions = list(range(100, 130))
    parts = reading.parts_of(source, positions, [5] * 30)  # 2 files a batch, 8 files a part

    assert [part.positions for part in parts] == [positions[start:start + 8] for start in range(0, 30, 8)]
    assert [part.size for part in parts] == [40, 40, 40, 30] and {part.of for part in parts} == {4}
    small = reading.parts_of(source, positions[:8], [5] * 8)
    assert [part.positions for part in small] == [positions[:8]]


def test_a_source_under_newest_stays_one_part_however_large(kit, repo, store, tmp_path, monkeypatch):
    # the index of its records reads every file of the entry: each part would read the whole source again
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)
    folder, _ = kit.git_project(tmp_path / "revisions", files={"a.ndjson": b"{}"}, executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "revisions", {
        "type": "git", "address": folder,
        "formats": [{"format": "records", "paths": ["*.ndjson"], "text": "text", "id": "id", "newest": "date"}]})

    [part] = reading.parts_of(repo.source(sid), list(range(20)), [5] * 20)
    assert part.positions == list(range(20)) and part.size == 100 and part.of == 1


def ids_of_tokens(store) -> int:
    return len(list((store.root / "refs" / "source-tokens").iterdir()))


def test_a_source_read_in_parts_gives_the_dataset_of_one_read_and_a_saved_part_is_not_read_again(
        kit, repo, store, tmp_path, monkeypatch):
    # the same recipe, on the same code, whether a source is one item or several that several workers read: the lock,
    # the dataset id, its documents and its shards are the same, a file that is not text and a Markdown another
    # source repeats included; a build that finds some parts saved reads the others only
    files = {f"en/many-{i:02d}.txt": f"many page {i} tells of a long road. ".encode() * 30 for i in range(30)}
    files |= {"en/many-00.txt": LF, "en/many-15.txt": b"\x80 not text " * 30}
    folder, _ = kit.git_project(tmp_path / "many", files=files, executable=set(), symlinks={})
    many = kit.add_source(repo, store, "example", "many", {"type": "git", "address": folder,
                                                           "tags": [{"tags": ["en"], "paths": ["**"]}]})
    other = lines_source(kit, repo, store, tmp_path / "other", "other", {"copy.txt": CRLF})
    one_pool(kit, repo, 30000)
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1, "max_epochs": 1000}]})
    whole = build_dataset(repo, "small", store, log=lambda _: None)
    assert ids_of_tokens(store) == 2
    lock = repo.generated("lock", "main")
    (repo.root / repo.relative("lock", "main")).unlink()  # the parts train it again

    monkeypatch.setattr(reading, "BATCH_BYTES", 2048)  # the asking process cuts: about 2 files a batch
    split = open_store((tmp_path / "split").as_uri())
    kit.download(repo, split, many, other)
    assert build_dataset(repo, "small", split, log=lambda _: None) == whole
    assert ids_of_tokens(split) > 3  # the large source's parts, each saved, and the other source

    assert repo.generated("lock", "main") == lock
    assert document_list(split, whole) == document_list(store, whole)
    one, parts = record_of(store, whole), record_of(split, whole)
    for field in ("sources", "documents", "pools", "contamination", "skipped"):
        assert parts[field] == one[field]
    assert [s["token_sha256"] for s in parts["shards"] + parts["held_out_shards"]] == \
           [s["token_sha256"] for s in one["shards"] + one["held_out_shards"]]
    assert [item["path"] for item in parts["sources"][many]["check"]["left_out"]] == ["en/many-15.txt"]

    next(iter((split.root / "refs" / "source-tokens").iterdir())).unlink()
    with Telemetry.start(split, "wpt dataset build", [], "small", env_file=tmp_path / ".env", environ={}) as telemetry:
        assert build_dataset(repo, "small", split, log=lambda _: None, telemetry=telemetry) == whole
    events = [json.loads(line) for line in split.read_record(telemetry.folder, "events.jsonl").splitlines()]
    assert len([e for e in events if e["kind"] == "event" and e["name"] == "store.ref_set"
                and e["attributes"]["ref"].startswith("source-tokens")]) == 1


def test_a_part_that_fails_takes_its_whole_source_out_once_and_nothing_of_it_counts(kit, repo, store, tmp_path,
                                                                                    monkeypatch):
    # as one read: the source is listed once, however many of its parts fail, and the page it shares with a later
    # source stays in that source, though a part of the failing one read that page without trouble
    other = lines_source(kit, repo, store, tmp_path / "other", "z-other", {"copy.txt": CRLF})
    every_file_chosen(kit, repo)
    before = build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    files = {f"en/many-{i:02d}.txt": f"many page {i} tells of a long road. ".encode() * 30 for i in range(30)}
    files |= {"en/many-00.txt": LF, "en/many-12.xhtml": b"<p>never closed", "en/many-28.xhtml": b"<p>nor this"}
    folder, _ = kit.git_project(tmp_path / "broken", files=files, executable=set(), symlinks={})
    broken = kit.add_source(repo, store, "example", "a-broken", {
        "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}],
        "formats": [{"format": "text", "paths": ["**/*.txt"]}, {"format": "xhtml", "paths": ["**/*.xhtml"]}]})
    assert broken < other
    monkeypatch.setattr(reading, "BATCH_BYTES", 2048)

    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))

    assert [s["source"] for s in record["skipped"]] == [broken]
    assert [d["id"] for d in document_list(store, record["id"])] == [d["id"] for d in document_list(store, before)]
    with pytest.raises(DatasetError):
        build_dataset(repo, "small", store, log=lambda _: None, strict=True)


def test_the_parts_after_one_that_failed_are_not_read(kit, repo, store, tmp_path, monkeypatch):
    # they would be fetched, converted and stored for a source that no longer counts: on one worker, parts run in
    # order, so only the parts before the broken one leave tokens
    lines_source(kit, repo, store, tmp_path / "other", "z-other", {"copy.txt": CRLF})
    every_file_chosen(kit, repo)
    build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    files = {f"en/many-{i:02d}.txt": f"many page {i} tells of a long road. ".encode() * 30 for i in range(30)}
    files |= {"en/many-12.xhtml": b"<p>never closed"}
    folder, _ = kit.git_project(tmp_path / "broken", files=files, executable=set(), symlinks={})
    broken = kit.add_source(repo, store, "example", "a-broken", {
        "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}],
        "formats": [{"format": "text", "paths": ["**/*.txt"]}, {"format": "xhtml", "paths": ["**/*.xhtml"]}]})
    monkeypatch.setattr(reading, "BATCH_BYTES", 2048)
    on_cores(monkeypatch, 1)
    before = ids_of_tokens(store)

    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))

    assert [s["source"] for s in record["skipped"]] == [broken]
    assert ids_of_tokens(store) - before == 1


class PoolHere:
    """workers.Pool on threads alone, two at once: a worker process can share no event with a test."""

    def __init__(self, repo, what, disk=None, folder=None):
        self.sizing = Sizing(what, Machine(3), disk=disk, folder=folder)

    def __enter__(self):
        return self

    def __exit__(self, *stopped):
        return False

    def call(self, function, *arguments, telemetry=None):
        return function(*arguments)


def test_a_source_s_error_is_its_first_part_s_in_their_order_whichever_fails_first_in_time(kit, repo, store, tmp_path,
                                                                                          monkeypatch):
    # the record and the dataset id hold the error, so it cannot depend on the timing: part 1 is running while part 2
    # fails, and goes on to fail itself; part 3 comes after both and is not read, though the progress counts it
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)
    monkeypatch.setattr(workers, "Pool", PoolHere)
    source = repo.source(many_files(kit, repo, store, tmp_path, count=16))
    parts = reading.parts_of(source, list(range(16)), [1] * 16)
    assert [part.number for part in parts] == [0, 1, 2, 3]
    second_failed, read = threading.Event(), []

    def reading_a_part(source, store, number, telemetry=None):
        read.append(number)
        if number == 1:
            assert second_failed.wait(30)
            raise DatasetError("part 1")
        if number == 2:
            second_failed.set()
            raise DatasetError("part 2")
        return number

    telemetry = Telemetry.off()
    with telemetry.progress("parts", len(parts), "parts") as progress:
        [(done, span)] = build.each_source(reading_a_part, parts, repo, store, telemetry, progress,
                                           lambda part: (part.number,), lambda part: 0, tmp_path)
        telemetry.end(span)

    assert str(done.error) == "part 1" and sorted(read) == [0, 1, 2] and progress.done == 4


def test_the_parts_of_a_source_give_its_documents_in_the_order_of_one_read(kit, repo, store, tmp_path, monkeypatch):
    # bundle order is not path order: the vocabulary reads a source's documents in path order, however it was cut
    sid = many_files(kit, repo, store, tmp_path, count=12)
    reads_here(monkeypatch, repo)
    source = repo.source(sid)
    order = list(range(12))[::-1]
    one = reading.read_markdown(source, store, order, tmp_path / "one", Telemetry.off())
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)  # each file a batch: 3 parts of 4
    parts = reading.parts_of(source, order, [1] * 12)
    assert len(parts) == 3
    reads = [reading.read_markdown(source, store, part.positions, tmp_path / "parts", Telemetry.off()) for part in parts]

    assert [(entry["path"], text) for entry, text in reading.documents_of(reads)] == \
           [(entry["path"], text) for entry, text in one.documents()]


class Encoding:
    """A Tokenizer that notes how many packs were written at each document it encodes."""

    def __init__(self, tokenizer, written):
        self.tokenizer, self.written, self.seen = tokenizer, written, []

    def encode(self, text):
        self.seen.append(len(self.written))
        return self.tokenizer.encode(text)


def test_a_worker_fetches_the_next_batch_while_it_converts_the_current_one_and_goes_no_further(
        kit, repo, store, tmp_path, monkeypatch):
    # memory and disk hold one batch more at most, and the tokens a source gets do not depend on the batches
    sid = many_files(kit, repo, store, tmp_path, count=6)
    one_pool(kit, repo, 700)
    _, vocabulary, _ = build.build_tokenizer(repo, "small", store, log=lambda _: None)
    reads_here(monkeypatch, repo)
    source, count = repo.source(sid), 6
    whole = reading.read_tokens(source, store, list(range(count)), vocabulary, tmp_path / "whole", None)
    for ref in (store.root / "refs" / "source-tokens").glob("*"):
        ref.unlink()

    converting, written, asked = [threading.Event() for _ in range(count)], [], []
    fetch, write_pack = Store.fetch_files, reading._write_pack

    def fetched(self, places):
        asked.append(len(written))
        if len(asked) > 1:  # goes on only while the batch before it is being converted: it would wait in vain otherwise
            assert converting[len(asked) - 2].wait(30)
        fetch(self, places)

    def wrote(*arguments):
        converting[len(written)].set()
        written.append(write_pack(*arguments))
        return written[-1]

    monkeypatch.setattr(Store, "fetch_files", fetched)
    monkeypatch.setattr(reading, "_write_pack", wrote)
    encoding = Encoding(reading.tokenizer_of(vocabulary, store), written)
    monkeypatch.setattr(reading, "tokenizer_of", lambda *_: encoding)
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)  # each file a batch

    batched = reading.read_tokens(source, store, list(range(count)), vocabulary, tmp_path / "batched", None)

    assert asked == [0, 0, 1, 2, 3, 4]  # at each fetch, the packs done: one batch ahead, never two
    assert encoding.seen == [1, 2, 3, 4, 5, 6]  # a batch's documents become tokens before the next batch is converted

    def listed(summary):
        return [{name: value for name, value in entry.items() if name != "start"} for entry in summary["documents"]]

    assert listed(batched) == listed(whole)
    for entry in batched["documents"]:
        assert hashlib.sha256(store.read_range(batched["tokens"], entry["start"], entry["tokens"] * 4)).hexdigest() \
            == entry["token_sha256"]


def test_a_fetch_ahead_that_fails_fails_the_source_with_its_own_error(kit, repo, store, tmp_path, monkeypatch):
    sid = many_files(kit, repo, store, tmp_path, count=3)
    reads_here(monkeypatch, repo)
    fetch, calls = Store.fetch_files, []

    def fetched(self, places):
        calls.append(places)
        if len(calls) == 2:
            raise CorruptObject("the second batch's bundle is broken")
        fetch(self, places)

    monkeypatch.setattr(Store, "fetch_files", fetched)
    monkeypatch.setattr(reading, "BATCH_BYTES", 1)

    with pytest.raises(CorruptObject, match="second batch"):
        reading.read_markdown(repo.source(sid), store, [0, 1, 2], tmp_path / "markdown", Telemetry.off())


def test_the_packs_a_cache_holds_change_nothing_a_build_makes(kit, repo, store, tmp_path, monkeypatch):
    # an earlier read left b.txt alone in a pack: the same recipe gives the same documents, on any machine
    sid = lines_source(kit, repo, store, tmp_path / "lines", "lines", {"a.txt": LF, "b.txt": CRLF})
    every_file_chosen(kit, repo)
    clean = build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    reads_here(monkeypatch, repo)
    cache = store.cache / "markdown"
    source = repo.source(sid)
    assert len(reading.read_markdown(source, store, [1], cache, Telemetry.off()).packs) == 1
    read = reading.read_markdown(source, store, [0, 1], cache, Telemetry.off())
    assert [entry["path"] for entry, _ in read.documents()] == ["a.txt", "b.txt"]
    for ref in (store.root / "refs" / "source-tokens").glob("*"):
        ref.unlink()

    assert build_dataset(repo, "small", store, log=lambda _: None) == clean
    assert [d["id"] for d in document_list(store, clean)] == [f"{sid}/a.txt"]


def test_of_two_sources_with_the_same_markdown_the_first_by_id_keeps_it_whichever_is_read_first(
        kit, repo, store, tmp_path, monkeypatch):
    first, second = sorted(lines_source(kit, repo, store, tmp_path / name, name, {"a.txt": ends})
                           for name, ends in (("one", LF), ("two", CRLF)))
    every_file_chosen(kit, repo)
    in_order = choice.by_source
    monkeypatch.setattr(choice, "by_source", lambda *found: in_order(*found)[::-1])  # the last by id is read first
    backward = build_dataset(repo, "small", store, log=lambda _: None)  # trains the vocabulary too
    assert [found for found, _ in trained.read(store, repo.generated("lock", "main")["documents_sha256"])] == \
        [f"{first}/a.txt"]
    monkeypatch.setattr(choice, "by_source", in_order)
    for ref in (store.root / "refs" / "datasets").glob("*"):
        ref.unlink()

    assert build_dataset(repo, "small", store, log=lambda _: None) == backward
    assert [d["id"] for d in document_list(store, backward)] == [f"{first}/a.txt"]


def test_every_stage_of_a_vocabulary_says_how_far_it_is(kit, repo, store, tmp_path, monkeypatch):
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 700)
    monkeypatch.setattr(wpt_telemetry, "PROGRESS_SECONDS", 0)  # every advance prints
    printed = []
    build.build_tokenizer(repo, "small", store, log=lambda _: None,
                          telemetry=Telemetry.off(echo=lambda text, error: printed.append(text)))
    assert any(line.startswith("vocabulary sample: 1 of 1 parts") for line in printed)
    assert any(re.match(r"vocabulary chunks: (\d+) of \1 documents", line) for line in printed)


def test_every_reading_of_the_source_files_in_a_build_says_how_far_it_is(kit, repo, store, tmp_path, monkeypatch):
    # the index, the conversations and the tasks each read every source file: minutes on a large repository
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 700)
    monkeypatch.setattr(wpt_telemetry, "PROGRESS_SECONDS", 0)  # every advance prints
    printed = []
    build_dataset(repo, "small", store, log=lambda _: None,
                  telemetry=Telemetry.off(echo=lambda text, error: printed.append(text)))
    assert sum(line == "source files: reading and validating 1" for line in printed) == 3
    assert sum(line.startswith("source files: 1 of 1 sources") for line in printed) >= 3


def test_the_index_and_the_choice_read_no_text(kit, repo, store, tmp_path):
    # the files of the store are gone, only its records stay: the index and the choice still answer
    sid = many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 100)
    for bundle in (store.root / "bundles").rglob("*"):
        if bundle.is_file():
            bundle.unlink()
    index = choice.build(repo, store, log=lambda _: None)
    assert len(index.rows) == 30 and not index.failed
    found = choice.Choice(index, repo.recipe("mixture", "main"))
    assert sum(found.take({"en": 400})["bytes"]) >= 400 > found.taken["en"]
    assert [source.id for source in index.sources] == [sid]


def index_of(repo, store):
    return choice.build(repo, store, log=lambda _: None)


def same_index(one, other):
    return (one.rows.tobytes() == other.rows.tobytes() and one.tag_sets == other.tag_sets
            and one.bundles == other.bundles and one.duplicates == other.duplicates
            and one.sources == other.sources and not one.failed and not other.failed)


def no_worker(monkeypatch):
    def refused(*_):
        raise AssertionError("a source was computed again")

    monkeypatch.setattr(workers, "Pool", refused)


def test_a_second_index_computes_no_source_and_is_the_same(kit, repo, store, tmp_path, monkeypatch):
    # every build and every --plan asks for the index: only the sources' saved parts are read again
    many_files(kit, repo, store, tmp_path, "one")
    many_files(kit, repo, store, tmp_path, "two", tag="fr")
    first = index_of(repo, store)
    no_worker(monkeypatch)
    assert len(first.rows) == 60 and same_index(index_of(repo, store), first)


def test_an_index_computes_only_the_source_whose_json_changed(kit, repo, store, tmp_path):
    many_files(kit, repo, store, tmp_path, "one")
    sid = many_files(kit, repo, store, tmp_path, "two")
    index_of(repo, store)
    document = repo.read("source", sid).document
    repo.save("source", sid, {**document, "tags": [{"tags": ["fr"], "paths": ["**"]}]}, repo.version("source", sid))
    index = index_of(repo, store)
    assert index.tag_sets == [frozenset({"en"}), frozenset({"fr"})] and len(index.rows) == 60


def test_a_saved_index_part_is_not_used_for_another_code(kit, repo, store, tmp_path, monkeypatch):
    sid = many_files(kit, repo, store, tmp_path)
    key = choice.index_key(repo.source(sid), store)
    monkeypatch.setattr(choice, "code_of", lambda module: "another code")
    assert choice.index_key(repo.source(sid), store) != key


def test_the_same_files_placed_elsewhere_move_the_index_with_them(kit, repo, store, tmp_path):
    # the upstream manifest and the final tree stay the same: the file list alone says where the files lie
    sid = many_files(kit, repo, store, tmp_path)
    index_of(repo, store)
    key = upstream_key(repo.source(sid))
    moved = {path: (mode, dataclasses.replace(placed, bundle="elsewhere/000001"))
             for path, (mode, placed) in stored_files(key, store).items()}
    store.set_ref("upstream-files", key, store.put_bytes(file_list_bytes(moved)))
    assert index_of(repo, store).bundles == ["elsewhere/000001"]


def test_a_source_that_fails_in_the_index_stays_failed_and_named_until_it_works(kit, repo, store, tmp_path):
    good = many_files(kit, repo, store, tmp_path, "good")
    sid = many_files(kit, repo, store, tmp_path, "bad")
    key = upstream_key(repo.source(sid))
    held = store.get_ref("upstream", key)
    store.delete_ref("upstream", key)
    answers = [index_of(repo, store) for _ in range(2)]
    assert [[(source.id, type(error), str(error)) for source, error in index.failed] for index in answers] == \
        [[(sid, NotDownloaded, f"{sid}: its upstream files are not in {store}; `wpt dataset download {sid}` fetches "
           "them")]] * 2
    assert [source.id for source in answers[1].sources] == [good]
    store.set_ref("upstream", key, held)
    index = index_of(repo, store)
    assert not index.failed and len(index.rows) == 60


def test_a_build_reads_only_the_documents_its_pools_need(two_kilobyte_groups, kit, repo, store, tmp_path):
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 700)
    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert 0 < record["documents"]["total"] < 30
    assert record["pools"][0]["share"] <= record["pools"][0]["tokens"]


def test_a_source_added_later_moves_no_document_before_another(kit, repo, store, tmp_path):
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 100)
    mixture = repo.recipe("mixture", "main")

    def order():
        found = choice.Choice(choice.build(repo, store, log=lambda _: None), mixture)
        return [bytes(row) for row in found.index.rows["sha256"][found.orders["en"]]]

    before = order()
    many_files(kit, repo, store, tmp_path, "later")
    after = order()
    assert len(after) == 60 and [sha for sha in after if sha in set(before)] == before


def synthetic_index(bundles, per_bundle, size=1000, first=0):
    """An Index of one pool "en": per_bundle files of size bytes, one after the other, in each of the named bundles."""
    rows = np.zeros(len(bundles) * per_bundle, dtype=choice.ROW)
    for number, name in enumerate(bundles):
        part = rows[number * per_bundle:(number + 1) * per_bundle]
        part["bundle"] = first + number
        part["start"] = np.arange(per_bundle) * size
        part["bytes"] = size
        part["sha256"] = [hashlib.sha256(f"{name} {i}".encode()).digest() for i in range(per_bundle)]
    names = ["x"] * first + list(bundles)
    return choice.Index(rows, [], [frozenset({"en"})], names, 0)


def test_the_choice_takes_whole_frames_and_a_source_added_later_moves_no_group(tmp_path):
    from wpt_store.blocks import FRAME
    mixture = {"pools": [{"name": "en", "select": "en", "weight": 1}]}
    per_bundle = 4 * FRAME // 1000  # 4 frames of 1 KB files in each of 3 bundles: 12 frames, about 12,000 files
    index = synthetic_index(["s/0", "s/1", "s/2"], per_bundle)
    found = choice.Choice(index, mixture)
    rows = found.take({"en": 3 * FRAME})
    touched = len(set(zip(rows["bundle"].tolist(), (rows["start"] // FRAME).tolist())))
    assert 3 <= touched <= 4 and int(rows["bytes"].sum()) >= 0.99 * touched * FRAME  # its frames are full, not sparse
    assert touched <= int(rows["bytes"].sum()) // FRAME + 1  # about as many frames as its bytes need

    def groups(found):
        rows = found.index.rows[found.orders["en"]]
        names = np.array(found.index.bundles)[rows["bundle"]]
        pairs = list(zip(names.tolist(), (rows["start"] // FRAME).tolist()))
        return [pair for at, pair in enumerate(pairs) if not at or pairs[at - 1] != pair]

    before = groups(found)
    assert len(before) == 12
    # another source, whose bundle is numbered before the others: no group of the first moves before another
    later = synthetic_index(["t/0", "t/1"], per_bundle, first=0)
    shifted = synthetic_index(["s/0", "s/1", "s/2"], per_bundle, first=2)
    both = choice.Index(np.concatenate([later.rows, shifted.rows]), [], later.tag_sets, ["t/0", "t/1", "s/0", "s/1", "s/2"], 0)
    after = groups(choice.Choice(both, mixture))
    assert len(after) == 20 and [pair for pair in after if pair in set(before)] == before


def test_a_pool_short_of_its_share_draws_again_until_its_share_is_exact(two_kilobyte_groups, kit, repo, store, tmp_path, monkeypatch):
    # a first guess far too low: the rounds that follow measure the bytes a token takes, and the share is still met
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 2000)
    monkeypatch.setattr(choice, "BYTES_PER_TOKEN", 0.01)
    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert record["pools"][0]["epochs"] <= 1 and record["pools"][0]["share"] == 2000
    drawn = sum(len(tokens) for tokens, *_ in rows_of(store, record["shards"]))
    assert drawn >= 2000
    assert len(list(store.root.joinpath("refs", "source-tokens").iterdir())) > 1  # one object per round


def test_a_vocabulary_trains_on_its_sample_in_the_pools_proportions(two_kilobyte_groups, kit, repo, store, tmp_path):
    many_files(kit, repo, store, tmp_path)
    many_files(kit, repo, store, tmp_path, "other", tag="fr")
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 0.75},
                                                         {"name": "fr", "select": "fr", "weight": 0.25}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "sample_bytes": 12000, "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 20000,
                                           "held_out": 0})
    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock = repo.generated("lock", "main")
    listed = [document for document, _ in trained.read(store, lock["documents_sha256"])]
    assert lock["document_count"] == len(listed) < record["documents"]["total"]
    chosen = {entry["id"] for entry in document_list(store, record["id"])}
    assert set(listed) <= chosen  # drawn from the dataset's own documents
    english = sum("/en/" in document for document in listed)
    assert english > len(listed) - english > 0  # the pools' weights, to the whole groups the sample ends on


def test_markdown_read_for_the_vocabulary_is_not_read_again_for_the_tokens(kit, repo, store, tmp_path):
    many_files(kit, repo, store, tmp_path, count=5)
    one_pool(kit, repo, 2000)
    build.build_tokenizer(repo, "small", store, log=lambda _: None)
    assert (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json").exists()
    assert not list((store.root / "refs").glob("source-tokens/*"))  # it stopped before any token
    packs = sorted((store.cache / "markdown").rglob("*.zst"))
    assert packs
    record = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert record["documents"]["total"] == 5
    assert sorted((store.cache / "markdown").rglob("*.zst")) == packs  # no file was read into a new one


def test_a_vocabulary_build_goes_on_past_a_failing_source_and_its_lock_lists_it(kit, repo, store, corpus, tmp_path):
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    twice = broken_sources(kit, repo, store, tmp_path)
    build.build_tokenizer(repo, "small", store, log=lambda _: None)
    assert repo.generated("lock", "main")["left_out"] == [{"source": sid, "error": "FormatMapError"}
                                                          for sid in sorted(twice)]


def test_a_strict_build_stops_at_a_source_not_in_the_store_and_locks_no_vocabulary(kit, repo, store, corpus, tmp_path):
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    folder, _ = kit.git_project(tmp_path / "later", files={"en/later.txt": b"not downloaded yet. " * 30},
                                executable=set(), symlinks={})
    kit.add_source(repo, store, "example", "later", {
        "type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]}, fetch=False)
    with pytest.raises(DatasetError, match="NotDownloaded"):
        build_dataset(repo, "small", store, log=lambda _: None, strict=True)
    assert repo.generated("lock", "main") is None


def test_a_vocabulary_s_plan_gives_each_pool_s_part_of_its_sample(two_kilobyte_groups, kit, repo, store, tmp_path, monkeypatch, capsys):
    # the sample is far smaller than what the dataset's first round takes: the plan speaks of the sample
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 7000, sample_bytes=6000)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert main(["--root", str(repo.root), "build-tokenizer", "small", "--plan"]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("pool en:"))
    files, size = re.fullmatch(r"pool en: weight 1, (\d+) files, ([\d,]+) bytes of the sample", line).groups()
    assert int(files) >= 1 and int(size.replace(",", "")) <= 6000


@pytest.mark.parametrize("command", ["build", "build-tokenizer"])
def test_a_plan_says_what_a_build_would_do_and_does_nothing(kit, repo, store, tmp_path, monkeypatch, capsys, command):
    many_files(kit, repo, store, tmp_path)
    one_pool(kit, repo, 700)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert main(["--root", str(repo.root), command, "small", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "pool en: weight 1" in out and "no lock: it trains on a sample of" in out and "reads " in out
    assert ("draws again" in out) == (command == "build")  # a vocabulary has no rounds
    assert ("computes no token" in out) == (command == "build-tokenizer")
    assert repo.generated("lock", "main") is None
    assert not list((store.root / "refs").glob("datasets/*")) and not list((store.root / "refs").glob("source-tokens/*"))
    assert not (store.cache / "markdown").exists()
