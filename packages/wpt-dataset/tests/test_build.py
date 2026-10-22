import hashlib
import json

import pytest

from wpt_dataset.build import build_dataset, document_list
from wpt_dataset.cli import main
from wpt_dataset.errors import ChecksumMismatch, LockedDocumentsMissing, LockMismatch, PoolError, TokenizerVersionNeeded
from wpt_store import open_store
from wpt_store.shards import read_shard, token_hash
from wpt_tokenizer import Tokenizer

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
    kit.add_source(repo, store, "example", "corpus", {
        "type": "git", "address": corpus[0], "revision": corpus[1],
        "tags": [{"tags": ["fr"], "paths": ["fr/**"]}, {"tags": ["literature"], "paths": ["fr/lit/**"]},
                 {"tags": ["en"], "paths": ["en/**"]}]})
    kit.write_recipe(repo, "mixture", "main", {"pools": [
        {"name": "literature", "select": "literature", "weight": 0.25},
        {"name": "fr", "select": "fr && !(en)", "weight": 0.25},  # also matches fr/lit/: first pool wins
        {"name": "en", "select": "en", "weight": 0.5}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 656})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main",
                                                "total_tokens": total_tokens, "held_out": held_out})


def record_of(store, dataset_id):
    return json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))


def rows_of(store, shards):
    return [row for shard in shards for row in read_shard(store.path(shard["object"]))]


def test_build_fills_pools_and_is_repeatable_on_another_machine(kit, repo, store, corpus, tmp_path):
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
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
    fresh = open_store((tmp_path / "other-store").as_uri())
    assert build_dataset(repo, "small", fresh, log=lambda _: None) == dataset_id
    again = record_of(fresh, dataset_id)
    assert again["tokenizer"]["vocabulary"] == record["tokenizer"]["vocabulary"]
    assert [s["token_sha256"] for s in again["shards"]] == [s["token_sha256"] for s in record["shards"]]


def test_held_out_documents_never_reach_training(kit, repo, store, tmp_path):
    # ids contain the (random) temporary path, so which documents are held out
    # varies between runs: 40 documents at 50% make an empty side vanishingly rare
    folder, commit = kit.git_project(tmp_path / "many", files={
        f"t{i:02d}.txt": f"document number {i} says hello. ".encode() * 5 for i in range(40)}, executable=set(), symlinks={})
    kit.add_source(repo, store, "example", "many", {"type": "git", "address": folder, "revision": commit,
                                                    "tags": [{"tags": ["all"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "all", {"pools": [{"name": "all", "select": "all", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "all", {"version": 1, "mixture": "all", "vocab_size": 556})
    kit.write_model_recipe(repo, "split", {"mixture": "all", "tokenizer": "all",
                                                "total_tokens": 100, "held_out": 0.5})

    record = record_of(store, build_dataset(repo, "split", store, log=lambda _: None))
    held = {doc for _, _, doc, _, _ in rows_of(store, record["held_out_shards"])}
    trained = {doc for _, _, doc, _, _ in rows_of(store, record["shards"])}
    assert held and trained and not held & trained
    assert record["documents"]["held_out"] == len(held)
    assert record["contamination"]["held_out_documents"] == len(held)

    # maintainer's decision, 2026-10-22: the build keeps, found from the
    # dataset id, every document's id, Markdown sha256 and tag, so a task
    # reader learns the held-out texts without converting any source
    dataset_id = record["id"]
    listed = document_list(store, dataset_id)
    assert {entry["id"] for entry in listed if entry["tag"] == "held-out"} == held
    training = {entry["id"] for entry in listed if entry["tag"] == "training"}
    assert len(training) == record["documents"]["training"] and trained <= training  # the shards draw some
    for entry in listed:  # the sha256 of the document's Markdown, the text training reads
        number = int(entry["id"].rsplit("/t", 1)[1].removesuffix(".txt"))
        assert f"document number {number} says hello.".encode() in store.get_bytes(entry["sha256"])


def test_too_many_passes_over_a_pool_is_refused(kit, repo, store, corpus):
    setup_recipes(kit, repo, store, corpus, total_tokens=10_000_000)
    with pytest.raises(PoolError, match="more than the 4 allowed"):
        build_dataset(repo, "small", store, log=lambda _: None)


def test_the_locked_vocabulary_stays_until_its_version_is_raised(kit, repo, store, corpus, tmp_path):
    # many datasets and models share one vocabulary, so new sources must not change it;
    # only a person raising the recipe's version trains a new one
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    first = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json")
    lock = json.loads(lock_path.read_text())
    assert lock["vocabulary_sha256"] == first["tokenizer"]["vocabulary"] and len(lock["documents"]) == 10
    assert first["tokenizer"]["recipe_version"] == 1

    folder, commit = kit.git_project(tmp_path / "more", files={"en/new.txt": b"zebras graze quietly. " * 30},
                                     executable=set(), symlinks={})
    kit.add_source(repo, store, "example", "more", {"type": "git", "address": folder, "revision": commit,
                                                    "tags": [{"tags": ["en"], "paths": ["**"]}]})
    second = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert second["documents"]["total"] == 11  # the new source is in the dataset...
    assert second["tokenizer"]["vocabulary"] == first["tokenizer"]["vocabulary"]  # ...but not in the vocabulary
    assert json.loads(lock_path.read_text()) == lock
    # another machine rebuilds the same vocabulary from the documents the lock lists
    fresh = open_store((tmp_path / "fresh").as_uri())
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
    assert new_lock["vocabulary_sha256"] != lock["vocabulary_sha256"] and len(new_lock["documents"]) == 11


def test_a_locked_vocabulary_that_cannot_be_rebuilt_the_same_is_refused(kit, repo, store, corpus, tmp_path):
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    build_dataset(repo, "small", store, log=lambda _: None)
    lock_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json")
    lock = json.loads(lock_path.read_text())
    lock_path.write_text(json.dumps({**lock, "vocabulary_sha256": "f" * 64}))  # as if another machine trained another
    with pytest.raises(LockMismatch):
        build_dataset(repo, "small", open_store((tmp_path / "fresh").as_uri()), log=lambda _: None)

    # a document the vocabulary was trained on is gone, and the store holds neither
    lock_path.write_text(json.dumps({**lock, "documents": lock["documents"] + [["example-gone/a.txt", "e" * 64]]}))
    with pytest.raises(LockedDocumentsMissing, match="1 of the documents .* such as example-gone/a.txt"):
        build_dataset(repo, "small", open_store((tmp_path / "fresh-2").as_uri()), log=lambda _: None)


def test_tokenizer_check_round_trips_every_document_and_reports_each_source(kit, repo, store, corpus, tmp_path,
                                                                           monkeypatch, capsys):
    # shows whether text the vocabulary never saw still encodes compactly enough
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    build_dataset(repo, "small", store, log=lambda _: None)
    folder, commit = kit.git_project(tmp_path / "more", files={"new.txt": "zèbres ✓ broutent. ".encode() * 30},
                                     executable=set(), symlinks={})
    late = kit.add_source(repo, store, "example", "more", {"type": "git", "address": folder, "revision": commit})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())

    assert main(["--root", str(repo.root), "tokenizer-check", "main"]) == 0
    lines = capsys.readouterr().out.splitlines()
    corpus_line = next(line for line in lines if line.startswith("example-corpus-"))
    assert ": 10 documents, " in corpus_line and "not in the vocabulary" not in corpus_line
    assert next(line for line in lines if line.startswith(late)).endswith("(not in the vocabulary's training)")
    assert lines[-1].startswith("every document decodes back exactly: 11 documents, ")


def test_a_streamed_build_gives_the_same_dataset_and_keeps_no_source_files(kit, repo, store, corpus, tmp_path):
    # a corpus larger than the disk is built one source at a time: the store
    # keeps tokens, never the sources' files, and the dataset is the one a
    # normal build gives
    setup_recipes(kit, repo, store, corpus, total_tokens=2000, held_out=0.2)
    # one pool of all 10 documents, so the held-out share always leaves it some
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "all", "select": "fr || en", "weight": 1}]})
    streamed_store = open_store((tmp_path / "streamed").as_uri())
    streamed_id = build_dataset(repo, "small", streamed_store, log=lambda _: None, stream=True)  # trains the vocabulary
    normal_store = open_store((tmp_path / "normal").as_uri())
    normal_id = build_dataset(repo, "small", normal_store, log=lambda _: None)  # rebuilds it from the lock

    assert streamed_id == normal_id
    streamed, normal = record_of(streamed_store, streamed_id), record_of(normal_store, normal_id)
    for part in ("tokenizer", "sources", "documents", "pools", "contamination"):
        assert streamed[part] == normal[part]
    assert [s["token_sha256"] for s in streamed["shards"] + streamed["held_out_shards"]] == \
           [s["token_sha256"] for s in normal["shards"] + normal["held_out_shards"]]
    files = kit.run("git", "ls-tree", "-r", "--name-only", corpus[1], cwd=corpus[0]).decode().split()
    for path in files:
        content = kit.run("git", "show", f"{corpus[1]}:{path}", cwd=corpus[0])
        assert not streamed_store.has(hashlib.sha256(content).hexdigest())
    vocabulary = Tokenizer.from_bytes(streamed_store.get_bytes(streamed["tokenizer"]["vocabulary"]))
    for tokens, *_ in rows_of(streamed_store, streamed["shards"]):
        assert not streamed_store.has(hashlib.sha256(vocabulary.decode(tokens.tolist())).hexdigest())  # its Markdown


def test_a_failing_source_stops_the_build_unless_skipping_is_asked(kit, repo, store, corpus, tmp_path, monkeypatch,
                                                                    capsys):
    # with thousands of sources, one site rebuilding its file must not block every
    # build; but leaving a source out is a choice someone makes, and the record says so
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = build_dataset(repo, "small", store, log=lambda _: None)  # also locks the vocabulary
    changed = kit.add_source(repo, store, "example", "changed", {
        "type": "git", "address": corpus[0], "revision": corpus[1], "checksum": "sha256:" + "1" * 64,
        "tags": [{"tags": ["en"], "paths": ["**"]}]}, compute_checksum=False)

    with pytest.raises(ChecksumMismatch, match=changed):
        build_dataset(repo, "small", store, log=lambda _: None)
    for stream in (False, True):
        skipped = build_dataset(repo, "small", store, log=lambda _: None, stream=stream, skip_failed=True)
        record = record_of(store, skipped)
        assert [s["source"] for s in record["skipped"]] == [changed]
        assert record["skipped"][0]["error"].startswith("ChecksumMismatch: ")
        assert changed not in record["sources"] and record["documents"] == record_of(store, first)["documents"]
        assert skipped != first  # the id says something is missing

    # the check goes on past a failure, so one run names every source to fix
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    capsys.readouterr()
    assert main(["--root", str(repo.root), "check", "--fetch", *repo.source_ids()]) == 1
    out, err = capsys.readouterr()
    assert "Markdown documents" in out and f"1 of 2 fetched sources failed:\n{changed}: ChecksumMismatch: " in err
