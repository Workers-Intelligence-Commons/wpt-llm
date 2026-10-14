import json

import pytest

from wpt_dataset.build import build_dataset
from wpt_dataset.errors import LockMismatch, PoolError
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
    kit.write_recipe(repo, "tokenizer", "main", {"mixture": "main", "vocab_size": 400})
    kit.write_recipe(repo, "dataset", "small", {"mixture": "main", "tokenizer": "main",
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
    kit.write_recipe(repo, "tokenizer", "all", {"mixture": "all", "vocab_size": 300})
    kit.write_recipe(repo, "dataset", "split", {"mixture": "all", "tokenizer": "all",
                                                "total_tokens": 100, "held_out": 0.5})

    record = record_of(store, build_dataset(repo, "split", store, log=lambda _: None))
    held = {doc for _, _, doc, _, _ in rows_of(store, record["held_out_shards"])}
    trained = {doc for _, _, doc, _, _ in rows_of(store, record["shards"])}
    assert held and trained and not held & trained
    assert record["documents"]["held_out"] == len(held)
    assert record["contamination"]["held_out_documents"] == len(held)


def test_too_many_passes_over_a_pool_is_refused(kit, repo, store, corpus):
    setup_recipes(kit, repo, store, corpus, total_tokens=10_000_000)
    with pytest.raises(PoolError, match="more than the 4 allowed"):
        build_dataset(repo, "small", store, log=lambda _: None)


def test_a_rebuilt_vocabulary_must_match_its_lock(kit, repo, store, corpus, tmp_path):
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    build_dataset(repo, "small", store, log=lambda _: None)
    lock_path = repo.lock_path("main")
    lock = json.loads(lock_path.read_text())
    lock["vocabulary_sha256"] = "f" * 64  # as if another machine had trained a different vocabulary
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(LockMismatch):
        build_dataset(repo, "small", open_store((tmp_path / "fresh").as_uri()), log=lambda _: None)
