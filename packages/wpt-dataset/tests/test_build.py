import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from wpt_dataset import build, conversations, license_check, markdown, sources, tasks, workers
from wpt_dataset.build import build_dataset, document_list, is_held_out
from wpt_dataset.cli import main
from wpt_dataset.errors import SourceGone, PoolError, TokenizerVersionNeeded, VocabularyMissing
from wpt_dataset.sources import stored_files, upstream_key
from wpt_store import CorruptObject, open_store
from wpt_store.shards import read_shard, token_hash
from wpt_telemetry import Telemetry
from wpt_tokenizer import Tokenizer
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
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 656})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main",
                                                "total_tokens": total_tokens, "held_out": held_out})
    return sid


def fresh_store(kit, repo, folder, sid, shared=None):
    """A new store (another machine) with the source downloaded into it,
    and the locked vocabularies of shared copied into it: a locked
    vocabulary reaches another machine through the store, never by
    training again."""
    fresh = open_store(folder.as_uri())
    kit.download(repo, fresh, sid)
    for name in repo.names("lock") if shared is not None else ():
        locked = repo.generated("lock", name)["vocabulary_sha256"]
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

    # the build keeps, found from the
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
    kit.write_recipe(repo, "mixture", "all", {"pools": [{"name": "all", "select": "all", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "all", {"version": 1, "mixture": "all", "vocab_size": 556})
    kit.write_model_recipe(repo, "split", {"mixture": "all", "tokenizer": "all", "total_tokens": 100,
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
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    first = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json")
    lock = json.loads(lock_path.read_text())
    assert lock["vocabulary_sha256"] == first["tokenizer"]["vocabulary"] and len(lock["documents"]) == 10
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


def test_a_change_of_the_tokenizer_code_asks_for_no_version_and_leaves_the_lock_as_it_is(kit, repo, store, corpus,
                                                                                         monkeypatch):
    # the lock holds no version of the tokenizer code; the code's hash is in the fingerprints, so changed code
    # builds the locked vocabulary again, and only a change of the recipe asks a person for a version
    setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    first = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    lock_path = repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json"
    locked = lock_path.read_bytes()
    assert set(json.loads(locked)) == {"recipe", "documents", "vocabulary_sha256"}
    assert "version" not in first["tokenizer"] and "recipe_version" in first["tokenizer"]

    monkeypatch.setattr(build, "code", lambda: "another tokenizer code")
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
    assert next(line for line in lines if line.startswith(late)).endswith("(not in the vocabulary's training)")
    assert lines[-1].startswith("every document decodes back exactly: 11 documents, ")


def test_a_streamed_build_gives_the_same_dataset_and_keeps_no_markdown(kit, repo, store, corpus, tmp_path):
    # a streamed build keeps tokens in the store, never the documents'
    # Markdown, and the dataset is the one a normal build gives
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000, held_out=0.2)
    # one pool of all 10 documents, so the held-out share always leaves it some
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "all", "select": "fr || en", "weight": 1}]})
    streamed_store = fresh_store(kit, repo, tmp_path / "streamed", sid)
    streamed_id = build_dataset(repo, "small", streamed_store, log=lambda _: None, stream=True)  # trains the vocabulary
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
        without = build_dataset(repo, "small", store, log=lambda _: None, stream=stream)  # no --skip-failed needed
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
    [missing] = [e["attributes"]["skipped"] for e in events
                 if e["kind"] == "span_end" and e["name"] == "source" and "skipped" in e["attributes"]]
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
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 560})
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
                   f"{license_check.text_id(files['other/NOTICE'])}; `wpt-dataset add-license-text` adds it, "
                   f"under review)"}]
    assert [(found["place"], found["matched"]) for found in check["statements"]] == [
        ({"file": "LICENSE"}, True), ({"file": "LICENSE"}, True), ({"file": "other/NOTICE"}, False)]
    assert f"{sid}: left out 3 files (entry), 1 files (not-text), 1 files (uncovered) (see the record's `check`)" \
        in logged


def test_a_broken_source_stops_the_build_unless_skipping_is_asked(kit, repo, store, corpus, tmp_path):
    # a source in the store whose files are broken is a fault to look at:
    # leaving it out is a choice someone makes, and the record says so
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = build_dataset(repo, "small", store, log=lambda _: None)  # also locks the vocabulary
    folder, commit = kit.git_project(tmp_path / "other", files={"en/other.txt": b"owls hunt at night. " * 30},
                                     executable=set(), symlinks={})
    broken = kit.add_source(repo, store, "example", "broken", {"type": "git", "address": folder, "tags": [{"tags": ["en"], "paths": ["**"]}]})
    [placed] = {placed for _, placed in stored_files(upstream_key(repo.source(broken)), store).values()}
    for bundle in (store.root / "bundles").glob(f"{placed.bundle}*"):
        bundle.write_bytes(bytes(b ^ 0xFF for b in bundle.read_bytes()))  # every byte changed

    with pytest.raises(CorruptObject):
        build_dataset(repo, "small", store, log=lambda _: None)
    for stream in (False, True):
        skipped = build_dataset(repo, "small", store, log=lambda _: None, stream=stream, skip_failed=True)
        record = record_of(store, skipped)
        assert [s["source"] for s in record["skipped"]] == [broken]
        assert record["skipped"][0]["error"].startswith("CorruptObject: ")
        assert record["documents"] == record_of(store, first)["documents"]


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
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 600})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 3000,
                                           "held_out": 0.2})
    build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    built, log = {}, tmp_path / "ends.txt"
    for cores in (1, 16):  # 1: one worker process at a time; 16: 12 at once, the first source ending last
        on_cores(monkeypatch, cores)
        if cores == 16:
            ended_out_of_order = first_ends_last(monkeypatch, build, "_tokenized" if stream else "_converted",
                                                 sorted(ids)[0], log)
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", ids[0], shared=store)
        kit.download(repo, fresh, *ids[1:])
        with Telemetry.start(fresh, "wpt-dataset-build", [], "small", env_file=tmp_path / ".env",
                             environ={}) as telemetry:
            dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None, stream=stream, telemetry=telemetry)
        built[cores] = dataset_id, record_of(fresh, dataset_id), document_list(fresh, dataset_id)
        events = [json.loads(line) for line in fresh.read_record(telemetry.folder, "events.jsonl").splitlines()]
        at_once = [e["attributes"]["limit"] for e in events if e["kind"] == "event" and e["name"] == "sources.at_once"]
        assert at_once and set(at_once) == {max(1, int(cores * 0.75))}
        kind = "source-tokens" if stream else "source-trees"  # what each source's conversion stores once
        refs = [source_of(events, e["span"]) for e in events
                if e["kind"] == "event" and e["name"] == "store.ref_set" and e["attributes"]["ref"].startswith(kind)]
        assert sorted(refs) == sorted(ids)  # recorded by the worker, under its source's span
    assert ended_out_of_order(log, sorted(ids)[0])
    (one_id, one, one_list), (many_id, many, many_list) = built[1], built[16]
    assert many_id == one_id and many_list == one_list
    assert one["documents"]["duplicates_removed"] == 3 and len(one["sources"]) == 4
    for part in ("sources", "documents", "pools", "contamination"):
        assert many[part] == one[part]
    assert [s["token_sha256"] for s in many["shards"] + many["held_out_shards"]] == \
           [s["token_sha256"] for s in one["shards"] + one["held_out_shards"]]


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
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 300,
                                           "held_out": 0})
    record_file = hashlib.sha256(records).hexdigest()

    build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    assert not list(store.cache.rglob(record_file))  # its records file was read from a copy, now gone
    normal = fresh_store(kit, repo, tmp_path / "normal", sid, shared=store)
    build_dataset(repo, "small", normal, log=lambda _: None)
    assert list(normal.cache.rglob(record_file))  # without --stream, the copy stays in the store's cache


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
    store.put_bytes(whole.get_bytes(held["tokenizer"]["vocabulary"]))  # and so is this machine

    here = build_dataset(repo, "small", store, log=lambda _: None)
    assert build_dataset(repo, "small", other, log=lambda _: None) == here
    missing = record_of(store, here)
    assert [s["source"] for s in missing["skipped"]] == [talks]
    assert held["fingerprints"]["text"] == missing["fingerprints"]["text"]
    assert held["shards"] == missing["shards"]


def test_a_dataset_whose_record_lacks_a_field_is_built_again(kit, repo, store, corpus):
    # a dataset whose record a build of another shape wrote is built again, so
    # wpt-train, which reads the current shape only, can use it
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
