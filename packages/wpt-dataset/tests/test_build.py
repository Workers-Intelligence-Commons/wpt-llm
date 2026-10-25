import hashlib
import json
import time

import pytest

from wpt_dataset import build, conversations, sources
from wpt_dataset.build import build_dataset, document_list, is_held_out
from wpt_dataset.cli import main
from wpt_dataset.errors import ChecksumMismatch, LockedDocumentsMissing, LockMismatch, PoolError, TokenizerVersionNeeded
from wpt_dataset.sources import stored_files
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
    return sid


def fresh_store(kit, repo, folder, sid):
    """A new store (another machine) with the source downloaded into it."""
    fresh = open_store(folder.as_uri())
    kit.download(repo, fresh, sid)
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
    fresh = fresh_store(kit, repo, tmp_path / "other-store", sid)
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
            "type": "git", "address": folder, "revision": commit, "held_out_by": pattern,
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
    more = kit.add_source(repo, store, "example", "more", {"type": "git", "address": folder, "revision": commit,
                                                           "tags": [{"tags": ["en"], "paths": ["**"]}]})
    second = record_of(store, build_dataset(repo, "small", store, log=lambda _: None))
    assert second["documents"]["total"] == 11  # the new source is in the dataset...
    assert second["tokenizer"]["vocabulary"] == first["tokenizer"]["vocabulary"]  # ...but not in the vocabulary
    assert json.loads(lock_path.read_text()) == lock
    # another machine rebuilds the same vocabulary from the documents the lock lists
    fresh = fresh_store(kit, repo, tmp_path / "fresh", sid)
    kit.download(repo, fresh, more)
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
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=1000)
    build_dataset(repo, "small", store, log=lambda _: None)
    lock_path = (repo.root / "data" / "recipes" / "tokenizers" / "main.lock.json")
    lock = json.loads(lock_path.read_text())
    lock_path.write_text(json.dumps({**lock, "vocabulary_sha256": "f" * 64}))  # as if another machine trained another
    with pytest.raises(LockMismatch):
        build_dataset(repo, "small", fresh_store(kit, repo, tmp_path / "fresh", sid), log=lambda _: None)

    # a document the vocabulary was trained on is gone, and the store holds neither
    lock_path.write_text(json.dumps({**lock, "documents": lock["documents"] + [["example-gone/a.txt", "e" * 64]]}))
    with pytest.raises(LockedDocumentsMissing, match="1 of the documents .* such as example-gone/a.txt"):
        build_dataset(repo, "small", fresh_store(kit, repo, tmp_path / "fresh-2", sid), log=lambda _: None)


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


def test_a_streamed_build_gives_the_same_dataset_and_keeps_no_markdown(kit, repo, store, corpus, tmp_path):
    # a streamed build keeps tokens in the store, never the documents'
    # Markdown, and the dataset is the one a normal build gives
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=2000, held_out=0.2)
    # one pool of all 10 documents, so the held-out share always leaves it some
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "all", "select": "fr || en", "weight": 1}]})
    streamed_store = fresh_store(kit, repo, tmp_path / "streamed", sid)
    streamed_id = build_dataset(repo, "small", streamed_store, log=lambda _: None, stream=True)  # trains the vocabulary
    normal_store = fresh_store(kit, repo, tmp_path / "normal", sid)
    normal_id = build_dataset(repo, "small", normal_store, log=lambda _: None)  # rebuilds it from the lock

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
    changed = kit.add_source(repo, store, "example", "changed", {
        "type": "git", "address": corpus[0], "revision": corpus[1], "checksum": "sha256:" + "1" * 64,
        "tags": [{"tags": ["en"], "paths": ["**"]}]}, compute_checksum=False)
    with pytest.raises(ChecksumMismatch, match=changed):  # upstream no longer gives its checksum: nothing kept
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
    assert "Markdown documents" in out and f"1 of 2 fetched sources failed:\n{changed}: ChecksumMismatch: " in err


def test_a_broken_source_stops_the_build_unless_skipping_is_asked(kit, repo, store, corpus, tmp_path):
    # a source in the store whose files are broken is a fault to look at:
    # leaving it out is a choice someone makes, and the record says so
    setup_recipes(kit, repo, store, corpus, total_tokens=2000)
    first = build_dataset(repo, "small", store, log=lambda _: None)  # also locks the vocabulary
    folder, commit = kit.git_project(tmp_path / "other", files={"en/other.txt": b"owls hunt at night. " * 30},
                                     executable=set(), symlinks={})
    broken = kit.add_source(repo, store, "example", "broken", {"type": "git", "address": folder, "revision": commit,
                                                               "tags": [{"tags": ["en"], "paths": ["**"]}]})
    [placed] = {placed for _, placed in stored_files(repo.source(broken).data["checksum"], store).values()}
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


@pytest.mark.parametrize("stream", [False, True])
def test_sources_converted_at_once_give_the_dataset_converted_one_at_a_time(kit, repo, store, tmp_path, monkeypatch,
                                                                            stream):
    # sources and documents are worked on several at once, finishing in any
    # order; the dataset id, the documents and the shards must not change, and
    # the record still ties each ref a source's conversion sets to that source
    ids = []
    for n in range(4):
        files = {f"en/{n}-{i}.txt": f"source {n} text {i}: the wind blows over hill {i}. ".encode() * 20
                 for i in range(3)}
        files["en/shared.txt"] = b"every source holds this same page. " * 20  # the first source keeps it
        folder, commit = kit.git_project(tmp_path / f"upstream-{n}", files=files, executable=set(), symlinks={})
        ids.append(kit.add_source(repo, store, "example", f"part-{n}", {
            "type": "git", "address": folder, "revision": commit, "tags": [{"tags": ["en"], "paths": ["**"]}]}))
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 600})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 3000,
                                           "held_out": 0.2})
    build_dataset(repo, "small", store, log=lambda _: None)  # locks the vocabulary
    order = {sid: n for n, sid in enumerate(ids)}
    converted, tokenized = build._converted, build.source_tokens

    def last_first(source, *args):  # the later a source, the sooner it is done
        time.sleep(0.05 * (len(ids) - order[source.id]))
        return converted(source, *args)

    def tokens_last_first(repo_, source, *args):
        time.sleep(0.05 * (len(ids) - order[source.id]))
        return tokenized(repo_, source, *args)

    monkeypatch.setattr(build, "_converted", last_first)
    monkeypatch.setattr(build, "source_tokens", tokens_last_first)
    built = {}
    for cores in (1, 16):  # 1: one at a time; 16: 12 at once
        monkeypatch.setattr(build, "Sizing", lambda memory, what, cores=cores: Sizing(memory, what, Machine(cores)))
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", ids[0])
        kit.download(repo, fresh, *ids[1:])
        with Telemetry.start(fresh, "wpt-dataset-build", [], "small", env_file=tmp_path / ".env",
                             environ={}) as telemetry:
            dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None, stream=stream, telemetry=telemetry)
        built[cores] = dataset_id, record_of(fresh, dataset_id), document_list(fresh, dataset_id)
        events = [json.loads(line) for line in fresh.read_record(telemetry.folder, "events.jsonl").splitlines()]
        spans = {e["span"]: e["attributes"]["source"] for e in events
                 if e["kind"] == "span_start" and e["name"] == "source"}
        kind = "source-tokens" if stream else "source-trees"  # what each source's conversion stores once
        refs = [spans.get(e["span"]) for e in events
                if e["kind"] == "event" and e["name"] == "store.ref_set" and e["attributes"]["ref"].startswith(kind)]
        assert sorted(refs) == sorted(ids)
    (one_id, one, one_list), (many_id, many, many_list) = built[1], built[16]
    assert many_id == one_id and many_list == one_list
    assert one["documents"]["duplicates_removed"] == 3 and len(one["sources"]) == 4
    for part in ("sources", "documents", "pools", "contamination"):
        assert many[part] == one[part]
    assert [s["token_sha256"] for s in many["shards"] + many["held_out_shards"]] == \
           [s["token_sha256"] for s in one["shards"] + one["held_out_shards"]]


def test_a_streamed_build_leaves_no_copy_of_a_source_s_files_on_this_machine(kit, repo, store, tmp_path):
    # --stream is for data larger than the disk: what a source's conversion
    # copies out of the store goes before the next source
    records = "".join(json.dumps({"id": str(i), "text": f"record {i} tells of the sea. " * 10}) + "\n"
                      for i in range(5)).encode()
    folder, commit = kit.git_project(tmp_path / "records", files={"data/part.jsonl": records}, executable=set(),
                                     symlinks={})
    sid = kit.add_source(repo, store, "example", "records", {
        "type": "git", "address": folder, "revision": commit, "formats": [{"format": "records", "paths": ["**"], "text": "text", "id": "id"}],
        "tags": [{"tags": ["en"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "main", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "main", {"version": 1, "mixture": "main", "vocab_size": 560})
    kit.write_model_recipe(repo, "small", {"mixture": "main", "tokenizer": "main", "total_tokens": 300,
                                           "held_out": 0})
    record_file = hashlib.sha256(records).hexdigest()

    build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    assert not list(store.cache.rglob(record_file))  # its records file was read from a copy, now gone
    normal = fresh_store(kit, repo, tmp_path / "normal", sid)
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
        "type": "git", "address": folder, "revision": commit,
        "formats": [{"format": "chat", "paths": ["*.jsonl"], "tree": {
            "id": "id", "parent": "parent", "role": "role", "text": "text",
            "roles": {"prompter": "user", "assistant": "assistant"}}}],
        "tags": [{"tags": ["answer-as-assistant"], "paths": ["**"]}]})
    messages_file = hashlib.sha256(messages).hexdigest()

    streamed_id = build_dataset(repo, "small", store, log=lambda _: None, stream=True)
    assert not list(store.cache.rglob(messages_file))  # read from a copy, now gone
    normal = fresh_store(kit, repo, tmp_path / "normal", sid)
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
        "type": "git", "address": folder, "revision": commit,
        "formats": [{"format": "chat", "paths": ["*.jsonl"], "tree": {
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
    other = fresh_store(kit, repo, tmp_path / "other", sid)
    for missing_it in (store, other):  # a store without the vocabulary trains it again, which needs every source
        missing_it.put_bytes(whole.get_bytes(held["tokenizer"]["vocabulary"]))

    here = build_dataset(repo, "small", store, log=lambda _: None)
    assert build_dataset(repo, "small", other, log=lambda _: None) == here
    missing = record_of(store, here)
    assert [s["source"] for s in missing["skipped"]] == [talks]
    assert held["fingerprints"]["text"] == missing["fingerprints"]["text"]
    assert held["shards"] == missing["shards"]


def test_a_dataset_whose_record_has_another_shape_is_built_again(kit, repo, store, corpus):
    # a dataset whose record a build of another shape wrote is built again, so
    # wpt-train, which reads the current shape only, can use it
    setup_recipes(kit, repo, store, corpus, total_tokens=500)
    dataset_id = build_dataset(repo, "small", store, log=lambda _: None)
    current = record_of(store, dataset_id)
    older = {k: v for k, v in current.items() if k != "fingerprints"} | {"schema": build.RECORD_SCHEMA - 1}
    store.set_ref("datasets", dataset_id, store.put_bytes(json.dumps(older).encode()))

    assert build_dataset(repo, "small", store, log=lambda _: None) == dataset_id
    assert record_of(store, dataset_id)["schema"] == build.RECORD_SCHEMA
    assert record_of(store, dataset_id)["fingerprints"] == current["fingerprints"]


@pytest.mark.parametrize("stream", [False, True])
def test_conversations_read_at_once_give_the_dataset_read_one_at_a_time(kit, repo, store, corpus, tmp_path,
                                                                        monkeypatch, stream):
    # chat sources are read several at once, finishing in any order; the
    # conversations and the dataset id must be the ones read one at a time give
    sid = setup_recipes(kit, repo, store, corpus, total_tokens=500)
    ids = [chat_source(kit, repo, store, tmp_path, f"talks-{n}", text=f"Answer number {n}.") for n in range(4)]
    order = {talk: n for n, talk in enumerate(ids)}
    read = conversations.of_file

    def last_first(source_id, *args):  # the later a source, the sooner it is done
        time.sleep(0.05 * (len(ids) - order[source_id]))
        return read(source_id, *args)

    monkeypatch.setattr(conversations, "of_file", last_first)
    built = {}
    for cores in (1, 16):  # 1: one at a time; 16: 12 at once
        monkeypatch.setattr(sources, "Sizing", lambda memory, what, cores=cores: Sizing(memory, what, Machine(cores)))
        fresh = fresh_store(kit, repo, tmp_path / f"store-{cores}", sid)
        kit.download(repo, fresh, *ids)
        dataset_id = build_dataset(repo, "small", fresh, log=lambda _: None, stream=stream)
        record = record_of(fresh, dataset_id)
        built[cores] = dataset_id, record["conversations"], fresh.get_bytes(record["conversations"]["object"])
    assert built[1] == built[16]
    assert built[1][1]["sources"] == sorted(ids) and built[1][1]["count"] == 4

