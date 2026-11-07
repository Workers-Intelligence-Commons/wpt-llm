import hashlib
import io
import os
import pickle
import subprocess
import tarfile

import boto3
import pytest
from moto import mock_aws

from wpt_store import (CorruptFile, File, Packing, SessionNeeded, copy_out, find_dataset, given, held, held_here,
                       has_model, index, keep_dataset, keep_failure, keep_model, latest_dataset, model_file, needed, new_id,
                       open_store, put_model, source_row)
from wpt_store.listing import listing
from wpt_store.sessions import remove_unowned_partials

MB = 1 << 20
KEY = "the-upstream-key"
ABOUT = {"key": KEY, "address": "https://example.org/a", "manifest": "m", "got": {"how": "downloaded"}}


@pytest.fixture(params=["local", "s3"])
def reopen(request, tmp_path, monkeypatch):
    """A function giving a new store object on the same store each call: a
    local folder, or an S3 bucket whose requests moto answers inside the test process."""
    if request.param == "local":
        yield lambda: open_store((tmp_path / "store").as_uri())
        return
    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        yield lambda: open_store("s3://wpt-test/store")


def text(seed: str, size: int) -> bytes:
    """About size bytes of text that compress, different for each seed."""
    words = [hashlib.sha256(f"{seed}{number}".encode()).hexdigest()[:7].encode() for number in range(4000)]
    block = b" ".join(words)
    return (block * (size // len(block) + 1))[:size]


def noise(size: int) -> bytes:
    return os.urandom(size)


def stored(store, source, files: dict, about=ABOUT, previous=None, pages=None, **options):
    """The source held in the store's session, with files ({path: bytes})."""
    packing = Packing(store, source, **options)
    for path, data in files.items():
        packing.keep(path, "100644", data)
    return packing.finish({path: "100644" for path in files}, pages or {}, about, previous)


def tar_of(root, session, source):
    """A source's tar as the zstd and tar programs read it, by hand."""
    return root / session / "sources" / f"{source}.tar.zst"


def test_a_source_s_tar_opens_by_hand_with_the_tar_program_whatever_its_paths_hold(tmp_path):
    # the point of the format: a person reaches a source with ordinary programs, this repository's code not needed
    store = open_store((tmp_path / "store").as_uri()).at("s1")
    long_path = "/".join(["folder-with-a-long-name"] * 12) + "/é-ü.txt"
    packing = Packing(store, "book")
    packing.keep("plain.txt", "100644", b"plain\n")
    packing.keep("same.txt", "100644", b"plain\n")  # same content: a hard link
    packing.keep("run.sh", "100755", b"#!/bin/sh\n")
    packing.keep(long_path, "100644", b"long\n")
    packing.keep("link", "120000", b"plain.txt")
    packing.finish({"plain.txt": "100644", "same.txt": "100644", "run.sh": "100755", long_path: "100644",
                    "link": "120000"}, {}, ABOUT)

    out = tmp_path / "opened"
    out.mkdir()
    decompressed = subprocess.run(["zstd", "-dc", str(tar_of(tmp_path / "store", "s1", "book"))], check=True,
                                  stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-xf", "-", "-C", str(out)], input=decompressed, check=True)
    assert (out / "plain.txt").read_bytes() == (out / "same.txt").read_bytes() == b"plain\n"
    assert (out / "plain.txt").stat().st_ino == (out / "same.txt").stat().st_ino
    assert os.access(out / "run.sh", os.X_OK)
    assert (out / long_path).read_bytes() == b"long\n"
    assert os.readlink(out / "link") == "plain.txt"
    assert {member.name for member in tarfile.open(fileobj=io.BytesIO(decompressed))} >= {"plain.txt", long_path}


def test_one_file_of_a_source_is_read_alone_by_range_and_the_frames_are_about_one_mib(reopen):
    store = reopen().at("s1")
    files = {f"{number}.txt": text(f"file {number}", 3 * MB) for number in range(4)}
    found = stored(store, "book", files)
    assert {path: store.read_file(placed) for path, (_, placed) in found.files.items()} == files

    other = reopen().at("s1")  # another machine, nothing of the tar on it
    other.cache = other.cache.parent / "elsewhere"
    other.cache.mkdir()
    _, placed = held(other, "book").files["2.txt"]
    assert other.read_file(placed) == files["2.txt"]
    table = other._table(other._found(_tar_file(found)))
    sizes = {after - before for before, after in zip(table.original, table.original[1:])}
    assert len(table.original) > 10 and max(sizes) <= MB + 64 * 1024


def _tar_file(found):
    return File(f"{found.session}/sources/{found.source}.tar.zst", found.sha256)


def test_a_file_whose_stored_bytes_changed_is_refused(tmp_path):
    root = tmp_path / "store"
    store = open_store(root.as_uri()).at("s1")
    found = stored(store, "book", {"a.txt": text("a", 2 * MB)})
    _, placed = found.files["a.txt"]
    path = tar_of(root, "s1", "book")
    other = stored(open_store(root.as_uri()).at("s2"), "book", {"a.txt": text("b", 2 * MB)})
    path.write_bytes(tar_of(root, "s2", "book").read_bytes())  # another's bytes under this name
    assert other.sha256 != found.sha256
    with pytest.raises(CorruptFile):
        open_store(root.as_uri()).at("s1").read_file(placed)


def test_fetch_files_copies_a_source_s_chosen_files_and_reads_them_from_the_copy(reopen, tmp_path):
    store = reopen().at("s1")
    files = {f"{number}.txt": text(f"file {number}", MB) for number in range(3)}
    found = stored(store, "book", files, stream=True)
    copies = reopen().at("s1").copies_in(tmp_path / "copies")
    wanted = [found.files["0.txt"][1], found.files["2.txt"][1]]
    copies.fetch_files(wanted)
    assert [copies.read_file(placed) for placed in wanted] == [files["0.txt"], files["2.txt"]]


def test_a_new_download_in_the_session_replaces_its_file_and_another_session_keeps_its_own(reopen):
    first = reopen().at("s1")
    old = stored(first, "book", {"a.txt": b"the first text\n"})
    newer = stored(first, "book", {"a.txt": b"the second text\n"}, previous=old)
    assert first.read_file(held(first, "book").files["a.txt"][1]) == b"the second text\n"
    assert newer.sha256 != old.sha256

    second = reopen().at("s2")
    assert held(second, "book") is None
    stored(second, "book", {"a.txt": b"another session's text\n"})
    assert first.read_file(held(first, "book").files["a.txt"][1]) == b"the second text\n"


def test_a_failed_download_keeps_the_file_the_session_held_and_records_the_error(reopen):
    store = reopen().at("s1")
    old = stored(store, "book", {"a.txt": b"the text\n"})
    failed = keep_failure(store, "book", old, "another-key", "https://example.org/b", OSError("no answer"),
                          [("https://example.org/b", OSError("no answer"))])

    assert failed.error["kind"] == "OSError" and failed.error["failures"][0]["message"] == "no answer"
    found = reopen().at("s1")
    assert held_here(found, "book").error["key"] == "another-key"
    assert found.read_file(held_here(found, "book").files["a.txt"][1]) == b"the text\n"
    # a source that never came records only its failure
    assert keep_failure(store, "new", None, KEY, "https://example.org/c", OSError("gone")).files == {}


def test_a_record_with_an_error_still_provides_the_good_file_of_its_download_and_a_good_file_of_the_old_session_is_found(
        reopen):
    store = reopen().at("s1")
    old = stored(store, "book", {"a.txt": b"the text\n"})
    failed = keep_failure(store, "book", old, KEY, "https://example.org/a", OSError("no answer"))
    assert not failed.holds(KEY) and failed.provides(KEY)  # not held (fetched again), but a build reads it
    assert not failed.provides("another-key")  # a good file of another download is no file of this one
    assert held(store, "book", KEY).provides(KEY)
    assert not keep_failure(store, "lost", None, KEY, "https://example.org/x", OSError("gone")).provides(KEY)

    # the old session's good file comes before this session's failure that has none
    old_session = reopen().at("old")
    stored(old_session, "lost", {"b.txt": b"old text\n"})
    new = reopen().at("new", start="old")
    keep_failure(new, "lost", None, KEY, "https://example.org/x", OSError("gone"))
    assert held(new, "lost").error  # asked for no key: the session's own record
    found = held(new, "lost", KEY)
    assert found.session == "old" and found.provides(KEY)
    assert held(new, "lost", "another-key").session == "new"  # no record provides it: the first there is


def test_a_later_success_clears_the_error(reopen):
    store = reopen().at("s1")
    failed = keep_failure(store, "book", None, KEY, "https://example.org/a", OSError("no answer"))
    assert failed.error
    stored(store, "book", {"a.txt": b"text\n"})
    assert held(reopen().at("s1"), "book").error is None


def test_a_command_started_from_another_id_reads_it_and_writes_under_its_own(reopen):
    old = reopen().at("old")
    stored(old, "book", {"a.txt": b"the old text\n"})
    keep_dataset(old, "x-aaaaaaaa", {"id": "d", "names": ["x"], "shards": []})

    store = reopen().at("new", start="old")
    found = held(store, "book")
    assert found.session == "old" and store.read_file(found.files["a.txt"][1]) == b"the old text\n"
    assert held_here(store, "book") is None  # the session itself holds nothing of it
    assert latest_dataset(store, "x")["id"] == "d" and find_dataset(store, "d")[0] == "old/datasets/x-aaaaaaaa/dataset.json"

    # what it writes records the ID it started from, and the old ID is left as it was
    written = keep_dataset(store, "y-bbbbbbbb", {"id": "e", "names": ["y"], "shards": []})
    assert written["from"] == "old"
    assert stored(store, "other", {"b.txt": b"b\n"}).record["from"] == "old"
    assert find_dataset(reopen().at("old"), "e") is None and held(reopen().at("old"), "other") is None

    # the session's own file comes first
    stored(store, "book", {"a.txt": b"the new text\n"})
    found = held(store, "book")
    assert found.session == "new" and store.read_file(found.files["a.txt"][1]) == b"the new text\n"


def test_a_store_view_sent_to_a_worker_process_stays_in_its_session(tmp_path):
    store = open_store((tmp_path / "store").as_uri()).at("new", start="old")
    again = pickle.loads(pickle.dumps(store))
    assert (again.session, again.start) == ("new", "old")


def test_every_session_needing_command_is_refused_with_no_id_and_a_named_failure(tmp_path):
    env_file = tmp_path / ".env"
    with pytest.raises(SessionNeeded):
        needed(environ={}, env_file=env_file)
    with pytest.raises(SessionNeeded):
        Packing(open_store((tmp_path / "store").as_uri()), "book")
    with pytest.raises(SessionNeeded):
        open_store((tmp_path / "store").as_uri()).needed()


def test_a_session_comes_from_the_option_then_the_environment_then_the_settings_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("WPT_SESSION=from-file\n", encoding="utf-8", newline="\n")
    assert given("from-option", {"WPT_SESSION": "from-env"}, env_file) == "from-option"
    assert given(None, {"WPT_SESSION": "from-env"}, env_file) == "from-env"
    assert given(None, {}, env_file) == "from-file"
    assert given(None, {}, tmp_path / "none") is None


def test_a_new_id_is_short_and_can_be_used_and_a_name_the_store_keeps_for_itself_cannot(tmp_path):
    assert len(new_id()) == 8 and new_id() != new_id()
    store = open_store((tmp_path / "store").as_uri())
    assert store.at(new_id()).session
    for reserved in ("index", "cache", "cut-key", "a/b", ".local"):
        with pytest.raises(ValueError):
            store.at(reserved)


def test_the_index_says_what_each_session_holds_and_is_rebuilt_from_the_files_alone(reopen):
    store = reopen().at("s1")
    one = stored(store, "book", {"a.txt": b"text\n"})
    failed = keep_failure(store, "lost", None, KEY, "https://example.org/x", OSError("no answer"))
    keep_dataset(store, "d-dddddddd", {"id": "d", "shards": []})
    keep_model(store, "m", {})
    stored(reopen().at("s2"), "book", {"a.txt": b"more text\n"})
    store.note([source_row(one), source_row(failed)])

    assert store.kind.list("index") or store.kind.list("index.zst")  # one file at the root, compressed when it pays
    states, sessions = listing(store)
    assert {(state.source, bool(state.error)) for state in states} == {("book", False), ("lost", True)}
    assert {row.session: row.held for row in sessions} == {"s1": {"datasets": 1, "models": 1, "sources": 2}}

    assert index.rebuild(store) == 5  # s2's source too: the files say so, whoever noted it
    states, sessions = listing(reopen())
    assert {(state.session, state.source) for state in states} == {("s1", "book"), ("s1", "lost"), ("s2", "book")}
    assert {row.session: row.held for row in sessions} == {"s1": {"datasets": 1, "models": 1, "sources": 2},
                                                           "s2": {"sources": 1}}

    store.kind.write("index", b"no index")  # a damaged index stops a reader, never a rebuild
    with pytest.raises(CorruptFile):
        listing(store)
    assert index.rebuild(store) == 5
    assert len(listing(store)[0]) == 3


def test_a_record_copied_from_a_session_names_the_id_this_session_started_from(reopen):
    first = reopen().at("bbbb", start="aaaa")
    assert keep_dataset(first, "d-dddddddd", {"id": "d", "names": ["d"], "shards": []})["from"] == "aaaa"
    second = reopen().at("cccc", start="bbbb")
    assert keep_dataset(second, "d-dddddddd", latest_dataset(second, "d"))["from"] == "bbbb"
    assert latest_dataset(reopen().at("cccc"), "d")["from"] == "bbbb"


def test_a_long_lived_reader_sees_what_another_process_downloaded(reopen):
    reader, other = reopen().at("s1"), reopen().at("s1")
    keep_failure(other, "book", None, KEY, "https://example.org/a", OSError("no answer"))
    assert held(reader, "book").error

    old = stored(other, "book", {"a.txt": text("old", 3 * MB)})
    found = held(reader, "book")
    assert found.error is None
    assert reader.read_file(found.files["a.txt"][1]) == text("old", 3 * MB)

    stored(other, "book", {"a.txt": text("new", 3 * MB)}, previous=old)  # same key: the tar replaced too
    found = held(reader, "book")
    assert reader.read_file(found.files["a.txt"][1]) == text("new", 3 * MB)


def test_a_store_whose_files_are_elsewhere_keeps_a_copy_of_the_tar_unless_the_download_streams(tmp_path, monkeypatch):
    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test",
                        "AWS_DEFAULT_REGION": "us-east-1", "XDG_CACHE_HOME": str(tmp_path / "cache"),
                        "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    def copies(store):
        return [path for path in store.cache.rglob("*.tar.zst*") if path.is_file()]

    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/store").at("s1")
        stored(store, "kept", {"a.txt": b"kept here\n"})
        assert len(copies(store)) == 1
        stored(store, "streamed", {"a.txt": b"not kept here\n"}, stream=True)
        assert len(copies(store)) == 1
        assert store.read_file(held(store, "streamed").files["a.txt"][1]) == b"not kept here\n"


def test_a_partial_tar_a_crashed_process_left_is_removed_by_the_next_run_and_a_discarded_one_leaves_nothing(tmp_path):
    store = open_store((tmp_path / "store").as_uri()).at("s1")
    packing = Packing(store, "book")
    packing.keep("a.txt", "100644", b"text\n")
    assert remove_unowned_partials(store.cache) == []  # its process lives
    packing.discard()
    assert held(store, "book") is None
    assert not [path for path in store.cache.iterdir() if "partial" in path.name]


def test_a_dataset_record_is_kept_under_its_session_and_the_latest_build_of_a_name_is_found_through_the_index(reopen):
    store = reopen().at("s1")
    assert latest_dataset(store, "d") is None
    first = keep_dataset(store, "d-11111111", {"id": "1" * 64, "names": ["d"], "shards": [{"key": "s1/datasets/d/0.parquet"}]})
    assert "time" in first  # stamped by the session
    keep_dataset(store, "other-22222222", {"id": "2" * 64, "names": ["other"], "shards": []})
    assert latest_dataset(reopen().at("s1"), "d")["shards"] == [{"key": "s1/datasets/d/0.parquet"}]
    second = keep_dataset(store, "d-33333333", {"id": "3" * 64, "names": ["d"], "shards": []})
    assert latest_dataset(reopen().at("s1"), "d")["id"] == second["id"]  # the later of two builds
    keep_dataset(store, "d-11111111", first)  # the first one found again by a later build: now the latest
    assert latest_dataset(reopen().at("s1"), "d")["id"] == first["id"]
    assert latest_dataset(reopen().at("s2"), "d") is None
    assert latest_dataset(reopen().at("s2", start="s1"), "d")["id"] == first["id"]
    # the session's own build comes before the one it started from, whatever their times
    own = keep_dataset(reopen().at("s2", start="s1"), "d-44444444", {"id": "4" * 64, "names": ["d"], "shards": []})
    assert latest_dataset(reopen().at("s2", start="s1"), "d")["id"] == own["id"]


def test_a_model_s_real_files_sit_in_its_folder_and_come_back_whole(reopen, tmp_path):
    folder = tmp_path / "model"
    (folder / "drafter").mkdir(parents=True)
    (folder / "config.json").write_bytes(b"{}\n")
    (folder / "drafter" / "weights.bin").write_bytes(noise(100_000))
    store = reopen().at("s1")
    keep_model(store, "fox", put_model(store, "fox", folder))

    found = reopen().at("s1")
    assert has_model(found, "fox") and not has_model(reopen().at("s2"), "fox")
    assert model_file(found, "fox", "config.json") == b"{}\n" and model_file(found, "fox", "no-such-file") is None
    assert any(key.startswith("s1/models/fox/drafter/weights.bin") for key in found.kind.list("s1/models/fox/"))
    copy_out(found, "fox", tmp_path / "copy")
    assert (tmp_path / "copy" / "drafter" / "weights.bin").read_bytes() == (folder / "drafter" / "weights.bin").read_bytes()
