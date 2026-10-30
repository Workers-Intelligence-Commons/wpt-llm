import os
import threading

import boto3
import pytest
from moto import mock_aws

from wpt_store import CorruptObject, MissingObject, Session, file_list_bytes, open_store, parse_file_list, sessions

MB = 1 << 20


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
        yield lambda: open_store("s3://wpt-test/sessions")


def named(session, placed, action=None):
    """Name a file list of {path: Placed}, as a download names the files it
    keeps: the end keeps only named files."""
    session.name({path: ("100644", where) for path, where in placed.items()}, action or (lambda listed: None))


def text(name, size):
    """size bytes of a downloaded text file named name."""
    line = f"{name}: a line of a downloaded text\n".encode()
    return (line * (size // len(line) + 1))[:size]


def test_a_session_s_files_go_into_bundles_cut_between_files_and_read_back_one_by_one(reopen):
    store = reopen()
    files = {"a.txt": text("a", 2 * MB), "b.txt": text("b", 1 * MB + 5), "big.txt": text("big", 9 * MB),
             "c.txt": text("c", 3)}
    with Session(store, "20261025T120000Z-download-x-0001", "download", "abc123", False, bundle=4 * MB) as session:
        placed = {path: session.add(data) for path, data in files.items()}
        named(session, placed)
        session.line("one-source", "downloaded", "abc123", got={"commit": "r1"}, address="https://example.org/one",
                     size=sum(map(len, files.values())))
    # a bundle closes once full, only between files; a file bigger than a bundle gets its own
    assert [placed[path].bundle.rsplit("/", 1)[1] for path in files] == ["000001", "000001", "000002", "000003"]
    assert (placed["b.txt"].start, placed["c.txt"].start) == (2 * MB, 0)

    later = reopen()  # another process, another machine for S3
    listed = parse_file_list(file_list_bytes({path: ("100644", where) for path, where in placed.items()}))
    for path, (mode, where) in listed.items():
        assert mode == "100644" and later.read_file(where) == files[path]

    with later.database() as found:
        assert found.execute("SELECT folder, command, wpt_commit, dirty, ended IS NOT NULL FROM sessions").fetchall() \
            == [("20261025T120000Z-download-x-0001", "download", "abc123", 0, 1)]
        assert found.execute("SELECT source, got, address, outcome, size, code FROM lines").fetchall() \
            == [("one-source", '{"commit": "r1"}', "https://example.org/one", "downloaded",
                 sum(map(len, files.values())), "abc123")]


def test_a_crash_loses_only_the_bundle_being_filled_and_the_lines_since_the_last_sealed(reopen):
    store = reopen()
    session = Session(store, "crashed", "download", "abc123", True, bundle=2 * MB)
    sealed = session.add(text("first", 2 * MB))  # fills the first bundle: sealed at once
    session.line("first", "downloaded", "abc123")  # recorded with the next bundle, which never comes
    unsealed = session.add(text("second", MB))
    # the process ends here, without end()
    later = reopen()
    assert later.read_file(sealed) == text("first", 2 * MB)
    with pytest.raises(MissingObject):
        later.read_file(unsealed)
    with later.database() as found:
        assert found.execute("SELECT folder, dirty, ended FROM sessions").fetchall() == [("crashed", 1, None)]
        assert found.execute("SELECT count(*) FROM lines").fetchone() == (0,)


def test_a_seal_puts_the_rows_in_the_database_only_once_the_bundles_since_the_last_put_outweigh_it(reopen):
    # every put reads and stores the whole database: bundles much smaller than it leave their rows in the
    # session's file until enough bytes were stored, or until the end
    store = reopen()
    with Session(store, "growing", "download", "abc123", False, bundle=MB) as session:
        session.line("small", "downloaded", "abc123")
        named(session, {"small.txt": session.add(text("small", MB))})  # compresses to far less than the database
        with reopen().database() as found:
            assert found.execute("SELECT count(*) FROM lines").fetchone() == (0,)
        session.line("large", "downloaded", "abc123")
        named(session, {"large.bin": session.add(os.urandom(40 * MB))})  # does not compress: outweighs it
        with reopen().database() as found:
            assert found.execute("SELECT source FROM lines ORDER BY number").fetchall() == [("small",), ("large",)]
        session.line("last", "reused", "abc123")
    with reopen().database() as found:
        assert found.execute("SELECT count(*) FROM lines").fetchone() == (3,)


def test_sessions_that_replace_the_database_at_once_both_keep_their_rows(reopen):
    # two machines end a session at the same moment: the second one's replace
    # finds the database changed since it read it, reads it again and merges
    first, second = reopen(), reopen()
    real = first.kind.replace
    raced = []

    def another_session_first(key, data, version):
        if not raced:
            raced.append(True)
            with Session(second, "other", "download", "def456", False) as other:
                other.line("other-source", "reused", "abc123")
        return real(key, data, version)

    first.kind.replace = another_session_first
    with Session(first, "mine", "download", "abc123", False) as mine:
        mine.line("my-source", "failed", "abc123", address="https://example.org/mine")
    assert raced
    with reopen().database() as found:
        assert found.execute("SELECT session, source, outcome FROM lines ORDER BY session").fetchall() \
            == [("mine", "my-source", "failed"), ("other", "other-source", "reused")]
        assert [row[0] for row in found.execute("SELECT folder FROM sessions ORDER BY folder")] == ["mine", "other"]


def test_a_mark_asks_for_a_new_download_until_one_is_recorded_after_it(reopen):
    # a person marks one source, or every source, for fresh content: a session that starts later sees it
    # marked, until a line says it was downloaded after the mark; a reuse or a failure leaves the mark
    store = reopen()
    with Session(store, "first", "download", "abc123", False) as session:
        session.line("one", "downloaded", "abc123")
        session.line("two", "downloaded", "abc123")
    store.mark(["one"])
    with Session(reopen(), "second", "download", "abc123", False) as session:
        assert (session.marked("one"), session.marked("two")) == (True, False)
        session.line("one", "failed", "abc123")
    with Session(reopen(), "third", "download", "abc123", False) as session:
        assert session.marked("one")
        session.line("one", "downloaded", "abc123")
    store.mark()  # every source
    with Session(reopen(), "fourth", "download", "abc123", False) as session:
        assert (session.marked("one"), session.marked("two"), session.marked("never-seen")) == (True, True, True)
        session.line("two", "downloaded", "abc123")
    with Session(reopen(), "fifth", "download", "abc123", False) as session:
        assert (session.marked("one"), session.marked("two")) == (True, False)


def test_lines_of_the_shape_that_pinned_a_revision_and_a_checksum_keep_them_in_what_they_got(tmp_path):
    # a store database written before lines recorded what a download got: its columns revision and checksum
    # move into got, and new lines go in beside them
    import sqlite3

    from wpt_store.blocks import compressed

    store = open_store((tmp_path / "store").as_uri())
    old = sqlite3.connect(":memory:")
    old.executescript("""
        CREATE TABLE sessions (folder TEXT PRIMARY KEY, command TEXT NOT NULL, started TEXT NOT NULL, ended TEXT,
                               wpt_commit TEXT NOT NULL, dirty INTEGER NOT NULL);
        CREATE TABLE lines (session TEXT NOT NULL REFERENCES sessions (folder), number INTEGER NOT NULL,
                            source TEXT NOT NULL, revision TEXT, checksum TEXT, address TEXT, time TEXT NOT NULL,
                            outcome TEXT NOT NULL, size INTEGER, code TEXT NOT NULL, PRIMARY KEY (session, number));
        INSERT INTO sessions VALUES ('old', 'download', '2026-10-01T00:00:00Z', '2026-10-01T01:00:00Z', 'abc', 0);
        INSERT INTO lines VALUES ('old', 1, 'one', 'r1', 'sha256:aa', 'https://example.org/one',
                                  '2026-10-01T00:30:00Z', 'downloaded', 10, 'abc');
        INSERT INTO lines VALUES ('old', 2, 'two', NULL, NULL, NULL, '2026-10-01T00:40:00Z', 'failed', NULL, 'abc');
    """)
    store.kind.write("store.sqlite.zst", compressed(old.serialize()))
    with Session(store, "new", "download", "abc123", False) as session:
        session.line("two", "downloaded", "abc123", got={"commit": "c2"})
    with store.database() as found:
        assert found.execute("SELECT session, source, got, outcome FROM lines ORDER BY session, number").fetchall() \
            == [("new", "two", '{"commit": "c2"}', "downloaded"),
                ("old", "one", '{"checksum": "sha256:aa", "revision": "r1"}', "downloaded"),
                ("old", "two", None, "failed")]


def test_an_outcome_the_database_does_not_know_is_refused(reopen):
    session = Session(reopen(), "s", "download", "abc123", False)
    with pytest.raises(ValueError):
        session.line("x", "checked", "abc123")


def test_a_bundle_that_could_not_be_stored_stops_the_session_and_its_lines_are_never_recorded(reopen, monkeypatch):
    # an upload fails while the second bundle is sealed: its file and the
    # line recorded with it must not reach the database as downloaded
    monkeypatch.setattr(sessions, "DATABASE_SHARE", 0)  # every seal puts its rows in the database
    store = reopen()
    real = store.keep_compressed
    sealed = []

    def fails_the_second_time(*args):
        sealed.append(args[0])
        if len(sealed) == 2:
            raise OSError("the upload failed")
        return real(*args)

    store.keep_compressed = fails_the_second_time
    with pytest.raises(OSError, match="the upload failed") as raised:
        with Session(store, "lost", "download", "abc123", False, bundle=2 * MB) as session:
            session.line("kept", "downloaded", "abc123")
            kept = session.add(text("kept", 2 * MB))  # the first bundle, sealed and recorded
            session.line("lost", "downloaded", "abc123")
            with pytest.raises(OSError, match="the upload failed"):
                session.add(text("lost", 2 * MB))
            # nothing more goes into a session that lost a bundle
            for more in (lambda: session.add(text("after", 10)), lambda: session.line("after", "reused", "abc123")):
                with pytest.raises(OSError, match="the upload failed"):
                    more()
    assert not getattr(raised.value, "__notes__", None)  # the end raised that same failure, not a second one
    assert len(sealed) == 2  # the end sealed no further bundle

    later = reopen()
    assert later.read_file(kept) == text("kept", 2 * MB)
    with later.database() as found:
        assert found.execute("SELECT folder, ended FROM sessions").fetchall() == [("lost", None)]
        assert found.execute("SELECT source, outcome FROM lines").fetchall() == [("kept", "downloaded")]


def test_files_go_on_into_the_next_bundle_while_a_full_one_is_being_stored(reopen):
    store = reopen()
    real = store.keep_compressed
    storing, go_on = threading.Event(), threading.Event()

    def slow(*args):
        storing.set()
        assert go_on.wait(30)
        return real(*args)

    store.keep_compressed = slow
    with Session(store, "busy", "download", "abc123", False, bundle=2 * MB) as session:
        placed = {}
        filling = threading.Thread(target=lambda: placed.update(first=session.add(text("first", 2 * MB))))
        filling.start()
        assert storing.wait(30)
        placed["second"] = session.add(text("second", MB))  # returns while the first bundle is still being stored
        assert filling.is_alive()
        go_on.set()
        filling.join(30)
        named(session, placed)
    assert [placed[name].bundle.rsplit("/", 1)[1] for name in ("first", "second")] == ["000001", "000002"]
    later = reopen()
    assert later.read_file(placed["first"]) == text("first", 2 * MB)
    assert later.read_file(placed["second"]) == text("second", MB)


@pytest.mark.parametrize("stream", [False, True])
def test_an_s3_store_keeps_a_local_copy_of_its_bundles_unless_the_session_streams(reopen, stream):
    store = reopen()
    data = text("kept", 3 * MB)
    with store.session("20261025T120000Z-download-x-0002", "download", "abc123", False, stream=stream) as session:
        placed = session.add(data)
        named(session, {"kept.txt": placed})
    copies = sorted(path.relative_to(store.cache).as_posix() for path in store.cache.rglob("bundles/*/*"))
    if store.local:  # a local store's bundles are on this machine already: no copy
        assert copies == []
    else:
        assert copies == ([] if stream else [f"bundles/{placed.bundle}.zst"])
    assert store.read_file(placed) == data
    assert reopen().file_path(placed).read_bytes() == data  # one file alone, checked, as a local file


def test_an_action_runs_once_the_files_added_before_it_are_stored_and_never_after_a_lost_bundle(reopen):
    store = reopen()
    seen = []
    with Session(store, "named", "download", "abc123", False, bundle=2 * MB) as session:
        placed = session.add(text("first", MB))
        named(session, {"first.txt": placed})
        session.then(lambda: seen.append(reopen().read_file(placed)))  # readable from any store object by then
        assert seen == []  # its bundle is still being filled
    assert seen == [text("first", MB)]

    real, ran = store.keep_compressed, []
    store.keep_compressed = lambda *args: (_ for _ in ()).throw(OSError("the upload failed"))
    with pytest.raises(OSError, match="the upload failed"):
        with Session(store, "lost-again", "download", "abc123", False) as session:
            named(session, {"lost.txt": session.add(text("lost", 10))})
            session.then(lambda: ran.append(1))
    store.keep_compressed = real
    assert ran == []  # what the action would name is not in the store


def test_a_file_reads_back_from_its_session_before_and_after_its_bundle_is_sealed(reopen):
    # what a download needs: the members of an archive read straight into
    # the bundles are read back (a cut, a patch) before the bundle is full
    store = reopen()
    files = {"a.txt": text("a", MB + 7), "b.txt": text("b", 10), "after.txt": text("after", 2 * MB),
             "c.txt": text("c", 3 * MB)}
    with Session(store, "read-back", "download", "abc123", False, bundle=4 * MB) as session:
        placed = {"a.txt": session.add(files["a.txt"]), "b.txt": session.add(files["b.txt"])}
        assert {path: session.read(where) for path, where in placed.items()} == {p: files[p] for p in placed}
        placed["after.txt"] = session.add(files["after.txt"])  # after the frame the reads ended early
        assert session.read(placed["after.txt"]) == files["after.txt"]
        placed["c.txt"] = session.add(files["c.txt"])  # too big for the rest: the first bundle is sealed
        assert placed["c.txt"].bundle.endswith("/000002")
        assert session.read(placed["c.txt"]) == files["c.txt"]
        assert session.read(placed["a.txt"]) == files["a.txt"]  # from the sealed bundle
        named(session, placed)
    later = reopen()
    assert {path: later.read_file(where) for path, where in placed.items()} == files


def test_the_end_rewrites_the_bundle_being_filled_without_the_bytes_no_file_list_names(reopen):
    # a try that failed left a file in the bundle being filled: the end stores that bundle without it, and the
    # file list naming the other files is stored with their new places
    store = reopen()
    lists = []
    with Session(store, "trimmed", "download", "abc123", False, bundle=8 * MB) as session:
        kept = session.add(text("kept", MB))
        failed = session.add(text("failed try", 2 * MB))  # no file list will name it
        after = session.add(text("after", MB + 3))
        named(session, {"kept.txt": kept, "after.txt": after}, lists.append)
    [listed] = lists
    files = parse_file_list(store.get_bytes(listed))
    assert files["kept.txt"][1] == kept and files["after.txt"][1].start == MB  # moved up, where the failed try was
    later = reopen()
    assert later.read_file(files["after.txt"][1]) == text("after", MB + 3)
    assert [key for key in later.kind.list("bundles/trimmed/")] in (["bundles/trimmed/000001.zst"],
                                                                     ["bundles/trimmed/000001"])
    with pytest.raises(CorruptObject):
        later.read_file(failed)  # those bytes are gone: what lies at their old place is another file


def test_the_end_rewrites_a_sealed_bundle_whose_bytes_a_file_list_does_not_name_and_names_the_new_places(reopen):
    # a mismatched archive's members went into a bundle sealed long before the end: the end makes a new bundle
    # of the named files, stores every list naming them again with their new places (its action runs again),
    # and removes the old bundle; a bundle holding no named file at all is removed
    store = reopen()
    lists = []
    with Session(store, "rewritten", "download", "abc123", False, bundle=2 * MB) as session:
        good = session.add(text("good", MB))
        session.add(text("mismatched member", MB))  # fills the first bundle: sealed with these bytes
        session.name({"good.txt": ("100644", good)}, lists.append, {"1" * 40: "good.txt"})
        session.add(text("lost alone", 2 * MB))  # a second bundle, of bytes no list names; its seal stores the list
        assert len(lists) == 1
        last = session.add(text("last", 10))
        named(session, {"last.txt": last})
    first, again = lists  # stored at the seal of the second bundle, then again with the new place
    assert parse_file_list(store.get_bytes(first))["good.txt"][1] == good
    moved = parse_file_list(store.get_bytes(again))["good.txt"][1]
    assert moved.bundle == "rewritten/000004" and moved.start == 0 and moved.sha256 == good.sha256
    later = reopen()
    assert later.read_file(moved) == text("good", MB)
    assert sorted(key.split("/")[2].split(".")[0] for key in later.kind.list("bundles/rewritten/")) == [
        "000003", "000004"]
    with later.database() as found:
        assert found.execute("SELECT object_id, bundle, start, length FROM git_objects").fetchall() == [
            ("1" * 40, "rewritten/000004", 0, MB)]


def rewrite_leaving_an_old_bundle(store, name):
    """A session that ends by rewriting a sealed bundle (a mismatched member among its bytes); gives the old
    bundle's name and the new place of its named file."""
    lists = []
    with Session(store, name, "download", "abc123", False, bundle=2 * MB) as session:
        good = session.add(text("good", MB))
        session.add(text("mismatched member", MB))  # fills the first bundle: sealed with these bytes
        session.name({"good.txt": ("100644", good)}, lists.append)
        last = session.add(text("last", 10))
        named(session, {"last.txt": last})
    return good, parse_file_list(store.get_bytes(lists[-1]))["good.txt"][1]


def bundles_of(store, name):
    return sorted(key.split("/")[2].split(".")[0] for key in store.kind.list(f"bundles/{name}/"))


def test_a_rewritten_bundle_s_old_copy_stays_until_every_session_that_started_before_the_rewrite_has_ended(reopen):
    # a session that started before the rewrite may hold the old place of a file: the old copy is marked for
    # removal, and the end of the last such session removes it
    store = reopen()
    early = Session(reopen(), "early", "download", "abc123", False)  # running while the other one ends
    old, moved = rewrite_leaving_an_old_bundle(store, "rewritten")
    assert moved.bundle != old.bundle
    assert bundles_of(reopen(), "rewritten") == ["000001", "000002", "000003"]  # the old copy is still there
    assert reopen().read_file(old) == text("good", MB)  # and early can read what it knew
    with reopen().database() as found:
        assert [row[0] for row in found.execute("SELECT bundle FROM removals")] == [old.bundle]

    with Session(reopen(), "middle", "download", "abc123", False):
        pass
    assert bundles_of(reopen(), "rewritten") == ["000001", "000002", "000003"]  # early still runs

    early.end()  # the last session that started before the rewrite: its end removes the old copy
    assert bundles_of(reopen(), "rewritten") == ["000002", "000003"]
    with reopen().database() as found:
        assert found.execute("SELECT bundle FROM removals").fetchall() == []
    with pytest.raises(MissingObject):
        reopen().read_file(old)


def test_a_rewritten_bundle_s_old_copy_is_removed_at_the_end_when_no_earlier_session_runs(reopen):
    store = reopen()
    old, _ = rewrite_leaving_an_old_bundle(store, "rewritten")
    assert bundles_of(reopen(), "rewritten") == ["000002", "000003"]
    with pytest.raises(MissingObject):
        reopen().read_file(old)


def test_the_end_keeps_the_bytes_the_git_map_names_when_no_file_list_does(reopen):
    # a blob only a history needs belongs to no tree: it is named for the map alone (no action), and the end's
    # rewrite keeps it next to the bytes of the failed try it drops
    store = reopen()
    ran = []
    with Session(store, "history", "download", "abc123", False, bundle=2 * MB) as session:
        kept = session.add(text("history only", MB))
        session.add(text("failed try", MB))  # fills the bundle: sealed with bytes nobody names
        session.name({"blob": ("100644", kept)}, None, {"a" * 40: "blob"})
        after = session.add(text("after", 10))
        named(session, {"after.txt": after}, ran.append)
    assert len(ran) == 1
    with Session(reopen(), "later", "download", "abc123", False) as later:
        [moved] = later.known_objects(["a" * 40]).values()
        assert moved.bundle != kept.bundle and later.read(moved) == text("history only", MB)
    with pytest.raises(MissingObject):
        reopen().read_file(kept)


def test_git_objects_a_session_names_are_known_to_later_sessions_while_their_bundle_is_held(reopen):
    # the map from git object id to sha256 is shared by every source: a later session gets where the store
    # holds a blob without downloading it; a blob whose bundle is gone is not known
    store = reopen()
    with Session(store, "first", "download", "abc123", False) as session:
        placed = session.add(text("blob", 100))
        other = session.add(text("other", 100))
        session.name({"a.txt": ("100644", placed), "b.txt": ("100644", other)}, lambda listed: None,
                     {"a" * 40: "a.txt", "b" * 40: "b.txt"})
        assert session.known_objects(["a" * 40, "c" * 40]) == {"a" * 40: placed}  # its own, before its seal
    with Session(reopen(), "second", "download", "abc123", False) as later:
        assert later.known_objects(["a" * 40, "b" * 40, "c" * 40]) == {"a" * 40: placed, "b" * 40: other}
        later.remember({"d" * 40: placed})  # a tree the store held already: the map learns its blobs
    store.drop_bundle(placed.bundle)
    with Session(reopen(), "third", "download", "abc123", False) as third:
        assert third.known_objects(["a" * 40, "d" * 40]) == {}
    with reopen().database() as found:
        assert sorted(row[0] for row in found.execute("SELECT object_id FROM git_objects")) == ["a" * 40, "b" * 40,
                                                                                               "d" * 40]


def test_git_objects_of_a_session_still_running_are_not_given_out_since_its_end_may_remove_their_bundle(reopen,
                                                                                                    monkeypatch):
    # a session seals a bundle holding a failed try next to a git file it names; another session running at the
    # same time must not name that file where it lies: the first one's end rewrites the bundle and removes it
    monkeypatch.setattr(sessions, "DATABASE_SHARE", 0)  # every seal puts its rows in the database
    store = reopen()
    first = Session(store, "first", "download", "abc123", False, bundle=2 * MB)
    good = first.add(text("good", MB))
    first.add(text("failed try", MB))  # fills the bundle: sealed with bytes no list names
    first.name({"good.txt": ("100644", good)}, lambda listed: None, {"a" * 40: "good.txt"})
    first.add(text("next", 10))
    first.then(lambda: None)
    first.add(text("fills the second", 2 * MB))  # its seal puts the git row of good.txt in the database
    with reopen().database() as found:
        assert found.execute("SELECT bundle FROM git_objects").fetchall() == [("first/000001",)]
    with Session(reopen(), "second", "download", "abc123", False) as second:
        assert second.known_objects(["a" * 40]) == {}  # first is still running
        second.remember({"a" * 40: good})  # nor does a tree first stored teach the map where it lies
    first.end()
    with Session(reopen(), "third", "download", "abc123", False) as third:
        [moved] = third.known_objects(["a" * 40]).values()  # first has ended: its rewritten bundle stays
        assert moved.bundle != good.bundle and third.read(moved) == text("good", MB)
    with pytest.raises(MissingObject):
        reopen().read_file(good)  # the bundle that held the failed try is gone


def test_a_new_session_removes_the_unfinished_bundles_no_live_session_owns(reopen):
    # a crash leaves the bundle being filled on this machine's disk; the next session removes it, and leaves
    # one another live session is filling
    from wpt_store.local_files import LOCK
    from wpt_store.sessions import PARTIAL
    from wpt_utils.files import lock_file, unlock_file

    store = reopen()
    store.cache.mkdir(parents=True, exist_ok=True)
    crashed, live = store.cache / f"{PARTIAL}crashed", store.cache / f"{PARTIAL}live"
    crashed.write_bytes(b"a crashed session's bytes")
    live.write_bytes(b"a live session's bytes")
    held = lock_file(store.cache / f"{PARTIAL}live{LOCK}")  # its owner is alive
    try:
        with Session(store, "next", "download", "abc123", False) as session:
            named(session, {"x.txt": session.add(b"x")})
        assert not crashed.exists() and live.exists()
    finally:
        unlock_file(held)
    with Session(store, "after", "download", "abc123", False):
        pass
    assert sorted(path.name for path in store.cache.glob(PARTIAL + "*")) == []


def test_a_local_bundle_is_on_the_disk_before_the_database_names_it(tmp_path, monkeypatch):
    # a power cut after the database names a bundle must not lose the bundle: its file and its folder's list
    # of names are synced first
    import os

    from wpt_store import database, kinds

    store = open_store((tmp_path / "store").as_uri())
    order = []
    real_fsync, real_put = os.fsync, database.put

    def fsync(descriptor):
        order.append(("synced", os.path.realpath(f"/proc/self/fd/{descriptor}") if os.name != "nt" else ""))
        return real_fsync(descriptor)

    monkeypatch.setattr(kinds.os, "fsync", fsync)
    monkeypatch.setattr(database, "put", lambda *args: order.append(("put", "")) or real_put(*args))
    with Session(store, "durable", "download", "abc123", False) as session:
        named(session, {"x.txt": session.add(text("x", MB))})
    put = order.index(("put", ""), 1)  # the first is the session's start
    assert any(kind == "synced" for kind, _ in order[:put])
    if os.name != "nt":
        synced = [path for kind, path in order[:put] if kind == "synced"]
        assert str(tmp_path / "store" / "bundles" / "durable") in synced  # the bundle's folder: its new name


CRASH = """
import os, sys
from wpt_store import Session, database, open_store, sessions

sessions.DATABASE_SHARE = 0  # every seal puts its rows in the database
real = database.put
calls = []


def put(*args):
    calls.append(args)
    if len(calls) == 2:  # the rows of the first sealed bundle are on this machine's disk, not yet in the store
        os._exit(0)
    return real(*args)


database.put = put
session = Session(open_store(sys.argv[1]), "crashed", "download", "abc123", False, bundle=1024)
session.line("first", "downloaded", "abc123")
session.add(b"x" * 1024)  # fills the bundle: sealed at once
"""


def test_a_session_that_crashed_before_its_rows_went_in_has_them_put_in_by_the_next_session(tmp_path):
    # a process dies between writing a sealed bundle's rows into its local file and putting them into the store
    # database: the next session on the machine puts them in, then removes the file
    import subprocess
    import sys

    from wpt_store.database import ROWS

    url = (tmp_path / "store").as_uri()
    subprocess.run([sys.executable, "-c", CRASH, url], check=True)
    store = open_store(url)
    with store.database() as found:
        assert found.execute("SELECT count(*) FROM lines").fetchone() == (0,)
    assert [path.name for path in store.cache.glob(ROWS + "crashed*") if path.name.endswith(".sqlite")]
    with Session(store, "next", "download", "abc123", False):
        pass
    with store.database() as found:
        assert found.execute("SELECT session, source, outcome FROM lines").fetchall() \
            == [("crashed", "first", "downloaded")]
        assert found.execute("SELECT folder, ended IS NULL FROM sessions ORDER BY folder").fetchall() \
            == [("crashed", 1), ("next", 0)]
    assert list(store.cache.glob(ROWS + "*")) == []


def test_rows_the_store_refused_at_the_end_are_put_in_by_the_next_session(reopen):
    # the store database's replace fails at a session's end: its last lines stay in its local file, and the
    # next session on the machine puts them in
    store = reopen()
    real = store.kind.replace
    failing = []

    def refuses_once(*args):
        if failing:
            failing.clear()
            raise OSError("the upload failed")
        return real(*args)

    store.kind.replace = refuses_once
    with pytest.raises(OSError, match="the upload failed"):
        with Session(store, "refused", "download", "abc123", False) as session:
            session.line("one", "downloaded", "abc123")
            failing.append(True)
    with store.database() as found:
        assert found.execute("SELECT count(*) FROM lines").fetchone() == (0,)
    with Session(store, "next", "download", "abc123", False):
        pass
    with reopen().database() as found:
        assert found.execute("SELECT session, source FROM lines").fetchall() == [("refused", "one")]
        assert found.execute("SELECT folder, ended IS NOT NULL FROM sessions ORDER BY folder").fetchall() \
            == [("next", 1), ("refused", 1)]  # it ended: its end is among the rows put in


def test_the_store_database_is_read_through_a_copy_on_this_machine_s_disk_removed_once_closed(reopen):
    # the database can outgrow what memory holds: each read opens a copy in the cache folder, which goes when
    # the connection closes; a copy a crashed process left is removed by the next read, one a live one holds
    # stays; a session leaves none behind
    from pathlib import Path

    from wpt_store.database import COPY
    from wpt_store.local_files import LOCK
    from wpt_utils.files import lock_file, unlock_file

    store = reopen()
    with Session(store, "first", "download", "abc123", False) as session:
        session.line("one", "downloaded", "abc123")
    store.cache.mkdir(parents=True, exist_ok=True)
    (store.cache / f"{COPY}crashed").write_bytes(b"a crashed process's copy")
    (store.cache / f"{COPY}crashed{LOCK}").write_bytes(b"")
    (store.cache / f"{COPY}live").write_bytes(b"a live process's copy")
    held = lock_file(store.cache / f"{COPY}live{LOCK}")
    try:
        with store.database() as found:
            _, _, path = found.execute("PRAGMA database_list").fetchone()
            assert Path(path).parent == store.cache and Path(path).stat().st_size > 0
            assert found.execute("SELECT session, source FROM lines").fetchall() == [("first", "one")]
        assert not Path(path).exists()
        assert sorted(item.name for item in store.cache.glob(COPY + "*")) == [f"{COPY}live", f"{COPY}live{LOCK}"]
    finally:
        unlock_file(held)
    with Session(store, "second", "download", "abc123", False) as session:
        assert not session.marked("one")
    assert sorted(item.name for item in store.cache.glob(".*") if not item.name.startswith(".partial-")) == []


def test_rows_files_a_crash_left_half_made_or_half_removed_do_not_stop_later_sessions(reopen):
    # a crash inside the removal of a session's rows file leaves its lock file alone; one while the file is made
    # leaves it without its tables: neither holds rows, and the next sessions start and leave neither behind
    from wpt_store.database import ROWS, ROWS_END
    from wpt_store.local_files import LOCK

    store = reopen()
    store.cache.mkdir(parents=True, exist_ok=True)
    (store.cache / f"{ROWS}gone{ROWS_END}{LOCK}").write_bytes(b"")
    (store.cache / f"{ROWS}empty{ROWS_END}").write_bytes(b"")
    (store.cache / f"{ROWS}empty{ROWS_END}{LOCK}").write_bytes(b"")
    for folder in ("next", "after"):
        with Session(store, folder, "download", "abc123", False) as session:
            session.line("one", "downloaded", "abc123")
    with store.database() as found:
        assert found.execute("SELECT folder FROM sessions ORDER BY folder").fetchall() == [("after",), ("next",)]
    assert list(store.cache.glob(ROWS + "*")) == []
