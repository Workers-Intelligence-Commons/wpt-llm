import threading

import boto3
import pytest
from moto import mock_aws

from wpt_store import MissingObject, Session, file_list_bytes, open_store, parse_file_list

MB = 1 << 20


@pytest.fixture(params=["local", "s3"])
def reopen(request, tmp_path, monkeypatch):
    """A function giving a new store object on the same store each call: a
    local folder, or an S3 bucket whose requests moto answers inside the test process."""
    if request.param == "local":
        yield lambda: open_store((tmp_path / "store").as_uri())
        return
    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        yield lambda: open_store("s3://wpt-test/sessions")


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
        session.line("one-source", "downloaded", "0.5.0", revision="r1", checksum="sha256:" + "1" * 64,
                     address="https://example.org/one", size=sum(map(len, files.values())))
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
        assert found.execute("SELECT source, revision, address, outcome, size, wpt_version FROM lines").fetchall() \
            == [("one-source", "r1", "https://example.org/one", "downloaded", sum(map(len, files.values())), "0.5.0")]


def test_a_crash_loses_only_the_bundle_being_filled_and_the_lines_since_the_last_sealed(reopen):
    store = reopen()
    session = Session(store, "crashed", "download", "abc123", True, bundle=2 * MB)
    sealed = session.add(text("first", 2 * MB))  # fills the first bundle: sealed at once
    session.line("first", "downloaded", "0.5.0")  # recorded with the next bundle, which never comes
    unsealed = session.add(text("second", MB))
    # the process ends here, without end()
    later = reopen()
    assert later.read_file(sealed) == text("first", 2 * MB)
    with pytest.raises(MissingObject):
        later.read_file(unsealed)
    with later.database() as found:
        assert found.execute("SELECT folder, dirty, ended FROM sessions").fetchall() == [("crashed", 1, None)]
        assert found.execute("SELECT count(*) FROM lines").fetchone() == (0,)


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
                other.line("other-source", "reused", "0.5.0")
        return real(key, data, version)

    first.kind.replace = another_session_first
    with Session(first, "mine", "download", "abc123", False) as mine:
        mine.line("my-source", "failed", "0.5.0", checksum="sha256:" + "2" * 64)
    assert raced
    with reopen().database() as found:
        assert found.execute("SELECT session, source, outcome FROM lines ORDER BY session").fetchall() \
            == [("mine", "my-source", "failed"), ("other", "other-source", "reused")]
        assert [row[0] for row in found.execute("SELECT folder FROM sessions ORDER BY folder")] == ["mine", "other"]


def test_an_outcome_the_database_does_not_know_is_refused(reopen):
    session = Session(reopen(), "s", "download", "abc123", False)
    with pytest.raises(ValueError):
        session.line("x", "checked", "0.5.0")


def test_a_bundle_that_could_not_be_stored_stops_the_session_and_its_lines_are_never_recorded(reopen):
    # an upload fails while the second bundle is sealed: its file and the
    # line recorded with it must not reach the database as downloaded
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
            session.line("kept", "downloaded", "0.5.0")
            kept = session.add(text("kept", 2 * MB))  # the first bundle, sealed and recorded
            session.line("lost", "downloaded", "0.5.0")
            with pytest.raises(OSError, match="the upload failed"):
                session.add(text("lost", 2 * MB))
            # nothing more goes into a session that lost a bundle
            for more in (lambda: session.add(text("after", 10)), lambda: session.line("after", "reused", "0.5.0")):
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
        session.then(lambda: seen.append(reopen().read_file(placed)))  # readable from any store object by then
        assert seen == []  # its bundle is still being filled
    assert seen == [text("first", MB)]

    real, ran = store.keep_compressed, []
    store.keep_compressed = lambda *args: (_ for _ in ()).throw(OSError("the upload failed"))
    with pytest.raises(OSError, match="the upload failed"):
        with Session(store, "lost-again", "download", "abc123", False) as session:
            session.add(text("lost", 10))
            session.then(lambda: ran.append(1))
    store.keep_compressed = real
    assert ran == []  # what the action would name is not in the store


def test_a_file_reads_back_from_its_session_before_and_after_its_bundle_is_sealed(reopen):
    # what fill-checksum needs: the members of an archive read straight into
    # the bundles are read back to check their text before the bundle is full
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
    later = reopen()
    assert {path: later.read_file(where) for path, where in placed.items()} == files
