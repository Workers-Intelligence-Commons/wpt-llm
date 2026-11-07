"""The download step: sources fetched into a session of the store, each
as its address serves it, several at once."""

import shutil
import threading
from types import SimpleNamespace

import pytest

from wpt import sessions
from wpt_dataset import archives, download, license_check, manifest, robots, sites
from wpt_dataset.errors import FetchFailed, NotDownloaded, SourceGone
from wpt_dataset.kept import kept
from wpt_dataset.sources import license_page, source_tree, stored_upstream, upstream_manifest, upstream_tree
from wpt_store import held, held_here
from wpt_store.listing import listing
from wpt_utils.programs import MissingTool


def two_sources(kit, repo, store, tmp_path, http_folder):
    """(git source id, http source id), neither downloaded yet."""
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    address, commit = kit.git_project(tmp_path / "project")
    git = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address},
                         fetch=False)
    book = kit.add_source(repo, store, "example", "book", {
        "type": "http", "address": f"{base}/book.txt"}, fetch=False)
    return git, book


def test_a_download_keeps_each_source_in_its_session_and_finds_it_there_next_time(
        kit, repo, store, tmp_path, http_folder):
    git, book = two_sources(kit, repo, store, tmp_path, http_folder)

    done = download.run(repo, [git, book], store, "download", [])
    assert [item.value for item in done] == ["downloaded", "downloaded"]

    [folder] = store.record_folders()
    assert folder.startswith(f"{store.session}/") and "-wpt-dataset-download-2-sources-" in folder
    for sid, files in ((git, kit.FILES), (book, {"book.txt": b"a public domain book\n"})):
        tree, _ = source_tree(repo.source(sid), store)
        for path, content in files.items():
            assert tree[path].placed.archive == f"{store.session}/sources/{sid}.tar.zst"
            assert manifest.content(store, tree[path]) == content
    commit = kit.run("git", "rev-parse", "HEAD", cwd=tmp_path / "project").decode().strip()
    for sid, got, size in ((git, {"commit": commit}, sum(map(len, kit.FILES.values())) + len("docs/guide.txt")),
                           (book, {}, len(b"a public domain book\n"))):
        found = held(store, sid)
        assert (found.got, found.address, found.size, found.error) == (got, repo.source(sid).data["address"], size, None)
        assert found.manifest == upstream_manifest(repo.source(sid), store)

    # upstream gone: the session still holds both, so nothing is downloaded again
    shutil.rmtree(tmp_path / "project")
    (http_folder[0] / "book.txt").unlink()
    assert [item.value for item in download.run(repo, [git, book], store, "download", [])] == ["reused", "reused"]

    states, sessions = listing(store)  # the index says what the session holds
    assert {(state.session, state.source, state.error) for state in states} == {
        (store.session, git, None), (store.session, book, None)}
    assert [(row.session, row.held["sources"], row.held["runs"]) for row in sessions] == [(store.session, 2, 2)]


def test_a_download_given_no_session_makes_a_new_one_and_says_it(kit, store, tmp_path, monkeypatch):
    monkeypatch.delenv("WPT_SESSION", raising=False)
    said = []
    plain = kit.open_store(store.root.as_uri())
    none = SimpleNamespace(session=None, from_id=None)
    view = sessions.fetching(plain, none, tmp_path / "none", log=said.append)
    assert said == [f"session {view.session}"]
    monkeypatch.setenv("WPT_SESSION", "named")
    assert sessions.fetching(plain, none, tmp_path / "none", log=said.append).session == "named"
    assert len(said) == 1


def test_a_git_source_downloads_exactly_its_revision_and_downloads_again_when_it_moves(kit, repo, store, tmp_path):
    # upstream's later commits may add files no license entry covers: the store holds the commit the source
    # names, whatever upstream's HEAD is, and a later commit enters only once the revision names it
    address, first = kit.git_project(tmp_path / "project")
    folder = tmp_path / "project"
    (folder / "README.md").write_bytes(b"a later readme\n")
    (folder / "added.bin").write_bytes(b"\x00\xff a later file\n")
    kit.run("git", "add", "README.md", "added.bin", cwd=folder)
    kit.run("git", "commit", "-q", "-m", "later", cwd=folder)
    later = kit.head(folder)
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": first})

    tree = upstream_tree(repo.source(sid), store)
    assert sorted(tree) == sorted([*kit.FILES, *kit.SYMLINKS])
    assert manifest.content(store, tree["README.md"]) == kit.FILES["README.md"]

    path = repo.root / "data" / "sources" / sid / "source.json"
    path.write_text(kit.dump_json({**repo.source(sid).data, "revision": later}), encoding="utf-8", newline="\n")
    assert held(store, sid).got == {"commit": first}
    kit.download(repo, store, sid)  # the moved revision is not what the session holds: it replaces the file

    tree = upstream_tree(repo.source(sid), store)
    assert manifest.content(store, tree["README.md"]) == b"a later readme\n" and "added.bin" in tree
    assert held(store, sid).got == {"commit": later}

    # a commit upstream does not serve (a branch rewritten) fails the source; the store keeps what it held
    path.write_text(kit.dump_json({**repo.source(sid).data, "revision": "1" * 40}), encoding="utf-8", newline="\n")
    with pytest.raises(FetchFailed, match=f"does not serve the commit {'1' * 40}"):
        kit.download(repo, store, sid)
    assert held_here(store, sid).got == {"commit": later} and held_here(store, sid).error["kind"] == "FetchFailed"


def test_a_download_started_from_an_old_session_fetches_only_what_the_old_one_lacks_and_reads_the_rest_there(
        kit, repo, store, tmp_path, http_folder):
    git, book = two_sources(kit, repo, store, tmp_path, http_folder)
    old = store.session
    download.run(repo, [git], store, "download", [])  # the old session holds the git source only
    new = kit.open_store(store.root.as_uri()).at("new", old)

    shutil.rmtree(tmp_path / "project")  # upstream of the git source is gone: only the old session has it
    done = download.run(repo, [git, book], new, "download", [])
    assert [item.value for item in done] == ["reused", "downloaded"]
    assert held_here(new, git) is None and held_here(new, book).session == "new"
    # a build, `check --fetch` and `manifest` read the git source from the old session's files
    assert held(new, git).session == old
    tree = upstream_tree(repo.source(git), new)
    assert manifest.content(new, tree["README.md"]) == kit.FILES["README.md"]
    assert tree["README.md"].placed.archive == f"{old}/sources/{git}.tar.zst"
    assert upstream_manifest(repo.source(git), new) == held(store, git).manifest

    # an old download made for other fetch options is not what the new one wants: it is fetched
    path = repo.root / "data" / "sources" / book / "source.json"
    path.write_text(kit.dump_json({**repo.source(book).data, "address": f"{http_folder[1]}/other.txt"}),
                    encoding="utf-8", newline="\n")
    (http_folder[0] / "other.txt").write_bytes(b"another book\n")
    other = kit.open_store(store.root.as_uri()).at("other", "new")
    assert [item.value for item in download.run(repo, [book], other, "download", [])] == ["downloaded"]
    assert held_here(other, book).session == "other"


def test_a_source_whose_last_attempt_failed_is_read_from_the_good_file_of_the_same_download(
        kit, repo, store, tmp_path):
    address, first = kit.git_project(tmp_path / "project")
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": first})
    path = repo.root / "data" / "sources" / sid / "source.json"
    description = repo.source(sid).data
    good = held(store, sid)

    # an attempt at another download fails: its good file is of a download this description does not name
    path.write_text(kit.dump_json({**description, "revision": "1" * 40}), encoding="utf-8", newline="\n")
    with pytest.raises(FetchFailed):
        kit.download(repo, store, sid)
    with pytest.raises(NotDownloaded):
        upstream_tree(repo.source(sid), store)

    # a new attempt at the download it holds a good file of fails: that file is still what a build reads
    path.write_text(kit.dump_json(description), encoding="utf-8", newline="\n")
    shutil.rmtree(tmp_path / "project")
    with pytest.raises(FetchFailed):
        kit.download(repo, store, sid)
    failed = held_here(store, sid)
    assert failed.error["kind"] == "SourceGone" and failed.sha256 == good.sha256 and failed.manifest == good.manifest
    tree = upstream_tree(repo.source(sid), store)
    assert manifest.content(store, tree["README.md"]) == kit.FILES["README.md"]
    assert upstream_manifest(repo.source(sid), store) == good.manifest

    # it is not held: a download fetches it again, and a source with no good file at all stays left out
    assert not failed.holds(failed.key) and failed.provides(failed.key)


def test_downloads_running_at_once_give_what_one_at_a_time_gives(kit, repo, store, tmp_path, http_folder,
                                                                  monkeypatch):
    folder, base = http_folder
    ids = []
    for n in range(6):
        (folder / f"book-{n}.txt").write_bytes(f"book number {n}\n".encode() * (n + 1))
        ids.append(kit.add_source(repo, store, "example", f"book-{n}", {
            "type": "http", "address": f"{base}/book-{n}.txt"}, fetch=False))
    results = {}
    for way, limits in (("at once", ({}, 6)), ("one at a time", ({}, 1))):
        target = kit.open_store((tmp_path / way).as_uri()).at("test-session")
        monkeypatch.setattr(sites, "limits", lambda limits=limits: (*limits, {}))
        done = download.download(repo, ids, target, log=lambda *_: None)
        results[way] = (done, [manifest.manifest_bytes(source_tree(repo.source(sid), target)[0]) for sid in ids])
    assert results["at once"] == results["one at a time"]
    assert [d.value for d in results["at once"][0]] == ["downloaded"] * 6


def test_a_site_s_limit_covers_the_hosts_under_it_and_every_other_site_takes_the_default():
    at_once, default, per_time = sites.limits()  # the file this package ships
    assert (at_once, default) == ({"wikimedia.org": 3, "github.com": 100, "huggingface.co": 16}, 2)
    assert per_time == {"huggingface.co": (3000, 300.0)}  # what the Hub publishes: 3,000 requests per 5 minutes
    assert sites.site_of("https://dumps.wikimedia.org/enwiki/latest/x.xml.bz2", at_once) == "wikimedia.org"
    assert sites.site_of("https://github.com/owner/project", at_once) == "github.com"
    assert sites.site_of("https://notgithub.com/owner/project", at_once) is None
    assert sites.site_of("https://huggingface.co/api/datasets/a/b", per_time) == "huggingface.co"
    assert sites.site_of("https://huggingface.co/datasets/a/b", at_once) == "huggingface.co"  # 16 at once as well
    # both kinds of limit for one site, side by side
    assert sites.limits("# a comment\n\nexample.org 5\nexample.org 10 per 60 seconds\ndefault 1\n") == (
        {"example.org": 5}, 1, {"example.org": (10, 60.0)})
    for wrong in ("example.org five\ndefault 1\n", "example.org 5 per minute\ndefault 1\n", "example.org 5\n",
                  "example.org 5\nexample.org 6\ndefault 1\n"):
        with pytest.raises(ValueError):
            sites.limits(wrong)


def test_a_limit_per_time_holds_back_the_requests_that_would_pass_it():
    # 3 requests per 10 seconds: the fourth waits until the first is 10 seconds old, and requests from
    # several threads share one count
    moments, slept = [0.0], []

    def sleep(seconds):
        slept.append(seconds)
        moments[0] += seconds

    pace = sites.Pace(3, 10, clock=lambda: moments[0], sleep=sleep)
    for at in (0.0, 1.0, 2.0):
        moments[0] = at
        pace.wait()
    assert slept == []
    moments[0] = 4.0
    pace.wait()
    assert slept == [6.0] and moments[0] == 10.0
    shared = sites.Pace(2, 60, sleep=lambda seconds: (_ for _ in ()).throw(TimeoutError(seconds)))
    done = []
    workers = [threading.Thread(target=lambda: done.append(shared.wait())) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert len(done) == 2
    with pytest.raises(TimeoutError):  # a third within the minute would wait
        shared.wait()


def test_every_request_to_a_site_with_a_limit_per_time_waits_its_turn(monkeypatch, http_folder):
    # the Hub's limit holds for each request wpt dataset sends itself (archives, its API, robots.txt): each
    # one asks the site's pace first
    folder, base = http_folder
    (folder / "file.txt").write_bytes(b"one file\n")
    asked = []
    monkeypatch.setattr(sites, "_paces", {"127.0.0.1": type("Counted", (), {"wait": lambda self: asked.append(1)})()})
    with archives.get(f"{base}/file.txt", "test") as answer:
        assert answer.read() == b"one file\n"
    robots.check(f"{base}/file.txt", "http", "test", {})
    assert len(asked) == 2


def test_only_an_address_that_failed_is_recorded_as_failed(kit, repo, store, http_folder, monkeypatch):
    folder, base = http_folder
    for part in ("main", "copy"):
        (folder / part).mkdir()
        (folder / part / "book.txt").write_bytes(b"a public domain book\n")
    (folder / "robots.txt").write_text("User-agent: GPTBot\nDisallow: /main/\n")
    data = {"type": "http", "address": f"{base}/main/book.txt", "licenses": [kit.entry(["book.txt"])]}
    mirrored = kit.add_source(repo, store, "example", "mirrored", {**data, "mirrors": [{"address": f"{base}/copy/book.txt"}]},
                              fetch=False)
    kit.download(repo, store, mirrored)  # the main address refuses; the mirror serves it

    # an interrupted download is no failure of its address
    (folder / "copy" / "other.txt").write_bytes(b"another book\n")
    stopped = kit.add_source(repo, store, "example", "stopped", {
        "type": "http", "address": f"{base}/copy/other.txt"}, fetch=False)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(download, "fetch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        kit.download(repo, store, stopped)

    found = held(store, mirrored)
    assert found.address == f"{base}/copy/book.txt"
    assert [failed["address"] for failed in found.record["failures"]] == [f"{base}/main/book.txt"]
    assert held_here(store, stopped) is None  # nothing says it failed


def test_a_license_page_is_checked_from_the_store_and_one_that_does_not_match_leaves_its_files_out(
        kit, repo, store, http_folder):
    # the build never contacts upstream: a page statement is read from the
    # copy the download stored, so the page's site going away changes
    # nothing; a page whose words are not the text the entry expects keeps
    # the entry's files out of training
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    (folder / "license.txt").write_bytes(kit.LICENSE_TEXT.encode())
    (folder / "other.txt").write_bytes(b"All rights reserved.\n")
    texts = license_check.LicenseTexts(repo.license_texts())
    matching = kit.add_source(repo, store, "example", "matching", {"type": "http", "address": f"{base}/book.txt",
        "licenses": [kit.entry(["book.txt"], place={"page": f"{base}/license.txt"}, expected=[kit.REVIEWED])]})
    other = kit.add_source(repo, store, "example", "other", {"type": "http", "address": f"{base}/book.txt?again",
        "licenses": [kit.entry(["book.txt"], place={"page": f"{base}/other.txt"}, expected=[kit.REVIEWED])]})
    for page in ("license.txt", "other.txt"):
        (folder / page).unlink()

    held = kept(repo.source(matching), store, texts)
    assert list(held.tree) == ["book.txt"] and held.left_out == []
    assert held.statements[0]["found"] == kit.REVIEWED and held.statements[0]["matched"]
    refused = kept(repo.source(other), store, texts)
    assert refused.tree == {} and refused.left_out == [
        {"why": "entry", "entry": 0, "license": "MIT", "files": 1,
         "detail": f"no license text holds the text there (its id would be "
                   f"{license_check.text_id(b'All rights reserved.' + bytes([10]))}; `wpt dataset add-license-text` "
                   f"adds it, under review)"}]


def test_a_license_page_is_stored_with_the_source_and_a_page_that_fails_stores_nothing(
        kit, repo, store, http_folder):
    # an entry whose statement is an outside page: the build checks it from the store, never upstream, so the
    # download keeps the page with the source's files; a page that cannot be had fails the download whole
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    (folder / "license.html").write_bytes(b"<p>Permission is hereby granted.</p>\n")
    entry = kit.entry(["book.txt"], place={"page": f"{base}/license.html"}, expected=[kit.REVIEWED])
    sid = kit.add_source(repo, store, "example", "paged", {"type": "http", "address": f"{base}/book.txt",
                                                           "licenses": [entry]})
    assert license_page(repo.source(sid), store, f"{base}/license.html") == b"<p>Permission is hereby granted.</p>\n"
    assert license_page(repo.source(sid), store, f"{base}/other.html") is None

    gone = kit.add_source(repo, store, "example", "unpaged", {
        "type": "http", "address": f"{base}/book.txt",
        "licenses": [{**entry, "place": {"page": f"{base}/missing.html"}}]}, fetch=False)
    with pytest.raises(SourceGone, match="license page"):
        kit.download(repo, store, gone)
    assert stored_upstream(repo.source(gone), store) is None  # its book is not kept either
    assert [failed["address"] for failed in held_here(store, gone).error["failures"]] == [f"{base}/missing.html"]


def test_a_failed_new_download_keeps_the_file_the_session_held_and_records_the_error(kit, repo, store, http_folder):
    # a source downloaded again whose second archive is gone: the first archive's members went into a new tar
    # that is dropped, and the session keeps its earlier file, with the failure in its record
    folder, base = http_folder
    (folder / "first.tar").write_bytes(kit.tar_bytes(top="a", files={"one.txt": b"version one\n"}, executable=set(),
                                                     symlinks={}))
    (folder / "second.tar").write_bytes(kit.tar_bytes(top="b", files={"two.txt": b"version one\n"}, executable=set(),
                                                      symlinks={}))
    sid = kit.add_source(repo, store, "example", "parts", {
        "type": "http", "address": f"{base}/first.tar", "addresses": [
            {"folder": "first", "address": f"{base}/first.tar"}, {"folder": "second", "address": f"{base}/second.tar"}],
        "licenses": [kit.entry(["first/**", "second/**"])]})
    before = held_here(store, sid)

    (folder / "first.tar").write_bytes(kit.tar_bytes(top="a", files={"one.txt": b"version two\n"}, executable=set(),
                                                     symlinks={}))
    data = repo.source(sid).data
    moved = [data["addresses"][0], {"folder": "second", "address": f"{base}/moved.tar"}]
    (repo.root / "data" / "sources" / sid / "source.json").write_text(kit.dump_json({**data, "addresses": moved}),
                                                                      encoding="utf-8", newline="\n")
    with pytest.raises(SourceGone, match="moved.tar"):
        kit.download(repo, store, sid)

    after = held_here(store, sid)
    assert after.sha256 == before.sha256 and after.error["kind"] == "SourceGone"
    assert store.read_file(after.files["first/a/one.txt"][1]) == b"version one\n"
    with pytest.raises(NotDownloaded):  # a failed source is left out of a build
        source_tree(repo.source(sid), store)


def test_a_missing_program_is_the_source_s_failure_and_its_record_says_so(kit, repo, store, http_folder, monkeypatch):
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    sid = kit.add_source(repo, store, "example", "book", {"type": "http", "address": f"{base}/book.txt"}, fetch=False)

    def no_program(*_):
        raise MissingTool("git is not installed")

    monkeypatch.setattr("wpt_dataset.download.fetch", no_program)
    [done] = download.run(repo, [sid], store, "download", [])
    assert isinstance(done.error, MissingTool)
    assert held_here(store, sid).error["kind"] == "MissingTool"
    assert [state.error for state in listing(store)[0]] == ["MissingTool: git is not installed"]


def test_a_new_session_removes_the_download_folders_a_stopped_process_left_and_keeps_the_others(
        kit, repo, store, tmp_path, http_folder, monkeypatch):
    # a download stopped by a signal or a crash leaves its temporary folder; the next session on the machine
    # removes it, leaves the one a live process still downloads into and one whose lock file it may not open
    # (another user's), and downloads all the same
    import os
    import stat
    import tempfile
    from pathlib import Path

    from wpt_store.local_files import LOCK
    from wpt_utils.files import lock_file, unlock_file

    if os.name != "nt" and os.geteuid() == 0:
        pytest.skip("root opens a read-only file for writing")
    git, book = two_sources(kit, repo, store, tmp_path, http_folder)
    # a temporary folder of this test's own, so that other tests' live downloads do not show in it; a short name,
    # so that socket paths inside it stay within the system's limit
    temporary = Path(tempfile.mkdtemp(prefix="wpt-test-"))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", None)  # found again from the environment
    names = ("stopped", "live", "foreign")
    stopped, live, foreign = (temporary / f"{download.WORK}{name}" for name in names)
    for folder in (stopped, live, foreign):
        (folder / "address-0" / "files").mkdir(parents=True)
        (folder / "address-0" / "files" / "a-file").write_bytes(b"bytes a download had fetched\n")
    (temporary / f"{stopped.name}{LOCK}").touch()  # its process is gone: nobody holds the lock
    held = lock_file(temporary / f"{live.name}{LOCK}")  # its process is alive
    foreign_lock = temporary / f"{foreign.name}{LOCK}"
    foreign_lock.touch()
    foreign_lock.chmod(stat.S_IREAD)  # this process may not open it for writing
    try:
        assert [item.value for item in download.run(repo, [git, book], store, "download", [])] == ["downloaded"] * 2
        left = sorted(path.name for path in temporary.glob(download.WORK + "*"))
        # the session's own folders are gone too
        assert left == sorted([foreign.name, foreign_lock.name, live.name, f"{live.name}{LOCK}"])
        assert (live / "address-0" / "files" / "a-file").is_file()
        assert (foreign / "address-0" / "files" / "a-file").is_file()
    finally:
        unlock_file(held)
        foreign_lock.chmod(stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(temporary, ignore_errors=True)
