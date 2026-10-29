"""`wpt-dataset download`: sources fetched into the store's bundles, each
as its address serves it, in a session recorded in the store, several
at once; and `wpt-store list`, which lists what the store database says."""

import json
import shutil
import threading

import pytest

from wpt_dataset import archives, download, license_check, manifest, robots, sites
from wpt_dataset.cli import main
from wpt_dataset.errors import FetchFailed, SourceGone
from wpt_dataset.kept import kept
from wpt_dataset.sources import (license_page, source_tree, stored_upstream, upstream_key, upstream_manifest,
                                 upstream_tree)
from wpt_store.cli import main as store_command


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


def test_download_keeps_each_source_in_bundles_records_its_session_and_reuses_it_next_time(
        kit, repo, store, tmp_path, http_folder, monkeypatch, capsys):
    git, book = two_sources(kit, repo, store, tmp_path, http_folder)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert main(["--root", str(repo.root), "download", git, book]) == 0

    [folder] = store.record_folders()
    assert "-wpt-dataset-download-2-sources-" in folder
    for sid, files in ((git, kit.FILES), (book, {"book.txt": b"a public domain book\n"})):
        tree, _ = source_tree(repo.source(sid), store)
        for path, content in files.items():
            assert tree[path].placed.bundle.startswith(folder + "/")  # in the session's own bundles
            assert manifest.content(store, tree[path]) == content
    with store.database() as found:
        assert found.execute("SELECT folder, command, ended IS NOT NULL FROM sessions").fetchall() == [
            (folder, "wpt-dataset-download", 1)]
        lines = [(source, outcome, json.loads(got), address, size) for source, outcome, got, address, size in
                 found.execute("SELECT source, outcome, got, address, size FROM lines ORDER BY source")]
    commit = kit.run("git", "rev-parse", "HEAD", cwd=tmp_path / "project").decode().strip()
    assert lines == [(sid, "downloaded", {**got, "manifest": upstream_manifest(repo.source(sid), store)},
                      repo.source(sid).data["address"], size)
                     for sid, got, size in sorted([(git, {"commit": commit}, sum(map(len, kit.FILES.values())) +
                                                    len("docs/guide.txt")),
                                                   (book, {}, len(b"a public domain book\n"))])]

    # upstream gone: the store still holds both, so nothing is downloaded again
    shutil.rmtree(tmp_path / "project")
    (http_folder[0] / "book.txt").unlink()
    assert main(["--root", str(repo.root), "download", git, book]) == 0
    capsys.readouterr()

    assert store_command(["list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "2 sessions:" and len([line for line in out if "wpt-dataset-download" in line]) == 2
    for sid in (git, book):
        [line] = [line for line in out if line.startswith(f"  {sid}: ")]
        assert line.startswith(f"  {sid}: reused; downloaded ")
        assert f" from {repo.source(sid).data['address']} at " in line
        assert line.endswith(f", manifest {upstream_manifest(repo.source(sid), store)}")
    [line] = [line for line in out if line.startswith(f"  {git}: ")]
    assert f" bytes, commit {commit}, manifest " in line  # what the download got, for the record


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
    kit.download(repo, store, sid)  # not marked: the moved revision is not what the store holds

    tree = upstream_tree(repo.source(sid), store)
    assert manifest.content(store, tree["README.md"]) == b"a later readme\n" and "added.bin" in tree
    with store.database() as found:
        got = [json.loads(row[0])["commit"] for row in
               found.execute("SELECT got FROM lines WHERE outcome = 'downloaded' ORDER BY time, number")]
    assert got == [first, later]

    # a commit upstream does not serve (a branch rewritten) fails the source; the store keeps what it held
    path.write_text(kit.dump_json({**repo.source(sid).data, "revision": "1" * 40}), encoding="utf-8", newline="\n")
    with pytest.raises(FetchFailed, match=f"does not serve the commit {'1' * 40}"):
        kit.download(repo, store, sid)


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
        target = kit.open_store((tmp_path / way).as_uri())
        monkeypatch.setattr(sites, "limits", lambda limits=limits: (*limits, {}))
        with kit.session(target) as session:
            done = download.download(repo, ids, target, session, log=lambda *_: None)
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
    # the Hub's limit holds for each request wpt-dataset sends itself (archives, its API, robots.txt): each
    # one asks the site's pace first
    folder, base = http_folder
    (folder / "file.txt").write_bytes(b"one file\n")
    asked = []
    monkeypatch.setattr(sites, "_paces", {"127.0.0.1": type("Counted", (), {"wait": lambda self: asked.append(1)})()})
    with archives.get(f"{base}/file.txt", "test") as answer:
        assert answer.read() == b"one file\n"
    robots.check(f"{base}/file.txt", "http", "test", {})
    assert len(asked) == 2


def test_only_an_address_that_failed_gets_a_failed_line(kit, repo, store, http_folder, monkeypatch):
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

    with store.database() as found:
        lines = found.execute("SELECT source, outcome, address FROM lines ORDER BY session, number").fetchall()
    assert lines == [(mirrored, "failed", f"{base}/main/book.txt"), (mirrored, "downloaded", f"{base}/copy/book.txt")]


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
                   f"{license_check.text_id(b'All rights reserved.' + bytes([10]))}; `wpt-dataset add-license-text` "
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
    assert stored_upstream(upstream_key(repo.source(gone)), store) is None  # its book is not kept either
    with store.database() as found:
        assert found.execute("SELECT outcome, address FROM lines WHERE source = ?", (gone,)).fetchall() == [
            ("failed", f"{base}/missing.html")]


def test_a_failed_new_download_leaves_what_the_store_held(kit, repo, store, http_folder):
    # a marked source downloaded again whose second archive is gone: the first archive's members went into
    # the session's bundles as they were read, but nothing names them, and the source keeps its earlier files
    folder, base = http_folder
    (folder / "first.tar").write_bytes(kit.tar_bytes(top="a", files={"one.txt": b"version one\n"}, executable=set(),
                                                     symlinks={}))
    (folder / "second.tar").write_bytes(kit.tar_bytes(top="b", files={"two.txt": b"version one\n"}, executable=set(),
                                                      symlinks={}))
    sid = kit.add_source(repo, store, "example", "parts", {
        "type": "http", "address": f"{base}/first.tar", "addresses": [
            {"folder": "first", "address": f"{base}/first.tar"}, {"folder": "second", "address": f"{base}/second.tar"}],
        "licenses": [kit.entry(["first/**", "second/**"])]})
    before = source_tree(repo.source(sid), store)[0]

    (folder / "first.tar").write_bytes(kit.tar_bytes(top="a", files={"one.txt": b"version two\n"}, executable=set(),
                                                     symlinks={}))
    (folder / "second.tar").unlink()
    store.mark([sid])
    with pytest.raises(SourceGone, match="second.tar"):
        kit.download(repo, store, sid)

    after = source_tree(repo.source(sid), store)[0]
    assert after == before and manifest.content(store, after["first/a/one.txt"]) == b"version one\n"
