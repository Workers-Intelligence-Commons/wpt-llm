"""`wpt-dataset download`: sources fetched into the store's bundles, each
checked against its checksum, in a session recorded in the store, several
at once; and `wpt-dataset store`, which lists what the store database says."""

import shutil
import threading

import pytest

from wpt_dataset import archives, download, manifest, robots, sites
from wpt_dataset.cli import main
from wpt_dataset.sources import source_tree


def two_sources(kit, repo, store, tmp_path, http_folder):
    """(git source id, http-file source id), neither downloaded yet."""
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    address, commit = kit.git_project(tmp_path / "project")
    git = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": commit},
                         fetch=False)
    book = kit.add_source(repo, store, "example", "book", {
        "type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-25"}, fetch=False)
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
        lines = found.execute("SELECT source, outcome, revision, checksum, address, size FROM lines"
                              " ORDER BY source").fetchall()
    assert lines == [(sid, "downloaded", repo.source(sid).data["revision"], repo.source(sid).data["checksum"],
                      repo.source(sid).data["address"], size)
                     for sid, size in sorted({git: sum(map(len, kit.FILES.values())) + len("docs/guide.txt"),
                                              book: len(b"a public domain book\n")}.items())]

    # upstream gone: the store still holds both, so nothing is downloaded again
    shutil.rmtree(tmp_path / "project")
    (http_folder[0] / "book.txt").unlink()
    assert main(["--root", str(repo.root), "download", git, book]) == 0
    capsys.readouterr()

    assert main(["--root", str(repo.root), "store"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "2 sessions:" and len([line for line in out if "wpt-dataset-download" in line]) == 2
    for sid in (git, book):
        [line] = [line for line in out if line.startswith(f"  {sid}: ")]
        assert line.startswith(f"  {sid}: reused; revision {repo.source(sid).data['revision']}, checksum "
                               f"{repo.source(sid).data['checksum']}; downloaded ")
        assert f" from {repo.source(sid).data['address']} at " in line and line.endswith(" bytes")


def test_downloads_running_at_once_give_what_one_at_a_time_gives(kit, repo, store, tmp_path, http_folder,
                                                                  monkeypatch):
    folder, base = http_folder
    ids = []
    for n in range(6):
        (folder / f"book-{n}.txt").write_bytes(f"book number {n}\n".encode() * (n + 1))
        ids.append(kit.add_source(repo, store, "example", f"book-{n}", {
            "type": "http-file", "address": f"{base}/book-{n}.txt", "revision": "2026-10-25"}, fetch=False))
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
    data = {"type": "http-file", "address": f"{base}/main/book.txt", "revision": "2026-10-25"}
    data["checksum"] = manifest.checksum(kit.upstream(data, f"{base}/copy/book.txt", store))
    mirrored = kit.add_source(repo, store, "example", "mirrored", {**data, "mirrors": [{"address": f"{base}/copy/book.txt"}]},
                              compute_checksum=False)
    kit.download(repo, store, mirrored)  # the main address refuses; the mirror serves it

    # an interrupted download is no failure of its address
    (folder / "copy" / "other.txt").write_bytes(b"another book\n")
    stopped = kit.add_source(repo, store, "example", "stopped", {
        "type": "http-file", "address": f"{base}/copy/other.txt", "revision": "2026-10-25"}, fetch=False)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(download, "fetch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        kit.download(repo, store, stopped)

    with store.database() as found:
        lines = found.execute("SELECT source, outcome, address FROM lines ORDER BY session, number").fetchall()
    assert lines == [(mirrored, "failed", f"{base}/main/book.txt"), (mirrored, "downloaded", f"{base}/copy/book.txt")]
