"""`wpt-dataset download`: sources fetched into the store's bundles, each
checked against its checksum, in a session recorded in the store, several
at once; and `wpt-store list`, which lists what the store database says."""

import shutil

import pytest

from wpt_dataset import download, manifest
from wpt_dataset.cli import main
from wpt_dataset.sources import source_tree
from wpt_store.cli import main as store_command


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

    assert store_command(["list"]) == 0
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
        monkeypatch.setattr(download, "site_limits", lambda limits=limits: limits)
        with kit.session(target) as session:
            done = download.download(repo, ids, target, session, log=lambda *_: None)
        results[way] = (done, [manifest.manifest_bytes(source_tree(repo.source(sid), target)[0]) for sid in ids])
    assert results["at once"] == results["one at a time"]
    assert [d.value for d in results["at once"][0]] == ["downloaded"] * 6


def test_a_site_s_limit_covers_the_hosts_under_it_and_every_other_site_takes_the_default():
    limits, default = download.site_limits()  # the file this package ships
    assert default >= 1 and {"github.com", "wikimedia.org", "huggingface.co"} <= set(limits)
    assert download.site_of("https://dumps.wikimedia.org/enwiki/latest/x.xml.bz2", limits) == "wikimedia.org"
    assert download.site_of("https://github.com/owner/project", limits) == "github.com"
    assert download.site_of("https://notgithub.com/owner/project", limits) is None
    assert download.site_limits("# a comment\n\nexample.org 5\ndefault 1\n") == ({"example.org": 5}, 1)


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
