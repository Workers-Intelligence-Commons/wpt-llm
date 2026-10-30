import hashlib
import http.server
import json
import io
import socket
import stat
import tarfile
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from wpt_dataset import cli, erasure, manifest
from wpt_dataset.errors import (AccessRefused, BadPath, CrawlForbidden, CutFailed, CutKeyMissing, FetchFailed, HostUnreachable, Incomplete,
                                LicenseMapError, NotDownloaded, NotText, PatchFailed, RobotsUnreadable, ServerError,
                                SourceGone, TooManyRequests, UnsupportedEntry)
from protego import Protego

from wpt_dataset import lfs, retry, robots
from wpt_dataset.repo import DATA_FOLDER
from wpt_dataset.sources import source_tree
from wpt_utils import programs


def test_git_mercurial_and_tar_of_one_project_give_one_manifest(kit, repo, store, git_upstream, hg_upstream, http_folder):
    # mirrors of different kinds can back each other up only if their manifests agree
    folder, base = http_folder
    (folder / "project-1.0.tar.gz").write_bytes(kit.tar_bytes())
    git_tree = kit.upstream({"type": "git"}, git_upstream[0], store)
    hg_tree = kit.upstream({"type": "mercurial"}, hg_upstream[0], store)
    tar_tree = kit.upstream({"type": "http", "extract": {"strip": 1}},
                                f"{base}/project-1.0.tar.gz", store)

    assert manifest.digest(git_tree) == manifest.digest(hg_tree) == manifest.digest(tar_tree)
    lines = manifest.manifest_bytes(git_tree).decode().splitlines()
    readme_sha = hashlib.sha256(kit.FILES["README.md"]).hexdigest()
    link_sha = hashlib.sha256(b"docs/guide.txt").hexdigest()
    assert lines == [
        f"100644 {readme_sha} README.md",
        f"100644 {hashlib.sha256(kit.FILES['docs/guide.txt']).hexdigest()} docs/guide.txt",
        f"120000 {link_sha} latest.txt",
        f"100755 {hashlib.sha256(kit.FILES['run.sh']).hexdigest()} run.sh",
    ]


def test_a_mirror_is_used_when_the_main_address_is_down(kit, repo, store, http_folder):
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"a public domain book\n")
    data = {"type": "http", "address": f"{base}/gone.txt", "extract": {"name": "book.txt"}, "mirrors": [{"address": f"{base}/book.txt"}]}
    sid = kit.add_source(repo, store, "example", "book", {**data, "licenses": [kit.entry(["book.txt"])]})

    tree, _ = source_tree(repo.source(sid), store)
    assert list(tree) == ["book.txt"]
    with store.database() as found:  # the main address failed, then the mirror gave it
        assert found.execute("SELECT address, outcome FROM lines").fetchall() == [
            (f"{base}/gone.txt", "failed"), (f"{base}/book.txt", "downloaded")]


def test_a_source_takes_what_its_address_serves_and_keeps_it_when_upstream_changes(kit, repo, store, http_folder):
    # nothing is pinned: a download takes what upstream serves when asked, and a source in the store is not
    # downloaded again because upstream changed (a stale store is accepted)
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"version one\n")
    sid = kit.add_source(repo, store, "example", "book", {"type": "http", "address": f"{base}/book.txt"},
                         fetch=False)
    with pytest.raises(NotDownloaded):
        source_tree(repo.source(sid), store)
    kit.download(repo, store, sid)
    (folder / "book.txt").write_bytes(b"upstream changed it\n")
    kit.download(repo, store, sid)

    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["book.txt"]) == b"version one\n"

    store.mark([sid])  # a person asks for fresh content: the next download takes what upstream serves now
    kit.download(repo, store, sid)
    kit.download(repo, store, sid)  # the mark holds until that download: then the store is used again
    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["book.txt"]) == b"upstream changed it\n"

    def named(content):  # the sha256 the store names a tree's manifest by
        entry = manifest.Entry(manifest.FILE, hashlib.sha256(content).hexdigest())
        return manifest.digest({"book.txt": entry})

    with store.database() as found:  # each line records which tree it got or used, for the record only
        assert [(source, outcome, json.loads(got)) for source, outcome, got in found.execute(
            "SELECT source, outcome, got FROM lines ORDER BY time")] == [
            (sid, "downloaded", {"manifest": named(b"version one\n")}),
            (sid, "reused", {"manifest": named(b"version one\n")}),
            (sid, "downloaded", {"manifest": named(b"upstream changed it\n")}),
            (sid, "reused", {"manifest": named(b"upstream changed it\n")})]


def offer_partial_fetches(folder):
    """Serve partial fetches from a local repository, as GitHub does."""
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        subprocess.run(["git", "config", setting, "true"], cwd=folder, check=True)


def test_a_git_source_never_stores_the_content_of_an_excluded_file_and_downloads_when_the_server_cannot_send_it(
        kit, store, tmp_path):
    # a repository's videos or datasets can weigh gigabytes the source leaves
    # out, and its first download may bring them, but they are never stored;
    # here the excluded file's content is gone from the server, so the fetch
    # with every content fails and the source downloads through the request
    # by id for the files it keeps
    address, commit = kit.git_project(tmp_path / "upstream", files={**kit.FILES, "video.bin": b"large\n"})
    offer_partial_fetches(address)
    blob = kit.run("git", "rev-parse", f"{commit}:video.bin", cwd=address).decode().strip()
    (Path(address) / ".git" / "objects" / blob[:2] / blob[2:]).unlink()

    tree = kit.upstream({"type": "git", "extract": {"exclude": ["video.bin"]}}, address, store)

    without, without_commit = kit.git_project(tmp_path / "without")  # the same project without that file
    assert manifest.digest(tree) == manifest.digest(
        kit.upstream({"type": "git"}, without, store))
    assert "video.bin" not in tree
    assert blob not in {entry.git for entry in tree.values()}
    assert hashlib.sha256(b"large\n").hexdigest() not in {entry.sha256 for entry in tree.values()}


def test_a_kept_submodule_gives_its_files_under_its_folder_and_an_excluded_one_is_skipped(kit, store, tmp_path):
    # a vendored library under its own allowed license enters with its own
    # license entry; one the source leaves out is never fetched
    library, library_commit = kit.git_project(tmp_path / "library", files={"lib.py": b"x = 1\n"}, symlinks={})
    offer_partial_fetches(library)
    parent, _ = kit.git_project(tmp_path / "parent", files={"main.py": b"import lib\n"}, symlinks={})
    for path in ("vendor/lib", "vendor/unknown"):
        kit.run("git", "update-index", "--add", "--cacheinfo", f"160000,{library_commit},{path}", cwd=parent)
    kit.run("git", "commit", "-q", "-m", "add the submodules", cwd=parent)
    data = {"type": "git", "extract": {"exclude": ["vendor/unknown"]}}

    tree = kit.upstream({**data, "submodules": {"vendor/lib": library}}, parent, store)

    assert sorted(tree) == ["main.py", "vendor/lib/lib.py"]
    assert tree["vendor/lib/lib.py"].sha256 == hashlib.sha256(b"x = 1\n").hexdigest()
    with pytest.raises(UnsupportedEntry, match="vendor/lib: a submodule with no address"):
        kit.upstream(data, parent, store)


def test_cuts_then_patches_produce_the_final_tree(kit, repo, store, git_upstream, monkeypatch):
    # an erasure is made with `wpt-dataset cut`, which names the passage by
    # its length and two hashes keyed by the store's cut key, never by its
    # text; the build removes it before the patches, written against the
    # cut text, apply
    original = kit.FILES["README.md"]
    start, end = original.index(b" and the dog"), original.index(b"\n")
    patch = (
        "diff --git a/docs/guide.txt b/docs/guide.txt\n"
        "deleted file mode 100644\n"
        "--- a/docs/guide.txt\n"
        "+++ /dev/null\n"
        "@@ -1,3 +0,0 @@\n" + "-le chat est sur le tapis et le chien dort\n" * 3 +
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1 +1 @@\n"
        "-the cat sat on the mat\n"
        "+the cat sat on the mat.\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100755\n"
        "new mode 100644\n"
    )
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": git_upstream[0]},
                         patches=[patch], fetch=False)
    kit.download(repo, store, sid)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert cli.main(["--root", str(repo.root), "cut", sid, "README.md", str(start), str(end),
                 "--version", repo.version("source", sid)]) == 0

    written = (repo.root / "data" / "sources" / sid / "source.json").read_text()
    assert b"dog" in original and "dog" not in written and repo.check() == []
    [cut] = repo.source(sid).data["cuts"]
    assert cut["length"] == end - start and set(cut) == {"path", "length", "hmac", "rolling"}
    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["README.md"]) == b"the cat sat on the mat.\n"
    assert "docs/guide.txt" not in tree
    assert tree["run.sh"].mode == manifest.FILE
    assert tree["latest.txt"].mode == manifest.SYMLINK  # untouched entries keep their mode


def test_a_cut_removes_every_run_of_its_passage_and_one_not_found_takes_the_source_out(kit, repo, store, tmp_path):
    # every place the passage is found in the file is removed; a passage
    # not in the file, or a store without the cut key, takes the whole
    # source out (CutFailed) until a person makes the cut again
    key = erasure.made_key(store)
    upstream = kit.git_project(tmp_path / "upstream", files={"a.txt": b"one SECRET two SECRET three\n"}, symlinks={})
    found = {"path": "a.txt", **erasure.cut(b" SECRET", key)}
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": upstream[0], "cuts": [found]})
    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["a.txt"]) == b"one two three\n"

    missing = {"path": "a.txt", **erasure.cut(b" NOT THERE", key)}
    gone = kit.add_source(repo, store, "example", "gone", {"type": "git", "address": upstream[0] + "/",
                                                            "cuts": [missing]})
    with pytest.raises(CutFailed, match="not in the file"):
        source_tree(repo.source(gone), store)
    other = kit.open_store((tmp_path / "keyless").as_uri())
    kit.download(repo, other, sid)
    with pytest.raises(CutKeyMissing):
        source_tree(repo.source(sid), other)


def test_a_commit_that_changes_only_data_finds_a_source_s_tree_again_and_a_change_to_the_code_does_not(
        kit, repo, store, git_upstream, monkeypatch):
    # a source's final tree and its text check are keyed by its own files
    # and the code outside data/ (wpt_utils.programs.code_outside):
    # a commit that adds another source changes the whole commit (code())
    # but not that, so neither is made again; a change to the code outside
    # data/ makes both again
    patch = ("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n"
             "-the cat sat on the mat and the dog sat on the log\n+the cat sat on the mat.\n")
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": git_upstream[0]},
                         patches=[patch])
    outside = programs.code_outside(DATA_FOLDER)
    for name in ("_CODE", "_OUTSIDE", "_GIVEN"):  # put back as they were when the test ends
        monkeypatch.setattr(programs, name, getattr(programs, name))

    def made() -> dict[str, int]:
        source_tree(repo.source(sid), store)
        return {kind: len(list((store.root / "refs" / kind).iterdir())) for kind in ("source-trees", "source-not-text")}

    first = made()
    assert first == {"source-trees": 1, "source-not-text": 1}
    kit.add_source(repo, store, "example", "other", {"type": "git", "address": git_upstream[0] + "/"})
    programs.set_code({"code": {"commit": "0" * 40, "changes": None}, "outside": {DATA_FOLDER: outside}})
    assert made() == first
    programs.set_code({"code": {"commit": "0" * 40, "changes": None}, "outside": {DATA_FOLDER: "1" * 64}})
    assert made() == {"source-trees": 2, "source-not-text": 2}


def test_a_patch_that_does_not_apply_is_refused(kit, repo, store, git_upstream):
    patch = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-not there\n+x\n"
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0]}, patches=[patch])
    with pytest.raises(PatchFailed):
        source_tree(repo.source(sid), store)


def test_every_path_needs_exactly_one_license(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0], "licenses": [kit.entry(kit.PROJECT),
                                                                  kit.entry(["docs/**"], license="CC0-1.0")]})
    with pytest.raises(LicenseMapError, match="docs/guide.txt matches 2"):
        source_tree(repo.source(sid), store)


def test_binary_files_must_be_excluded(kit, repo, store, tmp_path):
    upstream = tmp_path / "bin"
    upstream.mkdir()
    (upstream / "text.txt").write_bytes(b"words\n")
    (upstream / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    kit.run("git", "init", "-q", cwd=upstream)
    kit.run("git", "add", "-A", cwd=upstream)
    kit.run("git", "commit", "-q", "-m", "x", cwd=upstream)
    sid = kit.add_source(repo, store, "example", "bin", {"type": "git", "address": str(upstream)})
    with pytest.raises(NotText):
        source_tree(repo.source(sid), store)
    kept = kit.add_source(repo, store, "other", "bin", {"type": "git", "address": str(upstream), "extract": {"exclude": ["*.png"]}})
    assert list(source_tree(repo.source(kept), store)[0]) == ["text.txt"]


def test_paths_differing_only_by_case_are_refused(kit, store, http_folder):
    folder, base = http_folder
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("Notes.txt", "notes.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    (folder / "case.tar").write_bytes(buffer.getvalue())
    with pytest.raises(BadPath):
        kit.upstream({"type": "http"}, f"{base}/case.tar", store)


def test_a_robots_txt_refusal_skips_that_address_and_the_next_mirror_is_tried(kit, repo, store, http_folder):
    # a site's "no" to AI training crawlers holds for that site: nothing is
    # downloaded from it, but a mirror whose own rules allow it still serves the source
    folder, base = http_folder
    (folder / "copy").mkdir()
    for name in ("book.txt", "copy/book.txt"):
        (folder / name).write_bytes(b"a short story\n")
    (folder / "robots.txt").write_text("User-agent: GPTBot\nDisallow: /book.txt\n")
    data = {"type": "http", "address": f"{base}/book.txt"}
    with pytest.raises(CrawlForbidden, match="forbids .*book.txt to GPTBot"):
        kit.upstream(data, data["address"], store)

    alone = kit.add_source(repo, store, "example", "alone", {**data, "licenses": [kit.entry(["book.txt"])]},
                           fetch=False)
    with pytest.raises(CrawlForbidden, match="every address failed: CrawlForbidden: .*GPTBot"):
        kit.download(repo, store, alone)

    mirrored = kit.add_source(repo, store, "example", "mirrored", {
        **data, "mirrors": [{"address": f"{base}/copy/book.txt"}], "licenses": [kit.entry(["book.txt"])]})
    assert set(source_tree(repo.source(mirrored), store)[0]) == {"book.txt"}

    # refusing only assistants that fetch for a person is not a refusal of training
    (folder / "robots.txt").write_text("User-agent: Claude-User\nDisallow: /\n")
    assert set(kit.upstream(data, data["address"], store)) == {"book.txt"}


def test_robots_txt_is_asked_without_the_user_name_of_the_address(monkeypatch):
    # a submodule address like https://user@host/... is valid, but "user@host" is not a host to ask
    asked = []

    def rules(address, robots_url, user_agent):
        asked.append(robots_url)
        return Protego.parse("User-agent: *\nDisallow: /private\n")

    monkeypatch.setattr(robots, "_rules", rules)
    robots.check("https://LeoNatan@github.com:8443/LeoNatan/repo", "git", "wpt-test")
    assert asked == ["https://github.com:8443/robots.txt"]
    with pytest.raises(CrawlForbidden, match="forbids"):  # the rules still apply to the address
        robots.check("https://LeoNatan@github.com/private/repo", "git", "wpt-test")


def test_a_cache_reads_each_robots_txt_once_and_its_rules_still_apply(monkeypatch):
    # a program may ask one site thousands of times; a site may count each robots.txt read against its
    # limit of requests, so a program that keeps a cache reads each site's file once
    asked = []

    def rules(address, robots_url, user_agent):
        asked.append(robots_url)
        return Protego.parse("User-agent: *\nDisallow: /private\n")

    monkeypatch.setattr(robots, "_rules", rules)
    cache = {}
    robots.check("https://api.example.org/repos/a", "http", "wpt-test", cache)
    robots.check("https://api.example.org/repos/b", "http", "wpt-test", cache)
    with pytest.raises(CrawlForbidden, match="forbids"):
        robots.check("https://api.example.org/private/c", "http", "wpt-test", cache)
    robots.check("https://other.example.org/a", "http", "wpt-test", cache)
    assert asked == ["https://api.example.org/robots.txt", "https://other.example.org/robots.txt"]


@pytest.mark.parametrize("answer, usable",[(410, True), (403, False), (429, False)])
def test_only_a_missing_robots_txt_means_no_rules(kit, answer, usable, store):
    # 404 and 410 say the site has no policy; any other answer leaves its policy unknown
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/robots.txt":
                self.send_error(answer)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"a short story\n")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        address = f"http://127.0.0.1:{server.server_address[1]}/book.txt"
        data = {"type": "http"}
        if usable:
            assert set(kit.upstream(data, address, store)) == {"book.txt"}
        else:
            with pytest.raises(RobotsUnreadable, match=f"answered {answer}"):
                kit.upstream(data, address, store)
    finally:
        server.shutdown()


def test_a_download_failing_for_a_passing_reason_is_tried_again(kit, store):
    # a busy server must not fail a fetch of thousands of sources, while a file
    # that is not there is reported at once: asking again gives the same answer
    asked = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            asked[self.path] = asked.get(self.path, 0) + 1
            if self.path == "/book.txt" and asked[self.path] <= 2:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
            elif self.path == "/book.txt":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"a short story\n")
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base, data = f"http://127.0.0.1:{server.server_address[1]}", {"type": "http"}
        tree = kit.upstream(data, f"{base}/book.txt", store)
        assert store.get_bytes(tree["book.txt"].sha256) == b"a short story\n" and asked["/book.txt"] == 3
        with pytest.raises(FetchFailed, match="gone.txt: HTTP Error 404"):
            kit.upstream(data, f"{base}/gone.txt", store)
        assert asked["/gone.txt"] == 1
    finally:
        server.shutdown()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            info = zipfile.ZipInfo(path)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def test_each_of_several_addresses_fills_its_own_folder_of_one_tree(kit, repo, store, http_folder):
    # a publisher may put its newspaper issues in one archive and their
    # metadata in another; the exclusions read the paths with their folder
    folder, base = http_folder
    (folder / "fulltext.zip").write_bytes(zip_bytes({"9200396/issue-1.txt": b"the issue\n"}))
    (folder / "metadata.zip").write_bytes(zip_bytes({"issue-1.txt": b"its metadata\n", "broken.txt": b"\xff\n"}))
    sid = kit.add_source(repo, store, "example", "library", {
        "type": "http", "address": f"{base}/fulltext.zip", "addresses": [{"folder": "fulltext", "address": f"{base}/fulltext.zip"},
                      {"folder": "metadata", "address": f"{base}/metadata.zip"}],
        "extract": {"exclude": ["metadata/broken.txt"]}})

    tree, _ = source_tree(repo.source(sid), store)

    assert sorted(tree) == ["fulltext/9200396/issue-1.txt", "metadata/issue-1.txt"]
    assert manifest.content(store, tree["metadata/issue-1.txt"]) == b"its metadata\n"
    assert repo.check_source(sid) == []


def test_each_of_many_archive_addresses_is_unpacked_into_its_own_folder(kit, repo, store, http_folder):
    # a newspaper collection published as one archive per year (about 146 of them) is one http source: each
    # year's archive downloads and unpacks under its own folder, and no year's files land in another's
    folder, base = http_folder
    years = range(1774, 1920)
    for year in years:
        (folder / f"faro_{year}.tar.gz").write_bytes(
            kit.tar_bytes(top=f"faro_{year}", files={"scan-1.txt": f"the news of {year}\n".encode()},
                          executable=set(), symlinks={}))
    parts = [{"folder": f"faro_{year}", "address": f"{base}/faro_{year}.tar.gz"} for year in years]
    sid = kit.add_source(repo, store, "example", "newspapers", {"type": "http", "address": parts[0]["address"],
                                                                "addresses": parts, "extract": {"strip": 1}})

    tree, _ = source_tree(repo.source(sid), store)

    assert len(parts) == 146
    assert sorted(tree) == sorted(f"faro_{year}/scan-1.txt" for year in years)
    assert all(manifest.content(store, tree[f"faro_{year}/scan-1.txt"]) == f"the news of {year}\n".encode()
               for year in years)


def test_a_failed_download_names_the_address_it_was_for(kit, store, http_folder):
    folder, base = http_folder
    (folder / "fulltext.zip").write_bytes(zip_bytes({"issue-1.txt": b"the issue\n"}))
    data = {"type": "http", "address": f"{base}/fulltext.zip", "addresses": [{"folder": "fulltext", "address": f"{base}/fulltext.zip"},
                          {"folder": "metadata", "address": f"{base}/metadata.zip"}]}

    with pytest.raises(FetchFailed) as raised:
        kit.upstream(data, data["address"], store)

    assert str(raised.value).startswith(f"{base}/metadata.zip: ")


def lfs_pointer(content: bytes) -> bytes:
    """The blob git holds for a file it stores with LFS."""
    return (f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(content).hexdigest()}\n"
            f"size {len(content)}\n").encode()


def test_a_file_git_stores_with_lfs_is_read_as_the_object_it_points_to_and_fetched_once(
        kit, repo, store, tmp_path, monkeypatch):
    # a Hugging Face dataset is a git repository whose data files are LFS pointers: the source's file is the
    # object, fetched by its sha256, and a new download of the same commit fetches no object the store holds.
    # The LFS host's HTTP exchange is lfs.py's own test; here it is simulated
    records = b'{"text": "a public domain record"}\n' * 50
    address, _ = kit.git_project(tmp_path / "dataset", files={"README.md": b"# a dataset\n",
                                                              "data.jsonl": lfs_pointer(records)},
                                 executable=set(), symlinks={})
    fetched = []

    def download(where, sha256, size, target, user_agent):
        fetched.append((where, sha256, size))
        target.write_bytes(records)
        return target

    monkeypatch.setattr(lfs, "download", download)
    sid = kit.add_source(repo, store, "example", "dataset", {"type": "git", "address": address,
                                                             "licenses": [kit.entry(["README.md", "data.jsonl"])]})

    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["data.jsonl"]) == records
    assert manifest.content(store, tree["README.md"]) == b"# a dataset\n"
    assert fetched == [(address, hashlib.sha256(records).hexdigest(), len(records))]

    store.mark([sid])  # a person asks for fresh content: downloaded again, its LFS object taken from the store
    assert [done.value for done in kit.download(repo, store, sid)] == ["downloaded"]
    assert len(fetched) == 1


def unused_port() -> int:
    """A local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_each_way_upstream_fails_is_named(kit, store, monkeypatch):
    # a person reading a failed download's report, and the code that reacts to it, tell apart a source that is
    # gone, a site that refuses too many requests, a server that keeps failing, a host that cannot be reached,
    # a server that refuses access or asks for a password, and a transfer cut short, without reading the message
    monkeypatch.setattr(retry, "BASE", 0.0)
    refusals = {"/private": 401, "/forbidden": 403, "/censored": 451}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if code := refusals.get(self.path.split(".")[0].split("?")[0].split("/info/")[0]):
                self.send_response(code)  # a 401 asks for a password: git and Mercurial must not wait for it
                self.send_header("WWW-Authenticate", 'Basic realm="members"')
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path in ("/busy.txt", "/broken.txt", "/broken.git/info/refs?service=git-upload-pack"):
                self.send_response(429 if self.path == "/busy.txt" else 503)
                self.send_header("Retry-After", "0")
                self.end_headers()
            elif self.path == "/cut.txt":  # says 100 bytes, sends 10, closes
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"ten bytes\n")
            elif self.path == "/archive.tar.gz":
                self.send_response(200)
                self.send_header("Content-Length", str(len(archive) + 50))
                self.end_headers()
                self.wfile.write(archive)
            else:
                self.send_error(404)
            self.close_connection = True

        def log_message(self, *args):
            pass

    archive = kit.tar_bytes()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        nowhere = f"http://127.0.0.1:{unused_port()}"
        for address, kind, failure in (
                (f"{base}/gone.txt", "http", SourceGone), (f"{base}/busy.txt", "http", TooManyRequests),
                (f"{base}/broken.txt", "http", ServerError), (f"{base}/broken.git", "git", ServerError),
                (f"{nowhere}/book.txt", "http", HostUnreachable),
                (f"git://127.0.0.1:{unused_port()}/project", "git", HostUnreachable),
                (f"{base}/private.txt", "http", AccessRefused), (f"{base}/forbidden.txt", "http", AccessRefused),
                (f"{base}/censored.txt", "http", AccessRefused), (f"{base}/private.git", "git", AccessRefused),
                (f"{base}/forbidden.git", "git", AccessRefused), (f"{base}/private", "mercurial", AccessRefused),
                (f"{base}/forbidden", "mercurial", AccessRefused),
                (f"{base}/cut.txt", "http", Incomplete), (f"{base}/archive.tar.gz", "http", Incomplete)):
            with pytest.raises(failure, match=f"^{address}") as raised:
                kit.upstream({"type": kind, **({"revision": "0" * 40} if kind == "git" else {})}, address, store)
            assert type(raised.value) is failure
    finally:
        server.shutdown()


def test_a_source_whose_addresses_fail_in_different_ways_fails_with_its_main_address_kind(
        kit, repo, store, http_folder):
    # the main address is the one a person reads first and acts on; the mirrors' failures are listed with it
    folder, base = http_folder
    gone, nowhere = f"{base}/gone.txt", f"http://127.0.0.1:{unused_port()}/book.txt"
    for number, (main, mirror, failure) in enumerate(((gone, nowhere, SourceGone), (nowhere, gone, HostUnreachable))):
        sid = kit.add_source(repo, store, "example", f"mixed-{number}", {
            "type": "http", "address": main, "extract": {"name": "book.txt"}, "mirrors": [{"address": mirror}],
            "licenses": [kit.entry(["book.txt"])]}, fetch=False)
        with pytest.raises(failure, match=f"every address failed: {failure.__name__}: .*; "
                                          f"{({SourceGone, HostUnreachable} - {failure}).pop().__name__}: ") as raised:
            kit.download(repo, store, sid)
        assert type(raised.value) is failure


def test_an_lfs_object_is_asked_for_by_the_batch_request_and_checked_against_its_pointer(tmp_path):
    # the LFS exchange GitHub, GitLab and Hugging Face serve: a batch request at the repository's LFS endpoint
    # gives the object's address; the bytes must be the pointer's, and an object the host lacks is gone
    content = b"a large file\n" * 1000
    sha256 = hashlib.sha256(content).hexdigest()
    asked = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            asked.append((self.path, self.headers["Accept"]))
            wanted = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["objects"][0]
            if wanted["oid"] == sha256:
                found = {"oid": sha256, "actions": {"download": {"href": f"{base}/objects/{sha256}",
                                                                  "header": {"Authorization": "token"}}}}
            else:
                found = {"oid": wanted["oid"], "error": {"code": 404, "message": "Object does not exist"}}
            self._answer(json.dumps({"objects": [found]}).encode())

        def do_GET(self):
            asked.append((self.path, self.headers["Authorization"]))
            self._answer(content)

        def _answer(self, body):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        target = lfs.download(f"{base}/datasets/owner/name", sha256, len(content), tmp_path / "object", "wpt-test")
        assert target.read_bytes() == content
        assert asked == [("/datasets/owner/name.git/info/lfs/objects/batch", lfs.MEDIA),
                         (f"/objects/{sha256}", "token")]
        with pytest.raises(Incomplete, match="not the pointer's"):  # what arrived is not what the pointer names
            lfs.download(f"{base}/datasets/owner/name", sha256, len(content) - 1, tmp_path / "short", "wpt-test")
        with pytest.raises(SourceGone, match="Object does not exist"):
            lfs.fetch(f"{base}/datasets/owner/name", "0" * 64, 5, "wpt-test")
    finally:
        server.shutdown()
