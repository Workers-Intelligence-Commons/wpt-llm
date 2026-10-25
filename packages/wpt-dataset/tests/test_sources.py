import hashlib
import http.server
import io
import stat
import tarfile
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from wpt_dataset import manifest
from wpt_dataset.errors import (BadPath, ChecksumMismatch, CrawlForbidden, CutFailed, FetchFailed, LicenseMapError,
                                NotDownloaded, NotText, PatchFailed, UnsupportedEntry)
from protego import Protego

from wpt_dataset import huggingface, robots
from wpt_dataset.sources import source_tree


def test_git_mercurial_and_tar_of_one_project_give_one_checksum(kit, repo, store, git_upstream, hg_upstream, http_folder):
    # mirrors of different kinds can back each other up only if their manifests agree
    folder, base = http_folder
    (folder / "project-1.0.tar.gz").write_bytes(kit.tar_bytes())
    git_tree = kit.upstream({"type": "git", "revision": git_upstream[1]}, git_upstream[0], store)
    hg_tree = kit.upstream({"type": "mercurial", "revision": hg_upstream[1]}, hg_upstream[0], store)
    tar_tree = kit.upstream({"type": "http-archive", "revision": "2026-10-14", "extract": {"strip": 1}},
                                f"{base}/project-1.0.tar.gz", store)

    assert manifest.checksum(git_tree) == manifest.checksum(hg_tree) == manifest.checksum(tar_tree)
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
    data = {"type": "http-file", "address": f"{base}/gone.txt", "revision": "2026-10-14",
            "extract": {"name": "book.txt"}, "mirrors": [{"address": f"{base}/book.txt"}]}
    checksum = manifest.checksum(kit.upstream(data, f"{base}/book.txt", store))
    sid = kit.add_source(repo, store, "example", "book", {**data, "checksum": checksum}, compute_checksum=False,
                         fetch=True)

    tree, _ = source_tree(repo.source(sid), store)
    assert list(tree) == ["book.txt"]
    with store.database() as found:  # the main address failed, then the mirror gave it
        assert found.execute("SELECT address, outcome FROM lines").fetchall() == [
            (f"{base}/gone.txt", "failed"), (f"{base}/book.txt", "downloaded")]


def test_checksum_mismatch_is_refused_and_nothing_of_it_is_kept(kit, repo, store, http_folder):
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"version one\n")
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-14"}, fetch=False)
    (folder / "book.txt").write_bytes(b"upstream changed it\n")
    with pytest.raises(ChecksumMismatch):
        kit.download(repo, store, sid)

    with pytest.raises(NotDownloaded):
        source_tree(repo.source(sid), store)
    # only the failure is recorded, with the checksum upstream gives now
    changed = manifest.checksum(kit.upstream(repo.source(sid).data, f"{base}/book.txt", store))
    with store.database() as found:
        assert found.execute("SELECT source, outcome, checksum FROM lines").fetchall() == [(sid, "failed", changed)]


def offer_partial_fetches(folder):
    """Serve partial fetches from a local repository, as GitHub does."""
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        subprocess.run(["git", "config", setting, "true"], cwd=folder, check=True)


def test_a_git_source_never_downloads_the_content_of_an_excluded_file(kit, store, tmp_path):
    # a repository's videos or datasets can weigh gigabytes the source leaves
    # out; here the excluded file's content is gone from the server, so any
    # download of it would fail
    address, commit = kit.git_project(tmp_path / "upstream", files={**kit.FILES, "video.bin": b"large\n"})
    offer_partial_fetches(address)
    blob = kit.run("git", "rev-parse", f"{commit}:video.bin", cwd=address).decode().strip()
    (Path(address) / ".git" / "objects" / blob[:2] / blob[2:]).unlink()

    tree = kit.upstream({"type": "git", "revision": commit, "extract": {"exclude": ["video.bin"]}}, address, store)

    without, without_commit = kit.git_project(tmp_path / "without")  # the same project without that file
    assert manifest.checksum(tree) == manifest.checksum(
        kit.upstream({"type": "git", "revision": without_commit}, without, store))


def test_a_kept_submodule_gives_its_files_under_its_folder_and_an_excluded_one_is_skipped(kit, store, tmp_path):
    # a vendored library under its own allowed license enters with its own
    # license entry; one the source leaves out is never fetched
    library, library_commit = kit.git_project(tmp_path / "library", files={"lib.py": b"x = 1\n"}, symlinks={})
    offer_partial_fetches(library)
    parent, _ = kit.git_project(tmp_path / "parent", files={"main.py": b"import lib\n"}, symlinks={})
    for path in ("vendor/lib", "vendor/unknown"):
        kit.run("git", "update-index", "--add", "--cacheinfo", f"160000,{library_commit},{path}", cwd=parent)
    kit.run("git", "commit", "-q", "-m", "add the submodules", cwd=parent)
    commit = kit.run("git", "rev-parse", "HEAD", cwd=parent).decode().strip()
    data = {"type": "git", "revision": commit, "extract": {"exclude": ["vendor/unknown"]}}

    tree = kit.upstream({**data, "submodules": {"vendor/lib": library}}, parent, store)

    assert sorted(tree) == ["main.py", "vendor/lib/lib.py"]
    assert tree["vendor/lib/lib.py"].sha256 == hashlib.sha256(b"x = 1\n").hexdigest()
    with pytest.raises(UnsupportedEntry, match="vendor/lib: a submodule with no address"):
        kit.upstream(data, parent, store)


def test_cuts_then_patches_produce_the_final_tree(kit, repo, store, git_upstream):
    original = kit.FILES["README.md"]
    start, end = original.index(b" and the dog"), original.index(b"\n")
    after = original[:start] + original[end:]
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
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "cuts": [{"path": "README.md", "start": start, "end": end,
                  "sha256_after": hashlib.sha256(after).hexdigest()}],
    }, patches=[patch])

    tree, _ = source_tree(repo.source(sid), store)
    assert manifest.content(store, tree["README.md"]) == b"the cat sat on the mat.\n"
    assert "docs/guide.txt" not in tree
    assert tree["run.sh"].mode == manifest.FILE
    assert tree["latest.txt"].mode == manifest.SYMLINK  # untouched entries keep their mode


def test_a_cut_with_the_wrong_result_is_refused(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "cuts": [{"path": "README.md", "start": 0, "end": 3, "sha256_after": "0" * 64}]})
    with pytest.raises(CutFailed):
        source_tree(repo.source(sid), store)


def test_a_patch_that_does_not_apply_is_refused(kit, repo, store, git_upstream):
    patch = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-not there\n+x\n"
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]}, patches=[patch])
    with pytest.raises(PatchFailed):
        source_tree(repo.source(sid), store)


def test_every_path_needs_exactly_one_license(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "licenses": [{"license": "MIT", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT},
                     {"license": "CC0-1.0", "paths": ["docs/**"], "attribution": "Test Author", "text": "CC0 1.0 Universal\n"}]})
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
    commit = kit.run("git", "rev-parse", "HEAD", cwd=upstream).decode().strip()
    sid = kit.add_source(repo, store, "example", "bin", {"type": "git", "address": str(upstream), "revision": commit})
    with pytest.raises(NotText):
        source_tree(repo.source(sid), store)
    kept = kit.add_source(repo, store, "other", "bin", {"type": "git", "address": str(upstream), "revision": commit,
                                                    "extract": {"exclude": ["*.png"]}})
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
        kit.upstream({"type": "http-archive", "revision": "2026-10-14"}, f"{base}/case.tar", store)


def test_a_robots_txt_refusal_skips_that_address_and_the_next_mirror_is_tried(kit, repo, store, http_folder):
    # a site's "no" to AI training crawlers holds for that site: nothing is
    # downloaded from it, but a mirror whose own rules allow it still serves the source
    folder, base = http_folder
    (folder / "copy").mkdir()
    for name in ("book.txt", "copy/book.txt"):
        (folder / name).write_bytes(b"a short story\n")
    (folder / "robots.txt").write_text("User-agent: GPTBot\nDisallow: /book.txt\n")
    data = {"type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-14"}
    checksum = manifest.checksum(kit.upstream(data, f"{base}/copy/book.txt", store))
    with pytest.raises(CrawlForbidden, match="forbids .*book.txt to GPTBot"):
        kit.upstream(data, data["address"], store)

    alone = kit.add_source(repo, store, "example", "alone", {**data, "checksum": checksum}, compute_checksum=False)
    with pytest.raises(CrawlForbidden, match="every address failed; refused .*GPTBot"):
        kit.download(repo, store, alone)

    mirrored = kit.add_source(repo, store, "example", "mirrored", {
        **data, "checksum": checksum, "mirrors": [{"address": f"{base}/copy/book.txt"}]}, compute_checksum=False,
        fetch=True)
    assert set(source_tree(repo.source(mirrored), store)[0]) == {"book.txt"}

    # refusing only assistants that fetch for a person is not a refusal of training
    (folder / "robots.txt").write_text("User-agent: Claude-User\nDisallow: /\n")
    assert set(kit.upstream(data, data["address"], store)) == {"book.txt"}


def test_robots_txt_is_asked_without_the_user_name_of_the_address(monkeypatch):
    # a submodule address like https://user@host/... is valid, but "user@host" is not a host to ask
    asked = []

    def rules(robots_url, user_agent):
        asked.append(robots_url)
        return Protego.parse("User-agent: *\nDisallow: /private\n")

    monkeypatch.setattr(robots, "_rules", rules)
    robots.check("https://LeoNatan@github.com:8443/LeoNatan/repo", "git", "wpt-test")
    assert asked == ["https://github.com:8443/robots.txt"]
    with pytest.raises(CrawlForbidden, match="forbids"):  # the rules still apply to the address
        robots.check("https://LeoNatan@github.com/private/repo", "git", "wpt-test")


def test_a_cache_reads_each_robots_txt_once_and_its_rules_still_apply(monkeypatch):
    # a source drafter asks one site thousands of times; a site may count each robots.txt read against its
    # limit of requests, so a program that keeps a cache reads each site's file once
    asked = []

    def rules(robots_url, user_agent):
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
        data = {"type": "http-file", "revision": "2026-10-14"}
        if usable:
            assert set(kit.upstream(data, address, store)) == {"book.txt"}
        else:
            with pytest.raises(FetchFailed, match=f"answered {answer}"):
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
        base, data = f"http://127.0.0.1:{server.server_address[1]}", {"type": "http-file", "revision": "2026-10-16"}
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
    # metadata in another; the exclusions read the paths with their folder, as fill-checksum writes them
    folder, base = http_folder
    (folder / "fulltext.zip").write_bytes(zip_bytes({"9200396/issue-1.txt": b"the issue\n"}))
    (folder / "metadata.zip").write_bytes(zip_bytes({"issue-1.txt": b"its metadata\n", "broken.txt": b"\xff\n"}))
    sid = kit.add_source(repo, store, "example", "library", {
        "type": "http-archive", "address": f"{base}/fulltext.zip", "revision": "2026-10-23",
        "addresses": [{"folder": "fulltext", "address": f"{base}/fulltext.zip"},
                      {"folder": "metadata", "address": f"{base}/metadata.zip"}],
        "extract": {"exclude": ["metadata/broken.txt"]}})

    tree, _ = source_tree(repo.source(sid), store)

    assert sorted(tree) == ["fulltext/9200396/issue-1.txt", "metadata/issue-1.txt"]
    assert manifest.content(store, tree["metadata/issue-1.txt"]) == b"its metadata\n"
    assert repo.check_source(sid) == []


def test_a_failed_download_names_the_address_it_was_for(kit, store, http_folder):
    folder, base = http_folder
    (folder / "fulltext.zip").write_bytes(zip_bytes({"issue-1.txt": b"the issue\n"}))
    data = {"type": "http-archive", "address": f"{base}/fulltext.zip", "revision": "2026-10-23",
            "addresses": [{"folder": "fulltext", "address": f"{base}/fulltext.zip"},
                          {"folder": "metadata", "address": f"{base}/metadata.zip"}]}

    with pytest.raises(FetchFailed) as raised:
        kit.upstream(data, data["address"], store)

    assert str(raised.value).startswith(f"{base}/metadata.zip: ")


HUB = "https://huggingface.co/datasets/example/scans"


def simulated_hub(monkeypatch, files: dict[str, bytes]) -> list[str]:
    """The Hub's listing and downloads of a dataset holding files, every one
    but README.md stored with LFS; returns the paths downloaded, in order.
    The HTTP exchange with huggingface.co itself is not exercised."""
    downloaded = []
    listing = [{"type": "file", "path": path, "size": len(content)}
               | ({} if path == "README.md" else {"lfs": {"oid": hashlib.sha256(content).hexdigest()}})
               for path, content in files.items()]

    def fetch(address, revision, path, target, user_agent):
        downloaded.append(path)
        target.write_bytes(files[path])
        return hashlib.sha256(files[path]).hexdigest()

    monkeypatch.setattr(huggingface, "files", lambda address, revision, user_agent: listing)
    monkeypatch.setattr(huggingface, "fetch", fetch)
    monkeypatch.setattr(robots, "check", lambda *args: None)
    return downloaded


def test_a_dataset_s_archives_are_unpacked_into_its_files(kit, repo, store, monkeypatch):
    # a dataset holding one tar.gz of page scans per year: its files are
    # the scans, where unpacking each archive in place puts them, and an
    # excluded archive is never downloaded
    scans = {"paper_1770/1770-08-23_p1.json": b'{"page": 1}\n', "paper_1770/1770-08-30_p1.json": b'{"page": 2}\n'}
    downloaded = simulated_hub(monkeypatch, {
        "README.md": b"# Page scans\n",
        "paper_1770.tar.gz": kit.tar_bytes(top="paper_1770", files={p.split("/")[1]: c for p, c in scans.items()},
                                          executable=set(), symlinks={}),
        "paper_1931.tar.gz": kit.tar_bytes(top="paper_1931", files={"1931-01-01_p1.json": b"{}\n"},
                                          executable=set(), symlinks={})})
    sid = kit.add_source(repo, store, "example", "scans", {
        "type": "huggingface", "address": HUB, "revision": "a" * 40, "extract": {"exclude": ["paper_193[1-9].tar.gz"]}})

    tree, _ = source_tree(repo.source(sid), store)  # as the download kept it, checked against its checksum

    assert sorted(tree) == ["README.md", *scans]
    assert {path: manifest.content(store, tree[path]) for path in scans} == scans
    assert "paper_1931.tar.gz" not in downloaded
