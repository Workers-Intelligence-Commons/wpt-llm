import hashlib
import http.server
import io
import tarfile
import threading
import urllib.error

import pytest

from wpt_dataset import manifest
from wpt_dataset.errors import BadPath, ChecksumMismatch, CrawlForbidden, CutFailed, LicenseMapError, NotText, PatchFailed
from wpt_dataset.robots import RobotsUnreachable
from wpt_dataset.sources import compute_upstream, source_tree


def test_git_mercurial_and_tar_of_one_project_give_one_checksum(kit, repo, store, git_upstream, hg_upstream, http_folder):
    # mirrors of different kinds can back each other up only if their manifests agree
    folder, base = http_folder
    (folder / "project-1.0.tar.gz").write_bytes(kit.tar_bytes())
    git_tree = compute_upstream({"type": "git", "revision": git_upstream[1]}, git_upstream[0], store)
    hg_tree = compute_upstream({"type": "mercurial", "revision": hg_upstream[1]}, hg_upstream[0], store)
    tar_tree = compute_upstream({"type": "http-archive", "revision": "2026-10-14", "extract": {"strip": 1}},
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
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-14"})
    source = repo.source(sid)
    source.data["address"] = f"{base}/gone.txt"
    source.data["mirrors"] = [{"address": f"{base}/book.txt"}]

    tree, _ = source_tree(source, store)
    assert list(tree) == ["book.txt"]


def test_checksum_mismatch_is_refused(kit, repo, store, http_folder):
    folder, base = http_folder
    (folder / "book.txt").write_bytes(b"version one\n")
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-14"})
    (folder / "book.txt").write_bytes(b"upstream changed it\n")
    with pytest.raises(ChecksumMismatch):
        source_tree(repo.source(sid), store)


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
    assert store.get_bytes(tree["README.md"].sha256) == b"the cat sat on the mat.\n"
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
        compute_upstream({"type": "http-archive", "revision": "2026-10-14"}, f"{base}/case.tar", store)


def test_a_robots_txt_refusal_skips_that_address_and_the_next_mirror_is_tried(kit, repo, store, http_folder):
    # a site's "no" to AI training crawlers holds for that site: nothing is
    # downloaded from it, but a mirror whose own rules allow it still serves the source
    folder, base = http_folder
    (folder / "copy").mkdir()
    for name in ("book.txt", "copy/book.txt"):
        (folder / name).write_bytes(b"a short story\n")
    (folder / "robots.txt").write_text("User-agent: GPTBot\nDisallow: /book.txt\n")
    data = {"type": "http-file", "address": f"{base}/book.txt", "revision": "2026-10-14"}
    checksum = manifest.checksum(compute_upstream(data, f"{base}/copy/book.txt", store))
    with pytest.raises(CrawlForbidden, match="forbids .*book.txt to GPTBot"):
        compute_upstream(data, data["address"], store)

    alone = kit.add_source(repo, store, "example", "alone", {**data, "checksum": checksum}, compute_checksum=False)
    with pytest.raises(CrawlForbidden, match="every address failed; refused .*GPTBot"):
        source_tree(repo.source(alone), store)

    mirrored = kit.add_source(repo, store, "example", "mirrored", {
        **data, "checksum": checksum, "mirrors": [{"address": f"{base}/copy/book.txt"}]}, compute_checksum=False)
    assert set(source_tree(repo.source(mirrored), store)[0]) == {"book.txt"}

    # refusing only assistants that fetch for a person is not a refusal of training
    (folder / "robots.txt").write_text("User-agent: Claude-User\nDisallow: /\n")
    assert set(compute_upstream(data, data["address"], store)) == {"book.txt"}


@pytest.mark.parametrize("answer, usable", [(410, True), (403, False), (429, False)])
def test_only_a_missing_robots_txt_means_no_rules(answer, usable, store):
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
            assert set(compute_upstream(data, address, store)) == {"book.txt"}
        else:
            with pytest.raises(RobotsUnreachable, match=f"answered {answer}"):
                compute_upstream(data, address, store)
    finally:
        server.shutdown()


def test_a_download_failing_for_a_passing_reason_is_tried_again(store):
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
        tree = compute_upstream(data, f"{base}/book.txt", store)
        assert store.get_bytes(tree["book.txt"].sha256) == b"a short story\n" and asked["/book.txt"] == 3
        with pytest.raises(urllib.error.HTTPError, match="404"):
            compute_upstream(data, f"{base}/gone.txt", store)
        assert asked["/gone.txt"] == 1
    finally:
        server.shutdown()
