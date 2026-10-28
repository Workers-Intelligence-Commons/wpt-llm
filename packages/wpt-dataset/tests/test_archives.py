"""An archive an http source unpacks, read while it downloads, member by member: the
same manifest, documents, Markdown and `check --fetch` outcome as a copy
of the archive on the disk gives, and a zip's table of contents read
through a range request where the server serves ranges."""

import functools
import hashlib
import http.server
import io
import json
import stat
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path

import pytest

from wpt_dataset import archives, manifest, retry
from wpt_dataset.build import gather
from wpt_dataset.cli import main
from wpt_dataset.manifest import EXECUTABLE, FILE, SYMLINK, Entry
from wpt_dataset.sources import source_tree
from wpt_store import open_store
from wpt_store.sessions import Session

README, GUIDE, RUN = b"the cat sat on the mat\n", b"le chat est sur le tapis\n" * 40, b"#!/bin/sh\necho hello\n"


def tar_bytes(members, compression=""):
    """A tar of members (name, type, content or link target, mode)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=f"w:{compression}") as archive:
        for name, kind, value, mode in members:
            info = tarfile.TarInfo(name)
            info.type, info.mode = kind, mode
            if kind == tarfile.REGTYPE:
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
            else:
                info.linkname = value
                archive.addfile(info)
    return buffer.getvalue()


def zip_bytes(members):
    """A zip of members (name, unix mode, content)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, mode, content in members:
            info = zipfile.ZipInfo(name)
            info.create_system, info.external_attr = 3, mode << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def project_zip():
    return zip_bytes([("project/", stat.S_IFDIR | 0o755, b""), ("project/README.md", stat.S_IFREG | 0o644, README),
                      ("project/docs/guide.txt", stat.S_IFREG | 0o644, GUIDE),
                      ("project/run.sh", stat.S_IFREG | 0o755, RUN),
                      ("project/latest.txt", stat.S_IFLNK | 0o777, b"docs/guide.txt")])


PROJECT = {"README.md": (FILE, README), "docs/guide.txt": (FILE, GUIDE), "run.sh": (EXECUTABLE, RUN),
           "latest.txt": (SYMLINK, b"docs/guide.txt")}
PAGES = [{"identifier": 7, "date_modified": "2025-03-02T10:00:00Z", "html": "<p>7 newer</p>"},
         {"identifier": 8, "date_modified": "2025-01-01T00:00:00Z", "html": "<p>8 only</p>"},
         {"identifier": 7, "date_modified": "2025-03-01T10:00:00Z", "html": "<p>7 older</p>"}]
PAGES_FILE = "".join(json.dumps(row | {"license": [{"identifier": "CC0-1.0"}]}) + "\n" for row in PAGES).encode()


def case(name, kit):
    """(file name, its bytes, whether its server serves ranges, the source
    without its address, the tree it must give {path: (mode,
    content)}) of one kind of archive."""
    if name == "tar.gz":  # with a hard link to a kept member, which streaming takes from what it read
        members = [("project-1.0/README.md", tarfile.REGTYPE, README, 0o644),
                   ("project-1.0/docs/guide.txt", tarfile.REGTYPE, GUIDE, 0o644),
                   ("project-1.0/run.sh", tarfile.REGTYPE, RUN, 0o755),
                   ("project-1.0/latest.txt", tarfile.SYMTYPE, "docs/guide.txt", 0o777),
                   ("project-1.0/copy.md", tarfile.LNKTYPE, "project-1.0/README.md", 0o644),
                   ("project-1.0/empty", tarfile.DIRTYPE, "", 0o755)]
        return ("project.tar.gz", tar_bytes(members, "gz"), True, {"extract": {"strip": 1}},
                PROJECT | {"copy.md": (FILE, README)})
    if name == "tar":  # with a hard link to an excluded member: its content went by unread
        members = [("README.md", tarfile.REGTYPE, README, 0o644),
                   ("excluded/original.txt", tarfile.REGTYPE, GUIDE, 0o644),
                   ("kept-copy.txt", tarfile.LNKTYPE, "excluded/original.txt", 0o644)]
        return ("project.tar", tar_bytes(members), True, {"extract": {"exclude": ["excluded/**"]}},
                {"README.md": (FILE, README), "kept-copy.txt": (FILE, GUIDE)})
    if name in ("zip, ranges served", "zip, ranges served unannounced", "zip, no ranges"):
        return "project.zip", project_zip(), name != "zip, no ranges", {"extract": {"strip": 1}}, PROJECT
    assert name == "records, newest rule"
    archive = tar_bytes([("pages/pages_0.ndjson", tarfile.REGTYPE, PAGES_FILE, 0o644)], "gz")
    return ("pages.tar.gz", archive, True, {
        "extract": {"strip": 1},
        "licenses": [kit.entry(["*.ndjson"], license="CC0-1.0", place={"field": "license"},
                               expected=['[{"identifier": "CC0-1.0"}]'])],
        "formats": [{"format": "records", "paths": ["*.ndjson"], "text": "html", "as": "html", "id": "identifier",
                     "newest": "date_modified"}]}, {"pages_0.ndjson": (FILE, PAGES_FILE)})


def on_disk(address, extractor, tmp, user_agent):
    """The archive downloaded whole into a file, then read from that file."""
    with urllib.request.urlopen(urllib.request.Request(address, headers={"User-Agent": user_agent})) as answer:
        (tmp / "download").write_bytes(answer.read())
    manifest.read_archive(tmp / "download", extractor)


def added_to_bundles(monkeypatch) -> list[str]:
    """The sha256 of every content a session adds to its bundles from now on."""
    added, add = [], Session.add

    def counted(self, source):
        placed = add(self, source)
        added.append(placed.sha256)
        return placed

    monkeypatch.setattr(Session, "add", counted)
    return added


@pytest.mark.parametrize("name", ["tar.gz", "tar", "zip, ranges served", "zip, ranges served unannounced",
                                  "zip, no ranges", "records, newest rule"])
def test_an_archive_read_while_it_downloads_gives_what_a_copy_on_the_disk_gives(
        name, kit, repo, tmp_path, http_folder, ranged_folder, unannounced_ranged_folder, monkeypatch, capsys):
    # a zip's end is asked for through a range request even from a server that does not say it serves
    # ranges: a range in answer means it is read in pieces, the whole file in answer means it is saved
    file, data, ranges, fields, expected = case(name, kit)
    if name == "zip, ranges served unannounced":
        ranged_folder = unannounced_ranged_folder
    folder, base = ranged_folder[:2] if ranges else http_folder
    (folder / file).write_bytes(data)
    sid = kit.add_source(repo, None, "example", "archive", {
        "type": "http", "address": f"{base}/{file}",
        "licenses": [kit.entry(kit.covering(expected),
                               **kit.statement_in(repo, "README.md", expected.get("README.md", (FILE, b""))[1]))],
        **fields},
        fetch=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    results, asked, saved, save = {}, ranged_folder[2], [], archives._save
    monkeypatch.setattr(archives, "_save", lambda *a: saved.append(a[2]) or save(*a))
    added = added_to_bundles(monkeypatch)
    for way in ("read while it downloads", "a copy on the disk"):
        store = open_store((tmp_path / way).as_uri())
        monkeypatch.setenv("WPT_STORE", store.root.as_uri())
        if way == "a copy on the disk":
            monkeypatch.setattr(archives, "read", on_disk)
        asked.clear()
        added.clear()
        capsys.readouterr()

        code = main(["--root", str(repo.root), "check", "--fetch", sid])

        out = [line for line in capsys.readouterr().out.splitlines() if line.startswith(sid)]
        tree, _ = source_tree(repo.source(sid), store)
        documents, _, _ = gather(repo, store, log=lambda _: None)
        results[way] = (code, out, manifest.manifest_bytes(tree),
                        [(d.id, store.get_bytes(d.sha256)) for d in documents])
        if way == "read while it downloads":
            archive_gets = [wanted for path, wanted in asked if path == f"/{file}"]
            assert len(added) == len(set(added))  # each content in the bundles once
            streamed_saves = list(saved)

    assert results["read while it downloads"] == results["a copy on the disk"]
    code, out, listing, documents = results["read while it downloads"]
    assert code == 0 and out and documents
    assert listing == manifest.manifest_bytes({path: Entry(mode, hashlib.sha256(content).hexdigest())
                                               for path, (mode, content) in expected.items()})
    if name == "records, newest rule":
        assert [(i.rsplit("#", 1)[1], text) for i, text in documents] == [("7", b"7 newer\n"), ("8", b"8 only\n")]
    if name.startswith("zip, ranges served"):  # one GET for its first bytes, then its end through a range
        assert archive_gets[0] is None and any(wanted and wanted != "bytes=0-" for wanted in archive_gets[1:])
    if name == "tar.gz":
        assert archive_gets == [None]  # downloaded once, read as it came
    if name == "tar":  # read twice as it came, for the excluded member its hard link needs
        assert archive_gets == [None, None]
    assert len(streamed_saves) == (name == "zip, no ranges")  # only a zip with no ranges goes to the disk


def test_an_archive_whose_answer_is_cut_is_read_again_and_each_member_stored_once(kit, repo, store, tmp_path,
                                                                                   monkeypatch):
    folder, asked = tmp_path / "cut", []
    folder.mkdir()

    class CutOnce(kit.QuietHandler):
        """Serves a file, the first answer cut halfway through: it announces
        the whole file's size, sends half of it, and closes the connection."""

        def do_GET(self):
            if self.path != "/project.tar.gz":  # robots.txt
                self.send_error(404)
                return
            asked.append(self.path)
            data = Path(self.translate_path(self.path)).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:len(data) // 2] if len(asked) == 1 else data)
            self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(CutOnce, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    big = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(8000)).encode()  # compresses little
    tree = {"README.md": (FILE, README), "big.txt": (FILE, big), "guide.txt": (FILE, GUIDE)}
    (folder / "project.tar.gz").write_bytes(tar_bytes([(path, tarfile.REGTYPE, content, 0o644)
                                                       for path, (_, content) in tree.items()], "gz"))
    sid = kit.add_source(repo, store, "example", "project", {
        "type": "http", "address": f"http://127.0.0.1:{server.server_address[1]}/project.tar.gz",
        "licenses": [kit.entry(sorted(tree))]}, fetch=False)
    monkeypatch.setattr(retry, "BASE", 0)  # no wait before the new try
    added = added_to_bundles(monkeypatch)

    kit.download(repo, store, sid)
    server.shutdown()

    assert len(asked) == 2
    assert sorted(added) == sorted(hashlib.sha256(content).hexdigest() for _, content in tree.values())
    found, _ = source_tree(repo.source(sid), store)
    assert {path: manifest.content(store, entry) for path, entry in found.items()} == \
        {path: content for path, (_, content) in tree.items()}
