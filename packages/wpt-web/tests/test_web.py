import http.client
import json
import threading

import pytest

from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_web import pages
from wpt_web.server import make_server

ADDRESS = "https://example.org/book.git"
SID = f"example-book-{address_hash(ADDRESS)}"
DOCUMENT = {
    "schema": 4, "type": "git", "address": ADDRESS, "revision": "0" * 40, "checksum": "sha256:" + "0" * 64,
    "licenses": [{"license": "MIT", "paths": ["**"], "attribution": "Copyright (c) 2026 Test Author",
                  "text": "Copyright (c) 2026 Test Author\n\nPermission is hereby granted.\n"}],
    "formats": [{"format": "text", "paths": ["**"]}],
    "decision": {"decision": "refused", "by": "Test Reviewer", "date": "2026-10-15", "reason": "under review"},
}


@pytest.fixture
def web(tmp_path):
    """A repository with one source under review, served on a free port."""
    (tmp_path / "wpt" / "data" / "sources" / SID).mkdir(parents=True)
    path = tmp_path / "wpt" / "data" / "sources" / SID / "source.json"
    path.write_text(dump_json(DOCUMENT), encoding="utf-8", newline="\n")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>interface</p>")
    server = make_server(pages.routes(Repo(tmp_path / "wpt")), static)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def request(method, target, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", port)
        data = None if body is None else json.dumps(body).encode()
        connection.request(method, target, data, {} if body is None else {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        return response.status, (json.loads(raw) if response.getheader("Content-Type") == "application/json" else raw)

    yield request, path
    server.shutdown()
    server.server_close()


def test_a_source_is_listed_read_and_saved_only_through_the_source_rules(web):
    # the page replaces editing source.json by hand: a valid edit is written
    # as every data file is; anything the check would refuse is not written
    request, path = web
    status, listing = request("GET", "/api/sources")
    assert status == 200 and listing == [{"id": SID, "address": ADDRESS, "schema": 4, "decision": "refused",
                                          "reason": "under review", "licenses": ["MIT"], "problems": []}]
    status, detail = request("GET", f"/api/sources/{SID}")
    assert status == 200 and detail["document"] == DOCUMENT and detail["problems"] == []

    noted = {**DOCUMENT, "note": "checked the LICENSE file at this commit"}
    status, saved = request("PUT", f"/api/sources/{SID}", {"document": noted, "version": detail["version"]})
    assert status == 200 and path.read_text(encoding="utf-8") == dump_json(noted)

    before = path.read_text(encoding="utf-8")
    grey = {**noted, "licenses": [{**DOCUMENT["licenses"][0], "license": "CC-BY-SA-4.0"}]}
    status, refused = request("PUT", f"/api/sources/{SID}", {"document": grey, "version": saved["version"]})
    assert status == 422 and refused["problems"] == [f"{SID}: license CC-BY-SA-4.0 is not on the allowed list"]
    status, stale = request("PUT", f"/api/sources/{SID}", {"document": noted, "version": detail["version"]})
    assert status == 409 and "changed on disk" in stale["problems"][0]
    assert path.read_text(encoding="utf-8") == before


def test_the_interface_is_served_and_nothing_outside_its_folder(web):
    # the server hands out the built interface, never another file on the disk
    request, _ = web
    assert request("GET", "/") == (200, b"<p>interface</p>")
    assert request("GET", "/../wpt/data/sources")[0] == 404
