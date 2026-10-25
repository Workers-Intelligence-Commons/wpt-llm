"""The source drafter runs wpt-web starts (wpt_web.pages.source_drafters):
real `wpt-dataset source-drafter` processes on a repository and a store of
the test's own. A stop ends the drafts under way, in worker processes
too; a run that ends before its record says why on the page's list."""

import http.client
import json
import os
import threading
import time

import pytest

from wpt_dataset.repo import Repo
from wpt_dataset.source_drafter import report, runs
from wpt_store import open_store
from wpt_web import pages
from wpt_web.pages.source_drafters import Runs
from wpt_web.server import make_server

# A source drafter whose drafts mark when they start, take 3 seconds, then mark when they finish.
SLOW = '''
import time
from pathlib import Path

from wpt_dataset.source_drafter import Candidate, Skipped

SETTINGS = {"marks": ""}


def candidates(context):
    return [Candidate("a"), Candidate("b")]


def draft(candidate, context):
    marks = Path(context.settings["marks"])
    (marks / f"{candidate.name}.started").write_text("")
    time.sleep(3)
    (marks / f"{candidate.name}.finished").write_text("")
    return Skipped("slow")
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "wpt"
    (root / "data" / "sources").mkdir(parents=True)
    store = open_store((tmp_path / "store").as_uri())
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())  # the run's process reads it
    return Repo(root), store


@pytest.mark.skipif(os.name == "nt", reason="the stop of a process group is Ctrl+Break there, not Ctrl+C")
def test_a_stop_ends_the_drafts_under_way(setup, tmp_path):
    # given a run drafting (in worker processes when the machine lets several drafts run at once), a stop ends
    # every draft under way at once, and the run's record ends as stopped, so wpt-web starts it again next time
    repo, store = setup
    program, marks = tmp_path / "slow.py", tmp_path / "marks"
    program.write_text(SLOW)
    marks.mkdir()
    started = Runs(repo)
    started.start(["source-drafter", str(program), "--by", "T", "--set", f"marks={marks}"])
    deadline = time.monotonic() + 60
    while not list(marks.glob("*.started")):
        assert time.monotonic() < deadline, "the drafts never started"
        time.sleep(0.1)
    started.stop()
    time.sleep(4)  # longer than a draft takes
    assert list(marks.glob("*.finished")) == []
    [folder] = runs(store)
    assert report(store, folder)["end"]["exit_code"] == 130


def test_a_run_that_ends_before_its_record_is_listed_with_its_error(setup):
    # given a run whose process ends before it writes its record, the list of runs names it, its exit
    # code and its error, since nothing in the store does
    repo, store = setup
    started = Runs(repo)
    name = started.start(["source-drafter", "missing.py", "--by", "T"])
    server = make_server(pages.routes(repo, store, source_drafter_runs=started), repo.root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        deadline = time.monotonic() + 60
        while name in started.running():
            assert time.monotonic() < deadline, "the run never ended"
            time.sleep(0.1)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("GET", "/api/source-drafter-runs")
        response = connection.getresponse()
        assert response.status == 200
        listed = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
    assert listed["runs"] == []
    [ended] = listed["not_started"]
    assert ended["started_by"] == name and ended["exit_code"] != 0
    assert "missing.py: not a built-in source drafter" in ended["error"]
