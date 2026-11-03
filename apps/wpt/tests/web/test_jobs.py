"""The job routes: a `wpt` subcommand started as a child process of the
server with the arguments a request gives, and followed through the lines it
printed and the run record it wrote."""

import http.client
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wpt_dataset.repo import Repo
from wpt_store import open_store
from wpt.web import pages
from wpt.web.jobs import Jobs
from wpt.web.pages.jobs import JOBS
from wpt.web.server import make_server

# stands in for `python -m wpt`: it gets the subcommand words and the arguments, as `wpt` would
PROGRAM = """
import os, sys, time
words = sys.argv[1:]
if words[:2] == ["dataset", "check"]:
    print("checked", *words[2:])
    print("a warning", file=sys.stderr)
    sys.exit(3 if "--fail" in words else 0)
if words[:1] == ["train"]:
    gate = words[1]
    while not os.path.exists(gate):
        time.sleep(0.02)
    print("trained")
if words[:1] == ["eval"]:
    from pathlib import Path
    from wpt_store import open_store
    from wpt_telemetry import Telemetry
    with Telemetry.start(open_store(words[1]), "wpt eval", words[1:], "tiny", env_file=Path("no.env")) as telemetry:
        telemetry.log("measured")
if words[:1] == ["manager"]:
    print("waiting", flush=True)
    time.sleep(60)
if words[:1] == ["store"]:
    print(len(sys.stdin.read()), "bytes of input")
"""


@pytest.fixture
def web(tmp_path):
    """A repository served on a free port with jobs run by PROGRAM; gives request(method, target, body) and the
    store the jobs write their records in."""
    root = tmp_path / "wpt"
    (root / "data" / "sources").mkdir(parents=True)
    store = open_store((tmp_path / "store").as_uri())
    running = Jobs(root, command=(sys.executable, "-c", PROGRAM))
    server = make_server(pages.routes(Repo(root), store, running=running), tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def request(method, target, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.request(method, target, None if body is None else json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    yield request, store, running
    running.close()
    server.shutdown()
    server.server_close()


def followed(request, job):
    """The job's view once it is no longer running, and every line it printed."""
    for _ in range(500):
        status, view = request("GET", f"/api/jobs/{job}")
        assert status == 200
        if view["state"] != "running":
            return view, request("GET", f"/api/jobs/{job}/lines/0")[1]["lines"]
        time.sleep(0.02)
    raise AssertionError("the job did not end")


def test_a_job_runs_the_subcommand_with_the_arguments_given_and_its_lines_are_followed(web):
    request, _, _ = web
    status, started = request("POST", "/api/jobs/dataset/check", {"args": ["--fetch", "a b"]})
    assert status == 200 and started["command"] == "wpt dataset check --fetch a b"
    view, lines = followed(request, started["id"])
    assert view["state"] == "succeeded" and view["exit_code"] == 0 and view["ended"]
    assert lines == [{"stream": "stdout", "text": "checked --fetch a b"}, {"stream": "stderr", "text": "a warning"}]
    status, failed = request("POST", "/api/jobs/dataset/check", {"args": ["--fail"]})
    view, _ = followed(request, failed["id"])
    assert view["state"] == "failed" and view["exit_code"] == 3
    assert [job["id"] for job in request("GET", "/api/jobs")[1]] == [started["id"], failed["id"]]


def test_lines_are_followed_from_the_number_asked_for(web):
    request, _, _ = web
    job = request("POST", "/api/jobs/dataset/check", {})[1]["id"]
    followed(request, job)
    status, rest = request("GET", f"/api/jobs/{job}/lines/1")
    assert status == 200 and rest == {"first": 1, "next": 2, "lines": [{"stream": "stderr", "text": "a warning"}]}
    assert request("GET", f"/api/jobs/{job}/lines/5")[1] == {"first": 2, "next": 2, "lines": []}
    assert request("GET", f"/api/jobs/{job}/lines/x")[0] == 404


def test_any_number_of_jobs_run_at_once_and_a_job_reads_no_input(web, tmp_path):
    request, _, _ = web
    gates = [tmp_path / "first", tmp_path / "second"]
    first, second = (request("POST", "/api/jobs/train", {"args": [str(gate)]})[1]["id"] for gate in gates)
    assert {request("GET", f"/api/jobs/{job}")[1]["state"] for job in (first, second)} == {"running"}
    gates[1].write_text("")  # the second ends while the first goes on
    assert followed(request, second)[0]["state"] == "succeeded"
    assert request("GET", f"/api/jobs/{first}")[1]["state"] == "running"
    gates[0].write_text("")
    assert followed(request, first)[1] == [{"stream": "stdout", "text": "trained"}]
    job = request("POST", "/api/jobs/store/list", {})[1]["id"]
    assert followed(request, job)[1] == [{"stream": "stdout", "text": "0 bytes of input"}]


def test_the_run_record_of_a_job_is_found_by_the_job_and_not_by_another(web, tmp_path):
    request, store, _ = web
    mine = request("POST", "/api/jobs/eval", {"args": [store.root.as_uri()]})[1]["id"]
    other = request("POST", "/api/jobs/eval", {"args": [store.root.as_uri()]})[1]["id"]
    quiet = request("POST", "/api/jobs/dataset/check", {})[1]["id"]
    for job in (mine, other, quiet):
        followed(request, job)
    status, found = request("GET", f"/api/jobs/{mine}/records")
    assert status == 200 and len(found["records"]) == 1
    [record] = found["records"]
    assert record["start"]["job"] == mine and record["start"]["command"] == "wpt eval"
    assert "-wpt-eval-tiny-" in record["folder"]
    assert record["end"]["status"] == "succeeded" and record["ids"] == {"inputs": {}, "outputs": {}}
    assert request("GET", f"/api/jobs/{quiet}/records")[1] == {"records": []}  # it wrote none
    assert len(store.record_folders()) == 2


def test_a_job_is_known_to_the_server_that_started_it_only(web):
    request, _, _ = web
    for target in ("/api/jobs/0123abcd", "/api/jobs/0123abcd/lines/0", "/api/jobs/0123abcd/records"):
        status, refused = request("GET", target)
        assert status == 404 and "no job '0123abcd'" in refused["problems"][0]


def test_a_body_that_breaks_the_route_s_schema_starts_nothing(web):
    request, _, _ = web
    assert request("POST", "/api/jobs/train", {"args": "--steps 3"})[0] == 400
    assert request("POST", "/api/jobs/train", {"arguments": []})[0] == 400
    assert request("GET", "/api/jobs")[1] == []


def test_no_route_runs_a_command_that_reads_or_writes_a_persons_text(web):
    request, _, _ = web
    starred = ("read", "write", "conversation", "save-conversation", "task", "save-task", "save-rating", "save-probe")
    for command in starred:
        assert request("POST", f"/api/jobs/dataset/{command}", {"args": []})[0] == 404
    assert not any(name.startswith("web") for name in JOBS)
    # new-source runs without the options that carry a person's text, whatever prefix of them argparse takes
    for option in ("--file", "--file=a=b", "--fil", "--fi", "--license-text", "--license-t", "--license-text=x"):
        assert request("POST", "/api/jobs/dataset/new-source", {"args": ["mine", "notes", option]})[0] == 400, option
    assert request("POST", "/api/jobs/dataset/new-source", {"args": ["mine", "notes", "--license", "CC0-1.0"]})[0] == 200


def test_chat_runs_only_where_no_one_types(web):
    # wpt chat reads what a person types, so a job runs it only to print the model's notices
    request, _, _ = web
    assert request("POST", "/api/jobs/chat", {"args": ["runs/default"]})[0] == 400
    assert request("POST", "/api/jobs/chat", {"args": ["runs/default", "--chat"]})[0] == 400
    assert request("POST", "/api/jobs/chat", {})[0] == 400
    assert request("POST", "/api/jobs/chat", {"args": ["runs/default", "--about"]})[0] == 200


def test_every_job_of_the_command_line_has_a_route(web):
    request, _, _ = web
    operations = request("GET", "/api/openapi.json")[1]["paths"]
    started = {pattern.removeprefix("/api/jobs/") for pattern, methods in operations.items()
               if pattern.startswith("/api/jobs/") and "post" in methods}
    assert started == set(JOBS)
    for subcommand in ("train", "eval", "scaling", "chat", "manager", "store/list", "store/mark", "store/copy-out",
                       "agent/lock-image", "agent/build-image", "dataset/download", "dataset/build",
                       "dataset/migrate"):
        assert subcommand in started
    for pattern in ("/api/jobs", "/api/jobs/{id}", "/api/jobs/{id}/lines/{first}", "/api/jobs/{id}/records"):
        assert "get" in operations[pattern]


def test_a_stopping_server_interrupts_the_jobs_still_running(web):
    request, _, running = web
    job = request("POST", "/api/jobs/manager", {})[1]["id"]
    for _ in range(500):  # until the subcommand has started, so that the interruption finds it
        if request("GET", f"/api/jobs/{job}/lines/0")[1]["lines"]:
            break
        time.sleep(0.02)
    running.close()
    view, _ = followed(request, job)
    assert view["state"] == "failed" and view["exit_code"] != 0


def test_the_subcommands_the_routes_start_exist(tmp_path):
    # the routes name subcommands of the real command: each word list is accepted by its parser
    from wpt import COMMANDS

    assert {words[0] for words, _, _ in JOBS.values()} <= set(COMMANDS)
    help_of = subprocess.run([sys.executable, "-m", "wpt", "dataset", "--help"], capture_output=True, text=True,
                             cwd=tmp_path)
    commands = {words[1] for words, _, _ in JOBS.values() if words[0] == "dataset"}
    assert all(command in help_of.stdout for command in commands)
