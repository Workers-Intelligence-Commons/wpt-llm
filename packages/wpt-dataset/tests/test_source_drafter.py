"""The source drafter framework (wpt_dataset.source_drafter), run through
`wpt-dataset source-drafter` on a small source drafter written here, with
the real service layer and a store of the test's own: drafts saved, a run
resumed after a stop, skips, needs-a-person and failures, a source that
exists updated in place, and the runs a program started that it starts
again."""

import http.server
import json
import shutil
import threading

import pytest

from wpt_dataset import source_drafter
from wpt_dataset.cli import main
from wpt_dataset.repo import DRAFT_CHECKSUM, source_id
from wpt_utils.parallel import Sizing

# A source drafter driven by a plan file (its `plan` setting): its candidates, what draft() gives each, and a
# list of the candidates it was asked to draft (`calls`). A candidate's "stop" makes draft() stop the run as
# Ctrl+C would, the first time only; its site fact is computed once and counted in `facts`.
PROGRAM = '''
import json
from pathlib import Path

from wpt_dataset.repo import source_id
from wpt_dataset.source_drafter import Candidate, Drafted, NeedsPerson, Skipped

SETTINGS = {"plan": "", "revision": "2026-10-01"}


def candidates(context):
    plan = json.loads(Path(context.settings["plan"]).read_text())
    def listing():
        with open(plan["facts"], "a") as counted:
            counted.write("listed\\n")
        return plan["candidates"]
    for name in context.fact("listing", listing):
        yield Candidate(name, source_id("fake", name, f"https://example.org/{name}.zip"))


def draft(candidate, context):
    plan = json.loads(Path(context.settings["plan"]).read_text())
    with open(plan["calls"], "a") as calls:
        calls.write(candidate.name + "\\n")
    assert context.work.is_dir() and not any(context.work.iterdir())
    what = plan["drafts"].get(candidate.name, {})
    if what.get("stop") and not Path(plan["calls"] + ".stopped").exists():
        Path(plan["calls"] + ".stopped").write_text("")
        raise KeyboardInterrupt
    if what.get("skip"):
        return Skipped(what["skip"])
    if what.get("person"):
        return NeedsPerson(what["person"], {"files": ["LICENSE"]})
    if what.get("raise"):
        raise RuntimeError(what["raise"])
    return Drafted("fake", candidate.name, {
        "type": "http-archive", "address": f"https://example.org/{candidate.name}.zip",
        "revision": context.settings["revision"], "checksum": "sha256:" + "0" * 64,
        "licenses": [{"license": what.get("license", "MIT"), "paths": ["**"], "attribution": "A. Author",
                      "text": "Permission is hereby granted."}],
        "formats": [{"format": "text", "paths": what.get("paths", ["**"])}]},
        supersedes=tuple(what.get("supersedes", [])))
'''


@pytest.fixture
def program(tmp_path, repo, store, monkeypatch):
    """run(candidates, drafts, *options) runs the source drafter through the
    command; calls() lists the candidates draft() was asked to draft."""
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    path = tmp_path / "fake.py"
    path.write_text(PROGRAM)
    plan = tmp_path / "plan.json"
    calls, facts = tmp_path / "calls.txt", tmp_path / "facts.txt"

    class Program:
        @staticmethod
        def run(candidates, drafts, *options):
            plan.write_text(json.dumps({"candidates": candidates, "drafts": drafts, "calls": str(calls),
                                        "facts": str(facts)}))
            try:
                return main(["--root", str(repo.root), "source-drafter", str(path), "--by", "A. Drafter",
                             "--set", f"plan={plan}", *options])
            except SystemExit as stop:  # a run with failures ends so, its record saying it
                return stop.code

        @staticmethod
        def calls():
            return calls.read_text().split() if calls.exists() else []

        @staticmethod
        def listings():
            return len(facts.read_text().split()) if facts.exists() else 0

    return Program


def sid(name):
    return source_id("fake", name, f"https://example.org/{name}.zip")


class Machine:
    """A machine with plenty of everything, as wpt_utils.parallel measures it."""

    def __init__(self, cores):
        self.cores = cores

    def memory(self):
        return 1e12, 1e12

    def idle_cores(self):
        return float(self.cores)


def records(store):
    """(candidates.json, every outcomes.jsonl line) of each run, oldest first."""
    found = sorted((source_drafter.report(store, folder) for folder in source_drafter.runs(store)),
                   key=lambda run: run["started"])
    return [(run["candidates"], run["outcomes"]) for run in found]


def test_drafts_are_saved_under_review_and_every_outcome_is_recorded(program, repo, store):
    # given four candidates, a run saves the one drafted as a new source under review by the run's `by`, records
    # the others' outcomes with their reasons and notes, and exits 1 because one draft failed
    drafts = {"skipped": {"skip": "nothing published"}, "person": {"person": "a license file to read"},
              "broken": {"raise": "the site answered nonsense"}}
    assert program.run(["one", "skipped", "person", "broken"], drafts) == 1
    source = repo.source(sid("one"))
    assert source.data["decision"]["decision"] == "under-review"
    assert source.data["decision"]["by"] == "A. Drafter"
    assert source.data["checksum"] == DRAFT_CHECKSUM
    assert repo.source_ids() == [sid("one")]
    [(candidates, outcomes)] = records(store)
    assert candidates == {"candidates": 4, "chosen": 4, "done_before": 0, "to_draft": 4}
    assert [(o["candidate"], o["outcome"], o["change"]) for o in outcomes] == [
        ("one", "drafted", "new"), ("skipped", "skipped", None), ("person", "needs-a-person", None),
        ("broken", "failed", None)]
    assert outcomes[1]["reason"] == "nothing published"
    assert outcomes[2]["notes"] == {"files": ["LICENSE"]}
    assert "the site answered nonsense" in outcomes[3]["reason"]

    # a second run drafts only the failed candidate again: skips and needs-a-person are done
    assert program.run(["one", "skipped", "person", "broken"], {}) == 0
    calls = program.calls()  # the first run's drafts may run at once, in any order
    assert sorted(calls[:4]) == sorted(["one", "skipped", "person", "broken"]) and calls[4:] == ["broken"]
    assert records(store)[1][0] == {"candidates": 4, "chosen": 4, "done_before": 3, "to_draft": 1}
    assert repo.check() == []


def test_a_stopped_run_resumes_with_what_is_left(program, repo, store, monkeypatch):
    # given a run drafting one at a time (a machine of 1 core) stopped (Ctrl+C) while drafting its second
    # candidate, the same command drafts the second and third, never the first again, and reads its site facts
    # from the store instead of computing them again
    monkeypatch.setattr(source_drafter, "Sizing", lambda what: Sizing(what, Machine(1)))
    with pytest.raises(KeyboardInterrupt):
        program.run(["first", "second", "third"], {"second": {"stop": True}})
    assert repo.source_ids() == [sid("first")]
    assert program.run(["first", "second", "third"], {"second": {"stop": True}}) == 0
    assert program.calls() == ["first", "second", "second", "third"]
    assert repo.source_ids() == sorted(sid(name) for name in ("first", "second", "third"))
    assert program.listings() == 1
    stopped, resumed = records(store)
    assert [o["candidate"] for o in stopped[1]] == ["first"]
    assert resumed[0]["done_before"] == 1

    # --again drafts every chosen candidate, done or not; --only chooses them by name or source id
    assert program.run(["first", "second", "third"], {}, "--again", "--only", "first", "--only", sid("third")) == 0
    assert program.calls()[-2:] == ["first", "third"]
    assert [(o["candidate"], o["change"]) for o in records(store)[-1][1]] == [("first", "unchanged"),
                                                                            ("third", "unchanged")]


def test_an_existing_source_is_updated_in_place(program, kit, repo, store):
    # given admitted sources, a draft with the same license entries writes the new revision and its file-kind
    # map and keeps the decision; one whose license entries differ goes back under review by the run's `by`;
    # and a source with patches is left as it is, stopped with a note
    entry = {"license": "MIT", "paths": ["**"], "attribution": "A. Author", "text": "Permission is hereby granted."}
    for name in ("same", "other", "patched"):
        kit.add_source(repo, store, "fake", name, {"type": "http-archive", "address": f"https://example.org/{name}.zip",
                                                   "revision": "2026-01-01", "licenses": [entry]},
                       patches=["x"] if name == "patched" else (), compute_checksum=False)
    before = (repo.root / "data" / "sources" / sid("patched") / "source.json").read_bytes()
    drafts = {"same": {"paths": ["*.txt"]}, "other": {"license": "Apache-2.0"}}
    assert program.run(["same", "other", "patched"], drafts) == 0

    same = repo.source(sid("same")).data
    assert same["revision"] == "2026-10-01" and same["decision"] == kit.DECISION
    assert same["formats"] == [{"format": "text", "paths": ["*.txt"]}]
    other = repo.source(sid("other")).data
    assert other["revision"] == "2026-10-01" and other["licenses"][0]["license"] == "Apache-2.0"
    assert other["decision"]["decision"] == "under-review" and other["decision"]["by"] == "A. Drafter"
    assert (repo.root / "data" / "sources" / sid("patched") / "source.json").read_bytes() == before
    outcomes = {o["candidate"]: o for o in records(store)[0][1]}
    assert {name: (o["outcome"], o["change"]) for name, o in outcomes.items()} == {
        "same": ("drafted", "updated"), "other": ("drafted", "licenses-changed"), "patched": ("stopped", None)}
    assert "0001-change.patch" in outcomes["patched"]["reason"]
    assert "patched" not in program.calls()  # its candidate names its source: it stops before its draft


def test_drafts_in_worker_processes_give_what_drafts_one_at_a_time_give(program, repo, store, monkeypatch):
    # given the same candidates drafted one at a time (a machine of 1 core) and in worker processes (16 cores:
    # 12 at once) into an empty repository each time, both runs save the same sources and record the same
    # outcomes, in the candidates' order
    names = ["c", "a", "skipped", "b", "person"]
    drafts = {"skipped": {"skip": "nothing published"}, "person": {"person": "a license file to read"}}
    found = []
    for cores in (1, 16):
        for existing in repo.source_ids():
            shutil.rmtree(repo.root / "data" / "sources" / existing)
        monkeypatch.setattr(source_drafter, "Sizing", lambda what, cores=cores: Sizing(what, Machine(cores)))
        assert program.run(names, drafts, "--again") == 0
        outcomes = [(o["candidate"], o["outcome"], o["source"], o["change"], o["reason"]) for o in records(store)[-1][1]]
        found.append((outcomes, {s: repo.source(s).data for s in repo.source_ids()}))
    assert found[0] == found[1]
    assert [line[0] for line in found[0][0]] == names
    assert sorted(found[0][1]) == sorted(sid(name) for name in ("a", "b", "c"))


def test_a_program_starts_again_only_the_unfinished_runs_it_started(program, repo, store, tmp_path):
    # given runs a wpt-web started under its name (two stopped, with other --only candidates, and one
    # finished) and a stopped run under another name, it starts again both of its stopped runs and no other;
    # once one is started again and finishes, only the other is left
    here, elsewhere = "wpt-web-aaaaaaaa", "wpt-web-bbbbbbbb"
    candidates, stop = ["first", "second", "third", "fourth"], {name: {"stop": True} for name in ("first", "second", "fourth")}

    def stopped(name, started_by):
        (tmp_path / "calls.txt.stopped").unlink(missing_ok=True)
        with pytest.raises(KeyboardInterrupt):
            program.run(candidates, stop, "--only", name, "--started-by", started_by)

    stopped("first", f"{here}-00000001")
    stopped("second", f"{here}-00000002")
    assert program.run(candidates, {}, "--only", "third", "--started-by", f"{here}-00000003") == 0
    stopped("fourth", f"{elsewhere}-00000004")
    # a run's record the store cannot read (here its start is not zstd) is left out, never hiding the others
    store.kind.write(f"records/20261024T000000Z-{source_drafter.COMMAND}-broken/start.json.zst", b"not zstd")

    def left():
        return sorted(run["inputs"]["started_by"] for run in source_drafter.to_resume(store, here))

    assert left() == [f"{here}-00000001", f"{here}-00000002"]
    [first] = [run for run in source_drafter.to_resume(store, here) if "first" in run["arguments"]]
    assert main(["--root", str(repo.root), *source_drafter.rerun(first["arguments"]),
                 "--started-by", f"{here}-00000005"]) == 0
    assert left() == [f"{here}-00000002"]


def test_updating_replaces_the_file_kinds_submodules_and_history_start():
    # given an existing git source and a draft that gives other file kinds, other submodules and another history
    # start, all three are the draft's; a draft that gives no submodules removes them; the rest stays as a
    # person left it
    existing = {"revision": "a" * 40, "checksum": "sha256:" + "1" * 64, "licenses": [{"license": "MIT"}],
                "formats": [{"format": "text", "paths": ["**"]}], "submodules": {"x": "https://example.org/x.git"},
                "history": {"since": "b" * 40}, "tags": [{"tags": ["code"], "paths": ["**"], "note": "by a person"}],
                "decision": {"decision": "admitted", "by": "P", "date": "2026-01-01"}}
    drafted = {"revision": "c" * 40, "checksum": "sha256:" + "2" * 64, "licenses": [{"license": "MIT"}],
               "formats": [{"format": "code", "paths": ["**"]}], "submodules": {"y": "https://example.org/y.git"},
               "history": {"since": "d" * 40}}
    written, change = source_drafter.updated(existing, drafted, "A. Drafter", "2026-10-24")
    assert change == "updated"
    assert (written["formats"], written["submodules"], written["history"]) == (
        drafted["formats"], drafted["submodules"], drafted["history"])
    assert written["tags"] == existing["tags"] and written["decision"] == existing["decision"]
    without = {k: v for k, v in drafted.items() if k != "submodules"}
    assert "submodules" not in source_drafter.updated(existing, without, "A. Drafter", "2026-10-24")[0]


def test_a_draft_puts_the_sources_it_supersedes_back_under_review(program, kit, repo, store):
    # given two admitted sources and a refused one, a draft that supersedes one admitted and the refused one
    # puts both under review by the run's `by`, changing nothing else in them, and leaves the other admitted;
    # the new source starts under review
    entry = {"license": "MIT", "paths": ["**"], "attribution": "A. Author", "text": "Permission is hereby granted."}
    refused = {"decision": "refused", "by": "Test Reviewer", "date": "2026-01-01", "reason": "an older dump"}
    for name in ("old", "other", "gone"):
        kit.add_source(repo, store, "fake", name, {"type": "http-archive", "address": f"https://example.org/{name}.zip",
                                                   "revision": "2026-01-01", "licenses": [entry],
                                                   **({"decision": refused} if name == "gone" else {})},
                       compute_checksum=False)
    before = {name: repo.source(sid(name)).data for name in ("old", "gone")}
    assert program.run(["new"], {"new": {"supersedes": [sid("old"), sid("gone")]}}) == 0
    for name in ("old", "gone"):
        data = repo.source(sid(name)).data
        assert data["decision"] == {"decision": "under-review", "by": "A. Drafter", "date": data["decision"]["date"]}
        assert {k: v for k, v in data.items() if k != "decision"} == \
            {k: v for k, v in before[name].items() if k != "decision"}
    assert repo.source(sid("other")).data["decision"] == kit.DECISION
    assert repo.source(sid("new")).data["decision"]["decision"] == "under-review"
    [outcome] = records(store)[0][1]
    assert outcome["superseded"] == [sid("old"), sid("gone")]
    assert repo.check() == []


@pytest.fixture
def site():
    """(the address of a page, the page's state): the page answers its body with its ETag, or 304 (not
    changed) to a request whose If-None-Match names that ETag; state["asked"] keeps each request's
    If-None-Match. The site has no robots.txt."""
    state = {"body": b"first version", "etag": '"v1"', "asked": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/page":
                self.send_error(404)
                return
            state["asked"].append(self.headers.get("If-None-Match"))
            if self.headers.get("If-None-Match") == state["etag"]:
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", state["etag"])
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/page", state
    server.shutdown()


def test_a_page_fetched_again_is_asked_for_only_if_it_changed(repo, store, site):
    # given a page fetched once, a later run (another Context sharing the store) asks the site whether it
    # changed and, on 304, gets the kept page; once the site changes it, the new page comes, and is kept
    address, state = site

    def fetch():
        return source_drafter.Context(repo, store, {}, "fake", "T", "2026-10-25").fetch(address)

    assert fetch() == b"first version"
    assert fetch() == b"first version"
    state.update(body=b"second version", etag='"v2"')
    assert fetch() == b"second version"
    assert fetch() == b"second version"
    assert state["asked"] == [None, '"v1"', '"v1"', '"v2"']
