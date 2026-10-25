"""The agent measures: the shared run of attempts (wpt_eval.attempts) and
the measures that read it, each checked on attempt records made by hand."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import wpt_agent
from wpt_agent import images
from wpt_agent.loop import CONFIDENCE
from wpt_dataset.repo import address_hash, dump_json
from wpt_dataset.tasks import Task, Variant
from wpt_eval import find, main
from wpt_eval.attempts import Record

ROOT = Path(__file__).resolve().parents[3]  # the repository: its agent image holds the check library
TASK = {"schema": 1, "instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}, "needs": [["write", "bash"]]}
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}


def task_source(built, root, upstream, check="#!/bin/sh\necho 1\n", folders=("hello",)) -> str:
    """Writes data/images/tiny/Dockerfile and an admitted source of tasks
    tagged eval, so every one is measure-only; returns its id."""
    (root / "data" / "images" / "tiny").mkdir(parents=True, exist_ok=True)
    (root / "data" / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=ubuntu:24.04\nFROM ${BASE}\nARG PACKAGES\n")
    for folder in folders:
        (upstream / folder / "files").mkdir(parents=True)
        (upstream / folder / "task.json").write_text(json.dumps(TASK))
        (upstream / folder / "files" / "notes.txt").write_text("start\n")
        (upstream / folder / "check").write_text(check)
        (upstream / folder / "check").chmod(0o755)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()
    data = {"schema": 8, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Tasks, by Cy Author, 2026",
                          "text": "Cy Author dedicates these tasks to the public domain.\n"}],
            "formats": [{"format": "task", "paths": ["**"]}], "tags": [{"tags": ["eval"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-17"}}
    data["checksum"] = built.checksum_of(data)
    sid = f"example-tasks-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
    built.download(sid)
    return sid


COUNTER = """from wpt_eval import Result

def run(context):
    records = context.attempts(plain={plain})
    return Result({{"records": len(records), "scores": [r.score for r in records]}}, f"{{len(records)}} records")
"""


def test_the_measures_of_one_command_share_one_run_of_attempts_kept_in_its_record(trained, tmp_path, monkeypatch):
    # every measure of one wpt-eval reads the same records, played once for
    # each value of plain, at the measure-only tasks (tagged eval here), the
    # model always taking its likeliest token; plain offers the same variants as the varied run, under
    # their real names; the run's record keeps each attempt as a line of
    # attempts.jsonl with its changes to /work, an attempt Docker dropped
    # included, marked dropped, and only the scored ones reach the measures
    sid = task_source(trained, tmp_path, tmp_path / "tasks", folders=("look-or-ask/a", "read-the-manual/b"))
    played = []

    def play(task, offers, keep, **settings):
        played.append((task.id, [offer.names for offer in offers], settings["sampling"]))
        outcomes = [wpt_agent.Played(wpt_agent.Attempt(wpt_agent.opening(task, offer), [0], [False], "answered"),
                                     index / len(offers), None, f"changes {index}\n")
                    for index, offer in enumerate(offers, start=1)]
        if task.id.endswith("read-the-manual/b"):
            outcomes[1] = wpt_agent.SandboxError("docker: the container could not start")
        return [keep(index, outcome) for index, outcome in enumerate(outcomes, start=1)]

    monkeypatch.setattr(images, "built", lambda root, name: f"wpt-agent/{name}:test")
    monkeypatch.setattr(wpt_agent, "play", play)
    monkeypatch.setenv("WPT_ATTEMPTS", "2")
    tests = []
    for name, plain in (("first", False), ("second", False), ("plain", True)):
        tests.append(tmp_path / f"{name}.py")
        tests[-1].write_text(COUNTER.format(plain=plain))
    main([str(trained.folder), *map(str, tests)])

    assert [task for task, _, _ in played] == [f"{sid}/look-or-ask/a", f"{sid}/read-the-manual/b"] * 2
    assert {sampling for _, _, sampling in played} == {wpt_agent.GREEDY}
    for (_, varied, _), (_, plain, _) in zip(played[:2], played[2:]):
        for shown, needed in zip(varied, plain):  # one of write and bash, the same one, under its real name
            assert len(needed) == 1 and needed == {name: name for name in needed}
            assert set(needed) <= set(shown.values())
    for name in ("first", "second", "plain"):
        result = json.loads((trained.folder / "evals" / f"{name}.json").read_text())
        assert result["data"] == {"records": 3, "scores": [0.5, 1.0, 0.5]}
    [record] = trained.records("wpt-eval")
    lines = [json.loads(line) for line in trained.store.read_record(record.folder, "attempts.jsonl").splitlines()]
    assert [(line["task"], line["kind"], line["index"], line["plain"], line["dropped"], line.get("score"),
             line.get("changes")) for line in lines] == [
        (f"{sid}/look-or-ask/a", "look-or-ask", index, plain, False, index / 2, f"changes {index}\n")
        if part == "a" else (f"{sid}/read-the-manual/b", "read-the-manual", index, plain, False, 0.5, "changes 1\n")
        if index == 1 else (f"{sid}/read-the-manual/b", "read-the-manual", 2, plain, True, None, None)
        for plain in (False, True) for part in ("a", "b") for index in (1, 2)]
    assert lines[0]["conversation"]["messages"][0]["content"][0]["text"] == TASK["instructions"]
    dropped = lines[3]
    assert dropped["error"] == "docker: the container could not start"
    assert dropped["conversation"]["messages"][0]["content"][0]["text"] == TASK["instructions"]
    assert record.start["inputs"]["attempts"] == 2


def test_an_image_not_built_stops_the_agent_measures_and_names_the_command(trained, tmp_path):
    task_source(trained, tmp_path, tmp_path / "tasks")
    with pytest.raises(SystemExit, match="image 'tiny' is not built on this machine.*wpt-agent build-image tiny"):
        main([str(trained.folder), "tool-robustness"])
    assert not (trained.folder / "evals").exists()


@pytest.mark.xdist_group("docker")
def test_the_shared_run_plays_real_attempts_in_containers(docker_runs, trained, tmp_path, monkeypatch):
    # the trained model writes each attempt in a container and the task's
    # check scores it; every built-in measure then reads those attempts
    task_source(trained, tmp_path, tmp_path / "tasks")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image runs this check
    # the check library the measures read the attempts with, in the task's image
    shutil.copy(ROOT / "data" / "images" / "agent" / "wpt_check.py", tmp_path / "data" / "images" / "tiny")
    main([str(trained.folder), "--attempts", "2", "tool-robustness", "look-before-claim"])

    robustness = json.loads((trained.folder / "evals" / "tool-robustness.json").read_text())["data"]
    assert robustness["plain"]["attempts"] == robustness["varied"]["attempts"] == 2
    assert robustness["plain"]["mean_score"] == robustness["varied"]["mean_score"] == 1.0  # the check prints 1
    claims = json.loads((trained.folder / "evals" / "look-before-claim.json").read_text())["data"]
    assert claims["attempts"] == 2


# -- the measures, on records made by hand ----------------------------------------------------------

@pytest.fixture
def repository(tmp_path, monkeypatch):
    """tmp_path as a repository holding the agent image's check library;
    source(name, tags) writes an admitted source there and returns its id."""
    agent = tmp_path / "data" / "images" / "agent"
    agent.mkdir(parents=True)
    (tmp_path / "data" / "sources").mkdir()
    for name in ("Dockerfile", "wpt_check.py"):
        shutil.copy(ROOT / "data" / "images" / "agent" / name, agent / name)
    monkeypatch.chdir(tmp_path)

    def source(name, tags):
        address = f"https://example.org/{name}"
        data = {"schema": 8, "type": "git", "address": address, "revision": "0" * 40,
                "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "A", "text": "Public domain.\n"}],
                "formats": [{"format": "text", "paths": ["**"]}], "tags": tags, "checksum": "sha256:" + "0" * 64,
                "decision": {"decision": "admitted", "by": "T", "date": "2026-10-17"}}
        sid = f"example-{name}-{address_hash(address)}"
        (tmp_path / "data" / "sources" / sid).mkdir(parents=True)
        (tmp_path / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
        return sid

    return SimpleNamespace(source=source)


def task(folder, files=(), tool_files=(), check_files=(), needs=(), documents=()):
    entry = ("100644", b"x\n")
    return Task(id=f"example-tasks/{folder}", instructions="", image="agent", max_turns=20, max_tokens=4000,
                command_seconds=30, files={path: entry for path in files}, check=b"",
                tool_files={path: entry for path in tool_files}, check_files={path: entry for path in check_files},
                needs=tuple(tuple(Variant(tuple(tools)) for tools in need) for need in needs),
                documents=tuple(documents))


def record(task_, score=1.0, messages=(), plain=False, index=1, tools=None):
    """A record of an attempt at task_ whose assistant and tool turns are
    messages, given as ("call", name, arguments) or ("call", name,
    arguments, text), one assistant message with the text, when given, then
    the call, whose result follows in a tool turn; and ("answer", text), the
    final reply, after which the loop's confidence question and a reply of
    50 follow and metadata.answer holds both. tools maps the names the
    model saw to the real ones."""
    tools = tools or {"read": "read", "write": "write", "bash": "bash", "ask": "ask", "search": "search"}
    conversation = {"messages": [{"role": "user", "content": [{"type": "text", "text": "do it"}]}],
                    "metadata": {"tools": tools, "looking": ["read", "bash", "search"], "answered_by": ["read"],
                                 "answer": None}}
    for number, turn in enumerate(messages):
        if turn[0] == "answer":
            conversation["messages"] += [
                {"role": "assistant", "content": [{"type": "text", "text": turn[1]}]},
                {"role": "user", "content": [{"type": "text", "text": CONFIDENCE}]},
                {"role": "assistant", "content": [{"type": "text", "text": "50"}]}]
            conversation["metadata"]["answer"] = {"answer": turn[1], "confidence": 0.5}
        else:
            said = [{"type": "text", "text": turn[3]}] if len(turn) > 3 else []
            conversation["messages"] += [
                {"role": "assistant", "content": [*said, {"type": "tool_call", "id": f"c{number}", "name": turn[1],
                                                          "arguments": turn[2]}]},
                {"role": "tool", "content": [{"type": "tool_result", "call": f"c{number}",
                                              "content": [{"type": "text", "text": "ok"}]}]}]
    return Record(task_, task_.id.split("/")[1], index, plain, conversation, score)


def measure(name, varied, plain_records=(), tasks=None):
    """The data and text of the built-in measure `name` on those records:
    varied's from context.attempts(), plain_records' from
    context.attempts(plain=True); the measure tasks are tasks, by default
    those of the records."""
    _, path, _ = find(name)
    spec = importlib.util.spec_from_file_location(f"measure_{name.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tasks is None:
        tasks = list({record.task.id: record.task for record in [*varied, *plain_records]}.values())
    result = module.run(SimpleNamespace(attempts=lambda plain=False: list(plain_records if plain else varied),
                                        measure_tasks=lambda: tasks, store=None))
    assert result.text
    return result


def test_look_before_claim_counts_the_claims_on_files_a_look_opened_before(repository):
    # a claim is a file present in /work
    # at the start (.git/ left out, files the attempt wrote not counted),
    # named in an assistant message or in the answer, one per path and
    # message; it is looked first when an earlier looking call named that
    # file itself, a renamed read or a bash command naming it; a folder
    # listing does not count
    given = task("find-the-passage/a", files=["docs/a.md", "docs/b.md", "NOTES.md", ".git/config"])
    first = record(given, messages=[
        ("call", "view", {"path": "docs/a.md"}, "Let me look at docs/a.md."),  # a: its look comes after
        ("call", "shell", {"command": "ls docs"}, "docs/a.md says so, and docs/b.md too."),  # a looked, b not
        ("call", "write", {"path": "out.md", "content": "x"}, "Now out.md, from docs/b.md."),  # b: a listing only
        ("answer", "See /work/docs/a.md, and .git/config.")], tools={"view": "read", "shell": "bash", "write": "write"})
    second = record(given, messages=[("call", "bash", {"command": "cd /work && cat docs/b.md"}),
                                     ("call", "bash", {"command": "true"}, "I read docs/b.md."),
                                     ("answer", "docs/c.md and notes")])
    data = measure("look-before-claim", [first, second]).data
    assert data == {"attempts": 2, "attempts_with_claims": 2, "claims": 6, "looked_first": 3, "share": 0.5}


def test_not_in_the_source_gives_the_removed_and_the_answerable_halves_apart(repository):
    removed = task("evidence-removed/a", check_files=["missing.txt"])
    answerable = task("evidence-removed/b", check_files=["rest.txt"])
    other = task("find-the-passage/c", check_files=["rest.txt"])
    data = measure("not-in-the-source", [record(removed, 1.0), record(removed, 0.0, index=2),
                                         record(answerable, 1.0), record(answerable, 1.0, index=2),
                                         record(answerable, 0.0, index=3), record(other, 1.0)]).data
    assert data["removed"] == {"attempts": 2, "tasks": 1, "passed": 1, "pass_rate": 0.5, "mean_score": 0.5}
    assert data["answerable"]["attempts"] == 3 and data["answerable"]["pass_rate"] == pytest.approx(2 / 3)


def test_ask_versus_look_gives_each_half_s_pass_rate_and_share_that_asked(repository):
    noted = task("look-or-ask/a", needs=[[["read"], ["bash"]], [["write"]]])
    ask_only = task("look-or-ask/b", needs=[[["read"], ["bash"]], [["write"]], [["ask"]]])
    asked = ("call", "ask_user", {"question": "Which file?"})
    data = measure("ask-versus-look", [
        record(noted, 0.0, [("call", "read", {"path": "NOTES.md"}), asked], tools={"ask_user": "ask", "read": "read"}),
        record(noted, 1.0, [("call", "read", {"path": "NOTES.md"})], index=2),
        record(ask_only, 1.0, [asked], tools={"ask_user": "ask"}), record(ask_only, 1.0, [asked], index=2,
                                                                          tools={"ask_user": "ask"})]).data
    assert (data["in_file"]["pass_rate"], data["in_file"]["asked"], data["in_file"]["asked_share"]) == (0.5, 1, 0.5)
    assert (data["ask_only"]["pass_rate"], data["ask_only"]["asked"], data["ask_only"]["asked_share"]) == (1.0, 2, 1.0)


def test_sources_opened_first_counts_attempts_that_looked_before_changing_and_their_calls(repository):
    docs = task("find-the-passage/a", files=["docs/a.md", "quote.txt"])
    pages = task("find-the-passage/b", files=["quote.txt"], tool_files=["web/example.org/a"])
    repository_task = task("commit-and-push/c", files=["README.md"])
    data = measure("sources-opened-first", [
        record(docs, messages=[("call", "read", {"path": "docs/a.md"}), ("call", "write", {"path": "x", "content": ""})]),
        record(docs, messages=[("call", "write", {"path": "x", "content": ""})], index=2),
        record(pages, messages=[("call", "search", {"query": "lamp"}), ("call", "read", {"path": "quote.txt"}),
                                ("call", "bash", {"command": "echo hi"})]),
        record(repository_task, messages=[("call", "bash", {"command": "git push"})])]).data
    assert data == {"attempts": 3, "tasks": 2, "opened_first": 2, "share": pytest.approx(2 / 3),
                    "mean_tool_calls": 2.0}


def test_tool_robustness_compares_plain_and_varied_tools_on_the_same_tasks(repository):
    a, b, c = task("find-the-passage/a"), task("find-the-passage/b"), task("find-the-passage/c")
    varied = [record(a, 1.0), record(a, 0.0, index=2), record(b, 0.0), record(b, 0.0, index=2)]
    plain = [record(a, 1.0, plain=True), record(a, 1.0, plain=True, index=2), record(b, 1.0, plain=True),
             record(b, 0.0, plain=True, index=2), record(c, 1.0, plain=True)]  # c has no varied attempt
    data = measure("tool-robustness", varied, plain).data
    assert data["varied"]["pass_rate"] == 0.25 and data["plain"]["pass_rate"] == 0.75 and data["gap"] == 0.5
    assert data["plain"]["attempts"] == 4


def test_tool_robustness_says_whether_there_was_no_task_or_no_task_scored_in_both_runs(repository):
    # without a measure task there is no attempt; with tasks, Docker may
    # have dropped every attempt of each in one run
    assert measure("tool-robustness", [], tasks=[]).text == "No measure task, so no attempt.\n"
    a = task("find-the-passage/a")
    found = measure("tool-robustness", [], [record(a, 1.0, plain=True)])
    assert found.text.startswith("No measure task has scored attempts in both runs")
    assert found.data["gap"] is None


def test_code_versus_other_calls_a_task_code_when_a_document_it_points_to_is_tagged_code(repository):
    # any pointed document tagged code
    # makes a code task, by its file's tags (a repository pointer's commit:
    # test_history.py); a task pointing to none is another task
    code = repository.source("code", [{"tags": ["code"], "paths": ["**"]}])
    text = repository.source("text", [{"tags": ["en"], "paths": ["**"]}, {"tags": ["code"], "paths": ["src/**"]}])
    tasks = [task("find-the-passage/a", documents=[(text, "book.md", None), (code, "lib.py#3", None)]),
             task("find-the-passage/b", documents=[(text, "book.md", None)]),
             task("find-the-passage/c", documents=[(text, "src/main.c", None)]),
             task("look-or-ask/d")]
    data = measure("code-versus-other", [record(tasks[0], 1.0), record(tasks[1], 0.0), record(tasks[2], 0.0),
                                         record(tasks[3], 1.0), record(tasks[3], 1.0, index=2)]).data
    assert (data["code"]["attempts"], data["code"]["tasks"], data["code"]["pass_rate"]) == (2, 2, 0.5)
    assert (data["other"]["attempts"], data["other"]["tasks"], data["other"]["passed"]) == (3, 2, 2)
    assert data["gap"] == pytest.approx(0.5 - 2 / 3)
