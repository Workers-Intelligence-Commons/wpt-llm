"""The rules every write of the service layer follows (wpt_dataset/repo.py;
docs/architecture.md, "The service layer"): a write through a symbolic link
lands on its target inside the repository and nowhere outside it, one lock
around each version check and its write, across threads and processes,
JSON that no line splitter breaks (those three, maintainer's decisions,
2026-10-18), a save that keeps a file's layout where the meaning did not
change, conversations and tasks only in an empty source, a task's links
kept as they are, and a new source that leaves nothing behind when it
fails."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wpt_dataset.edit import unpack, write_patch
from wpt_dataset import repo as service
from wpt_dataset.errors import InvalidFile, NotText, RuleViolation, StaleFile
from wpt_dataset.repo import empty_source

TASK = {"schema": 1, "instructions": "Print hello.", "image": "tiny",
        "limits": {"max_turns": 2, "max_tokens": 100, "command_seconds": 5}, "needs": []}


def own_source(repo, store, formats):
    document = empty_source("notes", license="CC0-1.0", attribution="A. Person, notes",
                            text="CC0 1.0 Universal\n", by="A. Person", formats=formats)
    return repo.create_source("mine", "notes", document, {}, "first", store)


def edit(repo, store, sid, tmp_path, name, change):
    """Record change(folder) on the unpacked files as the next patch, as a person would with unpack and patch."""
    folder = tmp_path / name
    unpack(repo, repo.source(sid), store, folder)
    change(folder)
    write_patch(repo, repo.source(sid), store, folder, name)


def test_a_change_through_a_link_of_a_source_lands_on_its_target_inside_the_source_only(kit, repo, store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_bytes(b"not the source's\n")
    sid = own_source(repo, store, {"**": "text"})

    def links(folder):
        (folder / "notes.txt").write_bytes(b"my notes\n")
        (folder / "sub").mkdir()
        (folder / "sub" / "old.txt").write_bytes(b"old\n")
        os.symlink(str(outside / "kept.txt"), folder / "file-link")
        os.symlink(str(outside), folder / "folder-link")
        os.symlink("notes.txt", folder / "notes-link")
        os.symlink("sub", folder / "sub-link")
    edit(repo, store, sid, tmp_path, "links", links)

    version = repo.source_version(sid)
    for path in ("file-link", "folder-link/new.txt", "folder-link/kept.txt"):  # leads outside the source
        with pytest.raises(RuleViolation, match="outside the source"):
            repo.change_source_files(sid, {path: b"written\n"}, "through", version, store)
    assert sorted(p.name for p in outside.iterdir()) == ["kept.txt"]
    assert (outside / "kept.txt").read_bytes() == b"not the source's\n"

    repo.change_source_files(sid, {"notes-link": b"changed\n", "sub-link/new.txt": b"new\n"}, "inside", version, store)
    assert repo.read_source_file(sid, "notes.txt", store)[1] == b"changed\n"  # the link's target changed
    assert repo.read_source_file(sid, "sub/new.txt", store)[1] == b"new\n"
    _, files = repo.source_files(sid, store)
    assert sorted(files) == ["file-link", "folder-link", "notes-link", "notes.txt", "sub-link", "sub/new.txt",
                             "sub/old.txt"]
    assert files["notes-link"]["size"] == len("notes.txt")  # the links are still links

    repo.change_source_files(sid, {"file-link": None}, "unlink", repo.source_version(sid), store)  # the link itself may go
    repo.change_source_files(sid, {"folder-link": None}, "unlink-folder", repo.source_version(sid), store)
    repo.change_source_files(sid, {"sub-link/old.txt": None}, "remove-through", repo.source_version(sid), store)
    _, files = repo.source_files(sid, store)
    assert sorted(files) == ["notes-link", "notes.txt", "sub-link", "sub/new.txt"]



def test_a_write_through_a_link_lands_on_its_target_inside_the_repository_only(kit, repo, tmp_path):
    mixture = {"schema": 1, "pools": [{"name": "a", "select": "a", "weight": 1}]}
    mixtures = repo.root / "data" / "recipes" / "mixtures"
    outside = tmp_path / "outside" / "m.json"
    outside.parent.mkdir()
    outside.write_text("{}\n")
    os.symlink(outside, mixtures / "out.json")
    with pytest.raises(RuleViolation, match="outside the repository"):
        repo.save("mixture", "out", mixture, repo.version("mixture", "out"))
    assert outside.read_text() == "{}\n"

    inside = repo.root / "shared" / "in.json"  # inside the repository, outside data/
    inside.parent.mkdir()
    inside.write_text("{}\n")
    os.symlink(inside, mixtures / "in.json")
    version = repo.save("mixture", "in", mixture, repo.version("mixture", "in"))
    assert (mixtures / "in.json").is_symlink() and json.loads(inside.read_text()) == mixture
    assert repo.version("mixture", "in") == version
    (repo.root / "data" / "images").mkdir()
    os.symlink(repo.root / "shared", repo.root / "data" / "images" / "tiny", target_is_directory=True)  # a folder
    repo.save("dockerfile", "tiny", "FROM debian:trixie-slim\n", "")
    assert (repo.root / "shared" / "Dockerfile").read_text() == "FROM debian:trixie-slim\n"


def test_two_saves_of_one_version_cannot_both_pass(kit, repo, monkeypatch):
    # the version check and the write happen under one lock: the slower save finds the file changed
    mixture = {"schema": 1, "pools": [{"name": "a", "select": "a", "weight": 1}]}
    version = repo.save("mixture", "m", mixture, "")
    rules = repo._rules

    def slow(*args):
        time.sleep(0.2)  # long enough for the other save to reach its own check
        rules(*args)
    monkeypatch.setattr(repo, "_rules", slow)
    results = []

    def save(name):
        try:
            results.append(repo.save("mixture", "m", {**mixture, "pools": [{**mixture["pools"][0], "name": name}]},
                                     version))
        except StaleFile:
            results.append("stale")
    threads = [threading.Thread(target=save, args=(name,)) for name in ("b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("stale") == 1 and repo.version("mixture", "m") in results


LOCK_HOLDER = """
import sys
from wpt_dataset.repo import Repo
with Repo(sys.argv[1])._lock:
    print("held", flush=True)
    sys.stdin.readline()
"""


def test_the_lock_holds_across_processes_and_a_crashed_holder_leaves_none(repo):
    # wpt-web and wpt-dataset are two processes: one waits while the other holds the lock, and a holder that
    # dies (killed, as a crash would end it) releases it with its process
    def holder():
        process = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(repo.root)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        assert process.stdout.readline() == "held\n"
        return process

    def waiting():
        got = threading.Event()

        def take():
            with repo._lock:
                got.set()
        threading.Thread(target=take, daemon=True).start()
        return got

    process = holder()
    got = waiting()
    assert not got.wait(0.5)  # the other process holds it
    process.stdin.write("\n")
    process.stdin.flush()
    assert got.wait(30) and process.wait(30) == 0

    process = holder()
    got = waiting()
    assert not got.wait(0.5)
    process.kill()
    process.wait(30)
    assert got.wait(30)
    process.stdout.close()
    process.stdin.close()


def test_a_save_keeps_the_layout_where_the_meaning_did_not_change(kit, repo):
    path = repo.root / "data" / "recipes" / "mixtures" / "m.json"
    text = ('{"schema": 1,\n "pools": [\n    {"name": "a", "select": "a",   "weight": 0.5},\n'
            '    {"name": "b", "select": "b", "weight": 0.5}\n ]\n}\n')
    path.write_text(text, encoding="utf-8")
    document = json.loads(text)
    version = repo.version("mixture", "m")
    assert repo.save("mixture", "m", document, version) == version and path.read_text() == text  # nothing changed

    document["pools"][1]["select"] = "fr"
    repo.save("mixture", "m", {"pools": document["pools"], "schema": 1}, version)  # key order is no change either
    assert path.read_text() == text.replace('"select": "b"', '"select": "fr"')


def test_conversations_hold_line_and_paragraph_separators(kit, repo, store):
    sid = own_source(repo, store, {"notes.txt": "text"})
    said = "one two three\x85four"
    conversation = {"schema": 3, "messages": [{"role": "user", "content": [{"type": "text", "text": said}]},
                                              {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]}
    version = repo.save_conversation(sid, "sft/talks.jsonl", None, conversation, "talk", repo.source_version(sid),
                                     store)
    version = repo.save_conversation(sid, "sft/talks.jsonl", None, conversation, "again", version, store)
    _, data = repo.read_source_file(sid, "sft/talks.jsonl", store)
    assert data.count(b"\n") == 2 and b"\\u2028" in data
    assert repo.conversations(sid, "sft/talks.jsonl", store)[1] == [conversation, conversation]


def test_an_assistant_message_may_hold_no_block_but_no_other_message_may(kit):
    # a model that closes its turn at once is recorded as it happened (maintainer's decision, 2026-10-22);
    # the attempts files go through the same parse as a source's chat files
    from wpt_dataset import conversations
    from wpt_dataset.errors import InvalidFile

    def line(role):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": role, "content": []}]
        return (json.dumps({"schema": 3, "messages": messages}) + "\n").encode()

    parsed = conversations.parse("record", "attempts.jsonl", line("assistant"))
    assert parsed[0]["messages"][1] == {"role": "assistant", "content": []}
    for role in ("system", "user", "tool"):
        with pytest.raises(InvalidFile, match="should be non-empty"):
            conversations.parse("record", "attempts.jsonl", line(role))


def test_line_and_paragraph_separators_are_escaped_even_in_values_a_save_keeps(kit, repo, store):
    sid = own_source(repo, store, {"notes.txt": "text"})
    path = repo.root / "data" / "sources" / sid / "source.json"
    text = path.read_text(encoding="utf-8").replace("A. Person, notes", "A. Person\u2028notes\x85\u2029")
    path.write_text(text, encoding="utf-8")  # as a file written by hand may hold them raw
    repo.set_tags(sid, "notes.txt", ["sft"], repo.version("source", sid))  # the attribution does not change
    saved = path.read_text(encoding="utf-8")
    assert "\u2028" not in saved and "\x85" not in saved and "\u2029" not in saved
    assert '"A. Person\\u2028notes\\u0085\\u2029"' in saved
    assert repo.source(sid).data["licenses"][0]["attribution"] == "A. Person\u2028notes\x85\u2029"


def test_an_empty_license_text_is_allowed_only_while_the_source_is_not_admitted(kit, repo, store):
    from wpt_dataset import build, conversations, tasks
    from wpt_dataset.errors import InvalidFile

    document = empty_source("notes", license="CC0-1.0", attribution="A. Person, notes", text="", by="A. Person",
                            formats={"notes.txt": "text"})
    sid = repo.create_source("mine", "notes", document, {}, "first", store)
    assert repo.check() == [] and repo.source(sid).unwritten
    admitted = {**document, "decision": {"decision": "admitted", "by": "A. Person", "date": "2026-10-18"}}
    with pytest.raises(InvalidFile):
        repo.save("source", sid, admitted, repo.version("source", sid))
    refused = {**document, "decision": {"decision": "refused", "by": "A. Person", "date": "2026-10-22", "reason": "r"}}
    repo.save("source", sid, refused, repo.version("source", sid))
    assert repo.check() == [] and repo.source(sid).unwritten
    with pytest.raises(RuleViolation, match="license text first"):
        repo.create_source("mine", "other", document, {"notes.txt": b"x\n"}, "first", store)

    notes = []  # every reader leaves it out with a note naming it
    assert build.gather(repo, store, notes.append)[:2] == ([], {})
    assert conversations.read(repo, store, "sft", log=notes.append) == ([], {})
    assert tasks.read(repo, store, "grpo", 0, log=notes.append) == ([], {})
    assert [n for n in notes if n == repo.source(sid).note] == [repo.source(sid).note] * 3 and sid in notes[0]


def test_a_new_source_that_fails_leaves_nothing_and_keeps_its_error(kit, repo, store, monkeypatch):
    document = empty_source("notes", license="CC0-1.0", attribution="A. Person, notes", text="CC0 1.0 Universal\n",
                            by="A. Person", formats={"notes.txt": "text"})
    with pytest.raises(NotText):  # the error that stopped it, not one from the clean-up
        repo.create_source("mine", "notes", document, {"notes.txt": b"\xff\xfe"}, "first", store)
    assert repo.source_ids() == []

    rmtree = service.shutil.rmtree

    def broken(path, *args, **kwargs):  # the source's folder cannot be removed; temporary folders can
        if Path(path).parent == repo.root / "data" / "sources":
            raise PermissionError(f"cannot remove {path}")
        rmtree(path, *args, **kwargs)
    monkeypatch.setattr(service.shutil, "rmtree", broken)
    with pytest.raises(NotText) as raised:  # a clean-up that fails too is a note on the same error
        repo.create_source("mine", "notes", document, {"notes.txt": b"\xff\xfe"}, "first", store)
    assert "could not be removed" in raised.value.__notes__[0]


def test_conversations_and_tasks_are_written_only_in_an_empty_source(kit, repo, store, tmp_path):
    upstream = kit.git_project(tmp_path / "upstream", symlinks={})
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": upstream[0],
                                                            "revision": upstream[1]})
    conversation = {"schema": 3, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    with pytest.raises(RuleViolation, match="is a git source"):
        repo.save_conversation(sid, "talks.jsonl", None, conversation, "talk", repo.source_version(sid), store)
    with pytest.raises(RuleViolation, match="is a git source"):
        repo.save_task(sid, "tasks/hello", TASK, "", {}, "task", repo.source_version(sid), store)
    assert repo.source(sid).patches == []


def test_a_task_save_writes_only_what_changed_and_keeps_other_files_as_they_are(kit, repo, store, tmp_path):
    (repo.root / "data" / "images" / "tiny").mkdir(parents=True)
    (repo.root / "data" / "images" / "tiny" / "Dockerfile").write_text("FROM debian:trixie-slim\n")
    sid = own_source(repo, store, {"tasks/**": "task", "notes.txt": "text"})
    task_text = json.dumps(TASK, indent=4) + "\n"  # a layout of the person's own

    def task(folder):
        hello = folder / "tasks" / "hello"
        (hello / "files").mkdir(parents=True)
        (hello / "task.json").write_text(task_text)
        (hello / "check").write_text("#!/bin/sh\necho 1\n")
        (hello / "files" / "run.sh").write_text("#!/bin/sh\necho hello\n")
        (hello / "files" / "run.sh").chmod(0o755)
        os.symlink("README", hello / "files" / "latest")
        (hello / "files" / "README").write_text("say hello\n")
    edit(repo, store, sid, tmp_path, "hello", task)

    version, tasks = repo.tasks(sid, store)
    found = tasks["tasks/hello"]
    assert found["kept"] == {"files/latest": "a symbolic link"} and "latest" not in found["files"]
    files = {**found["files"], "README": "say hello, then stop\n", "run.sh": "#!/bin/sh\necho hello world\n"}
    version = repo.save_task(sid, "tasks/hello", found["task"], found["check"], files, "readme", version, store)

    patch = repo.source(sid).patches[-1].data
    assert b"README" in patch and b"run.sh" in patch and b"task.json" not in patch and b"check" not in patch
    _, entries = repo.source_files(sid, store)
    assert entries["tasks/hello/files/run.sh"]["mode"] == "100755"  # the execute bit stays
    assert entries["tasks/hello/files/latest"]["mode"] == "120000"  # the link stays a link
    _, task_json = repo.read_source_file(sid, "tasks/hello/task.json", store)
    assert task_json.decode() == task_text

    with pytest.raises(RuleViolation, match="latest is a symbolic link"):
        repo.save_task(sid, "tasks/hello", found["task"], found["check"], {**files, "latest": "text"}, "link",
                       version, store)
    with pytest.raises(RuleViolation, match="below the source's root"):
        repo.save_task(sid, "", TASK, "", {}, "root", version, store)
    assert len(repo.source(sid).patches) == 2

    # a save with nothing to change records nothing, reads the source once, and still refuses a stale version
    read = []
    source = repo.source
    repo.source = lambda *args: read.append(args) or source(*args)
    assert repo.save_task(sid, "tasks/hello", found["task"], found["check"], files, "same", version, store) == version
    assert len(read) == 1 and len(repo.source(sid).patches) == 2
    del repo.source
    stale = repo.source_version(sid)
    repo.set_tags(sid, "notes.txt", ["notes"], repo.version("source", sid))
    with pytest.raises(StaleFile):
        repo.save_task(sid, "tasks/hello", found["task"], found["check"], files, "same", stale, store)
    assert len(repo.source(sid).patches) == 2


def test_an_image_s_pool_of_tools_is_written_and_read_after_the_check(repo):
    # the pool (data/images/<name>/tools.json) is written like any file
    # people write, and a tool taking the name of one every task has is
    # refused before anything is written
    repo.save("dockerfile", "tiny", "FROM debian:trixie-slim\n", "")
    fetch = {"name": "fetch", "description": "Fetch a page's text.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
             "names": ["get_page"]}
    assert repo.image_tools("tiny") == []
    repo.save("image-tools", "tiny", {"schema": 1, "tools": [fetch]}, "")
    assert repo.image_tools("tiny") == [fetch] and repo.problems("image-tools", "tiny") == []
    version = repo.version("image-tools", "tiny")
    with pytest.raises(InvalidFile, match="the tool 'bash' is declared twice, or takes the name of a tool every task"):
        repo.save("image-tools", "tiny", {"schema": 1, "tools": [{**fetch, "name": "bash"}]}, version)
    assert repo.image_tools("tiny") == [fetch]
